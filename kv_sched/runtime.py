"""Hooks for exactly vLLM 0.11.2. Decisions never own KV resources."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from itertools import islice
import json
import os
import time

from .policy import (Config, CostModel, IntervalLedger, Victim, choose_victim,
                     eligible, percentiles, reorder_segments)


def gpu_probe(manager, request):
    if not manager.enable_caching or request.skip_reading_prefix_cache:
        return 0
    _, tokens = manager.coordinator.find_longest_cache_hit(request.block_hashes, request.num_tokens-1)
    return tokens


def cpu_probe(connector, request, gpu, kv_bytes=114688):
    """Mirror native lookup without touch/prepare_load; returns end, bytes, reason."""
    if connector is None or not hasattr(connector, "connector_scheduler"):
        return gpu, 0, "unsupported"
    cs = connector.connector_scheduler
    if cs is None or cs.manager.__class__.__name__ != "LRUOffloadingManager":
        return gpu, 0, "unsupported"
    size = cs.offloaded_block_size
    n = request.num_tokens // size
    if size*n-gpu < size:
        return gpu, 0, "too_short"
    start = gpu//size
    hashes = list(cs._get_block_hashes(request, start_idx=start, end_idx=n))
    hits = cs.manager.lookup(hashes)
    end = (start+hits)*size
    if hits == 0:
        first = cs.manager.blocks.get(hashes[0]) if hashes else None
        return gpu, 0, "pending" if first is not None and not first.is_ready else "miss"
    if end-gpu < size:
        return gpu, 0, "too_short"
    return end, (end-gpu)*kv_bytes, "ready"


def reclaimable(manager, request):
    seen = set()
    for group in manager.coordinator.get_blocks(request.request_id):
        for block in group:
            if block.block_id not in seen and not block.is_null and block.ref_cnt == 1:
                seen.add(block.block_id)
    return len(seen)


def invalidate_unexecuted_cache(manager, request):
    """Undo optimistic cache registration when scheduled work is rolled back.

    allocate_slots registers full-block hashes before the model has executed.
    Only blocks strictly below the committed progress may remain cache hits.
    """
    valid_blocks = request.num_computed_tokens // manager.block_size
    invalidated = 0
    for group in manager.coordinator.get_blocks(request.request_id):
        for block in group[valid_blocks:]:
            if block.block_hash is not None:
                if block.ref_cnt != 1:
                    raise RuntimeError('unexecuted block unexpectedly shared')
                manager.block_pool._maybe_evict_cached_block(block)
                invalidated += 1
    return invalidated


class SchedulerHooks:
    def __init__(self, scheduler):
        self.s = scheduler
        self.config = Config(**json.loads(os.environ.get("KV_SCHED_CONFIG", "{}")))
        if (scheduler.scheduler_config.async_scheduling or scheduler.use_pp
                or scheduler.num_spec_tokens or scheduler.is_encoder_decoder
                or scheduler.lora_config):
            raise ValueError("experiment supports synchronous TP=PP=1 dense text only")
        if scheduler.parallel_config.tensor_parallel_size != 1:
            raise ValueError("TP must be 1")
        groups = scheduler.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or groups[0].kv_cache_spec.__class__.__name__ != 'FullAttentionSpec':
            raise ValueError('single FullAttention KV group required')
        self.model = CostModel(self.config.cost_model)
        self.reset()

    def reset(self):
        self.counters = Counter()
        self.timings = {k: [] for k in ["schedule_ms", "probe_gpu_ms", "probe_cpu_ms", "reorder_ms", "victim_ms", "recovery_ms"]}
        self.events = []
        self.ledgers = {}
        self.probed = {}
        self.admissions = {}
        self.pending = {}
        self.load_predictions = {}
        self.preempted_since = {}
        self.step = 0

    def begin(self):
        self.started = time.perf_counter_ns()
        self.step += 1

    def probe(self, r, use_cpu):
        t = time.perf_counter_ns()
        g = gpu_probe(self.s.kv_cache_manager, r)
        self.timings["probe_gpu_ms"].append((time.perf_counter_ns()-t)/1e6)
        self.counters["gpu_probe_calls"] += 1
        c, b = g, 0
        if use_cpu and self.config.probe_cpu:
            t = time.perf_counter_ns()
            c, b, reason = cpu_probe(self.s.connector, r, g)
            self.timings["probe_cpu_ms"].append((time.perf_counter_ns()-t)/1e6)
            self.counters["cpu_probe_"+reason] += 1
        return g, c, b

    def reorder(self):
        cfg = self.config
        if cfg.admission == "fcfs" or cfg.window < 2:
            return
        if self.s.policy.value != "fcfs" or not self.s.kv_cache_manager.enable_caching:
            self.counters["gate_unsupported"] += 1
            return
        if self.s.kv_cache_manager.usage < cfg.threshold:
            self.counters["gate_low_usage"] += 1
            return
        queue = self.s.waiting
        window = list(islice(queue, cfg.window))
        if len(window) < 2:
            return
        t = time.perf_counter_ns()
        now = time.time()
        scores, waits = {}, {}
        tier = cfg.admission == "tier" and self.model.calibrated
        if cfg.admission == "tier" and not tier:
            self.counters["fallback_uncalibrated_gpu"] += 1
        for r in window:
            if not eligible(r):
                continue
            g, c, b = self.probe(r, tier)
            self.probed.setdefault(r.request_id, (g, c, self.step))
            if tier:
                self.load_predictions[r.request_id] = c-g
            wait = max(0, (now-r.arrival_time)*1000)
            waits[r.request_id] = wait
            saving = self.model.prefill(r.num_tokens)-self.model.path(r.num_tokens, g, c, b) if tier else g
            scores[r.request_id] = saving + (cfg.aging*wait if tier and cfg.fairness else 0)
        out = reorder_segments(window, lambda r: scores[r.request_id],
                               lambda r: cfg.fairness and cfg.max_wait_ms > 0 and waits[r.request_id] >= cfg.max_wait_ms)
        self.counters["reorder_calls"] += 1
        if out != window:
            # Replace only a bounded deque prefix; preserve all other positions.
            for _ in window:
                queue.pop_request()
            for r in reversed(out):
                queue.prepend_request(r)
            self.counters["reorder_changes"] += 1
            if cfg.trace:
                self.events.append({"event": "reorder", "step": self.step,
                                    "before": [r.request_id for r in window], "after": [r.request_id for r in out]})
        self.timings["reorder_ms"].append((time.perf_counter_ns()-t)/1e6)

    def select(self, request, new_tokens):
        t = time.perf_counter_ns()
        m = self.s.kv_cache_manager
        needed = m.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(request.num_computed_tokens+new_tokens+self.s.num_lookahead_tokens, self.s.max_model_len),
            new_computed_blocks=m.empty_kv_cache_blocks.blocks, num_encoder_tokens=0)
        required = max(1, needed-m.block_pool.get_num_free_blocks())
        candidates = []
        cs = self.s.connector.connector_scheduler if self.s.connector else None
        for r in self.s.running:
            protected = cs and (cs._reqs_being_loaded.get(r.request_id) or
                        (cs._reqs_being_stored.get(r.request_id) and not getattr(self.s.connector,'supports_preemption_flush',False)))
            if protected:
                self.counters["victim_pending_excluded"] += 1
                continue
            b = reclaimable(m, r)
            computed = r.num_computed_tokens
            cost = self.model.prefill(computed)
            if (self.config.preemption == "restore" and self.model.calibrated
                    and self.config.probe_cpu and cs is not None):
                c, payload, _ = cpu_probe(self.s.connector, r, 0)
                chunk = cs.offloaded_block_size
                c = min(c, computed//chunk*chunk)
                cost = self.model.path(computed, 0, c, c*114688)
            candidates.append(Victim(r, b, computed, r.num_tokens, len(r.output_token_ids), r.max_tokens, cost))
        victim = choose_victim(candidates, self.config.preemption, required,
                               self.config.resource_aware, self.config.progress_aware)
        if victim is not None:
            self.counters['non_tail_victims'] += int(victim is not self.s.running[-1])
        self.counters["victim_calls"] += 1
        self.timings["victim_ms"].append((time.perf_counter_ns()-t)/1e6)
        return victim

    def before_preempt(self, r):
        if self.s.connector:
            cs = self.s.connector.connector_scheduler
            if ((cs._reqs_being_stored.get(r.request_id) or cs._reqs_being_loaded.get(r.request_id))
                    and not getattr(self.s.connector,'supports_preemption_flush',False)):
                raise RuntimeError("CPU_PENDING_PREEMPTION_UNSAFE: native 0.11.2 releases unacknowledged transfer sources")
        self.counters["preemptions"] += 1
        self.preempted_since[r.request_id] = time.perf_counter()
        self.counters["computed_at_preemption"] += r.num_computed_tokens
        self.events.append({"event": "preempt", "id": r.request_id, "step": self.step,
                            "computed": r.num_computed_tokens,
                            "blocks": reclaimable(self.s.kv_cache_manager, r)})

    def rollback_cache(self, request):
        self.counters['rolled_back_cache_blocks'] += invalidate_unexecuted_cache(self.s.kv_cache_manager, request)

    def admitted(self, r, gpu, cpu):
        self.counters["gpu_hit_tokens"] += gpu
        self.counters["cpu_matched_tokens"] += cpu
        prediction = self.load_predictions.pop(r.request_id, None)
        if prediction is not None and prediction != cpu:
            self.counters["cpu_prediction_mismatches"] += 1
        if r.request_id not in self.admissions:
            self.admissions[r.request_id] = {"wait_ms": max(0, (time.time()-r.arrival_time)*1000),
                                              "gpu": gpu, "cpu": cpu}
            if r.request_id in self.probed:
                self.counters["prefix_loss_tokens"] += max(0, self.probed[r.request_id][0]-gpu)
        if cpu:
            self.pending[r.request_id] = cpu
        if self.config.trace:
            self.events.append({"event": "admit", "id": r.request_id, "gpu": gpu, "cpu": cpu, "step": self.step})

    def stage(self, output):
        meta = output.kv_connector_metadata
        if meta is not None and hasattr(meta, 'reqs_to_store'):
            self.counters['d2h_submitted_bytes'] += sum(len(spec[0].block_ids)*16*114688 for spec in meta.reqs_to_store.values())
            self.counters['h2d_submitted_bytes'] += sum(len(spec[1].block_ids)*16*114688 for spec in meta.reqs_to_load.values())
        self.execution = {rid: (self.s.requests[rid].num_computed_tokens,
                               self.s.requests[rid].num_computed_tokens+n,
                               self.s.requests[rid].num_prompt_tokens)
                          for rid, n in output.num_scheduled_tokens.items()}
        self.timings["schedule_ms"].append((time.perf_counter_ns()-self.started)/1e6)

    def committed(self, output, model_output):
        failed = set()
        co = model_output.kv_connector_output
        if co:
            if co.invalid_block_ids:
                raise RuntimeError('invalid KV load: run is not valid for performance measurement')
            for rid in co.finished_recving or []:
                n = self.pending.pop(rid, 0)
                self.counters["cpu_loaded_tokens"] += n
                self.counters["h2d_bytes"] += n*114688
        valid = model_output.req_id_to_index
        for rid, (a, b, prompt) in self.execution.items():
            if rid not in valid:
                self.counters["unconfirmed_executions"] += 1
                continue
            ledger = self.ledgers.setdefault(rid, IntervalLedger())
            if rid in self.preempted_since:
                self.timings['recovery_ms'].append((time.perf_counter()-self.preempted_since.pop(rid))*1000)
            self.counters["executed_tokens"] += b-a
            self.counters["recomputed_tokens"] += ledger.commit(a, b)
            self.counters["prefill_executed_tokens"] += max(0, min(b, prompt)-a)

    def export(self):
        return {"config": asdict(self.config), "cost_units": "ms" if self.model.calibrated else "token_proxy",
                "counters": dict(self.counters), "timings": {k: percentiles(v) for k,v in self.timings.items()},
                "admissions": self.admissions, "events": self.events, "steps": self.step}
