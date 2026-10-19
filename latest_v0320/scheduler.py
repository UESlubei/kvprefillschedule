"""Admission-only extension of native 0.32.0 sync and async schedulers.

Never owns blocks or changes allocation, running/holding queues or preemption.
"""
from __future__ import annotations
from collections import Counter
from itertools import islice
import json
import os
import time
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_offload.base import make_offload_key
from .common import DATA, KV_BYTES, dump, percentiles

def eligible(r):
    return (r.status.name=='WAITING' and r.num_computed_tokens==0
            and r.num_preemptions==0 and not r.skip_reading_prefix_cache
            and not getattr(r, 'num_output_placeholders', 0)
            and not getattr(r, 'num_stale_output_tokens', 0)
            and not getattr(r, 'resumable', False))

def gpu_probe(manager, request):
    if not manager.prefix_cache_lookup_enabled(request): return 0
    # The native public lookup can emit events. The coordinator lookup does not.
    _, tokens, _ = manager.coordinator.find_longest_cache_hit(request.block_hashes, request.num_tokens-1)
    return tokens

def cpu_probe(connector, request, gpu):
    if connector is None: return gpu, 0, 'disabled'
    cs = getattr(connector, 'connector_scheduler', None)
    if cs is None or type(cs.manager).__name__!='CPUOffloadingManager':
        raise ValueError('Tier admission requires the native CPU manager')
    if type(cs.manager._policy).__name__!='LRUCachePolicy':
        raise ValueError('Only the audited read-only LRU lookup is supported')
    configs = cs.config.kv_group_configs
    if len(configs)!=1 or configs[0].sliding_window_size_in_chunks is not None:
        raise ValueError('Tier probe supports one FullAttention group')
    cfg = configs[0]; chunk = cfg.tokens_per_chunk
    end_limit = request.num_tokens//chunk*chunk
    if end_limit-gpu < chunk: return gpu, 0, 'too_short'
    # Reproduce RequestOffloadState.update_offload_keys without creating state.
    keys = [make_offload_key(h, cfg.group_idx) for h in
            request.block_hashes[cfg.hashes_per_chunk-1::cfg.hashes_per_chunk]]
    end = gpu//chunk*chunk
    hit_keys=[]
    for key in keys[gpu//chunk:end_limit//chunk]:
        block = cs.manager._policy.get(key)  # LRU.get is a plain dict read.
        if block is None: break
        if not block.is_ready:
            # Native _maximal_prefix_lookup defers the whole lookup if any
            # matching chunk is pending, even after an earlier ready prefix.
            return gpu, 0, 'pending'
        hit_keys.append(key)
        end += chunk
    loading=getattr(cs,'_chunks_being_loaded',None)
    if loading and any(key in loading for key in hit_keys):return gpu,0,'inflight'
    return (end, (end-gpu)*KV_BYTES, 'ready') if end-gpu>=chunk else (gpu, 0, 'miss')

def reorder_segments(rows, scores, overdue):
    result = list(rows); i=0
    while i<len(rows):
        if not eligible(rows[i]): i+=1; continue
        j=i+1
        while j<len(rows) and eligible(rows[j]): j+=1
        result[i:j]=sorted(rows[i:j], key=lambda r:(not overdue[r.request_id],
            r.arrival_time if overdue[r.request_id] else -scores[r.request_id]))
        i=j
    return result

class CostModel:
    def __init__(self, path):
        self.data = json.loads(open(path).read())
        assert self.data['vllm']=='0.32.0+cu129' and self.data['kv_bytes_per_token']==KV_BYTES

    def prefill(self, length, cached=0):
        # Nearest context, then piecewise linear query interpolation. Batch=1
        # is the request-level ranking approximation, not a batch latency claim.
        q=max(0,length-cached)
        if not q: return 0.
        rows=[r for r in self.data['prefill'] if r['batch']==1]
        context=min({r['cached'] for r in rows},key=lambda c:abs(c-cached))
        points=sorted((r['length']-r['cached'],r['ms']) for r in rows if r['cached']==context)
        for (x0,y0),(x1,y1) in zip(points,points[1:]):
            if x0<=q<=x1: return y0+(y1-y0)*(q-x0)/(x1-x0)
        x,y=min(points,key=lambda p:abs(p[0]-q))
        return y*q/x

    def transfer(self, size):
        if not size: return 0.
        h=self.data['h2d']
        return h['fixed_ms'] + size/h['bytes_per_ms']

class AdmissionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cfg = json.loads(os.environ.get('KV032_POLICY','{}'))
        self.kv_mode = self.kv_cfg.get('admission','off')
        assert self.kv_mode in {'off','observe','gpu','tier'}
        assert self.policy.value=='fcfs' and self.parallel_config.tensor_parallel_size==1
        groups = self.kv_cache_config.kv_cache_groups
        assert len(groups)==1 and type(groups[0].kv_cache_spec).__name__=='FullAttentionSpec'
        self.kv_cost = CostModel(self.kv_cfg.get('cost_model',str(DATA/'cost_models/native.json'))) if self.kv_mode=='tier' else None
        self.kv_counts=Counter(); self.kv_times=[]; self.kv_probe_times=[]
        self.kv_first_probe={}; self.kv_local_hits={}; self.kv_pending={}; self.kv_ledgers={}
        self.kv_admissions={}; self.kv_initial={}; self.kv_reorders=[]
        self.kv_measured_exported=False
        dump(self.kv_cfg['output']+'.config.json',{
            'scheduler_class':type(self).__name__,
            'async':self.scheduler_config.async_scheduling,
            'reserve_full_isl':self.scheduler_config.scheduler_reserve_full_isl,
            'watermark':self.scheduler_config.watermark,
            'gpu_blocks':self.kv_cache_config.num_blocks,
            'max_num_seqs':self.scheduler_config.max_num_seqs,
            'compilation':str(self.vllm_config.compilation_config) if hasattr(self,'vllm_config') else None})

    def _kv_reorder(self):
        if self.kv_mode=='off': return
        q=self.waiting; window=list(islice(q,self.kv_cfg.get('window',8)))
        self.kv_counts['max_waiting']=max(self.kv_counts['max_waiting'],len(q))
        if len(window)<2 or self.kv_cache_manager.usage<self.kv_cfg.get('threshold',0): return
        now=time.time(); scores={}; overdue={}; begin=time.perf_counter()
        for r in window:
            if not eligible(r): continue
            g=gpu_probe(self.kv_cache_manager,r)
            self.kv_first_probe.setdefault(r.request_id,g)
            c,b,reason=(cpu_probe(self.connector,r,g) if self.kv_mode=='tier' else (g,0,'unused'))
            self.kv_counts['gpu_probe_calls']+=1
            if self.kv_mode=='tier': self.kv_counts['cpu_probe_'+reason]+=1
            wait=max(0,(now-r.arrival_time)*1000)
            overdue[r.request_id]=wait>=self.kv_cfg.get('max_wait_ms',2000)
            if self.kv_cost:
                remaining=self.kv_cost.prefill(r.num_tokens,c)+self.kv_cost.transfer(b)
                scores[r.request_id]=self.kv_cost.prefill(r.num_tokens)-remaining+.05*wait
            else: scores[r.request_id]=g
        self.kv_probe_times.append((time.perf_counter()-begin)*1000)
        self.kv_counts['window_evaluations']+=1
        if self.kv_mode=='observe': return
        ordered=reorder_segments(window,scores,overdue)
        if ordered!=window:
            for _ in window:q.pop_request()
            for r in reversed(ordered):q.prepend_request(r)
            self.kv_counts['reorder_changes']+=1
            if len(self.kv_reorders)<128:
                self.kv_reorders.append({'before':[r.request_id for r in window],
                    'after':[r.request_id for r in ordered]})

    def _get_local_prefix_cache_hit(self, request):
        result=super()._get_local_prefix_cache_hit(request)
        if self.kv_mode!='off': self.kv_local_hits[request.request_id]=result[1]
        return result

    def schedule(self, *args, **kwargs):
        ts=time.perf_counter()
        if self.kv_mode!='off' and not self.kv_initial and any('measure-' in r.request_id for r in self.waiting):
            self.kv_counts.clear();self.kv_times.clear();self.kv_probe_times.clear()
            self.kv_admissions.clear();self.kv_ledgers.clear();self.kv_first_probe.clear()
            self.kv_reorders.clear()
            from vllm import SamplingParams
            from vllm.v1.request import Request
            from vllm.v1.core.kv_cache_utils import get_request_block_hasher
            from vllm.utils.hashing import get_hash_fn_by_name
            from .common import prompt
            hasher=get_request_block_hasher(self.hash_block_size,get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo))
            for base in ([1000,2000] if self.connector else [1000]):
                for group in range(4):
                    r=Request('readonly-initial',prompt(base+group,2048)+prompt(23456,256),SamplingParams(max_tokens=1),None)
                    r.block_hashes=hasher(r)
                    g=gpu_probe(self.kv_cache_manager,r)
                    c,b,reason=cpu_probe(self.connector,r,g) if self.connector else (g,0,'disabled')
                    self.kv_initial[str(base+group)]={'gpu':g,'cpu_end':c,'bytes':b,'reason':reason}
        self._kv_reorder()
        output=super().schedule(*args, **kwargs)
        if self.kv_mode!='off':
            # Native _update_after_schedule has advanced computed tokens here.
            intervals=[]
            for rid,n in output.num_scheduled_tokens.items():
                r=self.requests[rid]; end=r.num_computed_tokens
                intervals.append((rid,end-n,end,r.num_prompt_tokens))
                if rid not in self.kv_admissions:
                    g=self.kv_local_hits.get(rid,0)
                    self.kv_admissions[rid]={'gpu':g,'wait_ms':max(0,(time.time()-r.arrival_time)*1000)}
                    self.kv_counts['prefix_loss_tokens']+=max(0,self.kv_first_probe.get(rid,g)-g)
            self.kv_pending[id(output)]=intervals
            self.kv_times.append((time.perf_counter()-ts)*1000)
            self.kv_counts['schedule_calls']+=1
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        result=super().update_from_output(scheduler_output,model_runner_output)
        if self.kv_mode!='off':
            for rid,a,b,p in self.kv_pending.pop(id(scheduler_output),[]):
                if rid not in model_runner_output.req_id_to_index:
                    self.kv_counts['unconfirmed_intervals']+=1; continue
                ledger=self.kv_ledgers.setdefault(rid,[])
                repeat=sum(max(0,min(b,y)-max(a,x)) for x,y in ledger)
                merged=[]
                for x,y in sorted([*ledger,(a,b)]):
                    if merged and x<=merged[-1][1]: merged[-1]=(merged[-1][0],max(y,merged[-1][1]))
                    else: merged.append((x,y))
                self.kv_ledgers[rid]=merged
                self.kv_counts['executed_tokens']+=b-a
                self.kv_counts['prefill_executed_tokens']+=max(0,min(b,p)-a)
                self.kv_counts['recomputed_tokens']+=repeat
        return result

    def _kv_export(self):
        dump(self.kv_cfg['output'],{'config':self.kv_cfg,'counters':dict(self.kv_counts),
            'schedule_ms':percentiles(self.kv_times),'probe_ms':percentiles(self.kv_probe_times),
            'admissions':self.kv_admissions,'initial_state':self.kv_initial,'reorders':self.kv_reorders})

    def reset_prefix_cache(self, *args, **kwargs):
        # The benchmark requests a native reset after stopping its timer. Export
        # there rather than relying on teardown, which can force-kill a core.
        if not self.kv_measured_exported:
            self._kv_export()
            if self.kv_initial:self.kv_measured_exported=True
        return super().reset_prefix_cache(*args, **kwargs)

    def shutdown(self):
        if not Path(self.kv_cfg['output']).exists():self._kv_export()
        return super().shutdown()

class SyncAdmissionScheduler(AdmissionMixin, Scheduler): pass
class AsyncAdmissionScheduler(AdmissionMixin, AsyncScheduler): pass
