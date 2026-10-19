"""Adaptations of #28951/#29870, deferred stores, and local lifecycle fixes.

Applied identically to instrumented baseline and optimized source. These are
compatibility changes, never counted as scheduling improvements.
Local changes also cover generated history, cancellation acknowledgements,
idle completion polling, and ready-load holders blocked behind a cold head.
"""
from pathlib import Path
import difflib

INSTALLED=Path('/mnt/ondeviceinfer/conda-envs/vllm-0.11.2/lib/python3.11/site-packages/vllm')


def sources():
    result={}
    def edit(rel,replacements):
        s=(INSTALLED/rel).read_text()
        for a,b in replacements:
            if s.count(a)!=1: raise RuntimeError('compat source mismatch '+rel+' '+a[:50])
            s=s.replace(a,b)
        compile(s,rel,'exec')
        result[rel]=s
    # Correct direction/size of the skipped sub-blocks (#28951).
    edit('v1/kv_offload/worker/cpu_gpu.py',[
        ('        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor\n        src_sub_block_count = src_blocks.size * src_block_size_factor\n\n        assert (\n            src_sub_block_count\n            == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip\n        )',
         '        dst_sub_block_count = dst_blocks.size * dst_block_size_factor\n        src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor\n        assert src_blocks.size * src_block_size_factor - src_sub_blocks_to_skip == dst_sub_block_count'),
        ('        src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)\n        expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])',
         '        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)\n        expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0], skip_count=src_sub_blocks_to_skip)'),
        ('            skip_count=dst_sub_blocks_to_skip,','            skip_count=0,'),
        ('    def get_finished(self) -> list[TransferResult]:',
         '    def wait(self, job_ids: set[int]) -> None:\n        for job_id in job_ids:\n            event = self.transfer_events.get(job_id)\n            if event is not None:\n                event.synchronize()\n\n    def get_finished(self) -> list[TransferResult]:')])
    edit('v1/kv_offload/worker/worker.py',[
        ('    def get_finished(self) -> list[TransferResult]:\n        """\n        Get transfers finished since last call.\n\n        Returns:\n            A list of (job_id, success) of transfers.\n        """\n        finished = []',
         '    def wait(self, job_ids: set[int]) -> None:\n        for handler in self.handlers:\n            handler.wait(job_ids)\n\n    def get_finished(self) -> list[TransferResult]:\n        """\n        Get transfers finished since last call.\n\n        Returns:\n            A list of (job_id, success) of transfers.\n        """\n        finished = []')])
    edit('v1/core/sched/output.py',[
        ('    # KV Cache Connector metadata.','    # Backport #29870: consumed before any freed blocks are overwritten.\n    preempted_req_ids: set[str] | None = None\n\n    # KV Cache Connector metadata.')])
    edit('v1/core/sched/scheduler.py',[
        ('                if new_blocks is None:\n                    # The request cannot be scheduled.\n                    break',
         '                if new_blocks is None:\n                    # Completed loads already own GPU blocks. If no running\n                    # request can release memory, let these holders finish\n                    # before a blocked head; otherwise the engine can deadlock.\n                    ready_holders = [r for r in self.waiting\n                        if r is not request\n                        and r.status == RequestStatus.WAITING_FOR_REMOTE_KVS\n                        and r.request_id in self.finished_recving_kv_req_ids] if not self.running and self.connector is not None else []\n                    if ready_holders:\n                        self.waiting.remove_requests(ready_holders)\n                        for holder in reversed(ready_holders):\n                            self.waiting.prepend_request(holder)\n                        hooks = getattr(self, "_kv", None)\n                        if hooks is not None:\n                            hooks.counters["native_ready_promotions"] += len(ready_holders)\n                        continue\n                    # The request cannot be scheduled.\n                    break'),
        ('        self.finished_req_ids: set[str] = set()',
         '        self.finished_req_ids: set[str] = set()\n        self._finished_kv_transfer_req_ids: set[str] = set()'),
        ('        if not delay_free_blocks:\n            self._free_blocks(request)',
         '        if delay_free_blocks:\n            self._finished_kv_transfer_req_ids.add(request_id)\n        else:\n            self._free_blocks(request)'),
        ('        return len(self.finished_req_ids) > 0',
         '        return bool(self.finished_req_ids or self._finished_kv_transfer_req_ids)'),
        ('        self.kv_cache_manager.free(request)\n        del self.requests[request.request_id]',
         '        self.kv_cache_manager.free(request)\n        self._finished_kv_transfer_req_ids.discard(request.request_id)\n        self.finished_recving_kv_req_ids.discard(request.request_id)\n        self.failed_recving_kv_req_ids.discard(request.request_id)\n        del self.requests[request.request_id]'),
        ('        for req_id in kv_connector_output.finished_recving or ():\n            logger.debug("Finished recving KV transfer for request %s", req_id)\n            self.finished_recving_kv_req_ids.add(req_id)\n        for req_id in kv_connector_output.finished_sending or ():\n            logger.debug("Finished sending KV transfer for request %s", req_id)\n            assert req_id in self.requests\n            self._free_blocks(self.requests[req_id])',
         '        cs = getattr(self.connector, "connector_scheduler", None)\n        for req_id in kv_connector_output.finished_recving or ():\n            request = self.requests.get(req_id)\n            if request is None:\n                continue  # Late completion after a cancelled request was released.\n            if request.is_finished():\n                if cs is None or req_id not in cs._reqs_being_stored:\n                    self._free_blocks(request)\n            else:\n                self.finished_recving_kv_req_ids.add(req_id)\n        for req_id in kv_connector_output.finished_sending or ():\n            request = self.requests.get(req_id)\n            if request is not None and (cs is None or req_id not in cs._reqs_being_loaded):\n                self._free_blocks(request)'),
        ('                    if is_ready:\n                        request.status = RequestStatus.WAITING',
         '                    if is_ready:\n                        # Preserve generated history when async CPU load resumes a preempted request.\n                        request.status = (RequestStatus.PREEMPTED if request.num_preemptions else RequestStatus.WAITING)'),
        ('        # NOTE(Kuntai): this function is designed for multiple purposes:',
         '        scheduler_output.preempted_req_ids = {r.request_id for r in preempted_reqs}\n\n        # NOTE(Kuntai): this function is designed for multiple purposes:')])
    edit('distributed/kv_transfer/kv_connector/v1/offloading_connector.py',[
        ('        request_being_stored = req_id in self._reqs_being_stored\n        return request_being_stored, None',
         '        # Both DMA source and destination remain owned until completion.\n        transfer_pending = req_id in self._reqs_being_stored or req_id in self._reqs_being_loaded\n        return transfer_pending, None'),
        ('    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:',
         '    supports_preemption_flush = True\n\n    def handle_preemptions(self, request_ids: set[str]):\n        assert self.connector_worker is not None\n        self.connector_worker.handle_preemptions(request_ids)\n\n    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:'),
        ('        self._reqs_to_load = {}\n        return meta',
         '        self._reqs_to_load = {}\n        for rid in scheduler_output.preempted_req_ids or ():\n            hashes = self._reqs_being_stored.get(rid)\n            if hashes:\n                self.manager.complete_store(hashes)\n                hashes.clear()\n            self._next_stored_block_idx.pop(rid, None)\n        return meta'),
        ('        self._job_counter = 0',
         '        self._job_counter = 0\n        self._unsubmitted_store_jobs = []'),
        ('    def _generate_job_id(self) -> int:',
         '    def _submit_stores(self):\n        for job_id, spec in self._unsubmitted_store_jobs:\n            assert self.worker.transfer_async(job_id, spec)\n        self._unsubmitted_store_jobs.clear()\n\n    def handle_preemptions(self, request_ids: set[str]):\n        self._submit_stores()\n        for rid in request_ids:\n            self.worker.wait(self._store_jobs.get(rid, set()))\n\n    def _generate_job_id(self) -> int:'),
        ('    def start_load_kv(self, metadata: OffloadingConnectorMetadata):\n        for req_id',
         '    def start_load_kv(self, metadata: OffloadingConnectorMetadata):\n        self._submit_stores()\n        for req_id'),
        ('            self._store_jobs[req_id].add(job_id)\n            assert self.worker.transfer_async(job_id, transfer_spec)',
         '            self._store_jobs[req_id].add(job_id)\n            self._unsubmitted_store_jobs.append((job_id, transfer_spec))')])
    edit('v1/worker/gpu_model_runner.py',[
        ('        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n        with record_function_or_nullcontext("gpu_model_runner: preprocess"):',
         '        if scheduler_output.preempted_req_ids and has_kv_transfer_group():\n            connector = get_kv_transfer_group()\n            if hasattr(connector, "handle_preemptions"):\n                connector.handle_preemptions(scheduler_output.preempted_req_ids)\n\n        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n        with record_function_or_nullcontext("gpu_model_runner: preprocess"):')])
    return result


def apply(target, patch_path):
    diffs=[]
    for rel,s in sources().items():
        (target/rel).write_text(s)
        diffs.extend(difflib.unified_diff((INSTALLED/rel).read_text().splitlines(True),s.splitlines(True),
                                        fromfile='a/vllm/'+rel,tofile='b/vllm/'+rel))
    patch_path.write_text(''.join(diffs))
