"""Build a Python-source overlay while reusing the pinned wheel's native libraries."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
INSTALLED = Path('/mnt/ondeviceinfer/conda-envs/vllm-0.11.2/lib/python3.11/site-packages/vllm')
COMMIT = '275de34170654274616082721348b7edd9741d32'


def patch_scheduler(text):
    def replace(a,b):
        nonlocal text
        if text.count(a) != 1:
            raise RuntimeError('source mismatch: '+a[:100])
        text = text.replace(a,b)
    replace('from vllm.config import VllmConfig', 'from kv_sched.runtime import SchedulerHooks\nfrom vllm.config import VllmConfig')
    replace('        self.use_pp = self.parallel_config.pipeline_parallel_size > 1',
            '        self.use_pp = self.parallel_config.pipeline_parallel_size > 1\n        self._kv = SchedulerHooks(self)')
    replace('    def schedule(self) -> SchedulerOutput:\n', '    def schedule(self) -> SchedulerOutput:\n        self._kv.begin()\n')
    replace('        if not preempted_reqs:\n            while self.waiting',
            '        if not preempted_reqs:\n            if token_budget > 0 and len(self.running) < self.max_num_running_reqs:\n                self._kv.reorder()\n            while self.waiting')
    replace('                    if self.policy == SchedulingPolicy.PRIORITY:\n                        preempted_req = max(',
            '                    if self.policy == SchedulingPolicy.PRIORITY or self._kv.config.preemption != "default":\n                        if self.policy == SchedulingPolicy.PRIORITY:\n                            preempted_req = max(')
    # Indent the original priority selection; custom policies share its rollback.
    replace('                            self.running,\n                            key=lambda r: (r.priority, r.arrival_time),\n                        )\n                        self.running.remove(preempted_req)',
            '                                self.running,\n                                key=lambda r: (r.priority, r.arrival_time),\n                            )\n                        else:\n                            preempted_req = self._kv.select(request, num_new_tokens)\n                            if preempted_req is None:\n                                break\n                        victim_index = self.running.index(preempted_req)\n                        self.running.remove(preempted_req)')
    replace('                            req_index -= 1\n                    else:',
            '                        if victim_index < req_index:\n                            req_index -= 1\n                    else:')
    replace('                            scheduled_running_reqs.remove(preempted_req)',
            '                            self._kv.counters["reserved_victim_rollbacks"] += 1\n                            self._kv.rollback_cache(preempted_req)\n                            scheduled_running_reqs.remove(preempted_req)')
    replace('                    self.kv_cache_manager.free(preempted_req)',
            '                    self._kv.before_preempt(preempted_req)\n                    self.kv_cache_manager.free(preempted_req)')
    replace('                # KVTransfer: the connector uses this info to determine',
            '                self._kv.admitted(request, num_new_local_computed_tokens, num_external_computed_tokens)\n\n                # KVTransfer: the connector uses this info to determine')
    replace('        with record_function_or_nullcontext("schedule: update_after_schedule"):',
            '        self._kv.stage(scheduler_output)\n        with record_function_or_nullcontext("schedule: update_after_schedule"):')
    replace('        sampled_token_ids = model_runner_output.sampled_token_ids',
            '        self._kv.committed(scheduler_output, model_runner_output)\n        sampled_token_ids = model_runner_output.sampled_token_ids')
    compile(text, '<patched scheduler>', 'exec')
    return text


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--offline',action='store_true')
    ap.add_argument('--compat',action='store_true')
    args=ap.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    records = {}
    # Critical baseline files must match the pinned upstream before experiments.
    critical = ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
                'distributed/kv_transfer/kv_connector/v1/offloading_connector.py',
                'v1/kv_offload/lru_manager.py', 'v1/kv_offload/worker/cpu_gpu.py']
    for rel in critical:
        local = (INSTALLED/rel).read_bytes()
        url = f'https://raw.githubusercontent.com/vllm-project/vllm/{COMMIT}/vllm/{rel}'
        if args.offline:
            existing=json.loads((DATA/'environment_manifest.json').read_text())
            if existing['source_validation'][rel] != hashlib.sha256(local).hexdigest():
                raise RuntimeError('installed source changed since upstream validation: '+rel)
        else:
            upstream = urllib.request.urlopen(url,timeout=30).read()
            if local != upstream:
                raise RuntimeError('installed source differs from pinned upstream: '+rel)
        records[rel] = hashlib.sha256(local).hexdigest()
    for flavor in ['stock','optimized']:
        target = DATA/'vendor'/flavor/'vllm'
        if not target.exists():
            for p in INSTALLED.rglob('*'):
                if '__pycache__' in p.parts:
                    continue
                q = target/p.relative_to(INSTALLED)
                if p.is_dir(): q.mkdir(parents=True,exist_ok=True)
                elif p.suffix == '.py': shutil.copy2(p,q)
                else: q.symlink_to(p)
    rel = 'v1/core/sched/scheduler.py'
    original = (INSTALLED/rel).read_text()
    if args.compat:
        from compat import apply
        for flavor in ['stock','optimized']:
            apply(DATA/'vendor'/flavor/'vllm',ROOT/'patches/upstream-compat-v0.11.2.patch')
        original=(DATA/'vendor/optimized/vllm'/rel).read_text()
    modified = patch_scheduler(original)
    (DATA/'vendor/optimized/vllm'/rel).write_text(modified)
    patch = ''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
                                       fromfile='a/vllm/'+rel,tofile='b/vllm/'+rel))
    (ROOT/'patches/scheduler-v0.11.2.patch').write_text(patch)
    manifest = {'commit':COMMIT,'source_validation':records,'native_wheel':str(INSTALLED),
                'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                'binary_strategy':'unaltered native files symlinked from existing 0.11.2 wheel; private Python overlay'}
    if args.compat:
        manifest['compat_patch_sha256']=hashlib.sha256((ROOT/'patches/upstream-compat-v0.11.2.patch').read_bytes()).hexdigest()
        manifest['compat_references']=json.loads((ROOT/'configs/upstream_compat.json').read_text())
    (DATA/'environment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
