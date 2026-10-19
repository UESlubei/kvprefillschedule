"""Audit actual smoke artifacts and native source, independently of console pass messages."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
PROJECT=Path(__file__).resolve().parents[1]

def main():
    env=json.loads((DATA/'manifests/environment.json').read_text())
    status=json.loads((DATA/'manifests/validation_status.json').read_text())
    assert status['state']=='completed',status
    assert env['vllm']=='0.32.0+cu129' and env['cuda_runtime']=='12.9'
    assert all(status['checks'][m]['returncode']==0 for m in ['torch','eager','offload','default'])
    runs={m:json.loads((DATA/f'raw/smoke-{m}.json').read_text()) for m in status['checks']}
    assert runs['torch']['matmul_passed'] and runs['torch']['h2d_d2h_passed']
    assert runs['torch']['capability']==[8,6]
    def counter(mode,point,name,labels):
        return sum(m.get('value',0) for m in runs[mode][point] if m['name']==name and all(m['labels'].get(k)==v for k,v in labels.items()))
    host=counter('offload','after_reuse','vllm:prompt_tokens_cached_by_source',{'source':'host'})-counter('offload','before_reuse','vllm:prompt_tokens_cached_by_source',{'source':'host'})
    h2d=counter('offload','after_reuse','vllm:kv_offload_total_bytes',{'transfer_type':'CPU_to_GPU'})-counter('offload','before_reuse','vllm:kv_offload_total_bytes',{'transfer_type':'CPU_to_GPU'})
    device=counter('eager','after_reuse','vllm:prompt_tokens_cached_by_source',{'source':'device'})-counter('eager','before_reuse','vllm:prompt_tokens_cached_by_source',{'source':'device'})
    assert host==4096 and h2d==4096*114688 and device==4096,(host,h2d,device)
    assert all(len(runs[m]['natural_outputs'])==2 for m in ['eager','offload','default'])
    source=Path(env['source']);tag_source=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/plans/latest_vllm_20261019')
    matches={}
    for rel in ['v1/core/sched/scheduler.py','config/scheduler.py']:
        actual=hashlib.sha256((source/rel).read_bytes()).hexdigest()
        reference=hashlib.sha256((tag_source/'vllm'/rel).read_bytes()).hexdigest()
        assert actual==reference,('native wheel source mismatch',rel)
        matches[rel]=actual
    cfg=runs['default']['resolved_config']
    result=dict(passed=True,generated_at=datetime.now(timezone.utc).isoformat(),vllm=env['vllm'],
        torch=env['torch'],cuda_runtime=env['cuda_runtime'],driver=subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        device=runs['torch']['device'],capability=[8,6],device_cached_tokens=device,
        host_recovered_tokens=host,h2d_bytes=h2d,native_source_hashes=matches,
        async_scheduling=cfg['scheduler_config']['async_scheduling'],
        cudagraph_mode=cfg['compilation_config']['cudagraph_mode'],
        compilation_mode=cfg['compilation_config']['mode'],
        reserve_full_isl=cfg['scheduler_config']['scheduler_reserve_full_isl'],
        watermark=cfg['scheduler_config']['watermark'],
        source_tag_commit=json.loads((DATA/'manifests/tag.json').read_text())['commits_in_release_page'],
        tests={m:status['checks'][m] for m in status['checks']},
        raw_sha256={m:hashlib.sha256((DATA/f'raw/smoke-{m}.json').read_bytes()).hexdigest() for m in runs},
        scope='verified Qwen3-1.7B compatibility, not a performance improvement or full hardware support claim')
    (DATA/'manifests/validation_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
