"""Compact read-only progress snapshot; never initializes torch/CUDA."""
from pathlib import Path
import json
from datetime import datetime
from zoneinfo import ZoneInfo

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    result={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}
    status=DATA/'pipeline_v3/status.json'
    result['stages']=json.loads(status.read_text()) if status.exists() else {}
    result['runs']={name:sum(1 for _ in (DATA/'raw'/name).rglob('metrics.json'))
                    for name in ['v3_http','v3_full','v3_extended','v3_long']}
    path=DATA/'pipeline_v3/followup_status.json'
    if path.exists():
        rows=json.loads(path.read_text())
        result['completed_followups']=[r['name'] for r in rows if r['returncode']==0]
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
