"""CPU-only package/binary provenance; does not initialize a CUDA context."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
WHEEL='https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
EXPECTED='9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b'

def main():
    import torch
    dist=metadata.distribution('vllm')
    source=Path(dist.locate_file('vllm'))
    critical=['v1/core/sched/scheduler.py','config/scheduler.py',
              'distributed/kv_transfer/kv_connector/v1/offloading_connector.py']
    hashes={rel:hashlib.sha256((source/rel).read_bytes()).hexdigest() for rel in critical if (source/rel).exists()}
    direct=dist.read_text('direct_url.json')
    packages={d.metadata['Name']:d.version for d in metadata.distributions()}
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),python=sys.version,
        executable=sys.executable,platform=platform.platform(),vllm=dist.version,
        torch=torch.__version__,cuda_runtime=torch.version.cuda,
        source=str(source),source_hashes=hashes,direct_url=json.loads(direct) if direct else None,
        expected_wheel_url=WHEEL,expected_wheel_sha256=EXPECTED,packages=packages,
        validation_scope='CPU import and provenance; GPU inference not implied')
    assert dist.version=='0.32.0+cu129',dist.version
    assert torch.version.cuda=='12.9',torch.version.cuda
    artifact=DATA/'artifacts'/WHEEL.rsplit('/',1)[-1]
    result['actual_wheel_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert result['actual_wheel_sha256']==EXPECTED
    if direct:
        declared=result['direct_url'].get('archive_info',{}).get('hashes',{}).get('sha256')
        if declared is not None:assert declared==EXPECTED,('wheel provenance mismatch',declared)
    (DATA/'manifests/environment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['vllm','torch','cuda_runtime','source','validation_scope']},indent=2))

if __name__=='__main__':main()
