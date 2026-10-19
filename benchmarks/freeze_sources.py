"""Freeze the experiment core, compatibility patches, and actual overlay files."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    files=[*sorted((ROOT/'kv_sched').glob('*.py')),ROOT/'benchmarks/run_trace.py',
           ROOT/'benchmarks/compat.py',ROOT/'benchmarks/prepare.py',
           *sorted((ROOT/'patches').glob('*.patch')),*sorted((ROOT/'configs').glob('*.json'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    from compat import sources
    vendor={str(DATA/'vendor'/flavor/'vllm'/rel):hashlib.sha256((DATA/'vendor'/flavor/'vllm'/rel).read_bytes()).hexdigest()
            for flavor in ['stock','optimized'] for rel in sources()}
    digest=hashlib.sha256(json.dumps({'project':hashes,'vendor':vendor},sort_keys=True).encode()).hexdigest()
    old=DATA/'v3_full_source_snapshot.json'
    if old.exists() and json.loads(old.read_text())['sha256']!=digest:
        if any((DATA/'raw/v3_full').rglob('metrics.json')):
            raise RuntimeError('existing formal results belong to another source; use a separate release label')
    dest=DATA/'source_snapshots'/digest
    for p in files:
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    payload={'sha256':digest,'path':str(dest),'hashes':hashes,'vendor_hashes':vendor,
             'environment_manifest':json.loads((DATA/'environment_manifest.json').read_text())}
    (dest/'source_manifest.json').write_text(json.dumps(payload,indent=2))
    (DATA/'v3_full_source_snapshot.json').write_text(json.dumps(payload,indent=2))
    print(dest)


if __name__=='__main__':main()
