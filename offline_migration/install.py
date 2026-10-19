"""Relocate only the unpacked copy; preserve the originating experiment release."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ORIGINAL_DATA = '/mnt/ondeviceinfer/experiments/vllm_kv_scheduling'
ORIGINAL_RUNTIME = '/mnt/ondeviceinfer/conda-envs/vllm-0.11.2'
ORIGINAL_MODEL = '/mnt/ondeviceinfer/models/Qwen3-1.7B'
ORIGINAL_PROJECT = '/home/sdn/wjh/Ondeviceinfer/source/vllm_kv_scheduling'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    project, data = root/'project', root/'data'
    runtime = root/'runtime'
    model = root/'models/Qwen3-1.7B'
    env_python = data/'environment/bin/python'
    changes = []
    mapping = {
        ORIGINAL_DATA: str(data), ORIGINAL_RUNTIME: str(runtime),
        ORIGINAL_MODEL: str(model), ORIGINAL_PROJECT: str(project),
        '/usr/local/cuda/bin/nsys': str(root/'tools/nsight/target-linux-x64/nsys'),
    }
    for parent in [project/'benchmarks', project/'tests']:
        for path in parent.glob('*.py'):
            before = path.read_text()
            after = before
            for old, new in mapping.items():
                after = after.replace(old, new)
            if path.name == 'finalize_full.py':
                after = after.replace("python=shutil.which('python')",
                                      f'python={str(root/"analysis/bin/python")!r}')
            if path.name == 'capture_model_manifest.py':
                after = after.replace('revision=root.name',
                                      "revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'")
            if after != before:
                old_sha = sha(path)
                path.write_text(after)
                changes.append(dict(file=str(path.relative_to(root)),
                                    original_sha256=old_sha, installed_sha256=sha(path)))
    # The source venv deliberately inherits the binary conda environment.
    # Recreate it, rather than copying its absolute interpreter symlink.
    subprocess.run([str(runtime/'bin/python'), '-m', 'venv', '--without-pip',
                    '--system-site-packages', str(data/'environment')], check=True)
    target_site = data/'environment/lib/python3.11/site-packages'
    shutil.copytree(root/'environment-site-packages', target_site, dirs_exist_ok=True)
    for name, module in [('pip', 'pip'), ('pip3', 'pip'), ('pytest', 'pytest'), ('py.test', 'pytest')]:
        entry = data/'environment/bin'/name
        entry.write_text('#!/bin/sh\nexec '+shlex.quote(str(env_python))+' -m '+module+' "$@"\n')
        entry.chmod(0o755)
    links = 0
    for link in (data/'vendor').rglob('*'):
        if not link.is_symlink():
            continue
        original = os.readlink(link)
        if original.startswith(ORIGINAL_RUNTIME+'/'):
            target = runtime/original.removeprefix(ORIGINAL_RUNTIME+'/')
            link.unlink()
            link.symlink_to(os.path.relpath(target, link.parent))
            links += 1
        elif os.path.isabs(original):
            raise RuntimeError('unmapped external overlay link: '+str(link)+' -> '+original)
        if not link.exists():
            raise RuntimeError('broken overlay link: '+str(link))
    manifest_path = data/'environment_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['native_wheel'] = str(runtime/'lib/python3.11/site-packages/vllm')
    manifest['binary_strategy'] = 'packed native conda environment; relocated private overlays'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    model_manifest = json.loads((data/'v3_model_manifest.json').read_text())
    model_manifest['path'] = str(model)
    (data/'v3_model_manifest.json').write_text(json.dumps(model_manifest, indent=2))
    for directory in ['raw', 'profiles', 'pipeline_v3', 'source_snapshots']:
        (data/directory).mkdir(exist_ok=True)
    env_text = '''#!/usr/bin/env bash
export KV_OFFLINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KV_EXP_PROJECT="$KV_OFFLINE_ROOT/project"
export KV_EXP_DATA="$KV_OFFLINE_ROOT/data"
export KV_EXP_PYTHON="$KV_EXP_DATA/environment/bin/python"
export KV_EXP_ANALYSIS_PYTHON="$KV_OFFLINE_ROOT/analysis/bin/python"
export CUDA_HOME="$KV_OFFLINE_ROOT/tools/cuda"
export PATH="$KV_EXP_DATA/environment/bin:$CUDA_HOME/bin:$KV_OFFLINE_ROOT/tools/nsight/target-linux-x64:$PATH"
export LD_LIBRARY_PATH="$KV_OFFLINE_ROOT/runtime/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$KV_EXP_DATA/vendor/optimized:$KV_EXP_PROJECT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=17
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS=4
export HF_HOME="$KV_EXP_DATA/cache/huggingface"
export XDG_CACHE_HOME="$KV_EXP_DATA/cache"
export TRITON_CACHE_DIR="$KV_EXP_DATA/cache/triton"
export TORCH_EXTENSIONS_DIR="$KV_EXP_DATA/cache/torch_extensions"
export MPLCONFIGDIR="$KV_EXP_DATA/cache/matplotlib"
'''
    (root/'env.sh').write_text(env_text)
    (project/'env.sh').write_text('#!/usr/bin/env bash\nsource "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"\n')
    source = json.loads((root/'BUNDLE_MANIFEST.json').read_text())
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{data}/vendor/optimized:{project}', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    subprocess.run([str(env_python), str(project/'benchmarks/freeze_sources.py')],
                   env=env, check=True)
    installed = json.loads((data/'v3_full_source_snapshot.json').read_text())
    receipt = dict(installed_at_unix=time.time(), installation_root=str(root),
                   origin_release=source['origin_release'],
                   installed_release=installed['sha256'], overlay_links_relocated=links,
                   migration_changes=changes,
                   performance_results_included=False,
                   note='Fresh experiments; source-machine results are reference documents only.')
    (root/'INSTALLATION.json').write_text(json.dumps(receipt, indent=2))
    (project/'PROGRESS.md').write_text('# 离线迁移环境\n\n已完成路径重定位并重新冻结源码。目标服务器实验尚未运行。\n\n'
        '原机器性能数字仅供参考，不代表本服务器结果。迁移明细见上级目录 INSTALLATION.json。\n')
    print('RELOCATED', links, 'overlay links;', len(changes), 'path-adapted files')


if __name__ == '__main__':
    main()
