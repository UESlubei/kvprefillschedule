"""Build and actually relocate a self-contained offline experiment bundle.

Run with the local Anaconda Python that already supplies conda-pack. No downloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
RUNTIME = Path('/mnt/ondeviceinfer/conda-envs/vllm-0.11.2')
ANALYSIS = Path('/home/sdn/anaconda3')
MODEL = Path('/mnt/ondeviceinfer/models/Qwen3-1.7B')
CUDA = Path('/usr/local/cuda-12.8')
NSIGHT = Path('/opt/nvidia/nsight-systems/2024.6.2')
NAME = 'vllm-kv-offline-linux-x86_64-20261019'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze_check():
    frozen = json.loads((DATA/'v3_full_source_snapshot.json').read_text())
    for relative, expected in frozen['hashes'].items():
        assert digest(PROJECT/relative) == expected, 'source changed: '+relative
    for path, expected in frozen['vendor_hashes'].items():
        assert digest(Path(path)) == expected, 'overlay changed: '+path
    return frozen


def archive(path, entries):
    """Stream file contents to parallel gzip; preserve overlay symbolic links."""
    if path.exists():
        print('REUSE', path, flush=True)
        return
    partial = path.with_suffix(path.suffix+'.partial')
    with partial.open('wb') as out:
        child = subprocess.Popen(['pigz', '-1', '-p', '4'], stdin=subprocess.PIPE, stdout=out)
        try:
            with tarfile.open(fileobj=child.stdin, mode='w|', dereference=False) as tar:
                for source, relative in entries:
                    tar.add(source, arcname=relative, filter=clean_member)
        finally:
            child.stdin.close()
        assert child.wait() == 0, 'pigz failed'
    partial.rename(path)
    print('PACKED', path.name, path.stat().st_size, flush=True)


def clean_member(info):
    if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.pyo')):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info


def conda_archive(prefix, path, analysis=False):
    if path.exists():
        print('REUSE', path, flush=True)
        return
    import conda_pack
    # The analysis environment's old package records contain unrelated missing
    # files. We select only Python, shared libraries, and plotting dependencies;
    # the resulting environment must pass the relocation/PNG check below.
    env = conda_pack.CondaEnv.from_prefix(str(prefix), ignore_missing_files=analysis)
    if analysis:
        env = env.exclude('*')
        for pattern in ['bin/python', 'bin/python3', 'bin/python3.12',
                        'lib/*.so*', 'lib/python3.12/*', 'include/python3.12/*']:
            env = env.include(pattern)
        env = env.exclude('lib/python*/site-packages/*')
        for pattern in ['numpy*', 'matplotlib*', 'mpl_toolkits*', 'mkl*', 'contourpy*', 'kiwisolver*',
                        'PIL*', 'pillow*', 'Pillow*', 'fontTools*', 'fonttools*',
                        'cycler*', 'pyparsing*', 'dateutil*', 'python_dateutil*',
                        'packaging*', 'six*', 'setuptools*', '_distutils_hack*',
                        'distutils-precedence.pth']:
            env = env.include('lib/python3.12/site-packages/'+pattern)
    env = env.exclude('*.pyc').exclude('*.pyo').exclude('*/__pycache__/*')
    print('CONDA_PACK_START', path.name, len(env.files), 'files', flush=True)
    partial = path.with_suffix(path.suffix+'.partial')
    env.pack(output=str(partial), format='tar.gz', compress_level=1, n_threads=4,
             force=True, verbose=False)
    partial.rename(path)
    print('CONDA_PACK_END', path.name, path.stat().st_size, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--skip-relocation-test', action='store_true')
    args = ap.parse_args()
    available = sorted(os.sched_getaffinity(0))
    separate = [i for i in available if 8 <= i <= 15]
    os.sched_setaffinity(0, separate or available[-min(4,len(available)):])
    os.nice(15)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(out).free > 55*1024**3, 'Need 55 GiB free for build and relocation verification.'
    frozen = freeze_check()
    bundle = out/'build'/NAME
    packages = bundle/'packages'
    packages.mkdir(parents=True, exist_ok=True)
    stage = out/'staging'
    if not (stage/'READY.json').exists():
        if stage.exists():
            raise RuntimeError('incomplete staging exists; use a fresh --output directory')
        stage.mkdir()
        shutil.copytree(PROJECT, stage/'project', symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
        target = stage/'data'
        target.mkdir()
        for directory in ['vendor', 'cost_models']:
            shutil.copytree(DATA/directory, target/directory, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
        for filename in ['environment_manifest.json', 'requirements-lock.txt',
                         'v3_machine_manifest.json', 'v3_model_manifest.json']:
            shutil.copy2(DATA/filename, target/filename)
        shutil.copy2(DATA/'v3_full_source_snapshot.json', target/'origin_release_source_snapshot.json')
        shutil.copytree(DATA/'environment/lib/python3.11/site-packages',
                        stage/'environment-site-packages',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
        reference = stage/'reference'
        reference.mkdir()
        shutil.copy2(PROJECT.parent/'vLLM_KV分层感知调度_单卡RTX3090实验方案_V3.md', reference/'实验方案_V3.md')
        (stage/'READY.json').write_text(json.dumps({'origin_release':frozen['sha256']}))
    assert json.loads((stage/'READY.json').read_text())['origin_release'] == frozen['sha256']
    recipe_path = out/'pack_recipe.json'
    if not recipe_path.exists() or json.loads(recipe_path.read_text()).get('version') != 2:
        history = out/'history'
        history.mkdir(exist_ok=True)
        for name in ['analysis.tar.gz', 'assets.tar.gz']:
            path = packages/name
            if path.exists():
                path.rename(history/(str(int(time.time()))+'-'+name))
        shutil.copytree(HERE, stage/'project/offline_migration', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for filename in ['install.sh', 'install.py', 'verify_offline.py', 'smoke_gpu.py',
                     'recalibrate.sh', 'README_离线迁移.md']:
        shutil.copy2(HERE/filename, bundle/filename)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        origin_release=frozen['sha256'], platform='Linux x86_64',
        python='3.11.16', vllm='0.11.2', torch='2.9.0+cu128', cuda_toolkit='12.8',
        model='Qwen/Qwen3-1.7B', model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        source_driver='570.86.15', source_glibc='2.31', nsight_systems='2024.6.2',
        packaged=['native conda environment', 'experiment venv overlay', 'stock/optimized vLLM overlays',
                  'model weights and tokenizer (materialized)', 'CUDA Toolkit', 'Nsight Systems',
                  'separate Python 3.12 plotting runtime', 'source, patches, cost calibration, manifests'],
        host_requirements=['Linux x86_64 with glibc >= 2.31', 'compatible NVIDIA driver',
                           'Ampere-or-newer GPU for the configured attention path',
                           'GCC/cc and normal Linux development headers for Triton JIT',
                           'bash, tar, gzip, sha256sum'],
        calibration='Bundled cost table was measured on RTX 3090; recalibrate on a different GPU.',
        fresh_target_runs=True, source_raw_results_included=False,
        downloads_used=False)
    (bundle/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    conda_archive(RUNTIME, packages/'runtime.tar.gz')
    conda_archive(ANALYSIS, packages/'analysis.tar.gz', analysis=True)
    entries = [(p, p.name) for p in stage.iterdir() if p.name != 'READY.json']
    # Resolve individual checkpoint links, not the entire archive: the vLLM
    # overlay's native-library links must remain links until installation.
    entries.extend((p.resolve(), 'models/Qwen3-1.7B/'+str(p.relative_to(MODEL.resolve())))
                   for p in sorted(MODEL.resolve().rglob('*')) if p.is_file())
    archive(packages/'assets.tar.gz', entries)
    archive(packages/'cuda.tar.gz', [(p, p.name) for p in CUDA.iterdir()])
    archive(packages/'nsight.tar.gz', [(p, p.name) for p in NSIGHT.iterdir()])
    recipe_path.write_text(json.dumps({'version':2}))
    checksum_files = sorted(p for p in bundle.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (bundle/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(bundle)}\n' for p in checksum_files))
    validation = dict(passed=False, scope='not run', gpu_test_performed=False)
    if not args.skip_relocation_test:
        relocated = out/'validation'/'relocated bundle with spaces v2'
        if relocated.exists():
            raise RuntimeError('relocation directory already exists; inspect validation log or use a fresh output')
        shutil.copytree(bundle, relocated, copy_function=os.link)
        env = os.environ.copy()
        for key in ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV']:
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        print('RELOCATION_TEST_START', relocated, flush=True)
        with (out/'relocation-validation.log').open('w') as log:
            subprocess.run(['bash', str(relocated/'install.sh')], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            run_env = env.copy()
            run_env.update(PYTHONPATH=f'{relocated}/data/vendor/optimized:{relocated}/project',
                           HF_HUB_OFFLINE='1', XDG_CACHE_HOME=str(relocated/'data/cache'))
            subprocess.run([str(relocated/'data/environment/bin/python'), '-m', 'pytest', '-q',
                            str(relocated/'project/tests/test_policy.py')], env=run_env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            profile = DATA/'profiles/v3_combined.nsys-rep'
            if profile.exists():
                target_profile = relocated/'data/portability_profile.nsys-rep'
                shutil.copy2(profile, target_profile)
                stats = subprocess.run([str(relocated/'tools/nsight/target-linux-x64/nsys'),
                    'stats', '--report', 'cuda_gpu_kern_sum,cuda_api_sum', '--format', 'csv',
                    '--force-export', 'true', str(target_profile)], env=run_env,
                    capture_output=True, text=True, check=True)
                assert 'does not contain CUDA kernel data' not in stats.stdout+stats.stderr
                assert 'cuda_gpu_kern_sum' in stats.stdout, stats.stdout[-1000:]
                (out/'relocation-nsys-stats.log').write_text(stats.stdout+stats.stderr)
        validation = json.loads((relocated/'data/offline_verification.json').read_text())
        validation.update(gpu_test_performed=False,
                          note='No GPU generation while source-machine background experiments are active.')
        print('RELOCATION_TEST_PASSED', validation, flush=True)
    (out/'VALIDATION.json').write_text(json.dumps(validation, indent=2))
    freeze_check()
    archive_path = out/(NAME+'.tar')
    partial = archive_path.with_suffix('.tar.partial')
    with tarfile.open(partial, mode='w', dereference=False) as tar:
        tar.add(bundle, arcname=NAME, filter=clean_member)
    partial.rename(archive_path)
    checksum = digest(archive_path)
    archive_path.with_suffix('.tar.sha256').write_text(f'{checksum}  {archive_path.name}\n')
    receipt = dict(bundle=str(archive_path), bytes=archive_path.stat().st_size,
                   sha256=checksum, validation=validation,
                   origin_release=frozen['sha256'], source_core_unchanged=True,
                   completed_unix=time.time())
    (out/'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
    print('OFFLINE_BUNDLE_READY', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
