"""Verify the installed offline bundle, with GPU work explicitly opt-in."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-smoke', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    data, project = root/'data', root/'project'
    assert platform.machine() == 'x86_64', 'This package contains Linux x86_64 binaries.'
    assert platform.system() == 'Linux'
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    glibc = libc.gnu_get_libc_version().decode()
    assert tuple(map(int, glibc.split('.')[:2])) >= (2, 31), f'glibc {glibc}: require >= 2.31'
    assert Path(sys.prefix).resolve() == (data/'environment').resolve(), 'source env.sh first'
    assert Path(sys.base_prefix).resolve() == (root/'runtime').resolve(), 'external base interpreter'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    manifest = json.loads((data/'v3_model_manifest.json').read_text())
    for item in manifest['files']:
        path = root/'models/Qwen3-1.7B'/item['file']
        assert path.stat().st_size == item['bytes']
        assert digest(path) == item['sha256'], str(path)
    frozen = json.loads((data/'v3_full_source_snapshot.json').read_text())
    for relative, expected in frozen['hashes'].items():
        assert digest(project/relative) == expected, relative
    for path, expected in frozen['vendor_hashes'].items():
        assert digest(Path(path)) == expected, path
    for path in (data/'vendor').rglob('*'):
        if path.is_symlink():
            assert path.exists(), str(path)
            assert Path(os.path.realpath(path)).is_relative_to(root), 'external link: '+str(path)
    import torch
    import vllm
    import vllm._C
    import pytest
    from transformers import AutoConfig, AutoTokenizer
    assert torch.__version__ == '2.9.0+cu128'
    assert Path(torch.__file__).resolve().is_relative_to(root/'runtime'), 'external torch installation'
    assert vllm.__version__ == '0.11.2'
    assert Path(vllm.__file__).resolve().is_relative_to(data/'vendor/optimized')
    tokenizer = AutoTokenizer.from_pretrained(root/'models/Qwen3-1.7B', local_files_only=True)
    config = AutoConfig.from_pretrained(root/'models/Qwen3-1.7B', local_files_only=True)
    assert config.model_type == 'qwen3' and len(tokenizer.encode('离线验证')) > 0
    cc = shutil.which('gcc') or shutil.which('cc')
    if not cc:
        raise RuntimeError('No C compiler: the target system needs GCC/cc for Triton JIT.')
    with tempfile.TemporaryDirectory(prefix='compiler-check-', dir=data) as tmp:
        c_source = Path(tmp)/'check.c'
        c_source.write_text('#include <Python.h>\nint main(void) { return 0; }\n')
        executable = Path(tmp)/'check'
        subprocess.run([cc, '-I'+str(root/'runtime/include/python3.11'),
                        str(c_source), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
    subprocess.run([str(root/'analysis/bin/python'), '-c',
                    "import mkl; import matplotlib; matplotlib.use('Agg'); "
                    "from mpl_toolkits.mplot3d import Axes3D; import matplotlib.pyplot as p; "
                    "p.plot([0,1]); p.savefig(__import__('sys').argv[1]); p.close()",
                    str(data/'offline_plot_check.png')], check=True)
    subprocess.run([str(root/'tools/cuda/bin/nvcc'), '--version'], check=True)
    subprocess.run([str(root/'tools/nsight/target-linux-x64/nsys'), '--version'], check=True)
    result = dict(passed=True, mode='cpu/import-and-integrity', root=str(root), glibc=glibc,
                  torch=torch.__version__, vllm=vllm.__version__, cuda=torch.version.cuda,
                  pytest=pytest.__version__, model_files=len(manifest['files']), compiler=cc,
                  origin_release=json.loads((root/'BUNDLE_MANIFEST.json').read_text())['origin_release'],
                  installed_release=frozen['sha256'])
    (data/'offline_verification.json').write_text(json.dumps(result, indent=2))
    print('OFFLINE_CPU_VERIFIED', result, flush=True)
    if args.gpu_smoke:
        subprocess.run([sys.executable, str(root/'smoke_gpu.py')], check=True)


if __name__ == '__main__':
    main()
