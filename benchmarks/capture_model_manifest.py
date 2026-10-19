"""Hash the local checkpoint and tokenizer once, outside any benchmark."""
import hashlib
import json
from pathlib import Path
from run_trace import DATA,MODEL


def main():
    root=Path(MODEL).resolve()
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix not in {'.json','.safetensors','.model','.tiktoken','.txt'}:continue
        digest=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        rows.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest.hexdigest()))
    assert any(r['file'].endswith('.safetensors') for r in rows)
    (DATA/'v3_model_manifest.json').write_text(json.dumps(dict(path=str(root),revision=root.name,files=rows),indent=2))
    print('MODEL_HASHED',len(rows),'files')


if __name__=='__main__':main()
