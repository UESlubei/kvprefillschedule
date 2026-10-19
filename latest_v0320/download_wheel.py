"""Resume verified byte ranges from the official GitHub release asset."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import time
import urllib.request

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
NAME='vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
URL='https://github.com/vllm-project/vllm/releases/download/v0.32.0/'+NAME
SHA='9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b'
SIZE=559420981
CHUNK=1024*1024


def main():
    import argparse,json,urllib.parse
    global NAME,URL,SHA,SIZE
    ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path);ap.add_argument('--workers',type=int,default=32)
    args=ap.parse_args()
    if args.asset:
        asset=json.loads(args.asset.read_text());URL=asset['url'].split('#')[0]
        NAME=urllib.parse.unquote(URL.rsplit('/',1)[-1]);SIZE=asset['size']
        SHA=asset['url'].split('#sha256=')[1]
    output=DATA/'artifacts'/NAME
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest()==SHA:
        print('ALREADY_VERIFIED',flush=True);return
    parts=DATA/'artifacts'/('parts-'+SHA[:16]);parts.mkdir(exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=30) as response:
        direct=response.geturl()
    def fetch(i):
        start=i*CHUNK;end=min(SIZE,start+CHUNK)-1;p=parts/f'{i:05d}'
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(4):
            try:
                req=urllib.request.Request(direct,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=30) as r:
                    assert r.status==206,(i,r.status)
                    assert r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}',(i,r.headers['Content-Range'])
                    data=r.read()
                assert len(data)==end-start+1,(i,len(data))
                tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(p);return
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
    count=(SIZE+CHUNK-1)//CHUNK;t=time.time();done=0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(fetch,i) for i in range(count)]
        for future in as_completed(futures):
            future.result();done+=1
            if done%32==0 or done==count:print('CHUNKS',done,'/',count,'SECONDS',round(time.time()-t,1),flush=True)
    digest=hashlib.sha256();tmp=output.with_suffix('.assembling')
    with tmp.open('wb') as dest:
        for i in range(count):
            data=(parts/f'{i:05d}').read_bytes();digest.update(data);dest.write(data)
    assert digest.hexdigest()==SHA,digest.hexdigest()
    tmp.replace(output);print('VERIFIED',output,SHA,flush=True)

if __name__=='__main__':main()
