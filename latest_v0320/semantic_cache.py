"""Natural-text retrieval across cold, device-cache and actual host recovery."""
import argparse
import json
from pathlib import Path
import time
from latest_v0320.common import DATA,MODEL,dump,prompt
from latest_v0320.teacher_forced import make_llm

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',choices=['N0','NT'],required=True)
    ap.add_argument('--track',choices=['eager','default'],required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    document=('Read this record and return only the exact requested field value.\n'
        'Project Atlas record: owner=Lina; launch_code=VIOLET-73; deadline=2027-06-18.\n'
        +('Neutral notes: requests arrive, memory blocks are allocated, and completed requests release references. '
          'The remaining notes do not modify the record at the beginning.\n')*85)
    fixtures=[]
    for field,expected in [('owner','Lina'),('launch_code','VIOLET-73'),('deadline','2027-06-18')]:
        text=tok.apply_chat_template([{'role':'user','content':document+f'\nReturn only the exact value of {field} from the Project Atlas record.'}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(text,add_special_tokens=False);assert 2048<len(ids)<7900
        fixtures.append({'field':field,'expected':expected,'ids':ids})
    llm=make_llm({'workload':'C','blocks':1024,'track':args.track},args.policy,args.output)
    def generate(ids,n=1,natural=False):
        return llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=n,
            ignore_eos=not natural),use_tqdm=False)[0]
    def reset():
        deadline=time.monotonic()+30
        while not llm.reset_prefix_cache(reset_connector=True):
            if time.monotonic()>deadline:raise RuntimeError('semantic reset failed')
            time.sleep(.03)
    def counters():
        values={}
        for m in llm.get_metrics():
            if not hasattr(m,'value'):continue
            if m.name=='vllm:prompt_tokens_cached_by_source':
                values[m.labels['source']]=values.get(m.labels['source'],0)+m.value
            if m.name=='vllm:kv_offload_total_bytes' and m.labels.get('transfer_type')=='CPU_to_GPU':
                values['h2d_bytes']=values.get('h2d_bytes',0)+m.value
        return values
    for n in [512,2048,4096,6144]:generate(prompt(99000+n,n),2)
    rows=[]
    for fixture in fixtures:
        for cache in ['cold','gpu','cpu']:
            reset()
            if cache!='cold':generate(fixture['ids'][:2048])
            if cache=='cpu':
                for g in range(4):generate(prompt(7700+g,4096))
            before=counters();out=generate(fixture['ids'],32,natural=True);after=counters()
            delta={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after)}
            text=out.outputs[0].text
            correct=fixture['expected'].casefold() in text.casefold()
            if cache=='cold':assert delta.get('device',0)==0 and delta.get('host',0)==0,delta
            if cache=='gpu':assert delta.get('device',0)==2048 and delta.get('host',0)==0,delta
            if cache=='cpu':assert delta.get('device',0)==0 and delta.get('host',0)==2048 and delta.get('h2d_bytes',0)>0,delta
            rows.append({'field':fixture['field'],'cache':cache,'expected':fixture['expected'],
                'text':text,'token_ids':list(out.outputs[0].token_ids),'correct':correct,'native_delta':delta})
            print(args.track,args.policy,fixture['field'],cache,correct,repr(text),delta,flush=True)
    result={'policy':args.policy,'track':args.track,'rows':rows,'passed':all(r['correct'] for r in rows),
        'exact_across_cache_modes':all(len({tuple(r['token_ids']) for r in rows if r['field']==f['field']})==1 for f in fixtures)}
    dump(args.output,result);llm.llm_engine.engine_core.shutdown()
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
