"""Capture identical-history logits at the first four generation positions.

Diagnostic only: clones on-device; transfers to CPU after the trace completes.
No synchronous tensor read is introduced into the forward pass.
"""
import os
import json
from pathlib import Path
import torch
import vllm
import run_trace

captured=[]
original=vllm.LLM
selected=set(os.environ.get('KV_CAPTURE_IDS','C-17-00004,C-17-00015,D-17-00015').split(','))
capture_positions=int(os.environ.get('KV_CAPTURE_POSITIONS','4'))
forced={r['id']:r['tokens'] for r in json.loads(Path(os.environ['KV_FORCE_TOKENS']).read_text())} if os.environ.get('KV_FORCE_TOKENS') else {}
decisions=[]
decision_path=os.environ.get('KV_CAPTURE_DECISIONS')


def construct(*args,**kwargs):
    llm=original(*args,**kwargs)
    core=llm.llm_engine.engine_core.engine_core
    runner=core.model_executor.driver_worker.worker.model_runner
    execution={}
    schedule_step=0
    if decision_path or not hasattr(core.scheduler,'_kv'):
        original_schedule=core.scheduler.schedule
        def schedule():
            nonlocal execution,schedule_step
            output=original_schedule();schedule_step+=1
            execution={rid:(core.scheduler.requests[rid].num_computed_tokens-n,
                core.scheduler.requests[rid].num_computed_tokens,
                core.scheduler.requests[rid].num_prompt_tokens)
                for rid,n in output.num_scheduled_tokens.items()}
            if decision_path and (any(rid in forced for rid in core.scheduler.requests)
                    or any(rid in forced for rid in output.finished_req_ids)):
                decisions.append({'scheduled':dict(output.num_scheduled_tokens),
                    'total_tokens':output.total_num_scheduled_tokens,
                    'new':[r.req_id for r in output.scheduled_new_reqs],
                    'cached':list(output.scheduled_cached_reqs.req_ids),
                    'resumed':sorted(output.scheduled_cached_reqs.resumed_req_ids),
                    'finished':sorted(output.finished_req_ids),
                    'states':{rid:[r.status.name,r.num_computed_tokens,r.num_output_tokens,r.num_preemptions]
                              for rid,r in core.scheduler.requests.items() if rid in forced}})
            return output
        core.scheduler.schedule=schedule
    compute=runner.model.compute_logits
    def logits(hidden):
        result=compute(hidden)
        current=execution if decision_path or not hasattr(core.scheduler,'_kv') else core.scheduler._kv.execution
        for j,rid in enumerate(runner.input_batch.req_ids):
            if rid not in selected or rid not in current:
                continue
            a,b,p=current[rid]
            request=core.scheduler.requests[rid]
            if p<=b<p+capture_positions and b==request.num_tokens:
                captured.append({'id':rid,'position':b-p,'logits':result[j].detach().clone(),
                                 'context':list(request.all_token_ids[:b]),'prompt_length':p,
                                 'step':schedule_step if decision_path or not hasattr(core.scheduler,'_kv') else core.scheduler._kv.step})
        return result
    runner.model.compute_logits=logits
    if forced:
        sample=runner.sampler.forward
        def forward(logits,sampling_metadata):
            output=sample(logits,sampling_metadata)
            for j,rid in enumerate(runner.input_batch.req_ids):
                if rid in forced:
                    position=len(runner.requests[rid].output_token_ids)
                    assert position==core.scheduler.requests[rid].num_output_tokens, 'worker lost generated history'
                    if position<len(forced[rid]):
                        output.sampled_token_ids[j,0]=forced[rid][position]
            return output
        runner.sampler.forward=forward
    return llm


vllm.LLM=construct
try:
    run_trace.main()
finally:
    for row in captured: row['logits']=row['logits'].cpu()
    dest=Path(os.environ['KV_LOGIT_CAPTURE'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(captured,dest)
    if decision_path:Path(decision_path).write_text(json.dumps(decisions,indent=2))
    print('CAPTURED',len(captured),str(dest),flush=True)
