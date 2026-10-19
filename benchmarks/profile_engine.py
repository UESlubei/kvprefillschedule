"""Capture post-warmup engine activity using nsys --capture-range=cudaProfilerApi."""
import torch
import cProfile
import pstats
import io
import json
import os
from kv_sched.runtime import SchedulerHooks
import run_trace

count=0
started=False
original=SchedulerHooks.reset
original_commit=SchedulerHooks.committed
profiler=cProfile.Profile()


def reset(self):
    global count,started
    original(self)
    count+=1
    if count==3:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('measured_trace_and_cache_seed')
        profiler.enable()
        started=True


SchedulerHooks.reset=reset


def committed(self,output,model_output):
    before=len(self.timings['recovery_ms'])
    ids=[rid for rid in self.execution
         if rid in model_output.req_id_to_index and rid in self.preempted_since]
    original_commit(self,output,model_output)
    recovered=self.timings['recovery_ms'][before:]
    assert len(ids)==len(recovered)
    for rid,ms in zip(ids,recovered):
        self.events.append({'event':'recovery','id':rid,'step':self.step,'ms':ms,
                            'note':'diagnostic wrapper; queue plus first resumed compute'})


SchedulerHooks.committed=committed
try:
    run_trace.main()
finally:
    if started:
        profiler.disable()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        dest=run_trace.DATA/'profiles'/(os.environ.get('KV_PROFILE_NAME','v3_combined')+'_cpu.pstats')
        dest.parent.mkdir(exist_ok=True)
        profiler.dump_stats(dest)
        text=io.StringIO()
        stats=pstats.Stats(profiler,stream=text)
        rows=[{'file':f,'line':line,'function':name,'primitive_calls':v[0],'calls':v[1],
               'self_seconds':v[2],'cumulative_seconds':v[3]}
              for (f,line,name),v in stats.stats.items() if 'kv_sched' in f or name=='schedule']
        dest.with_suffix('.json').write_text(json.dumps(rows,indent=2))
        stats.strip_dirs().sort_stats('cumulative').print_stats('runtime.py|policy.py|scheduler.py')
        dest.with_suffix('.txt').write_text(text.getvalue())
