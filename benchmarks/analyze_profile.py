"""Nsight timeline union and correctly matched victim/recovery diagnostics."""
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    db=DATA/'profiles/v3_combined.sqlite'
    assert db.exists()
    con=sqlite3.connect(db);con.row_factory=sqlite3.Row
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    intervals=[];counts={}
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        if table not in tables:continue
        rows=list(con.execute(f'SELECT start,"end" FROM {table} WHERE "end">start'))
        counts[table]=len(rows);intervals.extend((r[0],r[1]) for r in rows)
    assert counts.get('CUPTI_ACTIVITY_KIND_KERNEL',0)>0
    scope=None
    if 'NVTX_EVENTS' in tables:
        columns={r[1] for r in con.execute('PRAGMA table_info(NVTX_EVENTS)')}
        if 'text' in columns:
            candidates=list(con.execute('SELECT start,"end" FROM NVTX_EVENTS WHERE text=? AND "end">start',
                                        ('measured_trace_and_cache_seed',)))
            if candidates:scope=max(candidates,key=lambda r:r[1]-r[0])
    scope_note='named NVTX range, includes cache seed and profiling overhead'
    if scope is None:
        scope=(min(a for a,b in intervals),max(b for a,b in intervals))
        scope_note='first GPU activity to last GPU activity; excludes idle time outside this interval'
    start,end=scope
    union=[]
    for a,b in sorted((max(a,start),min(b,end)) for a,b in intervals if b>start and a<end):
        if union and a<=union[-1][1]:union[-1]=(union[-1][0],max(union[-1][1],b))
        else:union.append((a,b))
    busy=sum(b-a for a,b in union)
    points=[];latest={}
    source=DATA/'raw/v3_profile-combined/seed-17/metrics.json'
    for event in json.loads(source.read_text())['scheduler']['events']:
        if event['event']=='preempt':latest[event['id']]=event
        elif event['event']=='recovery' and event['id'] in latest:
            pre=latest.pop(event['id'])
            points.append(dict(id=event['id'],blocks=pre['blocks'],computed=pre['computed'],
                               recovery_ms=event['ms'],preempt_step=pre['step'],resume_step=event['step']))
    payload=dict(database=str(db),scope=scope_note,scope_ms=(end-start)/1e6,
                 gpu_activity_union_ms=busy/1e6,idle_within_scope_ms=((end-start)-busy)/1e6,
                 idle_fraction=((end-start)-busy)/max(1,end-start),counts=counts,
                 victim_recoveries=points,
                 note='Diagnostic only. Overlapping activity is merged, not summed. Recovery includes queueing.')
    (ROOT/'reports/v3_profile_analysis.json').write_text(json.dumps(payload,indent=2))
    print('PROFILE_ANALYZED',counts,'matched recoveries',len(points))


if __name__=='__main__':main()
