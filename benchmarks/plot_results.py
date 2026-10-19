"""Standalone publication/export figures; no GPU use."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
out=ROOT/'reports/figures'; out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

prefill=json.loads((DATA/'cost_models/prefill_raw.json').read_text())
transfer=json.loads((DATA/'cost_models/transfer_raw.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for fraction in [0,.25,.5,.75]:
    rs=sorted((r for r in prefill if r['batch']==1 and not r['validation']
               and abs(r['cached']/r['length']-fraction)<.001),key=lambda r:r['length'])
    ax[0].plot([r['length'] for r in rs],[r['ms'] for r in rs],marker='o',label=f'{fraction:.0%} cached')
ax[0].set(xlabel='Context tokens',ylabel='CUDA execution interval (ms)',title='Measured prefill cost (single request)')
ax[0].legend(); ax[0].grid(alpha=.2)
rs=transfer['native_handler']
ax[1].plot([r['bytes']/2**20 for r in rs],[r['job_ms'] for r in rs],marker='o',label='Native handler job wall time')
cs=transfer['continuous_copy']
ax[1].plot([r['bytes']/2**20 for r in cs],[r['cuda_ms'] for r in cs],marker='s',label='Continuous copy CUDA time')
ax[1].set(xlabel='Payload (MiB)',ylabel='Time (ms)',title='Pinned CPU to GPU transfer')
ax[1].legend(); ax[1].grid(alpha=.2)
fig.savefig(out/'cost_calibration.png',dpi=180); fig.savefig(out/'cost_calibration.pdf'); plt.close(fig)

summary=ROOT/'reports/formal_summary.json'
if summary.exists():
    data=json.loads(summary.read_text())
    groups={(r['gpu_blocks'],r['cpu_blocks'],r['group']):r for r in data['groups']}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    cases=[(axes[0,0],640,0,'B',['G0','G2','G3','G4'],'output_tok_s','Output tokens/s'),
           (axes[0,1],1024,0,'B',['G0','G2','G3','G4'],'recomputed_tokens','Recomputed tokens'),
           (axes[1,0],1024,128,'C',['B1','B2','O1','O2','O3'],'output_tok_s','Output tokens/s'),
           (axes[1,1],1024,128,'D',['B1','B2','O1','O2','O3'],'ttft_p95_ms','P95 TTFT (ms)')]
    for ax,gpu,cpu,work,policies,key,ylabel in cases:
        valid=[(p,groups[(gpu,cpu,f'{work}-{p}')]['metrics'][key]) for p in policies if (gpu,cpu,f'{work}-{p}') in groups]
        if not valid: continue
        labels=[p for p,_ in valid]; means=[d['mean'] for _,d in valid]
        errors=[[d['mean']-d['ci95'][0] for _,d in valid],[d['ci95'][1]-d['mean'] for _,d in valid]]
        ax.bar(labels,means,yerr=errors,capsize=3,color=['#64748b']+['#2563eb']*(len(valid)-1))
        ax.set(ylabel=ylabel,title=f'Workload {work}, {gpu} GPU blocks, {cpu} CPU blocks')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Qwen3-1.7B FP16 / RTX 3090 / local V1 engine; paired-run bootstrap 95% CI')
    fig.savefig(out/'formal_comparison.png',dpi=180); fig.savefig(out/'formal_comparison.pdf'); plt.close(fig)
print(out)
