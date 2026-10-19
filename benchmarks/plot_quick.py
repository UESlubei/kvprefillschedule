"""Export quick-validation plots with paired seed points (CPU only)."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'reports/quick_summary.json').read_text())
    rows = data['rows']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    specs = [
        ('A', ['G0', 'G1'], 'output_tok_s', 'Output tokens/s'),
        ('A', ['G0', 'G1'], 'gpu_hit_tokens', 'GPU hit tokens (including resumes)'),
        ('B', ['G0', 'G4'], 'recomputed_tokens', 'Actually recomputed tokens'),
        ('D', ['B1', 'O3'], 'output_tok_s', 'Output tokens/s'),
    ]
    for ax, (work, policies, key, ylabel) in zip(axes.flat, specs):
        by = [{r['seed']: r[key] for r in rows if r['group'] == f'{work}-{p}'} for p in policies]
        seeds = sorted(set.intersection(*(set(x) for x in by)))
        if not seeds:
            continue
        ax.bar(range(2), [statistics.mean(x[s] for s in seeds) for x in by],
               color=['#94a3b8', '#2563eb'], alpha=.75, width=.5)
        for i, seed in enumerate(seeds):
            ax.plot(range(2), [x[seed] for x in by], 'o--', lw=1,
                    color=['#b45309', '#15803d'][i % 2], label=f'Seed {seed}')
        ax.set(xticks=range(2), xticklabels=policies, ylabel=ylabel,
               title=f'Workload {work}, {len(seeds)} paired seeds')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('RTX 3090 / Qwen3-1.7B FP16 / local V1 engine\n'
                 'Quick validation: A/B 24 requests per run; D 32; exploratory results')
    out = ROOT / 'reports/figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'quick_comparison.png', dpi=180)
    fig.savefig(out / 'quick_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
