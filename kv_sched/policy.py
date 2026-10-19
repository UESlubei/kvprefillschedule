"""Pure scheduling rules. No allocation, transfers, or vLLM imports."""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    admission: str = "fcfs"
    preemption: str = "default"
    window: int = 8
    threshold: float = 0.0
    aging: float = 0.05
    max_wait_ms: float = 2000.0
    cost_model: str | None = None
    trace: bool = False
    probe_cpu: bool = True
    fairness: bool = True
    resource_aware: bool = True
    progress_aware: bool = True

    def __post_init__(self):
        if self.admission not in {"fcfs", "gpu", "tier"}:
            raise ValueError("unknown admission")
        if self.preemption not in {"default", "lcf", "pr_score", "recompute", "restore"}:
            raise ValueError("unknown preemption")
        if self.window < 0 or not 0 <= self.threshold <= 1:
            raise ValueError("invalid window/threshold")
        if self.aging < 0 or self.max_wait_ms < 0:
            raise ValueError("invalid fairness parameters")


class IntervalLedger:
    def __init__(self):
        self.intervals: list[tuple[int, int]] = []

    def commit(self, start: int, end: int) -> int:
        if start < 0 or end < start:
            raise ValueError("invalid execution interval")
        overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in self.intervals)
        merged: list[tuple[int, int]] = []
        for a, b in sorted([*self.intervals, (start, end)]):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            elif a < b:
                merged.append((a, b))
        self.intervals = merged
        return overlap


class CostModel:
    def __init__(self, path: str | None = None):
        self.data = json.loads(Path(path).read_text()) if path else None
        if self.data:
            if self.data.get("schema") != 1 or self.data.get("kv_bytes_per_token") != 114688:
                raise ValueError("cost model schema or KV layout mismatch")
            if self.data.get("model_revision") != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
                raise ValueError("model revision mismatch")
        self.calibrated = bool(self.data)

    def prefill(self, length: int, cached: int = 0, batch: int = 1) -> float:
        q = max(0, length - cached)
        if not q:
            return 0.0
        if not self.data:
            return float(q)  # Explicit token proxy, not a millisecond prediction.
        points = self.data["prefill"]
        exact = [p for p in points if p["length"] == length and p["cached"] == cached and p["batch"] == batch]
        if exact:
            return exact[0]["ms"]
        # Interpolate query length at the closest context and batch class.
        selected_batch = min({p["batch"] for p in points}, key=lambda b: abs(b-batch))
        context = min({p["cached"] for p in points if p["batch"] == selected_batch}, key=lambda c: abs(c-cached))
        rows = sorted((p["length"]-p["cached"], p["ms"]) for p in points
                      if p["batch"] == selected_batch and p["cached"] == context)
        xs = [r[0] for r in rows]
        j = bisect.bisect_left(xs, q)
        if j == 0:
            return rows[0][1] * q / max(rows[0][0], 1)
        if j == len(rows):
            return rows[-1][1] * q / max(rows[-1][0], 1)
        x0, y0 = rows[j-1]
        x1, y1 = rows[j]
        return y0 + (y1-y0)*(q-x0)/max(x1-x0, 1)

    def transfer(self, nbytes: int, blocks: int = 1) -> float:
        if not nbytes:
            return 0.0
        if not self.data:
            raise RuntimeError("CPU time prediction requires measured cost model")
        h = self.data["h2d"]
        return h["fixed_ms"] + nbytes / h["bytes_per_ms"] + blocks*h.get("per_block_ms", 0.0)

    def path(self, length: int, gpu: int, cpu_end: int, transfer_bytes: int) -> float:
        if cpu_end > gpu:
            return self.transfer(transfer_bytes) + self.prefill(length, cpu_end)
        return self.prefill(length, gpu)


def eligible(request) -> bool:
    return (request.status.name == "WAITING" and request.num_computed_tokens == 0
            and request.num_preemptions == 0 and not request.skip_reading_prefix_cache)


def reorder_segments(requests, score, overdue):
    """Ineligible slots are barriers; ties retain input order."""
    out = list(requests)
    i = 0
    while i < len(out):
        if not eligible(out[i]):
            i += 1
            continue
        j = i + 1
        while j < len(out) and eligible(out[j]):
            j += 1
        out[i:j] = sorted(out[i:j], key=lambda r: (not overdue(r),
                         getattr(r, 'arrival_time', 0) if overdue(r) else -score(r)))
        i = j
    return out


@dataclass
class Victim:
    request: object
    blocks: int
    computed: int
    tokens: int
    generated: int
    max_output: int
    recovery: float


def choose_victim(candidates: list[Victim], policy: str, required: int,
                  resource_aware: bool = True, progress_aware: bool = True):
    if not candidates:
        return None
    if policy == "default":
        return candidates[-1].request
    if policy == "lcf":
        return min(candidates, key=lambda x: x.computed).request
    pool = [v for v in candidates if v.blocks > 0]
    if not pool:
        return None
    if resource_aware and policy in {"recompute", "restore"}:
        sufficient = [v for v in pool if v.blocks >= required]
        pool = sufficient or pool

    def score(v):
        progress = 1 - v.generated / max(v.max_output, 1)
        if policy == "pr_score":
            return v.blocks/max(1, v.tokens) * progress
        return v.blocks/max(v.recovery, 0.01)*(max(0.05, progress) if progress_aware else 1.0)

    return max(pool, key=score).request


def percentiles(values):
    xs = sorted(values)
    if not xs:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    def q(f):
        p = (len(xs)-1)*f
        a = int(p)
        return xs[a] + (xs[min(a+1, len(xs)-1)]-xs[a])*(p-a)
    return {"count": len(xs), "mean": sum(xs)/len(xs), "p50": q(.5), "p95": q(.95), "p99": q(.99)}
