"""Parse A3R on-device benchmark logs (PREPARED; NOT YET RUN ON REAL DATA).

AffectScorer.cs emits one line every 5 s of the form
  [A3R_BENCHMARK] t=35.0s | RenderFps=60.1 | CamFps=29.8 | MpFps=4.4 | MpLat=27.3ms |
  ScorerMeanLat=0.52ms | ScorerP95Lat=2.70ms | PredLat=0.031ms | DropInf=0 | DropScore=0 |
  Face=True | HeapMB=1.70
Collect with:  adb logcat -v epoch -s Unity:I > run_<device>_<condition>_<rep>.txt

Usage:  python analyze_runtime_logs.py <log_dir> <out_dir>
File names must follow run_<device>_<condition>_<rep>.txt.  Each run is the
independent unit: per-run medians are computed over 5-s windows (after
discarding the first 30 s warm-up), and conditions are summarised across runs
by the median and inter-quartile range, with a percentile bootstrap CI of the
median over runs.  Windows within a run are never pooled as independent
observations.

Note: MpLat is measured from the start of the frame copy to the inference
callback (it includes GPU read-back and SetPixels32/Apply on the main thread),
and RenderFps is an exponentially smoothed estimate; report them as such.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAT = re.compile(
    r"\[A3R_BENCHMARK\] t=(?P<t>[\d.]+)s \| RenderFps=(?P<render_fps>[\d.]+) \| CamFps=(?P<cam_fps>[\d.]+) \| "
    r"MpFps=(?P<mp_fps>[\d.]+) \| MpLat=(?P<mp_lat_ms>[\d.]+)ms \| ScorerMeanLat=(?P<scorer_mean_ms>[\d.]+)ms \| "
    r"ScorerP95Lat=(?P<scorer_p95_ms>[\d.]+)ms \| PredLat=(?P<pred_ms>[\d.]+)ms \| DropInf=(?P<drop_inf>\d+) \| "
    r"DropScore=(?P<drop_score>\d+) \| Face=(?P<face>True|False) \| HeapMB=(?P<heap_mb>[\d.]+)")
NAME = re.compile(r"run_(?P<device>[^_]+)_(?P<condition>[^_]+)_(?P<rep>\d+)\.txt$")
METRICS = ["render_fps", "cam_fps", "mp_fps", "mp_lat_ms", "scorer_mean_ms", "scorer_p95_ms", "heap_mb"]


def parse_file(path: Path) -> pd.DataFrame:
    rows = [m.groupdict() for m in map(PAT.search, path.read_text(errors="ignore").splitlines()) if m]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in df.columns:
        if c != "face":
            df[c] = pd.to_numeric(df[c])
    df["face"] = df["face"] == "True"
    meta = NAME.search(path.name)
    for k, v in (meta.groupdict() if meta else {"device": "unknown", "condition": "unknown", "rep": 0}).items():
        df[k] = v
    return df


def summarise(windows: pd.DataFrame, warmup_s: float = 30.0, B: int = 2000, seed: int = 1) -> pd.DataFrame:
    w = windows[windows.t >= warmup_s]
    per_run = w.groupby(["device", "condition", "rep"])[METRICS + ["face"]].median().reset_index()
    rng = np.random.default_rng(seed)
    out = []
    for (dev, cond), g in per_run.groupby(["device", "condition"]):
        row = dict(device=dev, condition=cond, n_runs=len(g))
        for m in METRICS:
            v = g[m].to_numpy()
            bs = np.median(rng.choice(v, (B, len(v))), axis=1) if len(v) > 1 else np.array([v[0]])
            row.update({f"{m}_median": np.median(v), f"{m}_iqr": np.subtract(*np.percentile(v, [75, 25])),
                        f"{m}_ci_lo": np.percentile(bs, 2.5), f"{m}_ci_hi": np.percentile(bs, 97.5)})
        out.append(row)
    return pd.DataFrame(out)


def main(log_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frames = [parse_file(p) for p in sorted(Path(log_dir).glob("run_*.txt"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no [A3R_BENCHMARK] lines found")
    win = pd.concat(frames, ignore_index=True)
    win.to_csv(out / "runtime_windows.csv", index=False)
    summarise(win).to_csv(out / "runtime_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
