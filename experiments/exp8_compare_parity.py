"""E8c - Compare the original C# routines with the Python reference model.

Usage: python exp8_compare_parity.py <parity_dir> <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
py = pd.read_csv(d / "py_features.csv").set_index("i")
cs = pd.read_csv(d / "cs_features.csv").set_index("i")
res = {"feature_sets": int(len(py))}
for k in ("B", "E", "M", "yaw", "pitch", "C", "F"):
    res[f"max_abs_diff_{k}"] = float((py[k] - cs[k]).abs().max())
res["gate_decision_mismatches"] = int((py.gated != cs.gated).sum())
res["threshold_decision_mismatches_C_gt_0.40"] = int(((py.C > 0.40) != (cs.C > 0.40)).sum())

pf = pd.read_csv(d / "py_fires.csv")
cf = pd.read_csv(d / "cs_fires.csv")
traces = pd.read_csv(d / "traces.csv", usecols=["trace"]).trace.nunique()
res["traces"] = int(traces)
res["fires_python"] = int(len(pf)); res["fires_csharp"] = int(len(cf))
mism = 0
for k in range(traces):
    a = np.sort(pf[pf.trace == k].t_fire.to_numpy(np.float32))
    b = np.sort(cf[cf.trace == k].t_fire.to_numpy(np.float32))
    if len(a) != len(b) or not np.array_equal(a, b):
        mism += 1
res["traces_with_different_fire_times"] = mism
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
