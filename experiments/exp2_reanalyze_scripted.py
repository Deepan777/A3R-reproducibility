"""E2 - Reproduction and re-analysis of the scripted temporal evaluation.

The earlier project report states TP=87, FN=13, FP=0, TN=50 (F1=0.9305) and a
mean lead time of 2.483 s for "the trigger controller".  The only artefact in
the repository that produces these numbers is
python_tools/prediction/run_phase6_evaluation.py, which evaluates the
TemporalAffectPredictor (precursor/trend module, AffectPredictor.cs), not the
AdaptationController that gates guidance.

This script
  1. re-runs the original evaluation unchanged (5 seeds; the neutral-control
     noise is unseeded in the original) and records its outputs;
  2. derives the "ground-truth event" times analytically from the script
     parameters, showing that the event is the scripted score itself crossing
     0.60/0.65, so labels and predictions are not independent;
  3. applies the deployed AdaptationController (active scene configuration) to
     the same 150 scripted traces to show what the guidance gate actually does
     on them.

Outputs: results/E2_*.json / .csv
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT / "A3R_repo"
OUT = ROOT / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from a3r_core import ControllerConfig, run_controller  # noqa: E402


def rerun_original(seeds=(0, 1, 2, 3, 4)):
    from python_tools.prediction import run_phase6_evaluation as m
    out = []
    for s in seeds:
        np.random.seed(s)
        with contextlib.redirect_stdout(io.StringIO()):
            m.run_evaluation()
        summ = json.loads((REPO / "evaluation/phase6_prediction/prediction_metrics_summary.json").read_text())
        summ.pop("evaluation_timestamp", None)
        out.append({"seed": s, **summ})
    shutil.rmtree(REPO / "evaluation", ignore_errors=True)  # keep repository clean
    return out


def scripted_traces():
    """Regenerate the 150 classified scripted traces exactly as defined in
    run_phase6_evaluation.py (ramps run for their full 25 samples here)."""
    tr = []
    for i in range(40):
        slope, start = 0.06 + (i % 5) * 0.02, 0.20 + (i % 3) * 0.05
        t = np.arange(25) * 0.5
        c = np.minimum(1.0, start + slope * t)
        tr.append(dict(kind="confusion_ramp", slope=slope, start=start, t=t, C=c, F=np.full(25, 0.10),
                       event_level=0.60))
    for i in range(40):
        slope, start = 0.06 + (i % 4) * 0.02, 0.20 + (i % 3) * 0.05
        t = np.arange(25) * 0.5
        f = np.minimum(1.0, start + slope * t)
        tr.append(dict(kind="frustration_ramp", slope=slope, start=start, t=t, C=np.full(25, 0.15), F=f,
                       event_level=0.65))
    for i in range(20):
        t = np.arange(16) * 0.5
        tr.append(dict(kind="polarity_hesitation", slope=0.04, start=0.36, t=t, C=0.36 + 0.04 * t,
                       F=np.full(16, 0.15), event_time=3.5))
    rng = np.random.default_rng(0)
    for i in range(30):
        t = np.arange(20) * 0.5
        tr.append(dict(kind="neutral_control", t=t, C=0.10 + 0.04 * rng.uniform(-1, 1, 20),
                       F=0.08 + 0.03 * rng.uniform(-1, 1, 20)))
    for i in range(20):
        t = np.arange(20) * 0.5
        c = np.full(20, 0.15); c[8] = 0.52
        tr.append(dict(kind="spike_noise", t=t, C=c, F=np.full(20, 0.10)))
    return tr


def analyse():
    res = {}
    res["original_rerun"] = rerun_original()
    tr = scripted_traces()
    rows = []
    cfg = ControllerConfig()  # active scene configuration
    for k in tr:
        t = k["t"]
        lvl = k.get("event_level")
        if lvl is not None:
            sig = k["C"] if k["kind"] == "confusion_ramp" else k["F"]
            idx = np.nonzero(sig >= lvl)[0]
            ev = float(t[idx[0]]) if len(idx) else None
        else:
            ev = k.get("event_time")
        fired = run_controller(cfg, t, k["C"], k["F"]).fired
        # time at which the active confusion threshold is first exceeded
        x = np.nonzero((k["C"] > 0.40) | (k["F"] > 0.65))[0]
        cross = float(t[x[0]]) if len(x) else None
        rows.append(dict(kind=k["kind"], slope=k.get("slope"), start=k.get("start"),
                         scripted_event_time=ev, threshold_cross_time=cross,
                         controller_first_fire=fired[0] if fired else None,
                         controller_fire_minus_cross=(fired[0] - cross) if fired and cross is not None else None,
                         trace_duration=float(t[-1])))
    res["controller_on_scripted_traces"] = rows
    # Summary of controller behaviour per category
    summ = {}
    for kind in sorted(set(r["kind"] for r in rows)):
        rr = [r for r in rows if r["kind"] == kind]
        fired = [r for r in rr if r["controller_first_fire"] is not None]
        summ[kind] = dict(n=len(rr), n_fired=len(fired),
                          fire_minus_cross=sorted(set(round(r["controller_fire_minus_cross"], 3) for r in fired)),
                          scripted_event_times=sorted(set(r["scripted_event_time"] for r in rr
                                                          if r["scripted_event_time"] is not None)))
    res["controller_summary"] = summ
    # Analytic event-time dependence: lead time is fixed by (start, slope)
    res["note"] = ("Scripted event = first sample at which the scripted score reaches 0.60 (confusion) "
                   "or 0.65 (frustration), or t = 3.5 s (polarity). Labels are a deterministic function "
                   "of the evaluated signal; lead times are a function of the script's slope/start grid.")
    (OUT / "E2_scripted_reanalysis.json").write_text(json.dumps(res, indent=2, default=float))
    import csv
    with open(OUT / "E2_controller_on_scripted_traces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"original_rerun_counts": [r["contingency_matrix"] for r in res["original_rerun"]],
                      "lead": [r["lead_time_statistics_seconds"] for r in res["original_rerun"]][:1],
                      "breakdown": res["original_rerun"][0]["breakdown_by_category"],
                      "controller_summary": summ}, indent=1, default=float))


if __name__ == "__main__":
    analyse()
