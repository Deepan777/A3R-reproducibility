"""E7b - Analyse A3R's facial features on real faces (Pointing'04).

Unit of analysis: the person (n = 15). Images of one person are never treated
as independent; summaries are computed per person (averaging the two
sessions) and uncertainty is a percentile bootstrap over persons (B = 2,000).
The expressions in the database are not controlled (mostly neutral), so the
analysis addresses nuisance factors (identity, head pose, session), not affect.

Input : results/E7_real_faces.csv (exp7_real_faces_extract.py)
Output: results/E7_summary.json, results/E7_*.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
T = 0.40
rng = np.random.default_rng(20261114)


def boot_ci(v, B=2000, f=np.median):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return [float("nan")] * 2
    bs = f(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def main():
    d = pd.read_csv(RES / "E7_real_faces.csv")
    out = {"n_images": int(len(d)), "n_detected": int(d.detected.sum()),
           "n_persons": int(d.person.nunique())}
    # detection rate by pose (pooled counts, informational)
    det = d.groupby(["tilt", "pan"]).detected.mean().reset_index()
    det.to_csv(RES / "E7_detection_by_pose.csv", index=False)
    near = d[(d.pan.abs() <= 30) & (d.tilt.abs() <= 30)]
    out["detection_rate_within_30deg"] = float(near.detected.mean())
    out["detection_rate_beyond_60deg_pan"] = float(d[d.pan.abs() >= 60].detected.mean())

    f = d[d.detected == 1].copy()
    res = {}
    for conv in ("nat", "iso"):
        C, g = f"{conv}_C", f"{conv}_gated"
        fr = f[(f.tilt == 0) & (f.pan == 0)]
        per_sess = fr.groupby(["person", "series"])[[C, f"{conv}_r_B", f"{conv}_B", f"{conv}_E", f"{conv}_M"]].mean().reset_index()
        per_p = per_sess.groupby("person").mean(numeric_only=True)
        sess = per_sess.pivot(index="person", columns="series", values=C).dropna()
        r = dict(
            frontal_n_persons=int(per_p.shape[0]),
            frontal_C_median=float(per_p[C].median()), frontal_C_ci=boot_ci(per_p[C]),
            frontal_C_min=float(per_p[C].min()), frontal_C_max=float(per_p[C].max()),
            frontal_C_sd_between_persons=float(per_p[C].std(ddof=1)),
            frontal_persons_above_T=int((per_p[C] > T).sum()),
            frontal_rB_range=[float(per_p[f"{conv}_r_B"].min()), float(per_p[f"{conv}_r_B"].max())],
            frontal_persons_B_saturated=int((per_p[f"{conv}_B"] >= 0.999).sum()),
            frontal_persons_B_zero=int((per_p[f"{conv}_B"] <= 0.001).sum()),
            frontal_E_range=[float(per_p[f"{conv}_E"].min()), float(per_p[f"{conv}_E"].max())],
            frontal_M_range=[float(per_p[f"{conv}_M"].min()), float(per_p[f"{conv}_M"].max())],
            session_abs_diff_median=float((sess[1] - sess[2]).abs().median()) if sess.shape[1] == 2 else None,
            session_abs_diff_max=float((sess[1] - sess[2]).abs().max()) if sess.shape[1] == 2 else None,
            session_pairs=int(sess.shape[0]),
        )
        # pose effects: per person, C at pose minus own frontal C (sessions averaged)
        pp = f.groupby(["person", "tilt", "pan"])[[C, g, f"{conv}_yaw_ratio", f"{conv}_pitch_ratio"]].mean().reset_index()
        ref = pp[(pp.tilt == 0) & (pp.pan == 0)].set_index("person")[C]
        pp["dC"] = pp[C] - pp.person.map(ref)
        rows = []
        for (tilt, pan), gg in pp.groupby(["tilt", "pan"]):
            rows.append(dict(convention=conv, tilt=tilt, pan=pan, n_persons=int(gg.person.nunique()),
                             dC_median=float(gg.dC.median()), dC_q25=float(gg.dC.quantile(.25)),
                             dC_q75=float(gg.dC.quantile(.75)), dC_ci_lo=boot_ci(gg.dC)[0], dC_ci_hi=boot_ci(gg.dC)[1],
                             C_median=float(gg[C].median()), frac_persons_above_T=float((gg[C] > T).mean()),
                             gated_rate=float(gg[g].mean()),
                             yaw_ratio_median=float(gg[f"{conv}_yaw_ratio"].median()),
                             pitch_ratio_median=float(gg[f"{conv}_pitch_ratio"].median())))
        pose = pd.DataFrame(rows)
        pose.to_csv(RES / f"E7_pose_effects_{conv}.csv", index=False)
        # per-person slope of yaw ratio vs pan (|pan| <= 45, tilt 0) and pitch ratio vs tilt (pan 0)
        ys, ps = [], []
        for pid, gg in f.groupby("person"):
            a = gg[(gg.tilt == 0) & (gg.pan.abs() <= 45)]
            if a.pan.nunique() >= 3:
                ys.append(np.polyfit(a.pan, a[f"{conv}_yaw_ratio"], 1)[0] * 10)
            b = gg[(gg.pan == 0) & (gg.tilt.abs() <= 30)]
            if b.tilt.nunique() >= 3:
                ps.append(np.polyfit(b.tilt, b[f"{conv}_pitch_ratio"], 1)[0] * 10)
        acc0 = pose[(pose.tilt == 0)]
        accp = pose[(pose.pan == 0)]
        r.update(dict(
            yaw_ratio_slope_per10deg_median=float(np.median(ys)), yaw_ratio_slope_ci=boot_ci(ys), n_yaw_slopes=len(ys),
            pitch_ratio_slope_per10deg_median=float(np.median(ps)), pitch_ratio_slope_ci=boot_ci(ps), n_pitch_slopes=len(ps),
            gated_rate_by_pan_tilt0={int(k): float(v) for k, v in zip(acc0.pan, acc0.gated_rate)},
            gated_rate_by_tilt_pan0={int(k): float(v) for k, v in zip(accp.tilt, accp.gated_rate)},
            dC_by_pan_tilt0={int(k): [float(a), float(b), float(c)] for k, a, b, c in zip(acc0.pan, acc0.dC_median, acc0.dC_q25, acc0.dC_q75)},
            dC_by_tilt_pan0={int(k): [float(a), float(b), float(c)] for k, a, b, c in zip(accp.tilt, accp.dC_median, accp.dC_q25, accp.dC_q75)},
            frac_above_T_by_tilt_pan0={int(k): float(v) for k, v in zip(accp.tilt, accp.frac_persons_above_T)},
            frac_above_T_by_pan_tilt0={int(k): float(v) for k, v in zip(acc0.pan, acc0.frac_persons_above_T)},
        ))
        # all detected, non-extreme poses: share of person-poses above threshold
        mid = pp[(pp.pan.abs() <= 30) & (pp.tilt.abs() <= 30)]
        r["share_person_poses_above_T_within_30deg"] = float((mid[C] > T).mean())
        res[conv] = r
    out["conventions"] = res
    (RES / "E7_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
