"""E5 - Geometric analysis of the facial features, pose gate and scores.

Input: MediaPipe's canonical face mesh (468 vertices, cm) recovered from the
face_landmarker.task bundle shipped in the A3R repository
(extract_canonical_mesh.py).  The mesh is the mean face geometry used by
MediaPipe's face-geometry pipeline; it is NOT a recording of any person.

The mesh is posed (yaw, pitch), placed in front of a pinhole camera that
matches the runtime acquisition (WebCamTexture 640x480 rotated to portrait,
480x640 px), projected, converted to MediaPipe-style normalised coordinates
(x/W, y/H, depth scaled like x), and passed through the exact feature
extractor of the runtime (a3r_core.extract_features).

Analyses
  E5a  neutral frontal operating point (features, scores)
  E5b  head-pose sweep: pose-gate acceptance region, pose-induced score change
  E5c  viewing-distance sweep: scale dependence of each feature
  E5d  landmark-noise propagation to score noise
  E5e  controlled geometric deformations: score sensitivity per mm
  E5f  image-aspect (normalisation anisotropy) effect on the eye feature
  E5g  candidate scale-invariant mouth feature (D / IOD), evaluated only here

All outputs are deterministic (fixed seeds) and written to results/E5_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results"
sys.path.insert(0, str(HERE))
import a3r_core as core  # noqa: E402

W, H = 480, 640          # portrait image delivered to FaceLandmarker
F_PX = 480.0             # focal length (px); ~53 deg horizontal FOV in portrait
D0 = 40.0                # cm, nominal face-camera distance


def load_mesh():
    df = pd.read_csv(OUT / "canonical_face_468.csv")
    P = df[["x_cm", "y_cm", "z_cm"]].to_numpy()
    return np.vstack([P, np.zeros((10, 3))])   # pad to 478 (iris unused)


def rot(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def project(P, yaw=0.0, pitch=0.0, dist=D0, w=W, h=H, f=F_PX, noise_px=0.0, rng=None,
            isotropic=False):
    """Return (478, 3) landmarks in MediaPipe normalised image coordinates."""
    c = P[:468].mean(0)
    Q = (P - c) @ rot(yaw, pitch).T          # pose about the mesh centroid
    Xc, Yc, Zc = Q[:, 0], -Q[:, 1], dist - Q[:, 2]   # camera: x right, y down, z forward
    u = f * Xc / Zc + w / 2
    v = f * Yc / Zc + h / 2
    zpx = f * (Zc - Zc[:468].mean()) / Zc[:468].mean()
    if noise_px > 0:
        u = u + rng.normal(0, noise_px, u.shape)
        v = v + rng.normal(0, noise_px, v.shape)
        zpx = zpx + rng.normal(0, noise_px, zpx.shape)
    if isotropic:
        s = float(w)
        return np.stack([u / s, v / s, zpx / s], 1)
    return np.stack([u / w, v / h, zpx / w], 1)


def feats_scores(L):
    f = core.extract_features(L)
    C, Fr = core.scores(f.B, f.E, f.M)
    return dict(brow_raw=f.brow_raw, B=f.B, ear_raw=f.ear_raw, E=f.E, mouth_disp=f.mouth_raw,
                M=f.M, yaw_ratio=f.yaw_ratio, pitch_ratio=f.pitch_ratio, pose_gated=f.pose_gated,
                C=float(C), F=float(Fr))


def with_mid_range_brow(P, target=0.94):
    """Move landmarks 105/334 symmetrically so that the frontal r_B equals
    `target` (mid-point of the [0.90, 0.98] normalisation interval).  Used to
    study sensitivities at an unsaturated operating point."""
    Q = P.copy()
    iod = np.linalg.norm(Q[33] - Q[263])
    half = target * iod / 2
    for i, sgn in ((105, -1), (334, 1)):
        Q[i, 0] = sgn * half
    return Q


def main():
    rng = np.random.default_rng(20261113)
    P = load_mesh()
    res = {}

    # E5a ---------------------------------------------------------------
    a = {"canonical_runtime_coords": feats_scores(project(P)),
         "canonical_isotropic_coords": feats_scores(project(P, isotropic=True)),
         "midrange_brow_runtime_coords": feats_scores(project(with_mid_range_brow(P)))}
    # 3-D (unprojected) ratios for reference
    d = lambda i, j: float(np.linalg.norm(P[i] - P[j]))
    a["canonical_3d"] = dict(r_B=d(105, 334) / d(33, 263),
                             EAR=0.5 * (d(159, 145) / (2 * d(133, 33)) + d(386, 374) / (2 * d(362, 263))),
                             IOD_cm=d(33, 263), inner_brow_105_334_cm=d(105, 334),
                             x_105_cm=float(P[105, 0]), x_33_cm=float(P[33, 0]),
                             mouth_corner_below_lipmid_cm=float(0.5 * (P[13, 1] + P[14, 1]) - P[61, 1]))
    res["E5a_operating_point"] = a

    # E5b pose sweep ------------------------------------------------------
    # Three coordinate conventions separate the mechanisms:
    #   runtime  : (x/W, y/H, z/W) with 3-D distances  (exactly what the app does)
    #   iso3d    : (x, y, z) all in pixel units, 3-D distances
    #   2d       : (x/W, y/H) with z dropped, i.e. image-plane distances
    def conv(L, mode, Lpx=None):
        if mode == "runtime":
            return L
        if mode == "iso3d":
            return Lpx
        L2 = L.copy(); L2[:, 2] = 0.0
        return L2

    rows = []
    PPm = with_mid_range_brow(P)
    for mode in ("runtime", "iso3d", "2d"):
        for yaw in np.arange(-45, 45.1, 2.5):
            for pitch in np.arange(-35, 35.1, 2.5):
                L = project(PPm, yaw, pitch)
                Lpx = project(PPm, yaw, pitch, isotropic=True)
                r = feats_scores(conv(L, mode, Lpx))
                rows.append(dict(geometry="midrange_brow", coords=mode, yaw=yaw, pitch=pitch, **r))
    for yaw in np.arange(-45, 45.1, 2.5):
        for pitch in np.arange(-35, 35.1, 2.5):
            r = feats_scores(project(P, yaw, pitch))
            rows.append(dict(geometry="canonical", coords="runtime", yaw=yaw, pitch=pitch, **r))
    pose = pd.DataFrame(rows)
    pose.to_csv(OUT / "E5b_pose_sweep.csv", index=False)
    summ = {}
    for mode in ("runtime", "iso3d", "2d"):
        g = pose[(pose.geometry == "midrange_brow") & (pose.coords == mode)]
        acc = g[~g.pose_gated]
        fr = g[(g.yaw == 0) & (g.pitch == 0)].iloc[0]
        a0 = acc[acc.pitch == 0]; p0 = acc[acc.yaw == 0]
        summ[mode] = dict(
            accepted_fraction=float(len(acc) / len(g)),
            accepted_yaw_range_at_pitch0=[float(a0.yaw.min()), float(a0.yaw.max())] if len(a0) else None,
            accepted_pitch_range_at_yaw0=[float(p0.pitch.min()), float(p0.pitch.max())] if len(p0) else None,
            yaw_ratio_range=[float(g.yaw_ratio.min()), float(g.yaw_ratio.max())],
            pitch_ratio_range=[float(g.pitch_ratio.min()), float(g.pitch_ratio.max())],
            C_frontal=float(fr.C),
            C_range_accepted=[float(acc.C.min()), float(acc.C.max())] if len(acc) else None,
            frac_accepted_above_T=float((acc.C > core.ACTIVE["t_conf"]).mean()) if len(acc) else None,
            C_range_within_15deg=[float(g[(g.yaw.abs() <= 15) & (g.pitch.abs() <= 15)].C.min()),
                                  float(g[(g.yaw.abs() <= 15) & (g.pitch.abs() <= 15)].C.max())],
            gated_C_max=float(g[g.pose_gated].C.max()) if g.pose_gated.any() else None)
    res["E5b_pose"] = summ

    # E5c distance sweep -------------------------------------------------
    rows = []
    for dist in np.arange(25, 81, 2.5):
        r = feats_scores(project(with_mid_range_brow(P), dist=dist))
        L = project(with_mid_range_brow(P), dist=dist)
        iod_n = float(np.linalg.norm(L[33] - L[263]))
        rows.append(dict(distance_cm=dist, iod_norm=iod_n, M_scale_inv=float(-r["mouth_disp"] / iod_n), **r))
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(OUT / "E5c_distance_sweep.csv", index=False)
    res["E5c_distance"] = dict(
        M_at_30=float(dist_df.loc[dist_df.distance_cm == 30, "M"].iloc[0]),
        M_at_40=float(dist_df.loc[dist_df.distance_cm == 40, "M"].iloc[0]),
        M_at_60=float(dist_df.loc[dist_df.distance_cm == 60, "M"].iloc[0]),
        C_at_30=float(dist_df.loc[dist_df.distance_cm == 30, "C"].iloc[0]),
        C_at_60=float(dist_df.loc[dist_df.distance_cm == 60, "C"].iloc[0]),
        B_range=[float(dist_df.B.min()), float(dist_df.B.max())],
        E_range=[float(dist_df.E.min()), float(dist_df.E.max())],
        D_over_IOD_range=[float(dist_df.M_scale_inv.min()), float(dist_df.M_scale_inv.max())])

    # E5d landmark-noise propagation ------------------------------------
    rows = []
    PP = with_mid_range_brow(P)
    base = feats_scores(project(PP))
    for npx in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        for dist in (30.0, 40.0, 60.0):
            Cs, Bs, Es, Ms, gated = [], [], [], [], 0
            for _ in range(2000):
                r = feats_scores(project(PP, dist=dist, noise_px=npx, rng=rng))
                Cs.append(r["C"]); Bs.append(r["B"]); Es.append(r["E"]); Ms.append(r["M"])
                gated += r["pose_gated"]
            rows.append(dict(noise_px=npx, distance_cm=dist, C_sd=float(np.std(Cs)), B_sd=float(np.std(Bs)),
                             E_sd=float(np.std(Es)), M_sd=float(np.std(Ms)), C_mean=float(np.mean(Cs)),
                             gated_frac=gated / 2000))
    noise_df = pd.DataFrame(rows)
    noise_df.to_csv(OUT / "E5d_noise_propagation.csv", index=False)

    # E5e controlled deformations ----------------------------------------
    rows = []
    for mm in np.arange(0, 6.01, 0.5):      # inner-brow approach (each side, mm)
        Q = with_mid_range_brow(P); Q[105, 0] += mm / 10; Q[334, 0] -= mm / 10
        rows.append(dict(deformation="brow_approach_each_side_mm", amount=mm, **feats_scores(project(Q))))
    for frac in np.arange(0, 0.81, 0.1):     # upper-lid closure fraction
        Q = with_mid_range_brow(P)
        for top, bot in ((159, 145), (386, 374)):
            Q[top] = Q[top] + frac * (Q[bot] - Q[top])
        rows.append(dict(deformation="eyelid_closure_fraction", amount=frac, **feats_scores(project(Q))))
    for mm in np.arange(0, 4.01, 0.5):       # mouth-corner depression (mm)
        Q = with_mid_range_brow(P); Q[61, 1] -= mm / 10; Q[291, 1] -= mm / 10
        rows.append(dict(deformation="mouth_corner_depression_mm", amount=mm, **feats_scores(project(Q))))
    defo = pd.DataFrame(rows)
    defo.to_csv(OUT / "E5e_deformations.csv", index=False)

    def first_cross(kind):
        g = defo[defo.deformation == kind]
        x = g[g.C > core.ACTIVE["t_conf"]]
        return float(x.amount.iloc[0]) if len(x) else None
    res["E5e_threshold_crossing"] = {k: first_cross(k) for k in defo.deformation.unique()}
    res["E5e_C_at_zero"] = float(defo[defo.amount == 0].C.iloc[0])

    # E5f anisotropy -------------------------------------------------------
    rows = []
    for label, (w, h, iso) in {"portrait 480x640 (runtime)": (480, 640, False),
                               "landscape 640x480": (640, 480, False),
                               "isotropic (pixel units)": (480, 640, True)}.items():
        r = feats_scores(project(with_mid_range_brow(P), w=w, h=h, isotropic=iso))
        rows.append(dict(normalisation=label, **r))
    pd.DataFrame(rows).to_csv(OUT / "E5f_aspect.csv", index=False)
    res["E5f_aspect"] = rows

    (OUT / "E5_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print(noise_df.round(4).to_string())


if __name__ == "__main__":
    main()
