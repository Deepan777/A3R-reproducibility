"""E6 - Propagation of corner-placement error through four-corner registration.

Reproduces A3R_MarkerlessFoundationController.ComputeFourCornerPose():
    centre  = mean of the four tapped corners
    right   = normalise( normalise(TR - TL) + normalise(BR - BL) )
    up      = world vertical (forced), forward = right x up
    scale   = NOT estimated (fixed CAD dimensions, 82 x 54 mm board,
              2.54 mm pitch, 7.62 mm centre channel, A3R_BreadboardMap)
and evaluates the in-plane error of every terminal hole (300 holes) when each
tapped corner is displaced by isotropic Gaussian error (SD sigma per axis).

Also evaluates (i) a common translational bias caused by an error dh in the
detected table height, because corners are intersected with a plane 9 mm
above the detected plane: bias = dh / tan(elevation of the viewing ray); and
(ii) the closed-form approximation
    RMS_2D(r) ~ sqrt( sigma^2/2 + r^2 sigma^2 / W_eff^2 ),
where r is the hole's distance from the board centre.

This is an error-propagation analysis, NOT a measurement of registration
accuracy: no physical ground truth was available.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results"
W, D = 82.0, 54.0            # mm, A3R_BreadboardConfig
PITCH, GAP = 2.54, 7.62      # mm, A3R_BreadboardMap
HALF_PITCH = PITCH / 2


def terminal_holes():
    xs = (np.arange(1, 31) - 15.5) * PITCH
    zs = []
    for k in range(5):
        zs.append(-(GAP / 2 + k * PITCH)); zs.append(GAP / 2 + k * PITCH)
    X, Z = np.meshgrid(xs, np.array(zs))
    return np.stack([X.ravel(), Z.ravel()], 1)


def corners():
    return np.array([[-W / 2, D / 2], [W / 2, D / 2], [W / 2, -D / 2], [-W / 2, -D / 2]])  # TL TR BR BL


def estimate(pts):
    c = pts.mean(0)
    n = lambda v: v / np.linalg.norm(v)
    r = n(n(pts[1] - pts[0]) + n(pts[2] - pts[3]))
    th = np.arctan2(r[1], r[0])
    return c, th


def simulate(sigma, n=20000, rng=None, bias=(0.0, 0.0)):
    H = terminal_holes()
    C0 = corners()
    err = np.empty((n, len(H)))
    for k in range(n):
        p = C0 + rng.normal(0, sigma, C0.shape) + np.asarray(bias)
        c, th = estimate(p)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        err[k] = np.linalg.norm(c + H @ R.T - H, axis=1)
    return H, err


def main():
    rng = np.random.default_rng(20261113)
    rows, per_hole = [], []
    for sigma in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        H, err = simulate(sigma, rng=rng)
        r = np.linalg.norm(H, axis=1)
        rms = np.sqrt((err ** 2).mean(0))
        W_eff = W  # averaged two-edge estimate: SD(theta) ~ sigma / W
        pred = np.sqrt(sigma ** 2 / 2 + r ** 2 * sigma ** 2 / W_eff ** 2)
        rows.append(dict(sigma_mm=sigma, mean_err_mm=float(err.mean()), median_err_mm=float(np.median(err)),
                         p95_err_mm=float(np.percentile(err, 95)), max_hole_rms_mm=float(rms.max()),
                         min_hole_rms_mm=float(rms.min()),
                         frac_within_half_pitch=float((err < HALF_PITCH).mean()),
                         frac_within_one_pitch=float((err < PITCH).mean()),
                         closed_form_max_rel_dev=float(np.max(np.abs(pred - rms) / rms))))
        for (x, z), e in zip(H, rms):
            per_hole.append(dict(sigma_mm=sigma, x_mm=x, z_mm=z, rms_err_mm=e))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E6_registration_sweep.csv", index=False)
    pd.DataFrame(per_hole).to_csv(OUT / "E6_registration_per_hole.csv", index=False)
    # common bias from plane-height error
    bias_rows = []
    for dh in (1.0, 2.0, 5.0):
        for elev in (30.0, 45.0, 60.0, 75.0):
            bias_rows.append(dict(plane_height_error_mm=dh, view_elevation_deg=elev,
                                  translation_bias_mm=dh / np.tan(np.radians(elev))))
    pd.DataFrame(bias_rows).to_csv(OUT / "E6_plane_height_bias.csv", index=False)
    # sigma at which 95% of hole errors stay below half a pitch
    print(df.round(3).to_string())
    print(pd.DataFrame(bias_rows).round(2).to_string())
    (OUT / "E6_summary.json").write_text(json.dumps(dict(sweep=rows, bias=bias_rows), indent=1))


if __name__ == "__main__":
    main()
