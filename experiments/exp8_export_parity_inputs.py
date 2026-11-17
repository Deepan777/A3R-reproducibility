"""E8a - Export inputs and Python reference outputs for the C# parity test.

(1) Real-face landmarks: runs the app's FaceLandmarker on the Pointing'04 images
    and writes the normalised landmark arrays (float32, N x 478 x 3) to a
    PRIVATE scratch file together with the Python reference features/scores.
    The landmark file is an intermediate and is not distributed.
(2) Controller traces: 1,000 simulated steps (500 nominal, 500 noisy with face
    loss) from sim_model, with the reference model's firing times.

Usage (MediaPipe environment):
  python exp8_export_parity_inputs.py <image_root> <face_landmarker.task> <out_dir>
"""
import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a3r_core as core  # noqa: E402
from sim_model import SimParams, simulate  # noqa: E402


def landmarks(root, task, out):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=Path(task).read_bytes(),
                                     delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    det = vision.FaceLandmarker.create_from_options(opts)
    arrs, rows = [], []
    for p in sorted(Path(root).rglob("personne*.jpg")):
        if not p.parent.name.lower().startswith("personne"):
            continue
        res = det.detect(mp.Image.create_from_file(str(p)))
        if not res.face_landmarks:
            continue
        L = np.array([[q.x, q.y, q.z] for q in res.face_landmarks[0]], dtype=np.float32)
        f = core.extract_features(L.astype(np.float64))
        C, F = core.scores(f.B, f.E, f.M)
        arrs.append(L)
        rows.append([len(rows), f.B, f.E, f.M, f.yaw_ratio, f.pitch_ratio, int(f.pose_gated), float(C), float(F)])
    np.stack(arrs).astype("<f4").tofile(out / "landmarks_f32.bin")
    with open(out / "py_features.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["i", "B", "E", "M", "yaw", "pitch", "gated", "C", "F"]); w.writerows(rows)
    print("landmark sets:", len(rows))


def traces(out):
    rng = np.random.default_rng(424242)
    regimes = [("nominal", SimParams()), ("noisy_loss", replace(SimParams(), noise_sd=0.08, dropout_rate=4.0))]
    cfg = core.ControllerConfig()
    with open(out / "traces.csv", "w", newline="") as ft, open(out / "py_fires.csv", "w", newline="") as ff:
        wt, wf = csv.writer(ft), csv.writer(ff)
        wt.writerow(["trace", "t", "C", "valid"]); wf.writerow(["trace", "t_fire"])
        k = 0
        for _, p in regimes:
            for _ in range(500):
                tr = simulate(p, rng)
                # C# runtime receives C = F = 0 for frames without a valid face
                C = np.where(tr.valid, tr.C_obs, 0.0).astype(np.float32)
                t = tr.t_obs.astype(np.float32)          # Unity Time.time is single precision
                for ti, ci, vi in zip(t, C, tr.valid):
                    wt.writerow([k, repr(float(ti)), repr(float(ci)), int(vi)])
                fired = core.run_controller(cfg, t.astype(np.float64), C.astype(np.float64), None, tr.valid).fired
                for x in fired:
                    wf.writerow([k, repr(float(x))])
                k += 1
    print("traces:", k)


if __name__ == "__main__":
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    traces(out)
    landmarks(sys.argv[1], sys.argv[2], out)
