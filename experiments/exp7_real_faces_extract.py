"""E7a - Run A3R's own FaceLandmarker model on real faces with known head pose.

Dataset: Pointing'04 head-pose database (Gourier, Hall & Crowley, 2004), 15
people x 2 sessions x 93 head orientations (pan -90..+90 deg, tilt -90..+90 deg),
384 x 288 px colour images.  Copy used: Figshare 10.6084/m9.figshare.5142466.v2,
licence CC BY 4.0 (uploaded by a third party; original authors must be cited).
Pose labels are nominal (subjects turned their heads towards markers) and are
approximate.

The script uses the face_landmarker.task bundled with the A3R app, IMAGE mode,
CPU, one face, detection/presence confidence 0.5 (as in FaceLandmarkBridge),
and computes A3R's features and scores with the reference model (a3r_core)
in two coordinate conventions:
  native : MediaPipe's normalised coordinates of this 384x288 image
           (x/W, y/H, z) - exactly what the app's code would compute on it;
  iso    : all coordinates in pixel units (x*W, y*H, z*W) - removes the
           width/height normalisation anisotropy.
Only derived quantities (features, scores, ratios) are written; no images or
raw landmark coordinates of the depicted people are stored in the outputs.

Usage (MediaPipe environment):
  python exp7_real_faces_extract.py <image_root> <face_landmarker.task> <out_csv>
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3r_core as core  # noqa: E402

NAME = re.compile(r"personne(\d\d)(\d)(\d\d)([+-]\d+)([+-]\d+)\.jpg$", re.I)


def feats(P):
    f = core.extract_features(P)
    if not f.valid:
        return dict(valid=False)
    C, F = core.scores(f.B, f.E, f.M)
    return dict(valid=True, r_B=f.brow_raw, B=f.B, ear=f.ear_raw, E=f.E, mouth=f.mouth_raw, M=f.M,
                yaw_ratio=f.yaw_ratio, pitch_ratio=f.pitch_ratio, gated=int(f.pose_gated), C=float(C), F=float(F))


def main(root, task, out_csv):
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=Path(task).read_bytes(),
                                     delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
    det = vision.FaceLandmarker.create_from_options(opts)
    rows = []
    files = sorted(p for p in Path(root).rglob("*.jpg") if p.parent.name.lower().startswith("personne"))
    for p in files:
        m = NAME.search(p.name)
        if not m:
            continue
        person, series, _, tilt, pan = m.groups()
        img = mp.Image.create_from_file(str(p))
        W, H = img.width, img.height
        res = det.detect(img)
        base = dict(person=int(person), series=int(series), tilt=int(tilt), pan=int(pan), width=W, height=H)
        if not res.face_landmarks:
            rows.append({**base, "detected": 0})
            continue
        L = np.array([[q.x, q.y, q.z] for q in res.face_landmarks[0]])
        iso = L * np.array([W, H, W])
        nat = feats(L)
        isof = feats(iso)
        iod_px = float(np.linalg.norm(iso[33, :2] - iso[263, :2]))
        r = {**base, "detected": 1, "iod_px": iod_px}
        r.update({f"nat_{k}": v for k, v in nat.items()})
        r.update({f"iso_{k}": v for k, v in isof.items()})
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in base, k))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images, {sum(r['detected'] for r in rows)} with a face -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
