# A³R reproducibility code

Analysis code and result files for the A³R study: temporally gated adaptive guidance driven by facial-geometry heuristics for markerless mobile augmented-reality assembly training.

The application itself is at <https://github.com/vinay-0208/A3R>. All analyses refer to commit `c3c06890fda00d3a08ebf01cd1239f0e40fa2d42`.

**Authors:** Aarif Mohammed J K, Vinay Vishwanathan, Harshavardhan R, P. Deepanramkumar, A. Helen Sharmila — School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, India.

## Contents

| Path | What it is |
|---|---|
| `experiments/a3r_core.py` | Reference model of the app's decision path: features, pose gate, scores and the adaptation controller (dwell, cooldown, cap) |
| `experiments/sim_model.py` | Simulation model, with latent reference events independent of the controller |
| `experiments/test_a3r_core.py`, `test_analyze_runtime_logs.py` | Unit tests |
| `experiments/exp2_reanalyze_scripted.py` | E2: re-analysis of the earlier scripted evaluation |
| `experiments/exp3_exp4_controller.py` | E3/E4: controller ablation, duration response and sensitivity sweeps |
| `experiments/extract_canonical_mesh.py`, `exp5_feature_geometry.py` | E5: canonical face-mesh geometry (pose, distance, noise, deformation) |
| `experiments/exp6_registration.py` | E6: propagation of corner-placement error through four-corner registration |
| `experiments/exp7_real_faces_extract.py`, `exp7_real_faces_analyze.py` | E7: the app's landmark model run on the Pointing'04 head-pose database |
| `experiments/exp8_export_parity_inputs.py`, `parity_cs/`, `exp8_compare_parity.py` | E8: execution of the original C# routines and comparison with the reference model |
| `experiments/analyze_runtime_logs.py` | Analyser for future on-device logs (no device logs exist) |
| `experiments/make_figures.py`, `make_si_figures.py` | Data figures, read only from `results/` |
| `results/` | All result files: summaries, per-replicate simulation outputs (`*.csv.gz`), configurations with random seeds, and derived real-face features |

## Setup

Python 3.11 is required. The pinned analysis environment and the MediaPipe environment are kept separate, because MediaPipe installs newer NumPy builds.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Get the A3R application at the analysed commit. Some tests and E2 use it, as do the steps that need the bundled landmark model or the C# sources:

```bash
git clone https://github.com/vinay-0208/A3R A3R_repo
git -C A3R_repo checkout c3c06890fda00d3a08ebf01cd1239f0e40fa2d42
```

## Reproducing the results

Run all commands from the repository root. The runs are deterministic: seeds are fixed and recorded in `results/E3E4_config.json`.

| Step | Command | Output | Time |
|---|---|---|---|
| Tests | `python -m pytest experiments -q` | 13 passed | < 5 s |
| E2 | `python experiments/exp2_reanalyze_scripted.py` | `results/E2_*` | < 10 s |
| E3/E4 | `python experiments/exp3_exp4_controller.py` (`--quick` for a smoke test) | `results/E3_*`, `results/E4_*` | ~2 min |
| E5 | `python experiments/extract_canonical_mesh.py A3R_repo/unity_project/Assets/StreamingAssets/face_landmarker.task results/canonical_face_468.csv` then `python experiments/exp5_feature_geometry.py` | `results/E5*` | < 1 min |
| E6 | `python experiments/exp6_registration.py` | `results/E6_*` | < 30 s |
| E7 | see below | `results/E7_*` | ~5 min |
| E8 | see below | `results/E8_*` | ~2 min |
| Figures | `python experiments/make_figures.py` and `python experiments/make_si_figures.py` | `figures/panels/` | < 1 min |

A fresh run reproduces every file in `results/` exactly. For the `*.csv.gz` files the match is on the decompressed content, because the gzip header records the time the file was written.

### E7: real faces (Pointing'04)

The images are not redistributed. Download the Pointing'04 database (Figshare, <https://doi.org/10.6084/m9.figshare.5142466.v2>, licence CC BY 4.0) and cite Gourier, Hall & Crowley (2004), as the dataset record requires. Then:

```bash
python -m venv .venv_mp && .venv_mp/Scripts/activate
pip install -r requirements_mediapipe.txt
python experiments/exp7_real_faces_extract.py <pointing04_root> A3R_repo/unity_project/Assets/StreamingAssets/face_landmarker.task results/E7_real_faces.csv
```

Then switch back to the analysis environment and run:

```bash
python experiments/exp7_real_faces_analyze.py
```

`results/E7_real_faces.csv` contains only derived features and scores per image. No images or landmark coordinates are included.

### E8: original C# routines

The .NET 8 SDK is required.

1. In the MediaPipe environment, run `python experiments/exp8_export_parity_inputs.py <pointing04_root> <face_landmarker.task> parity_run`. This writes the landmark arrays to a local folder, which should not be shared.
2. Copy `experiments/parity_cs/*` to a build folder.
3. Copy the three original files `Affect/AffectFeatureExtractor.cs`, `Adaptation/AdaptationController.cs` and `Adaptation/AdaptationEvents.cs` from `A3R_repo/unity_project/Assets/Scripts/` into `<build>/original/`. Their SHA-256 hashes are in `results/E8_original_sources_sha256.txt`.
4. Run `dotnet run -c Release -- <path to parity_run>` in the build folder.
5. Run `python experiments/exp8_compare_parity.py parity_run results/E8_parity_summary.json`.

## Third-party data

* `results/canonical_face_468.csv` is recovered from the MediaPipe FaceLandmarker model bundled with the app (MediaPipe, Apache License 2.0).
* `results/E7_*` are derived from the Pointing'04 head-pose database (CC BY 4.0). N. Gourier, D. Hall and J. L. Crowley, "Estimating face orientation from robust detection of salient facial features", *Proc. Pointing 2004, ICPR International Workshop on Visual Observation of Deictic Gestures*, Cambridge, UK, 2004.
