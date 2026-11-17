"""Conformance tests for the Python reference of the A3R C# decision path.

Run:  python -m pytest experiments/test_a3r_core.py -q
The repository twin (python_tools/adaptation/adaptation_policy.py) is used as an
independent cross-check for the scenarios in which both implementations are
specified to agree (single step, no face loss, identical thresholds).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a3r_core import (ControllerConfig, AdaptationControllerCS, run_controller, scores,
                      extract_features)

REPO = HERE.parent / "A3R_repo"


def feed(cfg, seq, dt=0.5, step="s"):
    ctl = AdaptationControllerCS(cfg)
    fired = []
    for i, c in enumerate(seq):
        if ctl.update(i * dt, c, 0.0, step):
            fired.append(i * dt)
    return fired, ctl


def test_dwell_fires_at_first_sample_reaching_dwell():
    fired, _ = feed(ControllerConfig(), [0.5] * 10)
    assert fired == [3.0]          # t0 = 0, dwell 3.0 s reached at sample 6


def test_short_excursion_rejected():
    seq = [0.1] * 2 + [0.5] * 6 + [0.1] * 4  # 2.5 s above threshold
    fired, _ = feed(ControllerConfig(), seq)
    assert fired == []


def test_strict_inequality():
    fired, _ = feed(ControllerConfig(), [0.40] * 20)
    assert fired == []


def test_single_fire_per_run_without_cooldown():
    cfg = ControllerConfig(use_cooldown=False, use_step_cap=False)
    fired, _ = feed(cfg, [0.6] * 100)
    assert fired == [3.0]


def test_cooldown_reattempt_fires_when_cooldown_expires():
    # run 1 fires at 3.0; drop; run 2 starts at 5.0, dwell satisfied at 8.0,
    # suppressed by cooldown until 18.0, then fires at the first sample >= 18.0
    seq = [0.6] * 8 + [0.1] * 2 + [0.6] * 60
    fired, ctl = feed(ControllerConfig(), seq)
    assert fired == [3.0, 18.0]
    assert ctl.log.suppressed_cooldown > 0


def test_step_cap():
    seq = []
    for _ in range(6):
        seq += [0.6] * 40 + [0.1] * 2
    fired, ctl = feed(ControllerConfig(), seq)
    assert len(fired) == 3


def test_step_change_resets_cooldown_and_counter():
    cfg = ControllerConfig()
    ctl = AdaptationControllerCS(cfg)
    out = []
    for i in range(10):
        if ctl.update(i * 0.5, 0.6, 0, "a"):
            out.append(("a", i * 0.5))
    for i in range(10, 20):
        if ctl.update(i * 0.5, 0.6, 0, "b"):
            out.append(("b", i * 0.5))
    assert out == [("a", 3.0), ("b", 8.0)]


def test_face_loss_resets_in_runtime_semantics():
    seq_valid = [True] * 4 + [False] + [True] * 10
    ctl = AdaptationControllerCS(ControllerConfig())
    fired = [i * 0.5 for i, v in enumerate(seq_valid) if ctl.update(i * 0.5, 0.6, 0, "s", v)]
    assert fired == [5.5]          # run restarts at 2.5 s
    ctl = AdaptationControllerCS(ControllerConfig(dropout_policy="hold"))
    fired = [i * 0.5 for i, v in enumerate(seq_valid) if ctl.update(i * 0.5, 0.6, 0, "s", v)]
    assert fired == [3.0]


def test_frustration_branch_is_implied_by_confusion_branch():
    """Analytic claim: F > 0.65 implies C > 0.55 for all B, E, M in [0, 1]."""
    g = np.linspace(0, 1, 41)
    B, E, M = np.meshgrid(g, g, g, indexing="ij")
    C, F = scores(B, E, M)
    assert np.all(C[F > 0.65] > 0.55)
    assert np.min(C - F) >= -0.10 - 1e-12


def test_pose_gated_frames_cannot_cross_threshold():
    M = np.linspace(0, 1, 101)
    C, F = scores(0.0, 1.0, M)
    assert C.max() <= 0.30 + 1e-12 and F.max() <= 0.40 + 1e-12


def test_cross_check_against_repository_twin():
    sys.path.insert(0, str(REPO))
    try:
        from python_tools.adaptation.adaptation_policy import AdaptationController as Twin
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"repository twin unavailable: {exc}")
    rng = np.random.default_rng(7)
    for _ in range(200):
        seq = np.clip(0.35 + 0.15 * np.cumsum(rng.normal(0, 0.25, 160)) / 6, 0, 1)
        twin = Twin(confusion_threshold=0.40, frustration_threshold=0.65, dwell_time_seconds=3.0,
                    cooldown_seconds=15.0, max_adaptations_per_step=3)
        a = [i * 0.5 for i, c in enumerate(seq)
             if twin.update_affect_scores(i * 0.5, float(c), 0.0, 0.5, "s") is not None]
        b = run_controller(ControllerConfig(), np.arange(160) * 0.5, seq).fired
        assert a == b


def test_feature_extractor_on_synthetic_frontal_geometry():
    P = np.zeros((478, 3))
    # minimal frontal configuration in normalised image coordinates
    P[33] = [0.40, 0.40, 0]; P[263] = [0.60, 0.40, 0]
    P[133] = [0.46, 0.40, 0]; P[362] = [0.54, 0.40, 0]
    P[159] = [0.43, 0.39, 0]; P[145] = [0.43, 0.41, 0]
    P[386] = [0.57, 0.39, 0]; P[374] = [0.57, 0.41, 0]
    P[105] = [0.405, 0.35, 0]; P[334] = [0.595, 0.35, 0]
    P[13] = [0.5, 0.60, 0]; P[14] = [0.5, 0.62, 0]
    P[61] = [0.45, 0.61, 0]; P[291] = [0.55, 0.61, 0]
    P[1] = [0.5, 0.5, 0]; P[10] = [0.5, 0.25, 0]; P[152] = [0.5, 0.72, 0]
    f = extract_features(P)
    assert f.valid and not f.pose_gated
    assert abs(f.brow_raw - 0.95) < 1e-9 and abs(f.B - 0.375) < 1e-9
    assert abs(f.ear_raw - 0.02 / (2 * 0.06)) < 1e-9
    assert f.M == 0.0
