"""Faithful Python reference of the A3R runtime decision path.

Every function in this module mirrors a specific C# routine in the A3R Unity
project at commit c3c06890fda00d3a08ebf01cd1239f0e40fa2d42:

  extract_features()      <- Assets/Scripts/Affect/AffectFeatureExtractor.cs  Extract()
  scores()                <- Assets/Scripts/Affect/AffectScorer.cs            ProcessFrame()
  AdaptationControllerCS  <- Assets/Scripts/Adaptation/AdaptationController.cs UpdateAffectScores()

The repository already contains a Python "twin" (python_tools/adaptation/
adaptation_policy.py), but it diverges from the C# runtime in five respects
(default thresholds, tracking-loss handling, cooldown reset on step change,
action sequence and tutorial-state gating; see Research_Audit.md).  This module
was therefore written directly from the C# source and is the reference used by
all experiments in this study.

Controller variants used in the ablation study are implemented as options of a
single class so that every variant shares identical code except for the
mechanism under test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ----------------------------------------------------------------------------
# Landmark indices (AffectFeatureExtractor.cs)
# ----------------------------------------------------------------------------
LM_L_INNER_BROW, LM_R_INNER_BROW = 105, 334
LM_L_EYE_OUTER, LM_R_EYE_OUTER = 33, 263
LM_L_EYE_TOP, LM_L_EYE_BOT, LM_L_EYE_INNER = 159, 145, 133
LM_R_EYE_TOP, LM_R_EYE_BOT, LM_R_EYE_INNER = 386, 374, 362
LM_L_MOUTH, LM_R_MOUTH = 61, 291
LM_UPPER_LIP, LM_LOWER_LIP = 13, 14
LM_L_OUTER_BROW, LM_R_OUTER_BROW = 65, 295
LM_FACE_TOP, LM_FACE_BOT, LM_NOSE_TIP = 10, 152, 1

EAR_MIN, EAR_MAX = 0.04, 0.16
BROW_HI, BROW_LO = 0.98, 0.90          # r_B normalisation interval
MOUTH_SCALE = 0.008                    # normalised-image units
POSE_YAW = (0.90, 1.15)
POSE_PITCH = (0.85, 1.45)

# Score weights (AffectScorer.cs)
W_C = (0.40, 0.30, 0.30)   # brow, (1-eye), mouth
W_F = (0.35, 0.25, 0.40)

# Active scene configuration (A3R_Main.unity, AdaptationController/AffectScorer)
ACTIVE = dict(t_conf=0.40, t_frust=0.65, dwell=3.0, cooldown=15.0, max_per_step=3,
              update_interval=0.5)
# C# field initialisers (used only when a component is created outside the scene)
CODE_DEFAULT = dict(t_conf=0.40, t_frust=0.65, dwell=2.0, cooldown=15.0, max_per_step=3)


def clamp01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


@dataclass
class Features:
    brow_raw: float
    B: float
    ear_raw: float
    E: float
    mouth_raw: float
    M: float
    yaw_ratio: float
    pitch_ratio: float
    pose_gated: bool
    valid: bool = True


def _d(P, i, j):
    return float(np.linalg.norm(P[i] - P[j]))


def extract_features(P: np.ndarray) -> Features:
    """P: (>=468, 3) array of landmarks in MediaPipe normalised image
    coordinates (x right, y down, z depth), exactly as AffectFeatureExtractor
    receives them."""
    iod = _d(P, LM_L_EYE_OUTER, LM_R_EYE_OUTER)
    face_h = abs(P[LM_FACE_BOT, 1] - P[LM_FACE_TOP, 1])
    if iod < 1e-6 or face_h < 1e-6:
        return Features(0, 0, 0, 0, 0, 0, 0, 0, False, valid=False)
    brow_raw = _d(P, LM_L_INNER_BROW, LM_R_INNER_BROW) / iod
    B = float(clamp01((BROW_HI - brow_raw) / (BROW_HI - BROW_LO)))

    def ear(top, bot, inner, outer):
        h = _d(P, inner, outer)
        return _d(P, top, bot) / (2.0 * h) if h > 1e-6 else 0.0

    ear_raw = 0.5 * (ear(LM_L_EYE_TOP, LM_L_EYE_BOT, LM_L_EYE_INNER, LM_L_EYE_OUTER)
                     + ear(LM_R_EYE_TOP, LM_R_EYE_BOT, LM_R_EYE_INNER, LM_R_EYE_OUTER))
    E = float(clamp01((ear_raw - EAR_MIN) / (EAR_MAX - EAR_MIN)))

    lip_mid_y = 0.5 * (P[LM_UPPER_LIP, 1] + P[LM_LOWER_LIP, 1])
    disp = 0.5 * ((lip_mid_y - P[LM_L_MOUTH, 1]) + (lip_mid_y - P[LM_R_MOUTH, 1]))
    M = float(clamp01((0.0 - disp) / MOUTH_SCALE))

    nose = P[LM_NOSE_TIP]
    yaw = float(np.linalg.norm(nose - P[LM_L_EYE_OUTER]) / np.linalg.norm(nose - P[LM_R_EYE_OUTER]))
    pitch = float(np.linalg.norm(nose - P[LM_FACE_TOP]) / np.linalg.norm(nose - P[LM_FACE_BOT]))
    pose_ok = POSE_YAW[0] <= yaw <= POSE_YAW[1] and POSE_PITCH[0] <= pitch <= POSE_PITCH[1]
    if not pose_ok:
        B, E = 0.0, 1.0          # mouth term deliberately left unchanged (as in C#)
    return Features(brow_raw, B, ear_raw, E, disp, M, yaw, pitch, not pose_ok)


def scores(B, E, M):
    """Confusion and frustration scores (AffectScorer.cs).  Accepts scalars or
    arrays.  Because the weights are non-negative and sum to one and every
    feature lies in [0, 1], the Clamp01 in the C# source is never active."""
    C = clamp01(W_C[0] * B + W_C[1] * (1 - E) + W_C[2] * M)
    F = clamp01(W_F[0] * B + W_F[1] * (1 - E) + W_F[2] * M)
    return C, F


# ----------------------------------------------------------------------------
# Temporal adaptation controller
# ----------------------------------------------------------------------------
@dataclass
class ControllerConfig:
    t_conf: float = 0.40
    t_frust: float = 0.65
    dwell: float = 3.0
    cooldown: float = 15.0
    max_per_step: int = 3
    # Ablation switches -------------------------------------------------------
    use_dwell: bool = True        # False -> edge-triggered on threshold crossing
    use_cooldown: bool = True
    use_step_cap: bool = True
    # Candidate refinements (NOT in the deployed runtime) ---------------------
    hysteresis: float = 0.0       # exit threshold = T - hysteresis
    dropout_policy: str = "reset"  # "reset" (C# runtime: zeros reset dwell) or "hold"


@dataclass
class ControllerLog:
    fired: List[float] = field(default_factory=list)
    suppressed_cooldown: int = 0
    suppressed_cap: int = 0


class AdaptationControllerCS:
    """Line-by-line port of AdaptationController.UpdateAffectScores for the case
    in which the tutorial is in an armed placement state (the only state in
    which the C# method proceeds past its guard).

    Semantics reproduced exactly from the C# source:
      * elevated if C > T_c OR F > T_f (strict inequalities);
      * dwell measured as wall-clock time since the first above-threshold
        sample of the current run, evaluated only when a sample arrives;
      * one firing attempt per continuous run (_fsmFired latch);
      * cooldown is checked before the per-step cap; a suppressed attempt does
        NOT set the latch, so it is re-attempted at every subsequent sample of
        the same run (and therefore can fire once the cooldown expires);
      * a step change resets the run, the per-step counter and the cooldown;
      * a frame without a valid face produces C = F = 0 upstream, which resets
        the run (dropout_policy = "reset").
    """

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.log = ControllerLog()
        self._in_elev = False
        self._t0 = 0.0
        self._latched = False
        self._last_fire = -999.0
        self._n_step = 0
        self._step = None

    def _reset_run(self):
        self._in_elev = False
        self._t0 = 0.0
        self._latched = False

    def update(self, t: float, C: float, F: float, step_id: str = "s",
               face_valid: bool = True) -> bool:
        cfg = self.cfg
        if math.isnan(C) or math.isnan(F) or math.isinf(C) or math.isinf(F):
            return False
        if step_id != self._step:
            self._reset_run()
            self._n_step = 0
            self._last_fire = -999.0
            self._step = step_id
        if not face_valid:
            if cfg.dropout_policy == "hold":
                return False          # freeze: neither advance nor reset
            C, F = 0.0, 0.0           # C# runtime behaviour
        enter = (C > cfg.t_conf) or (F > cfg.t_frust)
        if cfg.hysteresis > 0 and self._in_elev:
            stay = (C > cfg.t_conf - cfg.hysteresis) or (F > cfg.t_frust - cfg.hysteresis)
        else:
            stay = enter
        above = stay if self._in_elev else enter

        if not above:
            self._reset_run()
            return False
        if not self._in_elev:
            self._in_elev = True
            self._t0 = t
            self._latched = False
        dwell_t = t - self._t0
        required = cfg.dwell if cfg.use_dwell else 0.0
        if self._latched or dwell_t < required:
            return False
        if cfg.use_cooldown and (t - self._last_fire) < cfg.cooldown:
            self.log.suppressed_cooldown += 1
            return False
        if cfg.use_step_cap and self._n_step >= cfg.max_per_step:
            self.log.suppressed_cap += 1
            return False
        self._latched = True
        self._last_fire = t
        self._n_step += 1
        self.log.fired.append(t)
        return True


def run_controller(cfg: ControllerConfig, t: np.ndarray, C: np.ndarray,
                   F: Optional[np.ndarray] = None, valid: Optional[np.ndarray] = None,
                   step_ids=None) -> ControllerLog:
    ctl = AdaptationControllerCS(cfg)
    if F is None:
        F = np.zeros_like(C)
    if valid is None:
        valid = np.ones_like(C, dtype=bool)
    for i in range(len(t)):
        sid = "s" if step_ids is None else step_ids[i]
        ctl.update(float(t[i]), float(C[i]), float(F[i]), sid, bool(valid[i]))
    return ctl.log


VARIANTS = {
    "V0 threshold only": dict(use_dwell=False, use_cooldown=False, use_step_cap=False),
    "V1 + dwell": dict(use_dwell=True, use_cooldown=False, use_step_cap=False),
    "V2 + cooldown": dict(use_dwell=True, use_cooldown=True, use_step_cap=False),
    "V3 full (A3R)": dict(use_dwell=True, use_cooldown=True, use_step_cap=True),
}
