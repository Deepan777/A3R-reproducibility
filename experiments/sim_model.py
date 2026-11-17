"""Generative observation model used for the controller experiments (E3, E4).

Reference labels are defined on a latent, noise-free *intent* process l(t) in
{0, 1} that is generated independently of any controller rule.  The controller
only sees what the deployed runtime would see: a score that is
  (i)   sampled at asynchronous FaceLandmarker callbacks (rate f_mp, jittered),
  (ii)  corrupted by temporally correlated measurement noise (AR(1)),
  (iii) delivered after an inference latency,
  (iv)  read by the scorer coroutine every ~0.5 s with sample-and-hold
        semantics (latest available landmark frame, as in FaceLandmarkBridge),
  (v)   replaced by "no face" (C = 0 in the runtime) during interruptions.

The model is a deliberately simple, fully specified surrogate.  It is NOT a
model of human confusion; it is a test bench for the question "does the
temporal gate respond to sustained score elevations and reject brief ones when
the score is observed the way the runtime observes it?".
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import lfilter

DT = 0.02  # s, resolution of the latent process


@dataclass
class SimParams:
    duration: float = 120.0          # s, one assembly step
    # latent event process --------------------------------------------------
    gap_mean: float = 12.0           # s, mean gap between latent events (exp.)
    gap_min: float = 3.0             # s
    p_sustained: float = 0.30        # probability an event is sustained
    sustained_range: tuple = (5.0, 20.0)   # s  -> intervention intended
    transient_range: tuple = (0.2, 2.0)    # s  -> intervention not intended
    fixed_durations: tuple | None = None   # override: list of single-event durations
    tau: float = 0.30                # s, first-order onset/offset time constant
    # score levels (relative to the confusion threshold T) -------------------
    T: float = 0.40
    base_margin: tuple = (0.06, 0.18)  # T - baseline  ~ U(.)
    elev_margin: tuple = (0.05, 0.20)  # elevated - T ~ U(.)
    # observation ------------------------------------------------------------
    noise_sd: float = 0.04           # SD of AR(1) score noise per callback
    noise_rho: float = 0.5           # AR(1) coefficient between callbacks
    f_mp: float = 4.5                # Hz, landmark callback rate
    mp_jitter: float = 0.2           # +/- fraction of callback interval
    latency: tuple = (0.015, 0.045)  # s, inference latency ~ U(.)
    extra_delay: float = 0.0         # s, additional delivery delay
    update_interval: float = 0.5     # s, scorer coroutine period
    frame_quant: float = 1 / 60      # s, coroutine resumes on frame boundaries
    dropout_rate: float = 0.0        # interruptions per minute (Poisson)
    dropout_range: tuple = (0.3, 3.0)  # s


@dataclass
class Trace:
    t_obs: np.ndarray
    C_obs: np.ndarray
    valid: np.ndarray
    events: list            # list of (onset, offset, kind)
    T: float


def latent_events(p: SimParams, rng):
    ev = []
    if p.fixed_durations is not None:
        t = 10.0
        for d in p.fixed_durations:
            ev.append((t, t + d, "probe"))
            t += d + 25.0
        return ev
    t = 5.0 + rng.exponential(p.gap_mean)
    while t < p.duration - 5.0:
        if rng.random() < p.p_sustained:
            d, kind = rng.uniform(*p.sustained_range), "sustained"
        else:
            d, kind = rng.uniform(*p.transient_range), "transient"
        off = min(t + d, p.duration - 1.0)
        ev.append((t, off, kind))
        t = off + p.gap_min + rng.exponential(p.gap_mean)
    return ev


def simulate(p: SimParams, rng) -> Trace:
    ev = latent_events(p, rng)
    dur = p.duration if p.fixed_durations is None else ev[-1][1] + 20.0
    n = int(dur / DT) + 1
    tt = np.arange(n) * DT
    l = np.zeros(n)
    for on, off, _ in ev:
        l[(tt >= on) & (tt < off)] = 1.0
    # first-order smoothing (facial movement is not instantaneous)
    a = DT / (p.tau + DT)
    ls = lfilter([a], [1.0, -(1.0 - a)], l)
    base = p.T - rng.uniform(*p.base_margin)
    elev = p.T + rng.uniform(*p.elev_margin)
    s_true = base + (elev - base) * ls

    # interruptions (face lost / pose rejected)
    lost = np.zeros(n, dtype=bool)
    if p.dropout_rate > 0:
        k = rng.poisson(p.dropout_rate * dur / 60.0)
        for st in rng.uniform(0, dur, k):
            lost[(tt >= st) & (tt < st + rng.uniform(*p.dropout_range))] = True

    # landmark callbacks
    step = 1.0 / p.f_mp
    caps = []
    t = rng.uniform(0, step)
    while t < dur:
        caps.append(t)
        t += step * rng.uniform(1 - p.mp_jitter, 1 + p.mp_jitter)
    caps = np.array(caps)
    idx = np.minimum((caps / DT).astype(int), n - 1)
    # stationary AR(1): z_k = rho z_{k-1} + sqrt(1-rho^2) e_k, e_k ~ N(0, sd^2)
    e = rng.normal(0, p.noise_sd, len(caps))
    z0 = rng.normal(0, p.noise_sd)
    noise, _ = lfilter([np.sqrt(1 - p.noise_rho ** 2)], [1.0, -p.noise_rho], e,
                       zi=[p.noise_rho * z0])
    cap_val = np.clip(s_true[idx] + noise, 0, 1)
    cap_face = ~lost[idx]
    avail = caps + rng.uniform(*p.latency, len(caps)) + p.extra_delay
    order = np.argsort(avail, kind="stable")
    avail, cap_val, cap_face = avail[order], cap_val[order], cap_face[order]

    # scorer ticks with sample-and-hold
    ticks = []
    t = p.update_interval
    while t < dur:
        ticks.append(t)
        t += p.update_interval + rng.uniform(0, p.frame_quant)
    ticks = np.array(ticks)
    j = np.searchsorted(avail, ticks, side="right") - 1
    has = j >= 0
    C_obs = np.where(has, cap_val[np.maximum(j, 0)], 0.0)
    valid = has & cap_face[np.maximum(j, 0)]
    C_obs = np.where(valid, C_obs, 0.0)
    return Trace(ticks, C_obs, valid, ev, p.T)


def score_trace(tr: Trace, fired: list, grace: float = 1.0):
    """Event-level scoring against the latent reference.

    eligible  = sustained latent episodes (intervention intended)
    hit       = >= 1 intervention in [onset, offset + grace]
    false     = intervention outside every eligible window
    redundant = interventions after the first within the same eligible window
    """
    elig = [(on, off) for on, off, k in tr.events if k in ("sustained", "probe")]
    hits, delays, redundant = 0, [], 0
    used = np.zeros(len(fired), dtype=bool)
    for on, off in elig:
        inwin = [i for i, f in enumerate(fired) if on <= f <= off + grace]
        if inwin:
            hits += 1
            delays.append(fired[inwin[0]] - on)
            redundant += len(inwin) - 1
        used[inwin] = True
    false = int((~used).sum())
    return dict(n_eligible=len(elig), hits=hits, delays=delays, false=false,
                redundant=redundant, n_fired=len(fired))
