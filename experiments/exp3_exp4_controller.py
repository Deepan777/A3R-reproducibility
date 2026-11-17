"""E3 (ablation, duration response) and E4 (sensitivity, robustness) of the
temporal adaptation controller.

Unit of analysis: one simulated assembly step (independent replicate with its
own seed, baseline, elevation level, event schedule and noise realisation).
All ratios are pooled over replicates; 95% CIs are percentile intervals of a
cluster bootstrap that resamples replicates (B = 1000), so score updates within
a step are never treated as independent observations.

Usage: python exp3_exp4_controller.py [--quick]
Outputs: results/E3_*.csv, results/E4_*.csv, results/E3E4_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
sys.path.insert(0, str(HERE))
from a3r_core import ControllerConfig, run_controller, VARIANTS  # noqa: E402
from sim_model import SimParams, simulate, score_trace  # noqa: E402

MASTER_SEED = 20261113
B_BOOT = 1000


def run_condition(args):
    """Simulate R replicates for one (SimParams, ControllerConfig) pair.
    Every controller variant listed in `cfgs` is applied to the SAME traces
    (common random numbers), which makes between-variant contrasts paired."""
    label, p, cfgs, R, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        tr = simulate(p, rng)
        for name, cfg in cfgs.items():
            log = run_controller(cfg, tr.t_obs, tr.C_obs, None, tr.valid)
            s = score_trace(tr, log.fired)
            dur_min = (tr.t_obs[-1] + p.update_interval) / 60.0
            rows.append(dict(condition=label, variant=name, rep=r, dur_min=dur_min,
                             n_eligible=s["n_eligible"], hits=s["hits"], false=s["false"],
                             redundant=s["redundant"], n_fired=s["n_fired"],
                             delay_sum=float(np.sum(s["delays"])), n_delay=len(s["delays"]),
                             delays=";".join(f"{d:.3f}" for d in s["delays"]),
                             valid_frac=float(tr.valid.mean())))
    return rows


def summarise(df: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    for (cond, var), g in df.groupby(["condition", "variant"], sort=False):
        H, N = g["hits"].to_numpy(), g["n_eligible"].to_numpy()
        Fp, D = g["false"].to_numpy(), g["dur_min"].to_numpy()
        Rd, Nf = g["redundant"].to_numpy(), g["n_fired"].to_numpy()
        ds, nd = g["delay_sum"].to_numpy(), g["n_delay"].to_numpy()
        R = len(g)
        idx = rng.integers(0, R, (B_BOOT, R))

        def ratio(a, b, scale=1.0):
            est = scale * a.sum() / max(b.sum(), 1e-12)
            bs = scale * a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-12)
            return est, *np.percentile(bs, [2.5, 97.5])

        hr = ratio(H, N)
        fr = ratio(Fp, D, 10.0)                # false interventions per 10 min
        rr = ratio(Rd, H)                      # redundant per hit episode
        md = ratio(ds, nd)                     # mean delay (s)
        all_d = np.array([float(x) for s in g["delays"] if s for x in s.split(";")])
        out.append(dict(condition=cond, variant=var, replicates=R,
                        eligible_episodes=int(N.sum()), interventions=int(Nf.sum()),
                        hit_rate=hr[0], hit_lo=hr[1], hit_hi=hr[2],
                        false_per10min=fr[0], false_lo=fr[1], false_hi=fr[2],
                        redundant_per_hit=rr[0], red_lo=rr[1], red_hi=rr[2],
                        mean_delay=md[0], delay_lo=md[1], delay_hi=md[2],
                        median_delay=float(np.median(all_d)) if len(all_d) else np.nan,
                        p90_delay=float(np.percentile(all_d, 90)) if len(all_d) else np.nan,
                        max_per_step=int(g["n_fired"].max()),
                        valid_frac=float(g["valid_frac"].mean())))
    return pd.DataFrame(out)


def build_jobs(quick: bool):
    R = 150 if quick else 1000
    Rs = 100 if quick else 500
    base = SimParams()
    full = ControllerConfig()
    ss = np.random.SeedSequence(MASTER_SEED)
    jobs = []

    def add(group, label, p, cfgs, R_):
        jobs.append((group, (label, p, cfgs, R_, int(ss.spawn(1)[0].generate_state(1)[0]))))

    ablation = {k: ControllerConfig(**v) for k, v in VARIANTS.items()}
    refinements = {
        "V3 full (A3R)": full,
        "V3 + hysteresis 0.05": replace(full, hysteresis=0.05),
        "V3 + hold on dropout": replace(full, dropout_policy="hold"),
        "V3 + both": replace(full, hysteresis=0.05, dropout_policy="hold"),
    }
    # E3a ablation under three observation regimes ---------------------------
    add("E3_ablation", "nominal (sd 0.04)", base, ablation, R)
    add("E3_ablation", "noisy (sd 0.08)", replace(base, noise_sd=0.08), ablation, R)
    add("E3_ablation", "noisy + 4 interruptions/min", replace(base, noise_sd=0.08, dropout_rate=4.0), ablation, R)
    add("E3_ablation", "persistent difficulty", replace(base, p_sustained=0.8, gap_mean=6.0,
                                                          sustained_range=(10.0, 40.0)), ablation, R)
    # E3b duration response ------------------------------------------------
    for sd in (0.04, 0.08):
        for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0):
            add("E3_duration", f"sd={sd}|d={d}", replace(base, fixed_durations=(d,), noise_sd=sd),
                {k: ablation[k] for k in ("V0 threshold only", "V1 + dwell")}, Rs)
    # E4 sensitivity (full controller, nominal observation) ----------------
    for T in (0.30, 0.35, 0.40, 0.45, 0.50):
        # the latent levels are defined relative to the nominal T = 0.40
        add("E4_threshold", f"T={T}", base, {"V3": replace(full, t_conf=T)}, Rs)
    for D in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        add("E4_dwell", f"D={D}", base, {"V3": replace(full, dwell=D)}, Rs)
    pd_ = replace(base, p_sustained=0.8, gap_mean=6.0, sustained_range=(10.0, 40.0))
    for cd in (0.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        add("E4_cooldown", f"cooldown={cd}", pd_, {"V3": replace(full, cooldown=cd)}, Rs)
    for cap in (1, 2, 3, 5, 1000):
        add("E4_cap", f"cap={cap}", pd_, {"V3": replace(full, max_per_step=cap)}, Rs)
    # E4 robustness (full controller and candidate refinements) -------------
    for sd in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        add("E4_noise", f"sd={sd}", replace(base, noise_sd=sd), refinements, Rs)
    for rho in (0.0, 0.5, 0.8, 0.95):
        add("E4_noise_corr", f"rho={rho}", replace(base, noise_sd=0.06, noise_rho=rho), refinements, Rs)
    for lam in (0.0, 1.0, 2.0, 4.0, 8.0):
        add("E4_dropout", f"rate={lam}/min", replace(base, dropout_rate=lam), refinements, Rs)
    for f in (2.0, 3.0, 4.5, 6.0, 10.0):
        add("E4_mp_rate", f"f_mp={f}", base, {"V3": full}, Rs)
    for ui in (0.25, 0.5, 1.0):
        add("E4_update_interval", f"interval={ui}", replace(base, update_interval=ui), {"V3": full}, Rs)
    for j in (0.0, 0.2, 0.5):
        add("E4_mp_jitter", f"jitter={j}", replace(base, mp_jitter=j), {"V3": full}, Rs)
    for dl in (0.0, 0.1, 0.25, 0.5, 1.0):
        add("E4_delay", f"delay={dl}", replace(base, extra_delay=dl), {"V3": full}, Rs)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = build_jobs(a.quick)
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(run_condition, [j for _, j in jobs]))
    else:  # deterministic serial execution (default)
        results = [run_condition(j) for _, j in jobs]
    print(f"{len(jobs)} conditions simulated in {time.time() - t0:.1f} s")
    rng = np.random.default_rng(MASTER_SEED + 1)
    groups = {}
    for (g, job), rows in zip(jobs, results):
        groups.setdefault(g, []).extend(rows)
    suffix = "_quick" if a.quick else ""
    for g, rows in groups.items():
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{g}_replicates{suffix}.csv.gz", index=False, compression="gzip")
        summarise(df, rng).to_csv(OUT / f"{g}_summary{suffix}.csv", index=False)
    cfg = dict(master_seed=MASTER_SEED, bootstrap=B_BOOT, sim_defaults=asdict(SimParams()),
               controller_active=asdict(ControllerConfig()), variants=VARIANTS,
               conditions=[dict(group=g, label=j[0], replicates=j[3], seed=j[4],
                                sim=asdict(j[1]), controllers={k: asdict(v) for k, v in j[2].items()})
                           for g, j in jobs])
    (OUT / f"E3E4_config{suffix}.json").write_text(json.dumps(cfg, indent=1, default=str))


if __name__ == "__main__":
    main()
