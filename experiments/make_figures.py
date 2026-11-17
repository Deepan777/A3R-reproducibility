"""Generate every data figure of the article from the result files.

All plotted values are read from results/ (produced by exp2-exp7); nothing is
typed in by hand.  Output: figures/panels/*.pdf (+ .eps).
Style: Arial 7 pt, 0.6-1.2 pt lines, white background, colour + marker +
line style redundancy (greyscale-safe). Width: 89 mm (single) / 183 mm (double).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "figures" / "panels"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import a3r_core as core  # noqa: E402
from sim_model import SimParams, simulate  # noqa: E402

MM = 1 / 25.4
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]   # validated categorical order
MARK = ["o", "s", "^", "D"]
LS = ["-", "--", "-.", ":"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d9d6"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
    "lines.markersize": 3.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tag(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Fig. 2a  landmark geometry on the canonical mesh;  Fig. 2b deformation response
# ---------------------------------------------------------------------------
def fig2():
    import exp5_feature_geometry as g5
    P = g5.load_mesh()
    L = g5.project(P)                       # runtime normalised coordinates
    u, v = L[:468, 0] * g5.W, L[:468, 1] * g5.H   # back to pixels for display
    fig = plt.figure(figsize=(183 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 0.22, 1, 1, 1], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    ax.scatter(u, v, s=0.6, color="#b9b8b2", linewidths=0, zorder=1)
    seg = lambda i, j, c, ls="-", lw=1.1: ax.plot([u[i], u[j]], [v[i], v[j]], color=c, ls=ls, lw=lw, zorder=3)
    seg(33, 263, INK2, "--", 0.8)                                  # IOD
    seg(105, 334, COL[0])                                          # brow pair
    for a, b in ((159, 145), (386, 374)):
        seg(a, b, COL[1])
    for a, b in ((133, 33), (362, 263)):
        seg(a, b, COL[1], ":", 0.9)
    seg(61, 291, COL[2], ":", 0.9)
    ax.plot([u[13], u[14]], [v[13], v[14]], color=COL[2], lw=1.1, zorder=3)
    for i, c in ((1, COL[3]), (10, COL[3]), (152, COL[3])):
        ax.plot(u[i], v[i], marker="x", color=c, ms=4, mew=0.9, zorder=4)
    # every used landmark is marked; only well-separated ones carry a number
    # (eye landmarks are identified in the legend to avoid label collisions)
    used = [105, 334, 33, 263, 159, 145, 386, 374, 133, 362, 61, 291, 13, 14, 1, 10, 152]
    lab = {105: (-15, -1), 334: (4, -1), 33: (-12, -3), 263: (4, -3), 61: (-12, -1), 291: (4, -1),
           1: (4, -1), 10: (4, -3), 152: (4, 1)}
    for i in used:
        ax.plot(u[i], v[i], "o", ms=2.0, color=INK, zorder=5)
        if i in lab:
            ax.annotate(str(i), (u[i], v[i]), xytext=lab[i], textcoords="offset points", fontsize=5.5,
                        color=INK2)
    ax.set_xlim(u.min() - 25, u.max() + 25); ax.set_ylim(v.max() + 20, v.min() - 20)
    ax.set_aspect("equal"); ax.axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[0], lw=1.1, label="Brow pair (105–334)"),
         Line2D([], [], color=INK2, ls="--", lw=0.8, label="Reference width (33–263)"),
         Line2D([], [], color=COL[1], lw=1.1, label="Eyelid gaps (159–145, 386–374)"),
         Line2D([], [], color=COL[1], ls=":", lw=0.9, label="Eye widths (133–33, 362–263)"),
         Line2D([], [], color=COL[2], lw=1.1, label="Lip pair (13, 14) and corners (61, 291)"),
         Line2D([], [], color=COL[3], marker="x", lw=0, label="Pose-gate points (1, 10, 152)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=1, fontsize=5.8,
              handlelength=2.2)
    tag(ax, "a", x=-0.02)

    # b-d: one panel per controlled deformation, each with its own physical unit
    d = pd.read_csv(RES / "E5e_deformations.csv")
    spec = [("brow_approach_each_side_mm", "Brow approach\nper side (mm)", 1.0, 0, "b"),
            ("mouth_corner_depression_mm", "Mouth-corner\ndepression (mm)", 1.0, 2, "c"),
            ("eyelid_closure_fraction", "Eyelid closure\n(% of opening)", 100.0, 1, "d")]
    first = None
    for j, (k, xl, scale, c, tg) in enumerate(spec):
        ax = fig.add_subplot(gs[j + 2], sharey=first)
        first = first or ax
        gg = d[d.deformation == k]
        ax.plot(gg.amount.to_numpy() * scale, gg.C, color=COL[c], marker=MARK[c], ms=3)
        ax.axhline(core.ACTIVE["t_conf"], color=INK, lw=0.8, ls=(0, (4, 2)))
        ax.set_xlabel(xl); ax.set_ylim(0, 0.7); grid(ax)
        if j == 0:
            ax.set_ylabel(r"Confusion score $C$")
            ax.text(0.03, core.ACTIVE["t_conf"] + 0.012, r"$\theta_C$", transform=ax.get_yaxis_transform(), fontsize=6.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        tag(ax, tg, x=-0.06 if j else -0.3)
    save(fig, "fig2_features")


# ---------------------------------------------------------------------------
# Fig. 3b  annotated controller timeline (simulated, illustrative)
# ---------------------------------------------------------------------------
def fig3_timeline():
    p = SimParams(duration=60.0, noise_sd=0.035, dropout_rate=0.0)
    rng = np.random.default_rng(11)
    # Hand-specified latent schedule (illustrative scenario)
    events = [(5.0, 6.4, "transient"), (10.0, 19.5, "sustained"), (21.0, 38.0, "sustained"), (44.0, 57.0, "sustained")]
    import sim_model as sm
    orig = sm.latent_events
    sm.latent_events = lambda p_, r_: events
    try:
        tr = simulate(p, rng)
    finally:
        sm.latent_events = orig
    # inject one face-loss interval
    lost = (tr.t_obs >= 46.0) & (tr.t_obs < 47.2)
    tr.valid[lost] = False; tr.C_obs[lost] = 0.0
    ctl = core.AdaptationControllerCS(core.ControllerConfig())
    fired, supp, runs, cur = [], [], [], None
    for t, c, v in zip(tr.t_obs, tr.C_obs, tr.valid):
        nsupp = ctl.log.suppressed_cooldown
        f = ctl.update(float(t), float(c), 0.0, "s", bool(v))
        above = c > 0.40
        if above and cur is None:
            cur = [t, t]
        elif above:
            cur[1] = t
        elif cur is not None:
            runs.append(cur); cur = None
        if f:
            fired.append(t)
        if ctl.log.suppressed_cooldown > nsupp:
            supp.append(t)
    if cur is not None:
        runs.append(cur)
    fig, ax = plt.subplots(figsize=(120 * MM, 46 * MM))
    for on, off, k in events:
        ax.axvspan(on, off, ymin=0, ymax=0.06, color="#9a9994" if k == "sustained" else "#d7d6d1", lw=0)
    ax.text(0.4, 0.075, "latent elevation (grey bars)", fontsize=5.3, color=INK2, transform=ax.get_xaxis_transform())
    ax.step(tr.t_obs, tr.C_obs, where="post", color=COL[0], lw=0.9, label=r"Observed score $C_k$ (2 Hz)")
    ax.plot(tr.t_obs[~tr.valid], np.zeros((~tr.valid).sum()) + 0.02, "v", color=COL[1], ms=3,
            label="No valid face ($C_k$ = 0)")
    ax.axhline(0.40, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.text(59.5, 0.415, r"$\theta_C$", ha="right", fontsize=6.5)
    for f in fired:
        ax.axvline(f, color=COL[3], lw=1.0)
        ax.annotate("", xy=(f, 0.86), xytext=(f, 0.97), arrowprops=dict(arrowstyle="-|>", color=COL[3], lw=0.8))
        ax.add_patch(plt.Rectangle((f, 0.80), 15.0, 0.035, color="#e4e1f7", lw=0))
    ax.text(fired[0] + 0.4, 0.845, "cooldown 15 s", fontsize=5.3, color=INK2)
    if supp:
        ax.plot(supp, np.full(len(supp), 0.74), "|", color=COL[1], ms=4, mew=0.8,
                label="Dwell met, suppressed by cooldown")
    ax.plot([], [], color=COL[3], lw=1.0, label="Intervention fired")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Time within step (s)"); ax.set_ylabel(r"Score $C$")
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2, fontsize=6)
    save(fig, "fig3b_timeline")
    (RES / "fig3b_timeline_events.json").write_text(json.dumps(
        dict(events=events, fired=[float(x) for x in fired], suppressed=[float(x) for x in supp],
             face_loss=[46.0, 47.2], seed=11), indent=1))


# ---------------------------------------------------------------------------
# Fig. 4  ablation and duration response
# ---------------------------------------------------------------------------
def fig4():
    s = pd.read_csv(RES / "E3_ablation_summary.csv")
    conds = list(dict.fromkeys(s.condition))
    vars_ = list(dict.fromkeys(s.variant))
    short = {"nominal (sd 0.04)": "N", "noisy (sd 0.08)": "H",
             "noisy + 4 interruptions/min": "H+L", "persistent difficulty": "P"}
    vlab = {"V0 threshold only": "V0 threshold only", "V1 + dwell": "V1 + dwell",
            "V2 + cooldown": "V2 + cooldown", "V3 full (A3R)": "V3 full (A³R)"}
    fig = plt.figure(figsize=(183 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.38)
    metrics = [("hit_rate", "hit_lo", "hit_hi", "Eligible episodes served", (0.7, 1.005), "a"),
               ("false_per10min", "false_lo", "false_hi", "Unintended interventions\nper 10 min", None, "b"),
               ("mean_delay", "delay_lo", "delay_hi", "Mean response delay (s)", (0, 6), "c")]
    x = np.arange(len(conds))
    for j, (m, lo, hi, lab, yl, tg) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        for i, v in enumerate(vars_):
            g = s[s.variant == v].set_index("condition").loc[conds]
            xx = x + (i - 1.5) * 0.18
            ax.errorbar(xx, g[m], yerr=[g[m] - g[lo], g[hi] - g[m]], fmt=MARK[i], color=COL[i],
                        ms=3.2, elinewidth=0.7, capsize=1.2, label=vlab[v])
        ax.set_xticks(x); ax.set_xticklabels([short[c] for c in conds], fontsize=6.5)
        ax.set_xlabel("Observation regime")
        ax.set_ylabel(lab)
        if m == "false_per10min":
            ax.set_yscale("symlog", linthresh=0.5); ax.set_ylim(-0.05, 150)
            ax.set_yticks([0, 0.5, 1, 5, 10, 50, 100]); ax.set_yticklabels(["0", "0.5", "1", "5", "10", "50", "100"])
        elif yl:
            ax.set_ylim(*yl)
        grid(ax); tag(ax, tg, x=-0.3)
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.42), ncol=4, fontsize=6.2)
    # duration response
    d = pd.read_csv(RES / "E3_duration_replicates.csv.gz")
    d[["sd", "dur"]] = d.condition.str.extract(r"sd=([\d.]+)\|d=([\d.]+)").astype(float)
    ax = fig.add_subplot(gs[1, 0:2])
    k = 0
    for sd, ls in ((0.04, "-"), (0.08, "--")):
        for i, v in enumerate(("V0 threshold only", "V1 + dwell")):
            g = d[(d.variant == v) & (d.sd == sd)].groupby("dur")
            pr = g.hits.sum() / g.n_eligible.sum()
            n = g.n_eligible.sum()
            z = 1.96
            lo = (pr + z * z / (2 * n) - z * np.sqrt(pr * (1 - pr) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            hi = (pr + z * z / (2 * n) + z * np.sqrt(pr * (1 - pr) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            c = COL[0] if i == 0 else COL[1]
            ax.fill_between(pr.index, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(pr.index, pr, ls=ls, marker=MARK[i], color=c, ms=3,
                    label=f"{'V0 threshold only' if i == 0 else 'V1 + dwell (D = 3 s)'}, σ = {sd}")
            k += 1
    ax.axvspan(0, 2.0, color="#f1f0ec", lw=0, zorder=0)
    ax.axvspan(5.0, 12.5, color="#eef3fb", lw=0, zorder=0)
    ax.text(1.0, 0.5, "brief\n(not intended)", ha="center", fontsize=5.8, color=INK2)
    ax.text(8.7, 0.08, "sustained (intended)", ha="center", fontsize=5.8, color=INK2)
    ax.axvline(3.0, color=INK, lw=0.6, ls=":")
    ax.set_xlim(0, 12.5); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Duration of latent score elevation (s)")
    ax.set_ylabel("P(intervention during elevation)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6, ncol=1)
    grid(ax, "both"); tag(ax, "d", x=-0.1)
    # per-step intervention distribution, persistent difficulty
    r = pd.read_csv(RES / "E3_ablation_replicates.csv.gz")
    r = r[r.condition == "persistent difficulty"]
    ax = fig.add_subplot(gs[1, 2])
    bins = np.arange(0, 13) - 0.5
    for i, v in enumerate(vars_[1:], start=1):
        vals = r[r.variant == v].n_fired
        hist, _ = np.histogram(vals, bins=bins)
        ax.plot(np.arange(12), hist / hist.sum(), drawstyle="steps-mid", color=COL[i], ls=LS[i], label=vlab[v])
    ax.set_xlabel("Interventions per step"); ax.set_ylabel("Fraction of steps")
    ax.set_xlim(-0.5, 11.5)
    ax.legend(fontsize=5.8, loc="upper right")
    grid(ax); tag(ax, "e", x=-0.3)
    save(fig, "fig4_ablation")


# ---------------------------------------------------------------------------
# Fig. 5  sensitivity and robustness
# ---------------------------------------------------------------------------
def fig5():
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 104 * MM))
    plt.subplots_adjust(hspace=0.62, wspace=0.42)

    def two(ax, df, xcol, xlab, tg, ref=None):
        ax2 = ax.twinx  # not used: single axis per panel (no dual axes)
        ax.errorbar(df[xcol], df.hit_rate, yerr=[df.hit_rate - df.hit_lo, df.hit_hi - df.hit_rate],
                    color=COL[0], marker="o", ms=3, capsize=1.2, elinewidth=0.6, label="Episodes served")
        ax.set_ylabel("Eligible episodes served"); ax.set_xlabel(xlab); ax.set_ylim(0.4, 1.02)
        if ref is not None:
            ax.axvline(ref, color=INK, lw=0.6, ls=":")
        grid(ax); tag(ax, tg, x=-0.32)

    # (a,b) dwell: hit vs false trade-off as separate panels
    d = pd.read_csv(RES / "E4_dwell_summary.csv"); d["D"] = d.condition.str[2:].astype(float)
    two(axs[0, 0], d, "D", "Dwell time D (s)", "a", ref=3.0)
    ax = axs[0, 1]
    ax.errorbar(d.D, d.false_per10min, yerr=[d.false_per10min - d.false_lo, d.false_hi - d.false_per10min],
                color=COL[1], marker="s", ms=3, capsize=1.2, elinewidth=0.6)
    ax.set_xlabel("Dwell time D (s)"); ax.set_ylabel("Unintended interventions\nper 10 min")
    ax.axvline(3.0, color=INK, lw=0.6, ls=":"); grid(ax); tag(ax, "b", x=-0.32)
    # (c) threshold
    t = pd.read_csv(RES / "E4_threshold_summary.csv"); t["T"] = t.condition.str[2:].astype(float)
    ax = axs[0, 2]
    ax.errorbar(t["T"], t.hit_rate, yerr=[t.hit_rate - t.hit_lo, t.hit_hi - t.hit_rate], color=COL[0],
                marker="o", ms=3, capsize=1.2, elinewidth=0.6, label="Episodes served")
    ax.set_ylim(0.4, 1.02); ax.set_xlabel(r"Threshold $\theta_C$ (latent levels fixed)")
    ax.set_ylabel("Eligible episodes served")
    for x_, y_ in zip(t["T"], t.false_per10min):
        if y_ > 0.05:
            ax.annotate(f"{y_:.1f}/10 min", (x_, t.loc[t['T'] == x_, 'hit_rate'].iloc[0]),
                        xytext=(4, -10), textcoords="offset points", fontsize=5.5, color=COL[1])
    ax.axvline(0.40, color=INK, lw=0.6, ls=":"); grid(ax); tag(ax, "c", x=-0.32)
    # (d) noise, (e) interruptions, (f) correlated noise: A3R vs candidate refinements
    vlab = {"V3 full (A3R)": "Deployed (A³R)", "V3 + hysteresis 0.05": "+ hysteresis 0.05",
            "V3 + hold on dropout": "+ hold on face loss", "V3 + both": "+ both"}
    for ax, f, col, xlab, tg in ((axs[1, 0], "E4_noise", "sd", "Score noise SD σ", "d"),
                                 (axs[1, 1], "E4_dropout", "rate", "Face-loss events per min", "e"),
                                 (axs[1, 2], "E4_noise_corr", "rho", "Noise autocorrelation ρ (σ = 0.06)", "f")):
        s = pd.read_csv(RES / f"{f}_summary.csv")
        s["x"] = s.condition.str.extract(r"=([\d.]+)").astype(float)
        for i, v in enumerate(vlab):
            g = s[s.variant == v]
            off = (i - 1.5) * (0.0025 if col == "sd" else 0.06 if col == "rate" else 0.008)
            ax.errorbar(g.x + off, g.hit_rate, yerr=[g.hit_rate - g.hit_lo, g.hit_hi - g.hit_rate], color=COL[i],
                        marker=MARK[i], ls=LS[i], ms=2.8, capsize=1.0, elinewidth=0.5, lw=0.9, label=vlab[v])
        ax.set_xlabel(xlab); ax.set_ylabel("Eligible episodes served"); ax.set_ylim(0.7, 1.0)
        grid(ax); tag(ax, tg, x=-0.32)
    axs[1, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.36), ncol=4, fontsize=6.2)
    save(fig, "fig5_sensitivity")


# ---------------------------------------------------------------------------
# Fig. 6  geometry robustness: pose, distance, landmark noise
# ---------------------------------------------------------------------------
def fig6():
    """Real faces (Pointing'04, n = 15 persons) in a-d; canonical-mesh geometry in e, f."""
    d = pd.read_csv(RES / "E7_real_faces.csv")
    f = d[d.detected == 1]
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM))
    plt.subplots_adjust(hspace=0.62, wspace=0.42)
    # (a) neutral frontal score per person and session
    ax = axs[0, 0]
    fr = f[(f.tilt == 0) & (f.pan == 0)].groupby(["person", "series"])[["nat_C", "iso_C"]].mean().reset_index()
    order = fr.groupby("person").nat_C.mean().sort_values().index
    pos = {p: i for i, p in enumerate(order)}
    for s, mk in ((1, "o"), (2, "s")):
        g = fr[fr.series == s]
        ax.plot([pos[p] for p in g.person], g.nat_C, mk, color=COL[0], ms=3, ls="none",
                label="Image coords" if s == 1 else None)
        ax.plot([pos[p] for p in g.person], g.iso_C, mk, color=COL[1], ms=3, ls="none", mfc="white",
                label="Pixel units" if s == 1 else None)
    ax.axhline(0.40, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(len(order))); ax.set_xticklabels([str(i + 1) for i in range(len(order))], fontsize=5)
    ax.set_xlabel("Person (ranked)"); ax.set_ylabel(r"Frontal, neutral $C$"); ax.set_ylim(0, 0.75)
    grid(ax); ax.legend(fontsize=5.2, loc="upper left", ncol=1); tag(ax, "a", x=-0.3)
    # (b,c) change of score with pan / tilt (per-person, relative to own frontal)
    for ax, axis, other, xl, tg in ((axs[0, 1], "pan", "tilt", "Head pan (deg; tilt 0)", "b"),
                                    (axs[0, 2], "tilt", "pan", "Head tilt (deg; pan 0; > 0 upwards)", "c")):
        for i, conv in enumerate(("nat", "iso")):
            pe = pd.read_csv(RES / f"E7_pose_effects_{conv}.csv")
            g = pe[pe[other] == 0].sort_values(axis)
            off = (i - 0.5) * 3
            ax.errorbar(g[axis] + off, g.dC_median, yerr=[g.dC_median - g.dC_q25, g.dC_q75 - g.dC_median],
                        color=COL[i], marker=MARK[i], ls=LS[i], ms=2.8, capsize=1.0, elinewidth=0.5, lw=0.9,
                        label="Image coords" if conv == "nat" else "Pixel units")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlabel(xl); ax.set_ylabel(r"$\Delta C$ vs own frontal"); ax.set_xticks([-90, -45, 0, 45, 90])
        grid(ax); tag(ax, tg, x=-0.3)
        if tg == "b":
            ax.legend(fontsize=5.6, loc="upper right")
    # (d) pose-gate rejection rate on real faces
    ax = axs[1, 0]
    pe = pd.read_csv(RES / "E7_pose_effects_nat.csv")
    a = pe[pe.tilt == 0].sort_values("pan"); b = pe[pe.pan == 0].sort_values("tilt")
    ax.plot(a.pan, a.gated_rate, color=COL[0], marker="o", ms=2.8, label="vs pan (tilt 0)")
    ax.plot(b.tilt, b.gated_rate, color=COL[3], marker="D", ls="--", ms=2.8, label="vs tilt (pan 0)")
    ax.set_xlabel("Head angle (deg)"); ax.set_ylabel("Frames rejected by pose gate"); ax.set_ylim(-0.03, 1.0)
    ax.set_xticks([-90, -45, 0, 45, 90]); grid(ax); ax.legend(fontsize=5.6, loc="upper center"); tag(ax, "d", x=-0.3)
    # (e) canonical mesh: distance
    dd = pd.read_csv(RES / "E5c_distance_sweep.csv")
    import exp5_feature_geometry as g5
    P = g5.load_mesh()
    can = [g5.feats_scores(g5.project(P, dist=x))["C"] for x in dd.distance_cm]
    axs = {2: axs[1, 1], 3: axs[1, 2]}
    ax = axs[2]
    ax.plot(dd.distance_cm, can, color=COL[0], marker="o", ms=2.5, label="Canonical")
    ax.plot(dd.distance_cm, dd.C, color=COL[1], marker="s", ls="--", ms=2.5, label="Brow mid-range")
    ax.axhline(0.40, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.set_xlabel("Face–camera distance (cm)"); ax.set_ylabel(r"Neutral frontal $C$")
    ax.set_ylim(0, 0.6); grid(ax); ax.legend(fontsize=5.6, loc="lower right", bbox_to_anchor=(1.0, 0.03)); tag(ax, "e", x=-0.3)
    n = pd.read_csv(RES / "E5d_noise_propagation.csv")
    ax = axs[3]
    for i, dist in enumerate((30.0, 40.0, 60.0)):
        g = n[n.distance_cm == dist]
        ax.plot(g.noise_px, g.C_sd, color=COL[i], marker=MARK[i], ls=LS[i], ms=2.8, label=f"{dist:.0f} cm")
    ax.set_xlabel("Landmark noise (px)"); ax.set_ylabel(r"SD of $C$ per frame")
    ax.set_ylim(0, 0.22); grid(ax); ax.legend(fontsize=5.8, title="Distance", title_fontsize=5.8); tag(ax, "f", x=-0.3)
    save(fig, "fig6_geometry")


# ---------------------------------------------------------------------------
# Fig. 7  registration error propagation
# ---------------------------------------------------------------------------
def fig7():
    s = pd.read_csv(RES / "E6_registration_sweep.csv")
    h = pd.read_csv(RES / "E6_registration_per_hole.csv")
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 52 * MM), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.35))
    ax = axs[0]
    g = h[h.sigma_mm == 2.0]
    sc = ax.scatter(g.x_mm, g.z_mm, c=g.rms_err_mm, cmap="Blues", s=9, vmin=1.3, vmax=1.8, edgecolors=INK2,
                    linewidths=0.2)
    ax.add_patch(plt.Rectangle((-41, -27), 82, 54, fill=False, ec=INK2, lw=0.6))
    for (x, z) in ((-41, 27), (41, 27), (41, -27), (-41, -27)):
        ax.plot(x, z, "o", ms=3.5, color=COL[1])
    ax.set_aspect("equal"); ax.set_xlabel("Board x (mm)"); ax.set_ylabel("Board z (mm)")
    ax.set_xlim(-46, 46); ax.set_ylim(-31, 31)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03); cb.set_label("RMS hole error (mm), σ = 2 mm", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    tag(ax, "a", x=-0.12)
    ax = axs[1]
    ax.plot(s.sigma_mm, s.frac_within_half_pitch, color=COL[0], marker="o", ms=3, label="< 1.27 mm (half pitch)")
    ax.plot(s.sigma_mm, s.frac_within_one_pitch, color=COL[1], marker="s", ls="--", ms=3, label="< 2.54 mm (one pitch)")
    ax.set_xlabel("Corner placement error SD σ (mm)"); ax.set_ylabel("Fraction of hole placements")
    ax.set_ylim(0, 1.03); grid(ax, "both"); ax.legend(fontsize=6, loc="upper right"); tag(ax, "b", x=-0.3)
    save(fig, "fig7_registration")


if __name__ == "__main__":
    only = sys.argv[1:] or ["fig2", "fig3_timeline", "fig4", "fig5", "fig6", "fig7"]
    for name in only:
        globals()[name]()
    print("figures written to", OUT)
