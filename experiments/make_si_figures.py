"""Supplementary figures S1-S4 (read from results/; same style as main figures)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import plt, np, pd, RES, MM, COL, MARK, LS, INK, INK2, save, tag, grid  # noqa: E402


def figS1():
    r = pd.read_csv(RES / "E2_controller_on_scripted_traces.csv")
    js = json.loads((RES / "E2_scripted_reanalysis.json").read_text())
    br = js["original_rerun"][0]["breakdown_by_category"]
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 55 * MM), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    for i, k in enumerate(("confusion_ramp", "frustration_ramp")):
        g = r[r.kind == k].drop_duplicates(["slope", "start"])
        ax.scatter(g.slope, g.scripted_event_time, s=14 + 60 * (g.start - 0.2), color=COL[i], marker=MARK[i],
                   label=k.replace("_", " "), edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Scripted ramp slope (s$^{-1}$)"); ax.set_ylabel("Scripted 'event' time (s)")
    ax.legend(fontsize=6); grid(ax, "both"); tag(ax, "a", x=-0.3)
    ax.text(0.98, 0.95, "marker size: ramp start\n(0.20, 0.25, 0.30)", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color=INK2)
    ax = axs[1]
    g = r.dropna(subset=["controller_first_fire"])
    for i, k in enumerate(("confusion_ramp", "frustration_ramp", "polarity_hesitation")):
        gg = g[g.kind == k]
        ax.scatter(gg.threshold_cross_time, gg.controller_first_fire, s=10, color=COL[i], marker=MARK[i],
                   label=k.replace("_", " "))
    x = np.linspace(0, 5, 2)
    ax.plot(x, x + 3.0, color=INK, lw=0.6, ls=":")
    ax.text(0.2, 3.5, "fire = crossing + 3.0 s", fontsize=5.5, color=INK2, rotation=28)
    ax.set_xlabel("First sample above threshold (s)"); ax.set_ylabel("Deployed gate fires (s)")
    ax.legend(fontsize=5.6, loc="lower right"); grid(ax, "both"); tag(ax, "b", x=-0.3)
    ax = axs[2]
    cats = [("confusion_struggle", "Confusion", "tp"), ("frustration_disengage", "Frustration", "tp"),
            ("polarity_hesitation", "Polarity", "tp"), ("neutral_negative_control", "Neutral", "true_negatives"),
            ("spike_rejection", "Spike", "spikes_rejected")]
    vals = [br[c][k] / br[c]["total"] for c, _, k in cats]
    ax.bar(range(len(cats)), vals, color=[COL[0]] * 3 + [COL[1]] * 2, width=0.6)
    for i, (c, _, k) in enumerate(cats):
        ax.text(i, vals[i] + 0.02, f"{br[c][k]}/{br[c]['total']}", ha="center", fontsize=5.8)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([l for _, l, _ in cats], fontsize=5.6, rotation=30, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Proportion classified as intended")
    grid(ax); tag(ax, "c", x=-0.3)
    save(fig, "figS1_scripted")


def figS2():
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM), gridspec_kw=dict(wspace=0.45))
    c = pd.read_csv(RES / "E4_cooldown_summary.csv"); c["x"] = c.condition.str.extract(r"=([\d.]+)").astype(float)
    k = pd.read_csv(RES / "E4_cap_summary.csv"); k["x"] = k.condition.str.extract(r"=(\d+)").astype(float)
    k.loc[k.x > 100, "x"] = 7
    ax = axs[0]
    ax.errorbar(c.x, c.hit_rate, yerr=[c.hit_rate - c.hit_lo, c.hit_hi - c.hit_rate], color=COL[0], marker="o", ms=3, capsize=1.2)
    ax.set_xlabel("Cooldown $T_c$ (s)"); ax.set_ylabel("Eligible episodes served"); ax.set_ylim(0.6, 1.0)
    grid(ax); tag(ax, "a", x=-0.3)
    ax = axs[1]
    ax.plot(c.x, c.redundant_per_hit, color=COL[1], marker="s", ms=3, label="Redundant per served episode")
    ax.plot(c.x, c.p90_delay / 10, color=COL[3], marker="D", ls="--", ms=3, label="90th-percentile delay / 10 s")
    ax.set_xlabel("Cooldown $T_c$ (s)"); ax.set_ylabel("Value"); ax.legend(fontsize=5.8, loc="upper left"); grid(ax); tag(ax, "b", x=-0.3)
    ax = axs[2]
    ax.errorbar(k.x, k.hit_rate, yerr=[k.hit_rate - k.hit_lo, k.hit_hi - k.hit_rate], color=COL[2], marker="^", ms=3, capsize=1.2)
    ax.set_xticks([1, 2, 3, 5, 7]); ax.set_xticklabels(["1", "2", "3", "5", "none"])
    ax.set_xlabel("Per-step cap $N_{\\max}$"); ax.set_ylabel("Eligible episodes served"); ax.set_ylim(0, 1.03)
    grid(ax); tag(ax, "c", x=-0.3)
    save(fig, "figS2_cooldown_cap")


def figS3():
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 50 * MM), gridspec_kw=dict(wspace=0.45))
    for ax, f, xl, tg in ((axs[0], "E4_mp_rate", "Landmark callback rate (Hz)", "a"),
                          (axs[1], "E4_update_interval", "Scoring interval (s)", "b"),
                          (axs[2], "E4_delay", "Additional delivery delay (s)", "c")):
        s = pd.read_csv(RES / f"{f}_summary.csv"); s["x"] = s.condition.str.extract(r"=([\d.]+)").astype(float)
        ax.errorbar(s.x, s.hit_rate, yerr=[s.hit_rate - s.hit_lo, s.hit_hi - s.hit_rate], color=COL[0], marker="o",
                    ms=3, capsize=1.2, label="Episodes served")
        ax.plot(s.x, s.mean_delay / 10, color=COL[1], marker="s", ls="--", ms=3, label="Mean delay / 10 s")
        ax.set_xlabel(xl); ax.set_ylabel("Value"); ax.set_ylim(0, 1.05); grid(ax); tag(ax, tg, x=-0.3)
    axs[0].legend(fontsize=5.8, loc="lower right")
    save(fig, "figS3_pipeline_timing")


def figS4():
    """Canonical-mesh pose maps (moved from the main Fig. 6)."""
    p = pd.read_csv(RES / "E5b_pose_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(120 * MM, 50 * MM), gridspec_kw=dict(wspace=0.55))
    for ax, mode, title, tg in ((axs[0], "runtime", "Runtime coordinates", "a"),
                                (axs[1], "iso3d", "Isotropic coordinates", "b")):
        g = p[(p.geometry == "midrange_brow") & (p.coords == mode)]
        Z = g.pivot(index="pitch", columns="yaw", values="C")
        im = ax.pcolormesh(Z.columns, Z.index, Z.values, cmap="Blues", vmin=0, vmax=0.7, shading="nearest",
                           rasterized=True)
        cs = ax.contour(Z.columns, Z.index, Z.values, levels=[0.40], colors=INK, linewidths=0.8)
        ax.clabel(cs, fmt={0.40: "0.40"}, fontsize=5.5)
        ax.set_xlabel("Head yaw (deg)"); ax.set_ylabel("Head pitch (deg; < 0: chin to camera)", fontsize=6)
        ax.set_title(title, fontsize=6.5); tag(ax, tg, x=-0.38)
        ax.spines[["top", "right"]].set_visible(True)
    cax = axs[1].inset_axes([1.05, 0.0, 0.06, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.ax.set_title(r"$C$", fontsize=6.5, pad=3); cb.ax.tick_params(labelsize=5.5)
    save(fig, "figS4_canonical_pose")


if __name__ == "__main__":
    figS1(); figS2(); figS3(); figS4()
