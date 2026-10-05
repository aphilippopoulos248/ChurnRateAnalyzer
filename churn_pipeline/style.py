"""Matplotlib charts (PNG) for the README and for Gemini to read."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, NEUTRAL, RISK_UP, RISK_DOWN = "#2a78d6", "#b9b8b2", "#e34948", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc1ee", "#2a78d6", "#123f7a"])
PCT = mtick.PercentFormatter(1.0, decimals=0)
MIN_N = 30

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, out_dir, name):
    d = out_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def churn_bars(ax, rows, overall, title):
    rows = rows[::-1]
    labels = [f"{r['label']}\nn={r['n']:,}" for r in rows]
    rates = [r["rate"] for r in rows]
    colors = [ACCENT if r > overall * 1.15 else NEUTRAL for r in rates]
    ax.barh(labels, rates, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.axvline(overall, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    xmax = max(max(rates) * 1.25, overall * 1.6)
    ax.set_xlim(0, xmax)
    for y, r in enumerate(rates):
        x = overall if (r <= overall and overall - r < xmax * 0.09) else r
        ax.text(x + xmax * 0.015, y, f"{r:.0%}", va="center", fontsize=9,
                fontweight="bold" if r > overall * 1.15 else "normal")
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.set_title(title)


def heatmap(ax, hm, title):
    vals = np.array([[c["rate"] if (c["rate"] is not None and c["n"] >= MIN_N) else np.nan
                      for c in row] for row in hm["cells"]], dtype=float)
    vmax = max(0.3, np.nanmax(vals) if np.isfinite(vals).any() else 0.3)
    im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    for i, row in enumerate(hm["cells"]):
        for j, c in enumerate(row):
            if c["n"] == 0:
                txt, col = "—", INK_2
            elif c["n"] < MIN_N:
                txt, col = f"n={c['n']}\n(too few)", INK_2
            else:
                txt = f"{c['rate']:.0%}\nn={c['n']:,}"
                col = "white" if c["rate"] > vmax * 0.55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(hm["cols"])), hm["cols"], fontsize=8.5)
    ax.set_yticks(range(len(hm["rows"])), hm["rows"], fontsize=8.5)
    ax.set_xlabel(hm["col_label"])
    ax.set_ylabel(hm["row_label"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(hm["cols"])), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hm["rows"])), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=PCT)
    cb.outline.set_visible(False)


def phase1_figures(res, cfg, out_dir):
    names = []
    top = res["ranked_features"][:6]
    ncols = 2
    nrows = (len(top) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, top):
        a = res["attributes"][c]
        churn_bars(ax, a["rows"][:10], res["overall"], a["label"])
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle(f"Churn rate by attribute (top {len(top)} of {len(res['ranked_features'])}, ranked by how much churn they explain)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.0, f"Dashed line = overall churn ({res['overall']:.1%}). Blue = group churns >15% above overall.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    names.append(save(fig, out_dir, "01_churn_by_attribute.png"))
    for i, (key, hm) in enumerate(res["heatmaps"].items(), start=2):
        fig, ax = plt.subplots(figsize=(min(12, 2.2 + 1.4 * len(hm["cols"])), 1.2 + 0.75 * len(hm["rows"])))
        heatmap(ax, hm, hm["title"])
        fig.tight_layout()
        names.append(save(fig, out_dir, f"{i:02d}_heatmap_{key}.png"))
    return names


# ---------------------------------------------------------------- phase 2
def phase2_figures(y, oof, metrics, chosen, drivers, segments, calibration, overall, out_dir, currency="$"):
    from sklearn.metrics import precision_recall_curve, roc_curve
    names = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (name, p), color in zip(oof.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        lw = 2.6 if name == chosen else 1.6
        a.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC {metrics[name]['roc_auc']:.3f})")
        b.plot(rec, prec, color=color, lw=lw, label=f"{name} (AP {metrics[name]['pr_auc']:.3f})")
    a.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 3)))
    b.axhline(y.mean(), color=INK_2, lw=1, ls=(0, (4, 3)))
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (5-fold out-of-fold)")
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, frameon=False, loc="lower right" if ax is a else "upper right")
    fig.tight_layout()
    names.append(save(fig, out_dir, "10_model_comparison.png"))

    d = drivers[:14][::-1]
    if d:
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.3))
        ors = [r["odds_ratio"] for r in d]
        labels = [f"{r['feature']}\n{r['comparison']}" if r["comparison"].startswith("vs") else r["feature"] for r in d]
        ax.barh(labels, [o - 1 for o in ors], left=1, color=[RISK_UP if o > 1 else RISK_DOWN for o in ors],
                height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.set_xscale("log")
        ax.axvline(1, color=INK_2, lw=1)
        lo, hi = min(min(ors), 0.5) / 1.6, max(max(ors), 2) * 2.2
        ax.set_xlim(lo, hi)
        ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128) if lo <= t <= hi]
        ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))
        ax.xaxis.set_minor_locator(mtick.NullLocator())
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"×{v:g}"))
        for yy, o in enumerate(ors):
            ax.text(o * (1.08 if o > 1 else 1 / 1.08), yy, f"×{o:.2f}", va="center",
                    ha="left" if o > 1 else "right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8, length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Odds ratio (log scale): red raises churn odds, blue lowers them")
        ax.set_title("Churn drivers (logistic regression)")
        fig.tight_layout()
        names.append(save(fig, out_dir, "11_churn_drivers.png"))

    if calibration:
        fig, ax = plt.subplots(figsize=(6, 5))
        m = max(0.2, max(max(c["predicted"], c["actual"]) for c in calibration) * 1.15)
        ax.plot([0, m], [0, m], color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.plot([c["predicted"] for c in calibration], [c["actual"] for c in calibration],
                color=ACCENT, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set(xlim=(0, m), ylim=(0, m), xlabel="Predicted churn probability", ylabel="Actual churn rate",
               title="Calibration: predicted vs actual churn")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        fig.tight_layout()
        names.append(save(fig, out_dir, "12_calibration.png"))

    s = segments[::-1]
    has_rev = s and s[0].get("expected_monthly_revenue_at_risk") is not None
    fig, axes = plt.subplots(1, 2 if has_rev else 1, figsize=(12 if has_rev else 7, 0.9 * len(s) + 1.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    labels = [f"{x['segment_id']}  {x['name'][:42]}\nn={x['subscribers']:,}" for x in s]
    colors = [ACCENT if x["risk_tier"] == "High" else NEUTRAL for x in s]
    a = axes[0]
    a.barh(labels, [x["predicted_churn_rate"] for x in s], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    a.axvline(overall, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    a.set_xlim(0, max(x["predicted_churn_rate"] for x in s) * 1.25)
    for yy, x in enumerate(s):
        a.text(x["predicted_churn_rate"] + 0.005, yy, f"{x['predicted_churn_rate']:.0%}", va="center",
               fontsize=9, fontweight="bold")
    a.xaxis.set_major_formatter(PCT)
    a.set_title("Predicted churn rate")
    if has_rev:
        b = axes[1]
        vals = [x["expected_monthly_revenue_at_risk"] for x in s]
        b.barh(labels, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        b.set_xlim(0, max(vals) * 1.35)
        for yy, x in enumerate(s):
            b.text(x["expected_monthly_revenue_at_risk"] + max(vals) * 0.02, yy,
                   f"{currency}{x['expected_monthly_revenue_at_risk']:,.0f} ({x['share_of_total_revenue_at_risk']:.0%})",
                   va="center", fontsize=9)
        b.set_title("Expected revenue at risk")
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelsize=8.5)
    fig.tight_layout()
    names.append(save(fig, out_dir, "13_risk_segments.png"))
    return names
