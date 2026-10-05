"""
SonicWave churn — data visualization phase.

Reads data/sonicwave_subscribers.csv and writes PNG charts to figures/.
Every chart answers one question about what drives churn.

Usage:
    python visualize_churn.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sonicwave_subscribers.csv"
OUT = ROOT / "figures"

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ACCENT = "#2a78d6"      # segments churning above the overall rate
NEUTRAL = "#b9b8b2"     # segments at/below the overall rate
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#9cc1ee", "#2a78d6", "#123f7a"]
)
MIN_N = 30  # segments smaller than this are greyed/flagged as unreliable

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

PCT = mtick.PercentFormatter(1.0, decimals=0)

ORDER = {
    "age_group": ["Under 25", "25-34", "35-49", "50 plus"],
    "plan_type": ["Free", "Basic", "Premium", "Family"],
    "content_mix": ["Low-usage", "Balanced", "Music-heavy", "Podcast-heavy"],
    "signup_channel": ["Web", "iOS", "Android", "Partner promo"],
    "payment_method": ["Credit card", "Debit", "PayPal", "Gift card"],
    "last_ticket_topic": ["No ticket", "Account", "Billing", "Content", "Playback"],
    "tickets_bucket": ["0", "1", "2", "3+"],
    "tenure_bucket": ["1-3", "4-6", "7-12", "13-24", "25-60"],
}
LABELS = {
    "age_group": "Age group",
    "plan_type": "Plan",
    "content_mix": "Content mix",
    "signup_channel": "Signup channel",
    "payment_method": "Payment method",
    "last_ticket_topic": "Last ticket topic",
    "tickets_bucket": "Support tickets (last 90 days)",
    "tenure_bucket": "Tenure (months)",
    "hours_bucket": "Avg weekly listening hours (quintile)",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["tickets_bucket"] = (
        df["support_tickets_90d"].clip(upper=3).astype(str).replace({"3": "3+"})
    )
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], [0, 3, 6, 12, 24, 60], labels=ORDER["tenure_bucket"]
    ).astype(str)
    q = pd.qcut(df["avg_weekly_hours"], 5)
    df["hours_bucket"] = q.map(lambda iv: f"{iv.left:.0f}-{iv.right:.0f}h").astype(str)
    ORDER["hours_bucket"] = [f"{iv.left:.0f}-{iv.right:.0f}h" for iv in q.cat.categories]
    return df


def rate_table(df, col):
    t = df.groupby(col)["churned"].agg(rate="mean", n="size")
    return t.reindex([o for o in ORDER[col] if o in t.index])


def churn_bars(ax, df, col, overall, title=None):
    """Horizontal churn-rate bars, top-to-bottom in ORDER, with overall reference line."""
    t = rate_table(df, col).iloc[::-1]
    colors = [ACCENT if r > overall * 1.15 else NEUTRAL for r in t["rate"]]
    labels = [f"{k}\nn={n:,}" for k, n in zip(t.index, t["n"])]
    ax.barh(labels, t["rate"], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.axvline(overall, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    xmax = max(t["rate"].max() * 1.25, overall * 1.6)
    ax.set_xlim(0, xmax)
    for y, r in enumerate(t["rate"]):
        x = r
        if r <= overall and overall - r < xmax * 0.09:  # keep value clear of the dashed line
            x = overall
        ax.text(x + xmax * 0.015, y, f"{r:.0%}", va="center", fontsize=9,
                color=INK, fontweight="bold" if r > overall * 1.15 else "normal")
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title or LABELS[col])


def heatmap(ax, df, row, col, title):
    rate = pd.crosstab(df[row], df[col], values=df["churned"], aggfunc="mean")
    n = pd.crosstab(df[row], df[col])
    rate = rate.reindex(index=ORDER[row], columns=ORDER[col])
    n = n.reindex(index=ORDER[row], columns=ORDER[col]).fillna(0).astype(int)
    masked = rate.where(n >= MIN_N)
    vmax = max(0.6, np.nanmax(masked.values))
    im = ax.imshow(masked.values, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            k = n.iat[i, j]
            if k == 0:
                txt, c = "—", INK_2
            elif k < MIN_N:
                txt, c = f"n={k}\n(too few)", INK_2
            else:
                v = rate.iat[i, j]
                txt = f"{v:.0%}\nn={k:,}"
                c = "white" if v > vmax * 0.55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=c)
    ax.set_xticks(range(len(rate.columns)), rate.columns)
    ax.set_yticks(range(len(rate.index)), rate.index)
    ax.set_xlabel(LABELS.get(col, col))
    ax.set_ylabel(LABELS.get(row, row))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # thin surface gaps between cells
    ax.set_xticks(np.arange(-0.5, rate.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, rate.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=PCT)
    cb.outline.set_visible(False)
    cb.set_label("Churn rate", color=INK_2)


def footer(fig, overall, n):
    fig.text(0.01, 0.005,
             f"Dashed line = overall churn rate ({overall:.1%}, n={n:,}). "
             f"Blue = segment churns >15% above overall.",
             fontsize=8, color=INK_2, ha="left", va="bottom")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- charts
def fig01_overview(df, overall):
    cols = ["plan_type", "content_mix", "signup_channel",
            "last_ticket_topic", "payment_method", "age_group"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5))
    for ax, c in zip(axes.flat, cols):
        churn_bars(ax, df, c, overall)
    fig.suptitle("Churn rate by subscriber segment", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    footer(fig, overall, len(df))
    save(fig, "01_churn_by_segment.png")


def fig02_tickets(df, overall):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    churn_bars(ax, df, "tickets_bucket", overall,
               "Churn jumps from ~5% to ~39% at 2+ support tickets")
    ax.set_ylabel("Tickets in last 90 days")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    footer(fig, overall, len(df))
    save(fig, "02_churn_by_support_tickets.png")


def fig03_tickets_x_topic(df):
    sub = df[df["last_ticket_topic"] != "No ticket"]
    order_backup = ORDER["tickets_bucket"]
    ORDER["tickets_bucket"] = ["1", "2", "3+"]
    ORDER["last_ticket_topic_t"] = ["Account", "Billing", "Content", "Playback"]
    sub = sub.assign(last_ticket_topic_t=sub["last_ticket_topic"])
    LABELS["last_ticket_topic_t"] = LABELS["last_ticket_topic"]
    fig, ax = plt.subplots(figsize=(8, 4))
    heatmap(ax, sub, "tickets_bucket", "last_ticket_topic_t",
            "Repeat tickets only predict churn when the topic is Billing")
    ORDER["tickets_bucket"] = order_backup
    fig.tight_layout()
    save(fig, "03_heatmap_tickets_x_topic.png")


def fig04_channel_x_mix(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    heatmap(ax, df, "signup_channel", "content_mix",
            "Partner-promo signups who barely listen churn at ~52%")
    fig.tight_layout()
    save(fig, "04_heatmap_channel_x_content_mix.png")


def fig05_plan_x_topic(df):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    heatmap(ax, df, "plan_type", "last_ticket_topic",
            "Premium churns 2-6x more than other plans outside of Billing tickets")
    fig.tight_layout()
    save(fig, "05_heatmap_plan_x_topic.png")


def fig06_tenure_hours(df, overall):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    churn_bars(axes[0], df, "tenure_bucket", overall)
    churn_bars(axes[1], df, "hours_bucket", overall)
    fig.suptitle("Tenure and listening hours show little churn signal on their own",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    footer(fig, overall, len(df))
    save(fig, "06_churn_by_tenure_and_hours.png")


def fig07_revenue(df):
    t = (df.groupby("plan_type")
           .apply(lambda g: pd.Series({
               "lost": g.loc[g.churned == 1, "monthly_spend"].sum(),
               "total": g["monthly_spend"].sum()}), include_groups=False)
           .reindex(ORDER["plan_type"]))
    t = t[t["total"] > 0].sort_values("lost")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(t.index, t["lost"], color=ACCENT, height=0.6, edgecolor=SURFACE, linewidth=2)
    xmax = t["lost"].max() * 1.45
    ax.set_xlim(0, xmax)
    for y, (lost, tot) in enumerate(zip(t["lost"], t["total"])):
        ax.text(lost + xmax * 0.015, y, f"${lost:,.0f}/mo", va="center",
                fontsize=9, fontweight="bold")
        ax.text(lost + xmax * 0.17, y, f"{lost / tot:.0%} of plan revenue",
                va="center", fontsize=8, color=INK_2)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    lost_all = t["lost"].sum()
    ax.set_title(f"Monthly revenue lost to churn: ${lost_all:,.0f} "
                 f"({lost_all / df['monthly_spend'].sum():.1%} of total) — "
                 f"Premium is the bulk")
    ax.set_xlabel("Monthly spend of churned subscribers (Free plan excluded, $0)")
    fig.tight_layout()
    save(fig, "07_revenue_lost_by_plan.png")


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    overall = df["churned"].mean()
    print(f"Loaded {len(df):,} subscribers, overall churn {overall:.1%}")
    fig01_overview(df, overall)
    fig02_tickets(df, overall)
    fig03_tickets_x_topic(df)
    fig04_channel_x_mix(df)
    fig05_plan_x_topic(df)
    fig06_tenure_hours(df, overall)
    fig07_revenue(df)


if __name__ == "__main__":
    main()
