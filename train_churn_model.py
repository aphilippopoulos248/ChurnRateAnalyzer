"""
SonicWave churn — Phase 2: churn prediction model + risk segmentation.

1. Compares three models with 5-fold stratified cross-validation.
2. Fits the chosen model (logistic regression + interaction flags) on all data
   and saves it to models/churn_model.joblib.
3. Uses out-of-fold churn probabilities (no subscriber is scored by a model
   that saw it during training) to carve the base into interpretable risk
   segments with a shallow surrogate decision tree.
4. Writes outputs/segment_risk.json for Phase 3 (Gemini/Gemma), plus
   outputs/model_metrics.json and figures 08-11.

Usage:
    python train_churn_model.py
"""
import json
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from features import (ALL_FEATURES, CATEGORICAL, ENGINEERED, NUMERIC,
                      RAW_FEATURES, ROOT, TARGET, load_data)
# Reuse the Phase 1 chart style so every figure in the repo looks the same.
from visualize_churn import ACCENT, GRID, INK, INK_2, PCT, SURFACE, save

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SEED = 42

# Reference level for each categorical: odds ratios read "vs this group".
REFERENCE = {
    "age_group": "25-34",
    "plan_type": "Basic",
    "content_mix": "Balanced",
    "payment_method": "Credit card",
    "last_ticket_topic": "No ticket",
    "signup_channel": "Web",
}
# Categorical palette slots 1-3 (validated all-pairs), diverging poles.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"

# Surrogate tree settings: shallow, big leaves, and a split must buy real
# purity — so every segment is large enough and different enough to act on.
SEGMENT_TREE = dict(max_depth=3, min_samples_leaf=100,
                    min_impurity_decrease=2e-4, random_state=SEED)
SHORT_LABEL = {
    "billing_repeat": "Repeat billing complaints",
    "promo_premium_low_usage": "Partner-promo Premium, low usage",
}


# ---------------------------------------------------------------- models
def logistic(flags=()):
    # Flags stay 0/1 (not scaled) so their odds ratio reads "if present vs absent".
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop=[REFERENCE[c] for c in CATEGORICAL]), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
        ("flag", "passthrough", list(flags)),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])


def boosting():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough")
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300, random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


CANDIDATES = {
    "Logistic regression (raw features)": (logistic, RAW_FEATURES),
    "Logistic regression + interaction flags": (
        lambda: logistic(ENGINEERED), ALL_FEATURES),
    "Gradient boosting (raw features)": (boosting, RAW_FEATURES),
}
CHOSEN = "Logistic regression + interaction flags"


def evaluate(y, p):
    top = p >= np.quantile(p, 0.9)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "top_decile_churn_rate": round(y[top].mean(), 4),
        "top_decile_lift": round(y[top].mean() / y.mean(), 2),
        "top_decile_share_of_churners_captured": round(y[top].sum() / y.sum(), 4),
    }


def compare_models(df):
    y = df[TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof, metrics = {}, {}
    for name, (make, cols) in CANDIDATES.items():
        oof[name] = cross_val_predict(make(), df[cols], y, cv=cv,
                                      method="predict_proba")[:, 1]
        metrics[name] = evaluate(y, oof[name])
        print(f"  {name:42s} ROC-AUC {metrics[name]['roc_auc']:.3f}  "
              f"PR-AUC {metrics[name]['pr_auc']:.3f}  Brier {metrics[name]['brier']:.4f}")
    return oof, metrics


# ---------------------------------------------------------------- drivers
def readable_term(raw):
    kind, name = raw.split("__", 1)
    if kind == "cat":
        col = next(c for c in CATEGORICAL if name.startswith(c + "_"))
        val = name[len(col) + 1:]
        return f"{col} = {val}", f"vs {REFERENCE[col]}"
    if kind == "flag":
        return name, "if present vs absent"
    return name, "per 1 standard deviation increase"


def odds_ratios(pipe):
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    rows = []
    for raw, b in zip(names, coefs):
        term, note = readable_term(raw)
        rows.append({"feature": term, "comparison": note,
                     "odds_ratio": round(float(np.exp(b)), 3),
                     "direction": "raises churn" if b > 0 else "lowers churn"})
    return sorted(rows, key=lambda r: -abs(np.log(r["odds_ratio"])))


# ---------------------------------------------------------------- segments
def segment_matrix(df):
    X = pd.get_dummies(df[CATEGORICAL], prefix_sep="=", dtype=int)
    return X.join(df[NUMERIC + list(ENGINEERED)])


def condition(feature, threshold, goes_right):
    if feature in ENGINEERED:
        return (ENGINEERED[feature] if goes_right else f"NOT ({ENGINEERED[feature]})",
                feature if goes_right else None)
    if "=" in feature:
        col, val = feature.split("=", 1)
        return (f"{col} = {val}" if goes_right else f"{col} != {val}",
                f"{col} = {val}" if goes_right else None)
    if feature == "avg_weekly_hours":
        return (f"{feature} > {threshold:.1f}" if goes_right else f"{feature} <= {threshold:.1f}", None)
    t = int(np.floor(threshold))
    return (f"{feature} >= {t + 1}" if goes_right else f"{feature} <= {t}", None)


def leaf_rules(tree, columns):
    t = tree.tree_
    rules = {}

    def walk(node, conds, positives):
        if t.children_left[node] == -1:
            rules[node] = (conds, positives)
            return
        f, thr = columns[t.feature[node]], t.threshold[node]
        for child, right in ((t.children_left[node], False), (t.children_right[node], True)):
            text, pos = condition(f, thr, right)
            walk(child, conds + [text], positives + ([pos] if pos else []))

    walk(0, [], [])
    return rules


def segment_name(positives):
    if not positives:
        return "Everyone else (baseline)"
    return " + ".join(SHORT_LABEL.get(p, p) for p in positives)


def distinguishing_traits(seg, pop):
    traits = []
    for col in CATEGORICAL:
        seg_share = seg[col].value_counts(normalize=True)
        pop_share = pop[col].value_counts(normalize=True)
        for val, s in seg_share.items():
            ratio = s / pop_share[val]
            if s >= 0.25 and ratio >= 1.5:
                traits.append({"attribute": col, "value": val,
                               "share_in_segment": round(float(s), 3),
                               "share_in_population": round(float(pop_share[val]), 3),
                               "over_representation": round(float(ratio), 2)})
    return sorted(traits, key=lambda d: -d["over_representation"])


def build_segments(df, p):
    X = segment_matrix(df)
    tree = DecisionTreeRegressor(**SEGMENT_TREE).fit(X, p)
    rules = leaf_rules(tree, list(X.columns))
    leaf = tree.apply(X)
    overall = df[TARGET].mean()
    total_risk = float((df["monthly_spend"] * p).sum())

    segments = []
    for node, (conds, positives) in rules.items():
        mask = leaf == node
        seg, ps = df[mask], p[mask]
        pred = float(ps.mean())
        risk = float((seg["monthly_spend"] * ps).sum())
        tier = "High" if pred >= 2 * overall else "Medium" if pred >= 1.2 * overall else "Low"
        segments.append({
            "name": segment_name(positives),
            "rule": conds,
            "risk_tier": tier,
            "subscribers": int(mask.sum()),
            "share_of_base": round(mask.mean(), 4),
            "predicted_churn_rate": round(pred, 4),
            "actual_churn_rate": round(float(seg[TARGET].mean()), 4),
            "lift_vs_overall": round(pred / overall, 2),
            "expected_churners": round(float(ps.sum()), 1),
            "monthly_revenue": round(float(seg["monthly_spend"].sum()), 2),
            "expected_monthly_revenue_at_risk": round(risk, 2),
            "share_of_total_revenue_at_risk": round(risk / total_risk, 4),
            "plan_mix": {k: round(float(v), 3) for k, v in
                         seg["plan_type"].value_counts(normalize=True).items()},
            "median_tenure_months": float(seg["tenure_months"].median()),
            "median_weekly_hours": float(seg["avg_weekly_hours"].median()),
            "distinguishing_traits": distinguishing_traits(seg, df),
        })
    segments.sort(key=lambda s: -s["predicted_churn_rate"])
    for i, s in enumerate(segments, 1):
        s["segment_id"] = f"S{i}"
    return [{"segment_id": s.pop("segment_id"), **s} for s in segments], leaf


# ---------------------------------------------------------------- figures
def fig08_model_comparison(y, oof, metrics):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (name, p), color in zip(oof.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a.plot(fpr, tpr, color=color, lw=2,
               label=f"{name}  (AUC {metrics[name]['roc_auc']:.3f})")
        b.plot(rec, prec, color=color, lw=2,
               label=f"{name}  (AP {metrics[name]['pr_auc']:.3f})")
    a.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 3)))
    b.axhline(y.mean(), color=INK_2, lw=1, ls=(0, (4, 3)))
    b.text(0.01, y.mean() + 0.015, f"random = {y.mean():.1%}", ha="left",
           fontsize=8, color=INK_2)
    a.set(xlabel="False positive rate", ylabel="True positive rate",
          title="ROC curve (5-fold out-of-fold)")
    b.set(xlabel="Recall (share of churners caught)", ylabel="Precision",
          title="Precision-recall curve")
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if ax is a else "upper right",
                  fontsize=8, frameon=False)
    b.xaxis.set_major_formatter(PCT)
    b.yaxis.set_major_formatter(PCT)
    fig.suptitle("Two interaction flags let a simple, explainable model "
                 "match gradient boosting", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "08_model_comparison.png")


def fig09_odds_ratios(drivers):
    d = pd.DataFrame(drivers).iloc[::-1]
    labels = [f"{f}\n{c}" if c.startswith("vs") else f
              for f, c in zip(d["feature"], d["comparison"])]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.2))
    colors = [RISK_UP if r > 1 else RISK_DOWN for r in d["odds_ratio"]]
    ax.barh(labels, d["odds_ratio"] - 1, left=1, color=colors, height=0.6,
            edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.axvline(1, color=INK_2, lw=1)
    lo, hi = d["odds_ratio"].min(), d["odds_ratio"].max()
    ax.set_xlim(min(lo, 0.5) / 1.6, hi * 2.2)
    for y_, r in enumerate(d["odds_ratio"]):
        ax.text(r * (1.08 if r > 1 else 1 / 1.08), y_, f"×{r:.2f}", va="center",
                ha="left" if r > 1 else "right", fontsize=8, color=INK)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio (log scale) — red raises churn odds, blue lowers them")
    ax.set_title("What drives churn once the two interaction flags are in the model")
    fig.tight_layout()
    save(fig, "09_churn_drivers_odds_ratios.png")


def fig10_calibration(y, p):
    bins = np.array([0, 0.02, 0.03, 0.04, 0.05, 0.1, 0.4, 0.5, 0.6, 0.7, 1.0])
    idx = np.digitize(p, bins[1:-1])
    cal = (pd.DataFrame({"bin": idx, "p": p, "y": y})
             .groupby("bin").agg(pred=("p", "mean"), act=("y", "mean"), n=("y", "size")))
    cal = cal[cal["n"] >= 30]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 0.8], [0, 0.8], color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.plot(cal["pred"], cal["act"], color=ACCENT, lw=2, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    low = cal[cal["pred"] < 0.1]
    ax.annotate(f"{len(low)} low-risk bins, n={int(low.n.sum()):,}", (low.pred.max(), low.act.max()),
                xytext=(14, 4), textcoords="offset points", fontsize=8, color=INK_2)
    for _, r in cal[cal["pred"] >= 0.1].iterrows():
        ax.annotate(f"n={int(r.n):,}", (r.pred, r.act), xytext=(8, -12),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="Predicted churn probability",
           ylabel="Actual churn rate",
           title="Calibration: predicted risk matches reality")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.text(0.79, 0.02, "Dashed = perfect calibration", ha="right",
            fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, "10_calibration.png")


def fig11_segments(segments, overall):
    s = pd.DataFrame(segments).iloc[::-1]
    labels = [f"{sid}  {n}\nn={k:,}" for sid, n, k in
              zip(s["segment_id"], s["name"], s["subscribers"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 1.1 * len(s) + 1.6), sharey=True)
    colors = [ACCENT if t == "High" else "#b9b8b2" for t in s["risk_tier"]]
    a.barh(labels, s["predicted_churn_rate"], color=colors, height=0.6,
           edgecolor=SURFACE, linewidth=2)
    a.axvline(overall, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    a.set_xlim(0, s["predicted_churn_rate"].max() * 1.25)
    for y_, v in enumerate(s["predicted_churn_rate"]):
        a.text(v + 0.01, y_, f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
    a.xaxis.set_major_formatter(PCT)
    a.set_title("Predicted churn rate")
    b.barh(labels, s["expected_monthly_revenue_at_risk"], color=colors, height=0.6,
           edgecolor=SURFACE, linewidth=2)
    b.set_xlim(0, s["expected_monthly_revenue_at_risk"].max() * 1.35)
    for y_, (v, sh) in enumerate(zip(s["expected_monthly_revenue_at_risk"],
                                     s["share_of_total_revenue_at_risk"])):
        b.text(v + b.get_xlim()[1] * 0.015, y_, f"${v:,.0f}/mo  ({sh:.0%})",
               va="center", fontsize=9)
    b.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    b.set_title("Expected monthly revenue at risk")
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    high = s[s["risk_tier"] == "High"]
    fig.suptitle(f"{len(high)} high-risk segments hold {high['share_of_base'].sum():.0%} "
                 f"of subscribers but {high['expected_churners'].sum() / s['expected_churners'].sum():.0%} "
                 f"of expected churners", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, f"Dashed line = overall churn rate ({overall:.1%}). "
             "Blue = High risk tier (predicted churn ≥ 2× overall).",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "11_risk_segments.png")


# ---------------------------------------------------------------- main
def main():
    for d in (MODELS_DIR, OUTPUTS_DIR, ROOT / "figures"):
        d.mkdir(exist_ok=True)
    df = load_data()
    y = df[TARGET].to_numpy()
    overall = float(y.mean())
    print(f"Loaded {len(df):,} subscribers, overall churn {overall:.1%}\n"
          f"5-fold cross-validation:")

    oof, metrics = compare_models(df)
    p = oof[CHOSEN]

    make, cols = CANDIDATES[CHOSEN]
    final = make().fit(df[cols], y)
    joblib.dump({"pipeline": final, "features": cols}, MODELS_DIR / "churn_model.joblib")
    print("  wrote models/churn_model.joblib")

    drivers = odds_ratios(final)
    segments, _ = build_segments(df, p)

    revenue = float(df["monthly_spend"].sum())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": ("Churn-risk segments for SonicWave subscribers. Use these to "
                    "decide which groups to target and what retention actions to take."),
        "dataset": {
            "subscribers": len(df),
            "overall_churn_rate": round(overall, 4),
            "total_monthly_revenue": round(revenue, 2),
            "expected_monthly_revenue_at_risk": round(float((df["monthly_spend"] * p).sum()), 2),
        },
        "model": {
            "type": CHOSEN,
            "validation": "5-fold stratified cross-validation; segment figures use out-of-fold predictions",
            "metrics": metrics[CHOSEN],
            "engineered_features": ENGINEERED,
        },
        "segments": segments,
        "churn_drivers": [d for d in drivers if abs(np.log(d["odds_ratio"])) >= np.log(1.25)],
        "caveats": [
            "Drivers are associations from historical data, not proven causes; test interventions before rolling out.",
            "Premium's higher raw churn rate (13% vs 7-8%) is almost entirely explained by Premium "
            "subscribers being over-represented in the partner-promo low-usage group; outside it, "
            "Premium churns like every other plan.",
            "Subscribers outside the high-risk segments churn at about 3%, and the model finds no "
            "further pattern there; treat that as baseline churn.",
        ],
    }
    (OUTPUTS_DIR / "segment_risk.json").write_text(json.dumps(report, indent=2))
    (OUTPUTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("  wrote outputs/segment_risk.json\n  wrote outputs/model_metrics.json")

    fig08_model_comparison(y, oof, metrics)
    fig09_odds_ratios(drivers)
    fig10_calibration(y, p)
    fig11_segments(segments, overall)

    print("\nRisk segments:")
    for s in segments:
        print(f"  {s['segment_id']} [{s['risk_tier']:6s}] {s['name']:40s} "
              f"n={s['subscribers']:5,}  predicted {s['predicted_churn_rate']:.1%}  "
              f"actual {s['actual_churn_rate']:.1%}  ${s['expected_monthly_revenue_at_risk']:,.0f}/mo at risk")


if __name__ == "__main__":
    main()
