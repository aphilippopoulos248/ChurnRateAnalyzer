"""
Phase 2 — churn model + risk segments for any dataset.

Compares logistic regression (raw), logistic regression + the interaction
flags found in Phase 1, and gradient boosting, with 5-fold stratified CV.
Picks the explainable model unless gradient boosting is clearly better,
then groups customers into risk segments with a shallow surrogate tree
fitted to out-of-fold probabilities.
"""
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

from . import style
from .explore import T, flag_mask

SEED = 42
GB_MARGIN = 0.01   # pick gradient boosting only if it beats the explainable model by this ROC-AUC


def add_flags(df, flags, corrections):
    df = df.copy()
    for f in flags:
        df[f["name"]] = flag_mask(df, f).astype(int)
    for c in corrections:
        df[c["name"]] = df[c["of"][0]] * df[c["of"][1]]
    return df


def references(df, cats):
    return {c: df[c].value_counts().index[0] for c in cats}


def logistic(cats, nums, flags, refs):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop=[refs[c] for c in cats], handle_unknown="ignore"), cats),
        ("num", StandardScaler(), nums),
        ("flag", "passthrough", flags),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=3000))])


def boosting(cats):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300, random_state=SEED))])


def evaluate(y, p):
    top = p >= np.quantile(p, 0.9)
    return {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "top_decile_churn_rate": round(float(y[top].mean()), 4),
            "top_decile_lift": round(float(y[top].mean() / y.mean()), 2),
            "top_decile_share_of_churners_captured": round(float(y[top].sum() / y.sum()), 4)}


# ---------------------------------------------------------------- drivers
def odds_ratios(pipe, cats, refs, flag_labels, skip):
    names = pipe.named_steps["pre"].get_feature_names_out()
    rows = []
    for raw, b in zip(names, pipe.named_steps["model"].coef_[0]):
        kind, name = raw.split("__", 1)
        if name in skip:
            continue
        if kind == "cat":
            col = next(c for c in sorted(cats, key=len, reverse=True) if name.startswith(c + "_"))
            term, note = f"{col} = {name[len(col) + 1:]}", f"vs {refs[col]}"
        elif kind == "flag":
            term, note = flag_labels.get(name, name), "if present vs absent"
        else:
            term, note = name, "per 1 standard deviation increase"
        rows.append({"feature": term, "comparison": note, "odds_ratio": round(float(np.exp(b)), 3),
                     "direction": "raises churn" if b > 0 else "lowers churn"})
    return sorted(rows, key=lambda r: -abs(np.log(r["odds_ratio"])))


# ---------------------------------------------------------------- segments
def _fmt_num(v):
    v = float(v)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")


def _structured(feature, thr, right, flags_by_name, integer):
    if feature in flags_by_name:
        return ("flag", feature, right)
    if "=" in feature:
        col, val = feature.split("=", 1)
        return ("cat", col, val, right)
    if integer.get(feature, False):
        t = int(np.floor(thr))
        return ("num", feature, ">=", t + 1) if right else ("num", feature, "<=", t)
    return ("num", feature, ">", thr) if right else ("num", feature, "<=", thr)


def _rule_text(c, flags_by_name):
    if c[0] == "flag":
        lab = flags_by_name[c[1]]
        return lab if c[2] else f"NOT ({lab})"
    if c[0] == "cat":
        return f"{c[1]} = {c[2]}" if c[3] else f"{c[1]} != {c[2]}"
    return f"{c[1]} {c[2]} {_fmt_num(c[3])}"


def _segment_name(conds, flags_by_name, df):
    """Readable name: flags first, then category and numeric conditions, simplified."""
    parts = [flags_by_name[c[1]] for c in conds if c[0] == "flag" and c[2]]
    cats = {}
    for c in conds:
        if c[0] == "cat":
            cats.setdefault(c[1], {"eq": None, "ne": set()})
            if c[3]:
                cats[c[1]]["eq"] = c[2]
            else:
                cats[c[1]]["ne"].add(c[2])
    for col, v in cats.items():
        if v["eq"] is not None:
            parts.append(f"{col} = {v['eq']}")
        else:
            rest = [x for x in df[col].astype(str).unique() if x not in v["ne"]]
            parts.append(f"{col} = {rest[0]}" if len(rest) == 1 else
                         f"{col} not {' / '.join(sorted(v['ne']))}")
    nums = {}
    for c in conds:
        if c[0] == "num":
            key = (c[1], c[2] in (">=", ">"))
            nums[key] = c  # deeper split on the same side wins
    for c in nums.values():
        parts.append(f"{c[1]} {c[2]} {_fmt_num(c[3])}")
    return " + ".join(parts) if parts else None


def build_segments(df, p, cfg, flags, overall):
    flags_by_name = {f["name"]: f["label"] for f in flags}
    X = pd.get_dummies(df[cfg["categorical"]], prefix_sep="=", dtype=int) if cfg["categorical"] else pd.DataFrame(index=df.index)
    X = X.join(df[cfg["numeric"] + [f["name"] for f in flags]])
    integer = {c: bool(np.allclose(df[c], np.round(df[c]))) for c in cfg["numeric"]}
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(100, int(0.01 * len(df))),
                                 min_impurity_decrease=0.005 * float(np.var(p)), random_state=SEED).fit(X, p)
    t, cols = tree.tree_, list(X.columns)
    rules = {}

    def walk(node, conds, pos):
        if t.children_left[node] == -1:
            rules[node] = (conds, pos)
            return
        f, thr = cols[t.feature[node]], t.threshold[node]
        for child, right in ((t.children_left[node], False), (t.children_right[node], True)):
            walk(child, conds + [_structured(f, thr, right, flags_by_name, integer)], pos)

    walk(0, [], [])
    leaf = tree.apply(X)
    rev = cfg.get("revenue_column")
    total_risk = float((df[rev] * p).sum()) if rev else None
    segs = []
    for node, (conds, pos) in rules.items():
        m = leaf == node
        seg, ps = df[m], p[m]
        pred = float(ps.mean())
        tier = "High" if pred >= 2 * overall else "Medium" if pred >= 1.2 * overall else "Low"
        traits = []
        for c in cfg["categorical"]:
            ss, pp = seg[c].value_counts(normalize=True), df[c].value_counts(normalize=True)
            for v, sh in ss.items():
                if sh >= 0.25 and sh / pp[v] >= 1.5:
                    traits.append({"attribute": c, "value": v, "share_in_segment": round(float(sh), 3),
                                   "share_in_population": round(float(pp[v]), 3),
                                   "over_representation": round(float(sh / pp[v]), 2)})
        numeric_profile = {c: {"segment_median": round(float(seg[c].median()), 2),
                               "overall_median": round(float(df[c].median()), 2)} for c in cfg["numeric"]}
        s = {"name": _segment_name(conds, flags_by_name, df), "rule": [_rule_text(c, flags_by_name) for c in conds],
             "risk_tier": tier,
             "subscribers": int(m.sum()), "share_of_base": round(float(m.mean()), 4),
             "predicted_churn_rate": round(pred, 4), "actual_churn_rate": round(float(seg[T].mean()), 4),
             "lift_vs_overall": round(pred / overall, 2), "expected_churners": round(float(ps.sum()), 1),
             "distinguishing_traits": sorted(traits, key=lambda d: -d["over_representation"]),
             "numeric_profile": numeric_profile, "_mask": m}
        if rev:
            risk = float((seg[rev] * ps).sum())
            s.update(monthly_revenue=round(float(seg[rev].sum()), 2),
                     expected_monthly_revenue_at_risk=round(risk, 2),
                     share_of_total_revenue_at_risk=round(risk / total_risk, 4) if total_risk else None)
        segs.append(s)
    segs.sort(key=lambda s: -s["predicted_churn_rate"])
    # name segments without a positive condition
    for s in segs:
        if s["name"] is None:
            s["name"] = "Everyone else (baseline)"
    low = segs[-1]
    if low["risk_tier"] == "Low" and low["share_of_base"] >= 0.25 and "baseline" not in low["name"]:
        low["name"] = f"Lowest risk: {low['name']}"
    for i, s in enumerate(segs, 1):
        s["segment_id"] = f"S{i}"
    return segs


def caveats(df, cfg, segs, overall, attributes):
    out = ["Drivers are associations from historical data, not proven causes; test interventions before rolling out."]
    high = [s for s in segs if s["risk_tier"] == "High"]
    if not high:
        return out
    in_high = np.zeros(len(df), dtype=bool)
    for s in high:
        in_high |= s["_mask"]
    rest = df[~in_high]
    base = float(rest[T].mean())
    names = ", ".join(s["segment_id"] for s in high)
    for c in cfg["categorical"]:
        for r in attributes[c]["rows"]:
            if r["n"] < 100 or r["rate"] < 1.25 * overall:
                continue
            sub = rest[rest[c].astype(str) == r["label"]]
            if len(sub) < 50:
                continue
            r_out = float(sub[T].mean())
            if r_out <= 1.25 * base:
                out.append(f"{c} = {r['label']} has a high raw churn rate ({r['rate']:.0%}), but outside the "
                           f"high-risk segments ({names}) it churns at {r_out:.1%}, close to the {base:.1%} baseline. "
                           f"Its raw rate comes from those segments, so only target {c} = {r['label']} in the combinations they define.")
    out.append(f"{cfg['entity'].capitalize()} outside the high-risk segments churn at about {base:.1%}; "
               "treat that as baseline churn.")
    return out


# ---------------------------------------------------------------- main
def run(df, cfg, explore_res, out_dir):
    flags, corrections = explore_res["flags"], explore_res["corrections"]
    df = add_flags(df, flags, corrections)
    y = df[T].to_numpy()
    overall = float(y.mean())
    cats, nums = cfg["categorical"], cfg["numeric"]
    flag_cols = [f["name"] for f in flags] + [c["name"] for c in corrections]
    refs = references(df, cats)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    cands = {"Logistic regression (raw features)": (lambda: logistic(cats, nums, [], refs), cats + nums)}
    if flags:
        cands["Logistic regression + interaction flags"] = (lambda: logistic(cats, nums, flag_cols, refs),
                                                            cats + nums + flag_cols)
    cands["Gradient boosting (raw features)"] = (lambda: boosting(cats), cats + nums)

    oof, metrics = {}, {}
    for name, (make, cols) in cands.items():
        oof[name] = cross_val_predict(make(), df[cols], y, cv=cv, method="predict_proba")[:, 1]
        metrics[name] = evaluate(y, oof[name])
        print(f"  {name:42s} ROC-AUC {metrics[name]['roc_auc']:.3f}  PR-AUC {metrics[name]['pr_auc']:.3f}  "
              f"Brier {metrics[name]['brier']:.4f}")
    explainable = "Logistic regression + interaction flags" if flags else "Logistic regression (raw features)"
    gb = "Gradient boosting (raw features)"
    if metrics[gb]["roc_auc"] > metrics[explainable]["roc_auc"] + GB_MARGIN:
        chosen = gb
        reason = (f"Gradient boosting beats the explainable model by more than {GB_MARGIN} ROC-AUC, so it scores "
                  "customers; logistic regression is still used to explain the drivers.")
    else:
        chosen = explainable
        reason = (f"{explainable} is within {GB_MARGIN} ROC-AUC of gradient boosting, so the explainable "
                  "model is preferred.")
    print(f"  chosen: {chosen}")
    p = oof[chosen]

    make, cols = cands[chosen]
    final = make().fit(df[cols], y)
    lr_make, lr_cols = cands[explainable]
    lr = final if chosen == explainable else lr_make().fit(df[lr_cols], y)
    drivers = odds_ratios(lr, cats, refs, {f["name"]: f["label"] for f in flags}, {c["name"] for c in corrections})

    segs = build_segments(df, p, cfg, flags, overall)
    cav = caveats(df, cfg, segs, overall, explore_res["attributes"])
    if corrections:
        cav.append("Some customers match more than one interaction flag; their risks overlap rather than stack, "
                   "so the model includes correction terms that are left out of the driver list.")

    bins = np.array([0, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    cal = pd.DataFrame({"b": np.digitize(p, bins[1:-1]), "p": p, "y": y}).groupby("b").agg(
        pred=("p", "mean"), act=("y", "mean"), n=("y", "size"))
    calibration = [{"predicted": round(float(r.pred), 4), "actual": round(float(r.act), 4), "n": int(r.n)}
                   for r in cal.itertuples() if r.n >= 30]

    def thin(xs, ys, n=150):
        idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
        return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]
    curves = {}
    for name, prob in oof.items():
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        curves[name] = {"roc": thin(fpr, tpr), "pr": thin(rec[::-1], prec[::-1])}

    rev = cfg.get("revenue_column")
    figs = style.phase2_figures(y, oof, metrics, chosen, drivers, segs, calibration, overall, out_dir,
                                cfg.get("currency", "$"))
    for s in segs:
        s.pop("_mask")
    segs = [{"segment_id": s.pop("segment_id"), **s} for s in segs]
    report = {
        "purpose": (f"Churn-risk segments for {cfg['entity']} of {cfg['business_context']}. Use these to decide "
                    "which groups to target and what retention actions to take."),
        "dataset": {"name": cfg["name"], "rows": len(df), "entity": cfg["entity"],
                    "business_context": cfg["business_context"],
                    "overall_churn_rate": round(overall, 4),
                    "total_monthly_revenue": round(float(df[rev].sum()), 2) if rev else None,
                    "expected_monthly_revenue_at_risk": round(float((df[rev] * p).sum()), 2) if rev else None,
                    "currency": cfg.get("currency", "$")},
        "model": {"type": chosen, "why": reason,
                  "validation": "5-fold stratified cross-validation; segment figures use out-of-fold predictions",
                  "metrics": metrics[chosen],
                  "engineered_features": {f["name"]: f["description"] for f in flags}},
        "segments": segs,
        "churn_drivers": [d for d in drivers if abs(np.log(d["odds_ratio"])) >= np.log(1.25)],
        "caveats": cav,
    }
    model_data = {"chosen_model": chosen, "explainable_model": explainable, "metrics": metrics,
                  "curves": curves, "calibration": calibration, "odds_ratios": drivers}
    return report, model_data, {"pipeline": final, "features": cols, "flags": flags,
                                "corrections": corrections}, figs
