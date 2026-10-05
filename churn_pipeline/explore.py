"""
Phase 1 — explore any churn dataset.

* Churn rate for every feature (numerics are binned), ranked by how much of
  the churn variation each one explains.
* Automatic interaction discovery: searches pairs of conditions
  (e.g. "plan = Premium" AND "tenure <= 6") for groups that churn far more than
  either condition does without the other, then refines each with a third
  condition if that sharpens it. The winners become model features.
* Heatmaps for the strongest pairs of columns, plus PNG charts.
"""
import numpy as np
import pandas as pd

from .config import order_of, readable
from . import style

T = "__target__"


# ---------------------------------------------------------------- binning
def bin_numeric(s: pd.Series):
    """Return (binned labels as strings, ordered label list)."""
    vals = s.dropna()
    uniq = np.sort(vals.unique())
    if len(uniq) <= 10 and np.allclose(uniq, np.round(uniq)):
        uniq = uniq.astype(int)
        share_ge = {v: (vals >= v).mean() for v in uniq}
        cap = next((v for v in uniq[1:] if share_ge[v] < 0.05), None)
        def lab(x):
            x = int(round(x))
            return f"{cap}+" if cap is not None and x >= cap else str(x)
        labels = [str(v) for v in uniq if cap is None or v < cap] + ([f"{cap}+"] if cap is not None else [])
        return s.map(lab), labels
    q = pd.qcut(s, 5, duplicates="drop")
    def fmt(v):
        return f"{v:,.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:,.1f}"
    cats = q.cat.categories
    names = [f"{fmt(max(iv.left, vals.min()))}–{fmt(iv.right)}" for iv in cats]
    mapping = dict(zip(cats, names))
    return q.map(mapping).astype(str), names


def binned_frame(df, cfg):
    """Every feature as display groups, plus each column's display order."""
    out, order = pd.DataFrame(index=df.index), {}
    for c in cfg["categorical"]:
        out[c] = df[c]
        order[c] = order_of(df, cfg, c)
    for c in cfg["numeric"]:
        out[c], order[c] = bin_numeric(df[c])
    return out, order


# ---------------------------------------------------------------- univariate
def rate_rows(y, groups, order):
    t = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].agg(["mean", "size"])
    return [{"label": str(g), "rate": round(float(t.at[g, "mean"]), 4), "n": int(t.at[g, "size"])}
            for g in order if g in t.index]


def signal(rows, overall):
    """Share of churn variance explained by the grouping (eta squared)."""
    n = sum(r["n"] for r in rows)
    between = sum(r["n"] * (r["rate"] - overall) ** 2 for r in rows)
    return between / (n * overall * (1 - overall)) if 0 < overall < 1 else 0.0


# ---------------------------------------------------------------- interactions
def conditions(df, cfg, min_n):
    """Candidate single conditions: category values and numeric thresholds."""
    conds = []
    for c in cfg["categorical"]:
        for v, k in df[c].value_counts().items():
            if k >= min_n:
                conds.append({"col": c, "op": "==", "value": v})
    for c in cfg["numeric"]:
        s = df[c]
        uniq = np.sort(s.unique())
        if len(uniq) <= 12:
            cuts = uniq[1:]
        else:
            cuts = np.unique(np.quantile(s, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        for t in cuts:
            t = float(t)
            if (s >= t).sum() >= min_n:
                conds.append({"col": c, "op": ">=", "value": t})
            if (s < t).sum() >= min_n:
                conds.append({"col": c, "op": "<", "value": t})
    return conds


def mask_of(df, cond):
    s = df[cond["col"]]
    if cond["op"] == "==":
        return (s.astype(str) == str(cond["value"])).to_numpy()
    if cond["op"] == ">=":
        return (s >= cond["value"]).to_numpy()
    return (s < cond["value"]).to_numpy()


def flag_mask(df, flag):
    m = np.ones(len(df), dtype=bool)
    for c in flag["conditions"]:
        m &= mask_of(df, c)
    return m


GENERIC_VALUES = {"yes", "no", "true", "false", "0", "1", "none", "other", "missing", "unknown", "n/a", "na"}


def fmt_num(v):
    v = float(v)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")


def describe(cond):
    if cond["op"] == "==":
        return f"{cond['col']} = {cond['value']}"
    return f"{cond['col']} {cond['op']} {fmt_num(cond['value'])}"


def short(cond):
    """Compact label: a distinctive category value on its own ('Billing'),
    otherwise column and value ('OnlineSecurity = No')."""
    if cond["op"] == "==":
        v = str(cond["value"])
        return describe(cond) if (v.lower() in GENERIC_VALUES or len(v) <= 3) else v
    return describe(cond).replace("_", " ")


def _rate(y, m):
    k = m.sum()
    return (y[m].mean() if k else 0.0), k


def find_interactions(df, cfg, overall, max_flags=4, top_features=12, feature_rank=None):
    y = df[T].to_numpy()
    n_rows = len(df)
    min_n = max(30, int(0.005 * n_rows))
    feats = feature_rank[:top_features] if feature_rank else cfg["categorical"] + cfg["numeric"]
    sub_cfg = {"categorical": [c for c in cfg["categorical"] if c in feats],
               "numeric": [c for c in cfg["numeric"] if c in feats]}
    conds = conditions(df, sub_cfg, min_n)
    M = np.array([mask_of(df, c) for c in conds])          # conditions x rows
    Mi = M.astype(np.int32)
    n_c = Mi.sum(1)
    churn_c = Mi @ y
    n_ab = Mi @ Mi.T
    ch_ab = (Mi * y) @ Mi.T
    cols = np.array([c["col"] for c in conds])

    cands = []
    iu = np.triu_indices(len(conds), 1)
    for a, b in zip(*iu):
        if cols[a] == cols[b] or n_ab[a, b] < min_n:
            continue
        r_ab = ch_ab[a, b] / n_ab[a, b]
        if r_ab < 2 * overall:
            continue
        na_only, nb_only = n_c[a] - n_ab[a, b], n_c[b] - n_ab[a, b]
        if na_only < min_n or nb_only < min_n:
            continue  # one condition is (nearly) inside the other: not an interaction
        r_a_only = (churn_c[a] - ch_ab[a, b]) / na_only
        r_b_only = (churn_c[b] - ch_ab[a, b]) / nb_only
        base = max(r_a_only, r_b_only)
        if r_ab < 2 * max(base, 0.01):
            continue
        cands.append({"idx": (a, b), "score": n_ab[a, b] * (r_ab - base),
                      "rate": r_ab, "base": base, "n": int(n_ab[a, b])})
    cands.sort(key=lambda c: -c["score"])

    # greedy: strongest first, skip near-duplicates (same columns or heavy overlap)
    chosen, pair_cols = [], []
    for c in cands:
        a, b = c["idx"]
        colset = {cols[a], cols[b]}
        if any(colset == pc for pc in pair_cols):
            continue
        m = M[a] & M[b]
        if any((m & ch["mask"]).sum() / (m | ch["mask"]).sum() > 0.3 for ch in chosen):
            continue
        flag = {"conditions": [conds[a], conds[b]], "mask": m}
        _refine(flag, M, conds, y, min_n)
        r, k = _rate(y, flag["mask"])
        out_r, _ = _rate(y, ~flag["mask"])
        flag.update(rate=float(r), n=int(k), rate_outside=float(out_r), base_rate=float(c["base"]))
        chosen.append(flag)
        pair_cols.append(colset)
        if len(chosen) >= max_flags:
            break

    flags = []
    for f in chosen:
        f["conditions"] = sorted(f["conditions"], key=lambda c: c["op"] != "==")
        label = " + ".join(short(c) for c in f["conditions"])
        name = "flag_" + "_".join(
            str(c["value"]).lower().replace(" ", "_") if c["op"] == "==" else
            f"{c['col']}_{'ge' if c['op'] == '>=' else 'lt'}_{c['value']:g}" for c in f["conditions"])
        name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)[:60]
        flags.append({
            "name": name, "label": label,
            "description": " AND ".join(describe(c) for c in f["conditions"]),
            "conditions": [{**c, "value": (str(c["value"]) if c["op"] == "==" else float(c["value"]))}
                           for c in f["conditions"]],
            "n": f["n"], "churn_rate": round(f["rate"], 4),
            "churn_rate_without_combination": round(f["base_rate"], 4),
        })
    # correction terms for overlapping flags (risks that overlap rather than stack)
    corrections = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            k = int((chosen[i]["mask"] & chosen[j]["mask"]).sum())
            if k >= 10:
                corrections.append({"name": f"overlap_{i + 1}_{j + 1}", "of": [flags[i]["name"], flags[j]["name"]],
                                    "n": k, "description": f"Has both: {flags[i]['label']} and {flags[j]['label']}"})
    top_pairs = []
    for c in cands:
        a, b = c["idx"]
        pair = (cols[a], cols[b])
        if not any(set(pair) == set(p) for p in top_pairs):
            top_pairs.append(pair)
        if len(top_pairs) >= 3:
            break
    return flags, corrections, top_pairs


def _refine(flag, M, conds, y, min_n):
    """Add a third condition if it isolates where the combination's churn really is."""
    m = flag["mask"]
    r, _ = _rate(y, m)
    used = {c["col"] for c in flag["conditions"]}
    best = None
    for i, c in enumerate(conds):
        if c["col"] in used:
            continue
        inside, outside = m & M[i], m & ~M[i]
        ri, ki = _rate(y, inside)
        ro, ko = _rate(y, outside)
        if ki < min_n or ko < 10 or ri < 1.1 * r or ro > 0.5 * r:
            continue
        score = ki * (ri - ro)
        if best is None or score > best[0]:
            best = (score, i, inside)
    if best:
        flag["conditions"].append(conds[best[1]])
        flag["mask"] = best[2]


# ---------------------------------------------------------------- main
def run(df, cfg, out_dir):
    y = df[T].to_numpy()
    overall = float(y.mean())
    binned, order = binned_frame(df, cfg)
    features = cfg["categorical"] + cfg["numeric"]

    attrs = {}
    for c in features:
        rows = rate_rows(y, binned[c], order[c])
        attrs[c] = {"label": readable(c), "rows": rows, "signal": round(signal(rows, overall), 4),
                    "kind": "numeric" if c in cfg["numeric"] else "categorical"}
    ranked = sorted(features, key=lambda c: -attrs[c]["signal"])
    flags, corrections, top_pairs = find_interactions(df, cfg, overall, feature_rank=ranked)

    heatmaps = {}
    for r, c in top_pairs:
        rate = pd.crosstab(binned[r], binned[c], values=df[T], aggfunc="mean")
        n = pd.crosstab(binned[r], binned[c])
        rows = [v for v in order[r] if v in rate.index]
        colv = [v for v in order[c] if v in rate.columns]
        cells = [[{"rate": None if pd.isna(rate.at[a, b]) else round(float(rate.at[a, b]), 4),
                   "n": int(n.at[a, b])} for b in colv] for a in rows]
        best = max((x for row in cells for x in row if x["rate"] is not None and x["n"] >= 30),
                   key=lambda x: x["rate"])
        heatmaps[f"{r}__{c}"] = {
            "title": f"{readable(r)} × {readable(c).lower()}: up to {best['rate']:.0%} churn",
            "row_label": readable(r), "col_label": readable(c),
            "rows": rows, "cols": colv, "cells": cells}

    revenue = None
    rev = cfg.get("revenue_column")
    if rev:
        revenue = {"column": rev,
                   "total": round(float(df[rev].sum()), 2),
                   "lost": round(float(df.loc[df[T] == 1, rev].sum()), 2)}

    result = {"overall": round(overall, 4), "rows": len(df), "ranked_features": ranked,
              "attributes": attrs, "heatmaps": heatmaps, "flags": flags,
              "corrections": corrections, "revenue": revenue}
    style.phase1_figures(result, cfg, out_dir)
    return result
