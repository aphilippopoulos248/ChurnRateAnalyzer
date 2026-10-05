"""
SonicWave churn — web dashboard builder.

Collects the results of all three phases into one JSON payload and injects
it into dashboard/template.html, producing a single self-contained page:

    docs/index.html    (GitHub Pages serves this folder)

Run after the pipeline:
    python visualize_churn.py
    python train_churn_model.py
    python generate_recommendations.py
    python build_dashboard.py
"""
import json
from datetime import datetime, timezone

import pandas as pd

from features import ROOT, TARGET
from visualize_churn import LABELS, ORDER, load, rate_table

OUTPUTS = ROOT / "outputs"
TEMPLATE = ROOT / "dashboard" / "template.html"
OUT_DIR = ROOT / "docs"
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"

ATTRIBUTES = ["plan_type", "content_mix", "signup_channel", "last_ticket_topic",
              "tickets_bucket", "payment_method", "age_group", "tenure_bucket",
              "hours_bucket"]
HEATMAPS = {
    "tickets_topic": ("tickets_bucket", "last_ticket_topic",
                      "Repeat tickets only predict churn when the topic is Billing"),
    "channel_mix": ("signup_channel", "content_mix",
                    "Partner-promo signups who barely listen churn at ~52%"),
    "plan_topic": ("plan_type", "last_ticket_topic",
                   "Billing tickets raise churn on every plan"),
}


def read_json(name, required=True):
    path = OUTPUTS / name
    if not path.exists():
        if required:
            raise SystemExit(f"outputs/{name} not found. Run the earlier pipeline steps first.")
        print(f"  note: outputs/{name} not found; that section will show as not run yet")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def phase1(df):
    overall = float(df[TARGET].mean())
    attrs = {}
    for col in ATTRIBUTES:
        t = rate_table(df, col)
        attrs[col] = {"label": LABELS[col],
                      "rows": [{"label": str(k), "rate": round(float(r.rate), 4), "n": int(r.n)}
                               for k, r in t.iterrows()]}
    heat = {}
    for key, (row, col, title) in HEATMAPS.items():
        rate = pd.crosstab(df[row], df[col], values=df[TARGET], aggfunc="mean")
        n = pd.crosstab(df[row], df[col])
        rows = [r for r in ORDER[row] if r in rate.index]
        cols = [c for c in ORDER[col] if c in rate.columns]
        heat[key] = {
            "title": title, "row_label": LABELS[row], "col_label": LABELS[col],
            "rows": rows, "cols": cols,
            "cells": [[{"rate": None if pd.isna(rate.at[r, c]) else round(float(rate.at[r, c]), 4),
                        "n": int(n.at[r, c]) if (r in n.index and c in n.columns) else 0}
                       for c in cols] for r in rows],
        }
    lost = (df[df[TARGET] == 1].groupby("plan_type")["monthly_spend"].sum()
              .reindex(ORDER["plan_type"]).fillna(0))
    total = df.groupby("plan_type")["monthly_spend"].sum().reindex(ORDER["plan_type"]).fillna(0)
    revenue = [{"plan": p, "lost": round(float(lost[p]), 2), "total": round(float(total[p]), 2)}
               for p in ORDER["plan_type"] if total[p] > 0]
    return {"overall": round(overall, 4), "attributes": attrs, "heatmaps": heat,
            "revenue_by_plan": revenue}


def main():
    df = load()
    segments = read_json("segment_risk.json")
    model = read_json("dashboard_model.json")
    metrics = read_json("model_metrics.json")
    recs = read_json("recommendations.json", required=False)

    data = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "phase1": phase1(df),
        "segments": segments,
        "model": {**model, "metrics": metrics},
        "recommendations": recs,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit("Data placeholder missing from dashboard/template.html")
    page = template.replace(PLACEHOLDER, payload)

    OUT_DIR.mkdir(exist_ok=True)
    head, sep, body = page.partition("<!--/head-->")
    if not sep:
        head, body = "", page
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head.strip() + '\n</head>\n<body>\n' + body.strip() + '\n</body>\n</html>\n')
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote docs/index.html ({len(html) / 1024:.0f} KB). Open it in a browser to view.")


if __name__ == "__main__":
    main()
