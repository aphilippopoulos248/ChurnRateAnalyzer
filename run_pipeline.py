"""
Run the whole churn pipeline on any CSV.

    python run_pipeline.py data/my_customers.csv --name telco
    python run_pipeline.py data/sonicwave_subscribers.csv --name sonicwave --context "a music and podcast streaming service"

First run on a CSV detects the schema and saves runs/<name>/config.json.
Check it (target, columns, revenue column, business context), edit if needed,
then rerun with just the name to reuse your edits:

    python run_pipeline.py --name telco

Options:
    --target COL        churn column, if it isn't detected automatically
    --context TEXT      what the business is, for Gemini (e.g. "a telecom provider")
    --entity WORD       what a row is called, e.g. subscribers, customers, members
    --redetect          re-detect the schema even if a config exists
    --skip-gemini       skip Phase 3 (keeps any previous recommendations)
    --dry-run           build the Gemini prompt without calling the API
    --model ID          Gemini model (default gemini-3.8-flash or GEMINI_MODEL)

Outputs go to runs/<name>/ (config, figures, outputs, dashboard.html), and
docs/index.html is rebuilt with every dataset you've run.
"""
import argparse
import json
import os
import sys
import time

import joblib

from churn_pipeline import config as C
from churn_pipeline import dashboard, explore, model, recommend


def main():
    ap = argparse.ArgumentParser(description="Churn pipeline for any CSV")
    ap.add_argument("csv", nargs="?", help="path to the CSV (omit to rerun a saved config)")
    ap.add_argument("--name")
    ap.add_argument("--target")
    ap.add_argument("--context")
    ap.add_argument("--entity")
    ap.add_argument("--redetect", action="store_true")
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", recommend.DEFAULT_MODEL))
    a = ap.parse_args()
    if not a.csv and not a.name:
        ap.error("give a CSV path, or --name of a dataset you've already run")

    name = C.slug(a.name or os.path.splitext(os.path.basename(a.csv))[0])
    cfg_path = C.run_dir(name) / "config.json"
    if a.csv and (a.redetect or not cfg_path.exists()):
        print(f"Detecting schema of {a.csv} ...")
        cfg = C.detect(a.csv, name=name, target=a.target, context=a.context)
        if a.entity:
            cfg["entity"] = a.entity
        C.save(cfg)
        print(f"  target: {cfg['target']} (churn = '{cfg['positive_label']}')")
        print(f"  categorical: {', '.join(cfg['categorical']) or '-'}")
        print(f"  numeric: {', '.join(cfg['numeric']) or '-'}")
        print(f"  revenue column: {cfg['revenue_column'] or 'none found'}")
        for col, why in cfg["excluded"].items():
            print(f"  excluded {col}: {why}")
        print(f"  saved {cfg_path.relative_to(C.ROOT)}. Edit it and rerun with --name {name} to change anything.")
    else:
        cfg = C.load(name)
        changed = False
        for key, val in (("csv", a.csv), ("business_context", a.context), ("entity", a.entity)):
            if val:
                cfg[key] = val if key != "csv" else os.path.relpath(os.path.abspath(val), C.ROOT)
                changed = True
        if a.target and a.target != cfg["target"]:
            sys.exit("To change the target, rerun with --redetect --target <column>.")
        if changed:
            C.save(cfg)

    run_dir = C.run_dir(name)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    df = C.load_data(cfg)
    print(f"\n[{cfg['name']}] {len(df):,} rows, churn rate {df['__target__'].mean():.1%}")

    t0 = time.time()
    print("Phase 1: explore")
    ex = explore.run(df, cfg, run_dir)
    print(f"  top attributes: {', '.join(ex['ranked_features'][:5])}")
    for f in ex["flags"]:
        print(f"  interaction: {f['label']}  (n={f['n']:,}, churn {f['churn_rate']:.0%} vs "
              f"{f['churn_rate_without_combination']:.0%} for either part alone)")
    if not ex["flags"]:
        print("  no strong interactions found")

    print("Phase 2: model")
    report, model_data, bundle, _ = model.run(df, cfg, ex, run_dir)
    joblib.dump(bundle, run_dir / "model.joblib")
    out = run_dir / "outputs"
    (out / "segment_risk.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "explore.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
    for s in report["segments"]:
        print(f"  {s['segment_id']} [{s['risk_tier']:6s}] {s['name'][:48]:48s} n={s['subscribers']:6,}  "
              f"predicted {s['predicted_churn_rate']:.1%}  actual {s['actual_churn_rate']:.1%}")

    print("Phase 3: recommend")
    recs = None
    rec_path = out / "recommendations.json"
    if a.skip_gemini:
        print("  skipped (--skip-gemini)")
    else:
        try:
            recs = recommend.run(report, run_dir, model=a.model, dry_run=a.dry_run)
        except SystemExit as e:
            print(f"  skipped: {e}")
        except Exception as e:  # keep the rest of the pipeline usable if the API call fails
            print(f"  Gemini call failed ({type(e).__name__}: {e}); continuing without new recommendations")
    if recs is None and rec_path.exists():
        recs = json.loads(rec_path.read_text(encoding="utf-8"))
        ids = {s["segment_id"] for s in report["segments"]}
        if {p["segment_id"] for p in recs["segment_plans"]} == ids:
            print("  using previous recommendations from outputs/recommendations.json")
        else:
            print("  previous recommendations don't match the new segments; leaving them out")
            recs = None

    page = dashboard.build_run(run_dir, dashboard.run_data(cfg, ex, report, model_data, recs))
    site = dashboard.build_site()
    print(f"\nDone in {time.time() - t0:.0f}s.")
    print(f"  dashboard: {page.relative_to(C.ROOT)}")
    if site:
        print(f"  site: {site[0].relative_to(C.ROOT)} ({', '.join(site[1])})")


if __name__ == "__main__":
    main()
