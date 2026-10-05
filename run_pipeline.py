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

Prefer a browser? Run `python app.py` and upload the CSV at http://127.0.0.1:5000
"""
import argparse
import os
import sys
from pathlib import Path

from churn_pipeline import config as C
from churn_pipeline import recommend, runner


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
                cfg[key] = val if key != "csv" else Path(os.path.relpath(os.path.abspath(val), C.ROOT)).as_posix()
                changed = True
        if a.target and a.target != cfg["target"]:
            sys.exit("To change the target, rerun with --redetect --target <column>.")
        if changed:
            C.save(cfg)

    print(f"\n[{cfg['name']}]")
    result = runner.run(cfg, use_gemini=not a.skip_gemini, gemini_model=a.model, dry_run=a.dry_run)
    print(f"\nDashboard: {result['dashboard'].relative_to(C.ROOT)}  (all datasets: docs/index.html)")

if __name__ == "__main__":
    main()
