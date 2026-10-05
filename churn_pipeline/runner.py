"""
Run every phase for one dataset config. Shared by the command line
(run_pipeline.py) and the web app (app.py); `progress` receives
(step, message) updates so either can show where the run is.
"""
import json
import time

import joblib

from . import config as C
from . import dashboard, explore, model, recommend

STEPS = ["load", "explore", "model", "recommend", "dashboard"]


def run(cfg, use_gemini=True, gemini_model=recommend.DEFAULT_MODEL, dry_run=False,
        progress=lambda step, msg: print(f"  {msg}")):
    run_dir = C.run_dir(cfg["name"])
    out = run_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    progress("load", "Loading and cleaning the data")
    df = C.load_data(cfg)
    progress("load", f"{len(df):,} rows, churn rate {df['__target__'].mean():.1%}")

    progress("explore", "Phase 1: churn by attribute and interaction search")
    ex = explore.run(df, cfg, run_dir)
    progress("explore", f"Top attributes: {', '.join(ex['ranked_features'][:5])}")
    for f in ex["flags"]:
        progress("explore", f"Interaction: {f['label']} (n={f['n']:,}, churn {f['churn_rate']:.0%} vs "
                            f"{f['churn_rate_without_combination']:.0%} for either part alone)")
    if not ex["flags"]:
        progress("explore", "No strong interactions found")
    (out / "explore.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")

    progress("model", "Phase 2: comparing three models with 5-fold cross-validation")
    report, model_data, bundle, _ = model.run(df, cfg, ex, run_dir)
    joblib.dump(bundle, run_dir / "model.joblib")
    (out / "segment_risk.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    m = report["model"]["metrics"]
    progress("model", f"Chosen: {report['model']['type']} (ROC-AUC {m['roc_auc']:.3f})")
    for s in report["segments"]:
        progress("model", f"{s['segment_id']} [{s['risk_tier']}] {s['name']}: n={s['subscribers']:,}, "
                          f"predicted {s['predicted_churn_rate']:.1%}, actual {s['actual_churn_rate']:.1%}")

    recs = None
    rec_path = out / "recommendations.json"
    if use_gemini:
        progress("recommend", f"Phase 3: asking {gemini_model} for a retention plan")
        try:
            recs = recommend.run(report, run_dir, model=gemini_model, dry_run=dry_run)
            if recs is not None:
                progress("recommend", f"Gemini returned {sum(len(p['actions']) for p in recs['segment_plans'])} actions"
                                      + (f", {len(recs['validation_warnings'])} validation warnings"
                                         if recs["validation_warnings"] else ", all numbers check out"))
        except SystemExit as e:
            progress("recommend", f"Skipped Gemini: {e}")
        except Exception as e:  # keep the rest of the run usable if the API call fails
            progress("recommend", f"Gemini call failed ({type(e).__name__}: {e})")
    else:
        progress("recommend", "Phase 3 skipped")
    if recs is None and rec_path.exists():
        prev = json.loads(rec_path.read_text(encoding="utf-8"))
        if {p["segment_id"] for p in prev["segment_plans"]} == {s["segment_id"] for s in report["segments"]}:
            recs = prev
            progress("recommend", "Using the previous recommendations for this dataset")

    progress("dashboard", "Building the dashboard")
    page = dashboard.build_run(run_dir, dashboard.run_data(cfg, ex, report, model_data, recs))
    dashboard.build_site()
    progress("dashboard", f"Done in {time.time() - t0:.0f}s")
    return {"dashboard": page, "report": report, "recommendations": recs}
