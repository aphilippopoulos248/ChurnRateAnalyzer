"""
Churn pipeline web app.

    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:5000

Upload a CSV, check the detected columns, and the app runs all three phases
(exploration, churn model, Gemini retention plan) and opens the dashboard.
Put GEMINI_API_KEY in .env to enable Phase 3.
"""
import json
import os
import threading
import traceback
from datetime import datetime, timezone

import pandas as pd
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

from churn_pipeline import config as C
from churn_pipeline import recommend, runner

try:
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
except ImportError:
    pass

UPLOADS = C.ROOT / "uploads"
app = Flask(__name__, template_folder="churn_pipeline/web/templates", static_folder="churn_pipeline/web/static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-dev-only")

JOBS = {}            # name -> {"state", "step", "log", "error", "started"}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()   # one pipeline run at a time (matplotlib isn't thread-safe)


def has_api_key():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def past_runs():
    runs = []
    for f in sorted(C.RUNS.glob("*/outputs/dashboard_data.json"), key=lambda p: -p.stat().st_mtime):
        d = json.loads(f.read_text(encoding="utf-8"))
        ds = d["segments"]["dataset"]
        runs.append({"name": d["config"]["name"], "display_name": d["config"]["display_name"],
                     "rows": ds["rows"], "churn": ds["overall_churn_rate"], "built_at": d["built_at"],
                     "segments": len(d["segments"]["segments"]),
                     "high": sum(s["risk_tier"] == "High" for s in d["segments"]["segments"]),
                     "has_recs": d["recommendations"] is not None})
    return runs


def column_profile(cfg):
    raw = pd.read_csv(C.csv_path(cfg), nrows=50000)
    raw.columns = [c.strip() for c in raw.columns]
    rows = []
    for c in raw.columns:
        s = raw[c]
        role = ("target" if c == cfg["target"] else "categorical" if c in cfg["categorical"]
                else "numeric" if c in cfg["numeric"] else "excluded")
        samples = ", ".join(str(v) for v in s.dropna().unique()[:4])
        rows.append({"name": c, "role": role, "unique": int(s.nunique()), "samples": samples[:80],
                     "reason": cfg["excluded"].get(c, ""), "missing": float(s.isna().mean())})
    return rows


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return render_template("index.html", runs=past_runs(), has_key=has_api_key())


@app.post("/upload")
def upload():
    f = request.files.get("csv")
    if not f or not f.filename:
        flash("Choose a CSV file to upload.")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".csv"):
        flash("That isn't a .csv file. Export your data as CSV and try again.")
        return redirect(url_for("index"))
    name = C.slug(request.form.get("name") or os.path.splitext(f.filename)[0])
    UPLOADS.mkdir(exist_ok=True)
    path = UPLOADS / f"{name}.csv"
    f.save(path)
    try:
        cfg = C.detect(path, name=name, context=request.form.get("context") or None)
    except SystemExit as e:
        # detection couldn't find a churn column: let the user pick one
        try:
            cols = list(pd.read_csv(path, nrows=5).columns)
        except Exception:
            flash("Couldn't read that file as a CSV.")
            return redirect(url_for("index"))
        return render_template("pick_target.html", name=name, columns=cols, message=str(e),
                               context=request.form.get("context", ""), entity=request.form.get("entity", ""))
    except Exception as e:
        flash(f"Couldn't read that file: {e}")
        return redirect(url_for("index"))
    if request.form.get("entity"):
        cfg["entity"] = request.form["entity"].strip()
    cfg["display_name"] = request.form.get("display_name") or f.filename.rsplit(".", 1)[0].replace("_", " ").title()
    C.save(cfg)
    _retire_old_recommendations(name)
    return redirect(url_for("configure", name=name))


def _retire_old_recommendations(name):
    """A new upload is new data: don't let an older Gemini plan be reused for it."""
    old = C.run_dir(name) / "outputs" / "recommendations.json"
    if old.exists():
        old.replace(old.with_name("recommendations.previous.json"))


@app.post("/pick-target/<name>")
def pick_target(name):
    path = UPLOADS / f"{C.slug(name)}.csv"
    try:
        cfg = C.detect(path, name=name, target=request.form["target"], context=request.form.get("context") or None)
    except SystemExit as e:
        flash(str(e))
        return redirect(url_for("index"))
    if request.form.get("entity"):
        cfg["entity"] = request.form["entity"].strip()
    cfg["display_name"] = name.replace("_", " ").title()
    C.save(cfg)
    _retire_old_recommendations(name)
    return redirect(url_for("configure", name=name))


@app.get("/configure/<name>")
def configure(name):
    cfg = C.load(C.slug(name))
    return render_template("configure.html", cfg=cfg, columns=column_profile(cfg), has_key=has_api_key(),
                           model=os.getenv("GEMINI_MODEL", recommend.DEFAULT_MODEL))


@app.post("/run/<name>")
def run(name):
    name = C.slug(name)
    cfg = C.load(name)
    form = request.form
    cats, nums, excluded = [], [], {}
    for col in [r["name"] for r in column_profile(cfg)]:
        if col == cfg["target"]:
            continue
        role = form.get(f"role__{col}", "excluded")
        if role == "categorical":
            cats.append(col)
        elif role == "numeric":
            nums.append(col)
        else:
            excluded[col] = cfg["excluded"].get(col) or "excluded by you"
    if not cats and not nums:
        flash("Include at least one column as a feature.")
        return redirect(url_for("configure", name=name))
    cfg.update(categorical=cats, numeric=nums, excluded=excluded,
               revenue_column=form.get("revenue_column") or None,
               business_context=form.get("business_context", "").strip() or cfg["business_context"],
               entity=form.get("entity", "").strip() or cfg["entity"],
               display_name=form.get("display_name", "").strip() or cfg.get("display_name") or name,
               currency=form.get("currency", "$").strip() or "$")
    C.save(cfg)
    use_gemini = form.get("use_gemini") == "on" and has_api_key()
    with JOBS_LOCK:
        if JOBS.get(name, {}).get("state") == "running":
            return redirect(url_for("status", name=name))
        JOBS[name] = {"state": "running", "step": "load", "log": [], "error": None,
                      "started": datetime.now(timezone.utc).isoformat()}
    threading.Thread(target=_work, args=(name, cfg, use_gemini), daemon=True).start()
    return redirect(url_for("status", name=name))


def _work(name, cfg, use_gemini):
    job = JOBS[name]

    def progress(step, msg):
        job["step"] = step
        job["log"].append({"step": step, "msg": msg})
    try:
        if RUN_LOCK.locked():
            progress("load", "Waiting for another run to finish")
        with RUN_LOCK:
                runner.run(cfg, use_gemini=use_gemini, gemini_model=os.getenv("GEMINI_MODEL", recommend.DEFAULT_MODEL),
                       progress=progress)
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["log"].append({"step": job["step"], "msg": "Failed: " + job["error"]})
        traceback.print_exc()


@app.get("/status/<name>")
def status(name):
    name = C.slug(name)
    if name not in JOBS:
        return redirect(url_for("dashboard", name=name))
    cfg = C.load(name)
    return render_template("status.html", name=name, display_name=cfg.get("display_name", name), steps=runner.STEPS)


@app.get("/api/status/<name>")
def api_status(name):
    job = JOBS.get(C.slug(name))
    if not job:
        abort(404)
    return jsonify(job)


@app.get("/dashboard/<name>")
def dashboard(name):
    page = C.run_dir(C.slug(name)) / "dashboard.html"
    if not page.exists():
        abort(404)
    html = page.read_text(encoding="utf-8")
    bar = (f'<div class="appbar"><a href="{url_for("index")}">← Upload another CSV</a>'
           f'<a href="{url_for("configure", name=name)}">Change settings and rerun</a>'
           f'<a href="{url_for("files", name=name, path="outputs/recommendations.md")}">Gemini report (.md)</a>'
           f'<a href="{url_for("files", name=name, path="outputs/segment_risk.json")}">Segment report (.json)</a></div>')
    style = ("<style>.appbar{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;padding:10px 0;"
             "border-bottom:1px solid var(--line)}.appbar a{color:var(--accent);text-decoration:none}"
             ".appbar a:hover{text-decoration:underline}</style>")
    return html.replace("</head>", style + "</head>", 1).replace('<div class="wrap">', '<div class="wrap">' + bar, 1)


@app.get("/runs/<name>/<path:path>")
def files(name, path):
    if not (path.startswith("outputs/") or path.startswith("figures/")):
        abort(404)
    return send_from_directory(C.run_dir(C.slug(name)), path)


@app.errorhandler(413)
def too_large(_):
    flash("That file is over 100 MB. Upload a smaller extract.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Churn pipeline app: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
