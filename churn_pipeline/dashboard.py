"""Build the web dashboard: one page per run and a combined site with a dataset switcher."""
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, RUNS

TEMPLATE = Path(__file__).resolve().parent / "templates" / "dashboard.html"
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"
SITE = ROOT / "docs" / "index.html"


def run_data(cfg, explore_res, report, model_data, recs):
    return {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": {k: cfg.get(k) for k in ("name", "csv", "target", "entity", "business_context", "currency")}
                  | {"display_name": cfg.get("display_name") or cfg["name"].replace("_", " ").title()},
        "phase1": {k: explore_res[k] for k in ("overall", "ranked_features", "attributes", "heatmaps",
                                               "flags", "corrections")}
                  | {"n_features": len(explore_res["ranked_features"])},
        "segments": report,
        "model": model_data,
        "recommendations": recs,
    }


def render(datasets: dict) -> str:
    payload = json.dumps({"datasets": datasets}, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    head, _, body = page.partition("<!--/head-->")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")


def build_site():
    """Combine every run that has dashboard data into docs/index.html."""
    datasets = {}
    for f in sorted(RUNS.glob("*/outputs/dashboard_data.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        datasets[d["config"]["name"]] = d
    if not datasets:
        return None
    SITE.parent.mkdir(exist_ok=True)
    SITE.write_text(render(datasets), encoding="utf-8")
    return SITE, list(datasets)


def build_run(run_dir, data):
    (run_dir / "outputs" / "dashboard_data.json").write_text(json.dumps(data), encoding="utf-8")
    page = run_dir / "dashboard.html"
    page.write_text(render({data["config"]["name"]: data}), encoding="utf-8")
    return page
