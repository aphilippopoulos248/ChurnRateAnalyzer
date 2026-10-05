"""
Dataset configuration: detect the schema of any churn CSV, save it as an
editable config.json, and load + clean the data the same way every time.

A config looks like:
{
  "name": "sonicwave",
  "csv": "data/sonicwave_subscribers.csv",
  "target": "churned", "positive_label": "1",
  "id_column": "subscriber_id",
  "revenue_column": "monthly_spend",
  "categorical": [...], "numeric": [...],
  "excluded": {"column": "reason", ...},
  "category_order": {"age_group": ["Under 25", ...]},
  "entity": "subscribers",
  "business_context": "a music and podcast streaming service",
  "currency": "$"
}
Edit it and rerun to override anything the detector got wrong.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

TARGET_HINTS = ["churned", "churn", "exited", "attrition", "cancelled", "canceled",
                "left", "is_churn", "churn_flag", "attrition_flag"]
REVENUE_HINTS = ["monthly_spend", "monthly_charges", "monthlycharges", "monthly_revenue",
                 "mrr", "monthly_fee", "spend", "revenue", "charges", "price", "fee", "amount"]
POSITIVE_VALUES = {"1", "yes", "y", "true", "t", "churned", "churn", "exited", "left",
                   "cancelled", "canceled", "attrited customer", "attrited"}
MAX_CATEGORIES = 25          # more than this -> keep top values, rest become "Other"
MIN_CATEGORY_SHARE = 0.005   # categories rarer than this fold into "Other"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "dataset"


def _norm(col):
    return re.sub(r"[^a-z0-9]", "", col.lower())


def _maybe_numeric(s: pd.Series):
    """Strings like ' 29.85' or '1,200' -> numbers, if almost all parse."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s
    parsed = pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    ok = parsed.notna().mean()
    blank = s.astype(str).str.strip().isin(["", "nan", "None", "NA", "N/A"]).mean()
    return parsed if ok + blank >= 0.98 and ok > 0.5 else s


def _binary_target(s: pd.Series, positive=None):
    vals = s.dropna().astype(str).str.strip()
    uniq = vals.str.lower().unique()
    if len(uniq) != 2:
        return None, None
    if positive is None:
        hits = [u for u in uniq if u in POSITIVE_VALUES]
        positive = hits[0] if len(hits) == 1 else sorted(uniq)[-1]
    y = s.astype(str).str.strip().str.lower().eq(str(positive).lower()).astype(int)
    return y, str(positive)


def detect(csv_path, name=None, target=None, context=None):
    """Inspect a CSV and propose a config."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)
    excluded = {}

    # target
    if target is None:
        by_norm = {_norm(c): c for c in cols}
        for hint in TARGET_HINTS:
            c = by_norm.get(_norm(hint))
            if c is not None and _binary_target(df[c])[0] is not None:
                target = c
                break
        if target is None:
            for c in cols:
                if "churn" in c.lower() and _binary_target(df[c])[0] is not None:
                    target = c
                    break
    if target is None or target not in cols:
        raise SystemExit("Couldn't find the churn column. Pass it with --target <column name>.")
    y, positive = _binary_target(df[target])
    if y is None:
        raise SystemExit(f"Target column '{target}' must have exactly two values (e.g. 0/1, Yes/No).")

    numeric, categorical = [], []
    id_column, revenue_column = None, None
    n = len(df)
    for c in cols:
        if c == target:
            continue
        s = _maybe_numeric(df[c])
        nunique = s.nunique(dropna=True)
        lname = c.lower()
        if nunique <= 1:
            excluded[c] = "only one value"
            continue
        if (lname == "id" or lname.endswith("id") or lname.endswith("_id") or "identifier" in lname) \
                and nunique >= 0.95 * n:
            id_column = id_column or c
            excluded[c] = "identifier"
            continue
        if pd.api.types.is_bool_dtype(s):
            categorical.append(c)
        elif not pd.api.types.is_numeric_dtype(s):
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9 and not s.astype(str).str.fullmatch(r"[A-Za-z ]+").all():
                    excluded[c] = "date/time (reserved for change-over-time analysis)"
                    continue
            except (TypeError, ValueError):
                pass
            if nunique >= 0.5 * n:
                excluded[c] = "free text or identifier (too many unique values)"
                continue
            categorical.append(c)
        elif nunique <= 2:
            categorical.append(c)  # 0/1 style flags read better as categories
        else:
            numeric.append(c)

    # revenue column: first numeric column matching a hint
    by_norm = {_norm(c): c for c in numeric}
    for hint in REVENUE_HINTS:
        match = by_norm.get(_norm(hint)) or next((c for c in numeric if _norm(hint) in _norm(c)), None)
        if match:
            revenue_column = match
            break

    cfg = {
        "name": slug(name or csv_path.stem),
        "csv": str(csv_path.resolve().relative_to(ROOT)) if csv_path.resolve().is_relative_to(ROOT)
               else str(csv_path.resolve()),
        "target": target,
        "positive_label": positive,
        "id_column": id_column,
        "revenue_column": revenue_column,
        "categorical": categorical,
        "numeric": numeric,
        "excluded": excluded,
        "category_order": {},
        "entity": "customers",
        "business_context": context or "a subscription business",
        "currency": "$",
    }
    # Drop features that are fully determined by a categorical (e.g. price set by plan):
    # they repeat the same information and blur the model's drivers.
    data = clean(df, cfg)
    for c in list(cfg["numeric"]):
        for cat in cfg["categorical"]:
            if data.groupby(cat, observed=True)[c].nunique().max() == 1:
                cfg["numeric"].remove(c)
                cfg["excluded"][c] = f"fully determined by {cat} (redundant)"
                break
    return cfg


def clean(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply a config to a raw dataframe: target -> 0/1, numerics parsed and imputed,
    categoricals as strings with rare values folded into 'Other'."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    y, _ = _binary_target(df[cfg["target"]], cfg["positive_label"])
    out["__target__"] = y
    for c in cfg["numeric"]:
        s = pd.to_numeric(_maybe_numeric(df[c]), errors="coerce")
        out[c] = s.fillna(s.median())
    for c in cfg["categorical"]:
        s = df[c].astype(str).str.strip().replace({"nan": "Missing", "": "Missing", "None": "Missing"})
        share = s.value_counts(normalize=True)
        keep = share[share >= MIN_CATEGORY_SHARE].index[:MAX_CATEGORIES]
        out[c] = s.where(s.isin(keep), "Other")
    rev = cfg.get("revenue_column")
    if rev and rev not in out:
        s = pd.to_numeric(_maybe_numeric(df[rev]), errors="coerce")
        out[rev] = s.fillna(s.median())
    return out


def order_of(df, cfg, col):
    """Display order for a categorical: config order if given, else by size."""
    given = cfg.get("category_order", {}).get(col)
    present = list(df[col].value_counts().index)
    if given:
        return [v for v in given if v in present] + [v for v in present if v not in given]
    return present


def run_dir(name):
    return RUNS / slug(name)


def save(cfg):
    d = run_dir(cfg["name"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return d / "config.json"


def load(name):
    path = run_dir(name) / "config.json"
    if not path.exists():
        raise SystemExit(f"No config at {path}. Run with a CSV first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_data(cfg):
    csv = Path(cfg["csv"])
    csv = csv if csv.is_absolute() else ROOT / csv
    return clean(pd.read_csv(csv), cfg)


def readable(col):
    """'support_tickets_90d' -> 'Support tickets 90d', 'InternetService' -> 'Internet service'."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col)).replace("_", " ").split()
    words = [w if (w.isupper() and len(w) > 1) else w.lower() for w in words]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))
