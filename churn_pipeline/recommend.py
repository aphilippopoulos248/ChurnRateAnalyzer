"""
Phase 3 — retention recommendations with Gemini, for any dataset.

Sends the segment report and key charts to Gemini, requires a fixed JSON
schema back, and checks Gemini's numbers against the model's.
"""
import json
import os
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from .config import ROOT

DEFAULT_MODEL = "gemini-3.8-flash"


class Action(BaseModel):
    title: str = Field(description="Short imperative name of the action")
    description: str = Field(description="What to do, concretely, in 2-3 sentences")
    owner: str = Field(description="Team that would own it")
    effort: Literal["Low", "Medium", "High"]
    timeline: str = Field(description="e.g. '2 weeks', '1 quarter'")
    assumed_churn_reduction: float = Field(
        description="Assumed relative reduction in this segment's churn, 0-1 (0.2 = 20% fewer churners). Be conservative.")
    estimated_churners_saved: float = Field(description="segment expected_churners x assumed_churn_reduction")
    estimated_monthly_revenue_saved: float = Field(
        description="segment expected_monthly_revenue_at_risk x assumed_churn_reduction; 0 if revenue is not available")
    kpi: str = Field(description="Metric that shows whether it worked")


class Experiment(BaseModel):
    hypothesis: str
    design: str = Field(description="How to A/B test it: groups, size, duration")
    success_metric: str


class SegmentPlan(BaseModel):
    segment_id: str = Field(description="Must match a segment_id from the data, e.g. S1")
    segment_name: str
    priority_rank: int = Field(description="1 = tackle first")
    priority_rationale: str
    root_cause_hypothesis: str
    evidence: list[str] = Field(description="Specific numbers from the data or charts")
    actions: list[Action]
    experiment: Experiment


class ChartObservation(BaseModel):
    chart: str
    observation: str = Field(description="What this chart adds beyond the JSON numbers")


class RetentionPlan(BaseModel):
    executive_summary: str = Field(description="4-6 sentences for a non-technical manager")
    segment_plans: list[SegmentPlan]
    chart_observations: list[ChartObservation]
    what_not_to_do: list[str] = Field(description="Tempting actions the data argues against, with the reason")
    risks_and_caveats: list[str]


def system_prompt(report):
    ds = report["dataset"]
    has_rev = ds.get("expected_monthly_revenue_at_risk") is not None
    rank_by = ("expected monthly revenue at risk and how actionable they are" if has_rev
               else "expected churners and how actionable they are")
    return f"""You are a senior retention strategist. The business is {ds['business_context']}, \
and each row in the data is one of its {ds['entity']}. You receive the output of a churn model (risk segments, churn drivers, \
caveats) and charts from the analysis. Produce a retention plan.

Rules:
- Ground every claim in the provided data or charts. Quote the actual numbers. Never invent statistics, customer \
quotes, or product features the data doesn't mention.
- Write one segment plan per segment in the data, using its exact segment_id.
- Rank segments by {rank_by}, not by churn rate alone. A large segment with slightly lower churn can matter more.
- Drivers are associations, not proven causes: state root causes as hypotheses and give each high-risk segment an \
A/B test to confirm the fix works before full rollout.
- Low-risk/baseline segments churn at a normal background rate. Recommend only light, low-cost actions there.
- Respect every caveat. If a caveat says a group's raw churn is explained by a segment, don't target that group on \
its own anywhere in the plan, including in chart observations.
- Every section must agree with the others: chart observations, actions and what_not_to_do must not contradict.
- Savings estimates must be conservative: assumed_churn_reduction between 0.05 and 0.5, and estimated figures must \
equal the segment's expected numbers times that reduction.{'' if has_rev else ' Revenue is not available: set estimated_monthly_revenue_saved to 0.'}
- 2-4 actions per high-risk segment, 1-2 for other segments."""


def chart_list(fig_dir):
    pngs = sorted(p.name for p in fig_dir.glob("*.png"))
    keep = [n for n in pngs if n.startswith(("01_", "02_", "03_", "04_", "11_", "13_"))]
    return keep


def build_contents(report, fig_dir, types):
    parts = [types.Part.from_text(text="Churn segment report (JSON):\n\n" + json.dumps(report, indent=2))]
    attached = []
    for name in chart_list(fig_dir):
        parts.append(types.Part.from_text(text=f"Chart {name}"))
        parts.append(types.Part.from_bytes(data=(fig_dir / name).read_bytes(), mime_type="image/png"))
        attached.append(name)
    parts.append(types.Part.from_text(text="Using the report and the charts above, write the retention plan."))
    return parts, attached


def load_api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit(f"No API key found. Add GEMINI_API_KEY=your-key to {ROOT / '.env'} or set it as an environment variable.")
    return key


def call_gemini(model, contents, report, attempts=3):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=load_api_key())
    config = types.GenerateContentConfig(system_instruction=system_prompt(report),
                                         response_mime_type="application/json", response_schema=RetentionPlan)
    for i in range(1, attempts + 1):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=config)
            plan = r.parsed if r.parsed is not None else RetentionPlan.model_validate_json(r.text)
            return plan, r.usage_metadata
        except Exception as e:
            if i == attempts:
                raise
            print(f"  attempt {i} failed ({type(e).__name__}: {e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def validate(plan, report):
    segs = {s["segment_id"]: s for s in report["segments"]}
    w = []
    for sid in segs.keys() - {p.segment_id for p in plan.segment_plans}:
        w.append(f"{sid} has no plan.")
    for p in plan.segment_plans:
        seg = segs.get(p.segment_id)
        if seg is None:
            w.append(f"Plan refers to unknown segment {p.segment_id}.")
            continue
        at_risk = seg.get("expected_monthly_revenue_at_risk")
        if at_risk is not None and sum(a.estimated_monthly_revenue_saved for a in p.actions) > at_risk + 1:
            w.append(f"{p.segment_id}: claimed revenue saved exceeds the revenue at risk.")
        if sum(a.estimated_churners_saved for a in p.actions) > seg["expected_churners"] + 0.5:
            w.append(f"{p.segment_id}: claimed churners saved exceeds the expected churners.")
        for a in p.actions:
            if not 0.05 <= a.assumed_churn_reduction <= 0.5:
                w.append(f"{p.segment_id} / '{a.title}': assumed reduction {a.assumed_churn_reduction:.0%} is outside 5-50%.")
            exp_ch = seg["expected_churners"] * a.assumed_churn_reduction
            if abs(a.estimated_churners_saved - exp_ch) > max(1, 0.1 * exp_ch):
                w.append(f"{p.segment_id} / '{a.title}': churners saved doesn't match the stated reduction.")
            if at_risk is not None:
                exp = at_risk * a.assumed_churn_reduction
                if abs(a.estimated_monthly_revenue_saved - exp) > max(5, 0.1 * exp):
                    w.append(f"{p.segment_id} / '{a.title}': revenue saved doesn't match the stated reduction.")
    if len({p.priority_rank for p in plan.segment_plans}) != len(plan.segment_plans):
        w.append("Priority ranks are not unique.")
    return w


def render_markdown(plan, report, model, warnings):
    segs = {s["segment_id"]: s for s in report["segments"]}
    d, cur = report["dataset"], report["dataset"].get("currency", "$")
    has_rev = d.get("expected_monthly_revenue_at_risk") is not None
    out = [f"# Retention Plan: {d['name']}", "",
           f"*Generated by `{model}` from the churn model ({report['model']['type']}, ROC-AUC "
           f"{report['model']['metrics']['roc_auc']}). Segment numbers come from the model; strategies and savings "
           "are Gemini's untested assumptions.*", "",
           f"**Base:** {d['rows']:,} {d['entity']}, {d['overall_churn_rate']:.1%} churn"
           + (f", {cur}{d['expected_monthly_revenue_at_risk']:,.0f} of {cur}{d['total_monthly_revenue']:,.0f} monthly revenue at risk." if has_rev else "."),
           "", "## Executive summary", "", plan.executive_summary, "", "## Priorities", "",
           "| Rank | Segment | Size | Predicted churn | " + ("At risk / mo | Est. saved / mo |" if has_rev else "Expected churners | Est. churners saved |"),
           "|---|---|---|---|---|---|"]
    plans = sorted(plan.segment_plans, key=lambda p: p.priority_rank)
    for p in plans:
        s = segs.get(p.segment_id, {})
        if has_rev:
            tail = f"{cur}{s.get('expected_monthly_revenue_at_risk', 0):,.0f} | {cur}{sum(a.estimated_monthly_revenue_saved for a in p.actions):,.0f} |"
        else:
            tail = f"{s.get('expected_churners', 0):,.0f} | {sum(a.estimated_churners_saved for a in p.actions):,.0f} |"
        out.append(f"| {p.priority_rank} | {p.segment_id} {s.get('name', p.segment_name)} | {s.get('subscribers', 0):,} | "
                   f"{s.get('predicted_churn_rate', 0):.0%} | {tail}")
    for p in plans:
        out += ["", f"## {p.priority_rank}. {p.segment_id} — {p.segment_name}", "",
                f"**Why this rank:** {p.priority_rationale}", "", f"**Root-cause hypothesis:** {p.root_cause_hypothesis}",
                "", "**Evidence:**", ""] + [f"- {e}" for e in p.evidence]
        out += ["", "**Actions:**", "", "| Action | Owner | Effort | Timeline | Assumed reduction | Est. saved | KPI |",
                "|---|---|---|---|---|---|---|"]
        for a in p.actions:
            saved = f"{cur}{a.estimated_monthly_revenue_saved:,.0f}/mo" if has_rev else f"{a.estimated_churners_saved:,.0f} {d['entity']}"
            out.append(f"| **{a.title}** — {a.description} | {a.owner} | {a.effort} | {a.timeline} | "
                       f"{a.assumed_churn_reduction:.0%} | {saved} | {a.kpi} |")
        e = p.experiment
        out += ["", "**Experiment:**", "", f"- *Hypothesis:* {e.hypothesis}", f"- *Design:* {e.design}",
                f"- *Success metric:* {e.success_metric}"]
    out += ["", "## What not to do", ""] + [f"- {x}" for x in plan.what_not_to_do]
    out += ["", "## What the charts show", ""] + [f"- **{c.chart}:** {c.observation}" for c in plan.chart_observations]
    out += ["", "## Risks and caveats", ""] + [f"- {x}" for x in plan.risks_and_caveats]
    out += ["", "## Validation", ""] + ([f"- ⚠️ {x}" for x in warnings] if warnings else
                                       ["- All segment IDs, savings totals and per-action estimates check out against the model."])
    return "\n".join(out) + "\n"


def run(report, run_dir, model=DEFAULT_MODEL, dry_run=False):
    from google.genai import types
    fig_dir = run_dir / "figures"
    contents, attached = build_contents(report, fig_dir, types)
    print(f"  prompt: {len(report['segments'])} segments + {len(attached)} charts")
    out = run_dir / "outputs"
    if dry_run:
        preview = ["# Gemini prompt preview", "", f"Model: `{model}`", "", "## System prompt", "",
                   system_prompt(report), "", "## Attached charts", ""] + [f"- {c}" for c in attached]
        (out / "gemini_prompt.md").write_text("\n".join(preview), encoding="utf-8")
        print("  dry run: wrote outputs/gemini_prompt.md (no API call)")
        return None
    print(f"  calling {model} ...")
    t0 = time.time()
    plan, usage = call_gemini(model, contents, report)
    print(f"  done in {time.time() - t0:.1f}s")
    warnings = validate(plan, report)
    result = {"model": model, "validation_warnings": warnings, **plan.model_dump()}
    (out / "recommendations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "recommendations.md").write_text(render_markdown(plan, report, model, warnings), encoding="utf-8")
    for w in warnings:
        print(f"  warning: {w}")
    return result
