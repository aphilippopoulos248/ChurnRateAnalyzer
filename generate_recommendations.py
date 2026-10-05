"""
SonicWave churn — Phase 3: retention recommendations with Gemini.

Sends Phase 2's segment report (outputs/segment_risk.json) and the key charts
from figures/ to Gemini, asks for a retention plan in a fixed JSON schema,
checks the answer against the data, and writes:

    outputs/recommendations.json   structured plan (machine-readable)
    outputs/recommendations.md     readable report

Setup:
    pip install -r requirements.txt
    Put GEMINI_API_KEY=your-key in a .env file in this folder (git-ignored),
    or set it as an environment variable.

Usage:
    python generate_recommendations.py               # call Gemini
    python generate_recommendations.py --dry-run     # build the prompt only, no API call
    python generate_recommendations.py --model gemini-3.1-pro-preview
"""
import argparse
import json
import os
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from features import ROOT

OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
SEGMENT_REPORT = OUTPUTS / "segment_risk.json"
DEFAULT_MODEL = "gemini-3.8-flash"

# Charts Gemini sees, with a one-line caption so it knows what each one is.
CHARTS = {
    "01_churn_by_segment.png": "Phase 1: churn rate by each subscriber attribute",
    "03_heatmap_tickets_x_topic.png": "Phase 1: churn by ticket count x last ticket topic",
    "04_heatmap_channel_x_content_mix.png": "Phase 1: churn by signup channel x content mix",
    "07_revenue_lost_by_plan.png": "Phase 1: monthly revenue lost to churn by plan",
    "09_churn_drivers_odds_ratios.png": "Phase 2: logistic regression odds ratios",
    "11_risk_segments.png": "Phase 2: risk segments, predicted churn and revenue at risk",
}


# ---------------------------------------------------------------- response schema
class Action(BaseModel):
    title: str = Field(description="Short imperative name of the action")
    description: str = Field(description="What to do, concretely, in 2-3 sentences")
    owner: str = Field(description="Team that would own it, e.g. Billing, Support, Partnerships, Product, Marketing")
    effort: Literal["Low", "Medium", "High"]
    timeline: str = Field(description="e.g. '2 weeks', '1 quarter'")
    assumed_churn_reduction: float = Field(
        description="Assumed relative reduction in this segment's churn rate, 0-1 (e.g. 0.2 = 20% fewer churners). Be conservative.")
    estimated_churners_saved: float = Field(
        description="segment expected_churners x assumed_churn_reduction")
    estimated_monthly_revenue_saved: float = Field(
        description="segment expected_monthly_revenue_at_risk x assumed_churn_reduction")
    kpi: str = Field(description="Metric that shows whether it worked")


class Experiment(BaseModel):
    hypothesis: str
    design: str = Field(description="How to A/B test it: groups, size, duration")
    success_metric: str


class SegmentPlan(BaseModel):
    segment_id: str = Field(description="Must match a segment_id from the data, e.g. S1")
    segment_name: str
    priority_rank: int = Field(description="1 = tackle first. Rank by expected monthly revenue at risk and how actionable the segment is")
    priority_rationale: str
    root_cause_hypothesis: str = Field(description="Most likely reason this group churns, stated as a hypothesis")
    evidence: list[str] = Field(description="Specific numbers from the data or charts that support the hypothesis")
    actions: list[Action]
    experiment: Experiment


class ChartObservation(BaseModel):
    chart: str = Field(description="Chart file name")
    observation: str = Field(description="What this chart adds beyond the JSON numbers")


class RetentionPlan(BaseModel):
    executive_summary: str = Field(description="4-6 sentences for a non-technical manager")
    segment_plans: list[SegmentPlan]
    chart_observations: list[ChartObservation]
    what_not_to_do: list[str] = Field(description="Tempting actions the data argues against, with the reason")
    risks_and_caveats: list[str]


SYSTEM_PROMPT = """You are a senior retention strategist for SonicWave, a music and podcast \
streaming service. You receive the output of a churn model (risk segments, churn drivers, \
caveats) and charts from the analysis. Produce a retention plan.

Rules:
- Ground every claim in the provided data or charts. Quote the actual numbers. Never invent \
statistics, customer quotes, or features of the product that the data doesn't mention.
- Write one segment plan per segment in the data, using its exact segment_id.
- Rank segments by expected monthly revenue at risk and how actionable they are, not by churn \
rate alone. A large segment with slightly lower churn can matter more.
- Drivers are associations, not proven causes: state root causes as hypotheses and give each \
high-risk segment an A/B test to confirm the fix works before full rollout.
- Low-risk/baseline segments churn at a normal background rate. Recommend only light, \
low-cost actions there; don't spend effort trying to eliminate normal churn.
- Respect the caveats in the data. In particular, don't recommend changing a plan's price or \
features if the caveats say that plan's raw churn is explained by another segment.
- Savings estimates must be conservative: assumed_churn_reduction between 0.05 and 0.5, and \
estimated figures must equal the segment's expected numbers times that reduction.
- 2-4 actions per high-risk segment, 1-2 for low-risk segments."""


# ---------------------------------------------------------------- prompt
def build_contents(report, types):
    parts = [types.Part.from_text(text=(
        "Phase 2 churn segment report (JSON):\n\n" + json.dumps(report, indent=2)))]
    attached = []
    for name, caption in CHARTS.items():
        path = FIGURES / name
        if not path.exists():
            print(f"  warning: {name} not found, skipping (run Phase 1/2 first)")
            continue
        parts.append(types.Part.from_text(text=f"Chart {name}: {caption}"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
        attached.append(name)
    parts.append(types.Part.from_text(text=(
        "Using the report and the charts above, write the retention plan.")))
    return parts, attached


def load_api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("No API key found. Add GEMINI_API_KEY=your-key to a .env file in "
                 f"{ROOT} or set it as an environment variable.")
    return key


def call_gemini(model, contents, attempts=3):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RetentionPlan,
    )
    for i in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config)
            if response.parsed is not None:
                return response.parsed, response.usage_metadata
            return RetentionPlan.model_validate_json(response.text), response.usage_metadata
        except Exception as e:  # network errors, rate limits, malformed JSON
            if i == attempts:
                raise
            wait = 2 ** i
            print(f"  attempt {i} failed ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------- validation
def validate(plan: RetentionPlan, report):
    """Check Gemini's answer against the data. Returns a list of warnings."""
    segs = {s["segment_id"]: s for s in report["segments"]}
    warnings = []
    seen = {p.segment_id for p in plan.segment_plans}
    for sid in segs.keys() - seen:
        warnings.append(f"{sid} has no plan.")
    for p in plan.segment_plans:
        seg = segs.get(p.segment_id)
        if seg is None:
            warnings.append(f"Plan refers to unknown segment {p.segment_id}.")
            continue
        total_rev = sum(a.estimated_monthly_revenue_saved for a in p.actions)
        total_churn = sum(a.estimated_churners_saved for a in p.actions)
        if total_rev > seg["expected_monthly_revenue_at_risk"] + 1:
            warnings.append(
                f"{p.segment_id}: actions claim ${total_rev:,.0f}/mo saved, more than the "
                f"${seg['expected_monthly_revenue_at_risk']:,.0f}/mo at risk.")
        if total_churn > seg["expected_churners"] + 0.5:
            warnings.append(
                f"{p.segment_id}: actions claim {total_churn:.0f} churners saved, more than the "
                f"{seg['expected_churners']:.0f} expected.")
        for a in p.actions:
            expected = seg["expected_monthly_revenue_at_risk"] * a.assumed_churn_reduction
            if abs(a.estimated_monthly_revenue_saved - expected) > max(5, 0.1 * expected):
                warnings.append(
                    f"{p.segment_id} / '{a.title}': revenue saved ${a.estimated_monthly_revenue_saved:,.0f} "
                    f"doesn't match {a.assumed_churn_reduction:.0%} of at-risk revenue (${expected:,.0f}).")
    if len(segs) and len({p.priority_rank for p in plan.segment_plans}) != len(plan.segment_plans):
        warnings.append("Priority ranks are not unique.")
    return warnings


# ---------------------------------------------------------------- report
def render_markdown(plan: RetentionPlan, report, model, warnings):
    segs = {s["segment_id"]: s for s in report["segments"]}
    d = report["dataset"]
    out = [
        "# SonicWave Retention Plan",
        "",
        f"*Generated by `{model}` from the Phase 2 churn model "
        f"({report['model']['type']}, ROC-AUC {report['model']['metrics']['roc_auc']}). "
        "Numbers in the tables come from the model; the strategy and savings estimates "
        "come from Gemini and are untested assumptions.*",
        "",
        f"**Base:** {d['subscribers']:,} subscribers, {d['overall_churn_rate']:.1%} churn, "
        f"${d['expected_monthly_revenue_at_risk']:,.0f} of ${d['total_monthly_revenue']:,.0f} "
        "monthly revenue at risk.",
        "",
        "## Executive summary",
        "",
        plan.executive_summary,
        "",
        "## Priorities",
        "",
        "| Rank | Segment | Subscribers | Predicted churn | Revenue at risk / mo | Est. saved / mo |",
        "|---|---|---|---|---|---|",
    ]
    plans = sorted(plan.segment_plans, key=lambda p: p.priority_rank)
    for p in plans:
        s = segs.get(p.segment_id, {})
        saved = sum(a.estimated_monthly_revenue_saved for a in p.actions)
        out.append(
            f"| {p.priority_rank} | {p.segment_id} {s.get('name', p.segment_name)} | "
            f"{s.get('subscribers', 0):,} | {s.get('predicted_churn_rate', 0):.0%} | "
            f"${s.get('expected_monthly_revenue_at_risk', 0):,.0f} | ${saved:,.0f} |")
    for p in plans:
        out += ["", f"## {p.priority_rank}. {p.segment_id} — {p.segment_name}", "",
                f"**Why this rank:** {p.priority_rationale}", "",
                f"**Root-cause hypothesis:** {p.root_cause_hypothesis}", "",
                "**Evidence:**", ""]
        out += [f"- {e}" for e in p.evidence]
        out += ["", "**Actions:**", "",
                "| Action | Owner | Effort | Timeline | Assumed reduction | Est. saved / mo | KPI |",
                "|---|---|---|---|---|---|---|"]
        for a in p.actions:
            out.append(f"| **{a.title}** — {a.description} | {a.owner} | {a.effort} | "
                       f"{a.timeline} | {a.assumed_churn_reduction:.0%} | "
                       f"${a.estimated_monthly_revenue_saved:,.0f} | {a.kpi} |")
        e = p.experiment
        out += ["", "**Experiment:**", "",
                f"- *Hypothesis:* {e.hypothesis}",
                f"- *Design:* {e.design}",
                f"- *Success metric:* {e.success_metric}"]
    out += ["", "## What not to do", ""] + [f"- {x}" for x in plan.what_not_to_do]
    out += ["", "## What the charts show", ""]
    out += [f"- **{c.chart}:** {c.observation}" for c in plan.chart_observations]
    out += ["", "## Risks and caveats", ""] + [f"- {x}" for x in plan.risks_and_caveats]
    out += ["", "## Validation", ""]
    out += ([f"- ⚠️ {w}" for w in warnings] if warnings
            else ["- All segment IDs, savings totals and per-action estimates check out against the model's numbers."])
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the prompt and save it to outputs/gemini_prompt.md without calling the API")
    args = ap.parse_args()

    if not SEGMENT_REPORT.exists():
        sys.exit("outputs/segment_risk.json not found. Run train_churn_model.py first.")
    report = json.loads(SEGMENT_REPORT.read_text())

    from google.genai import types
    contents, attached = build_contents(report, types)
    print(f"Prompt: segment report ({len(report['segments'])} segments) + {len(attached)} charts")

    if args.dry_run:
        preview = ["# Gemini prompt preview", "", f"Model: `{args.model}`", "",
                   "## System prompt", "", SYSTEM_PROMPT, "",
                   "## Attached charts", ""] + [f"- {c}" for c in attached] + [
                   "", "## Response schema", "", "```json",
                   json.dumps(RetentionPlan.model_json_schema(), indent=2), "```"]
        (OUTPUTS / "gemini_prompt.md").write_text("\n".join(preview), encoding="utf-8")
        print("  dry run: wrote outputs/gemini_prompt.md (no API call)")
        return

    print(f"Calling {args.model} ...")
    t0 = time.time()
    plan, usage = call_gemini(args.model, contents)
    print(f"  done in {time.time() - t0:.1f}s"
          + (f", {usage.prompt_token_count:,} input / {usage.candidates_token_count:,} output tokens"
             if usage and usage.prompt_token_count else ""))

    warnings = validate(plan, report)
    result = {"model": args.model, "validation_warnings": warnings, **plan.model_dump()}
    (OUTPUTS / "recommendations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUTS / "recommendations.md").write_text(
        render_markdown(plan, report, args.model, warnings), encoding="utf-8")
    print("  wrote outputs/recommendations.json\n  wrote outputs/recommendations.md")

    print("\nPriorities:")
    for p in sorted(plan.segment_plans, key=lambda p: p.priority_rank):
        saved = sum(a.estimated_monthly_revenue_saved for a in p.actions)
        print(f"  {p.priority_rank}. {p.segment_id} {p.segment_name}: "
              f"{len(p.actions)} actions, est. ${saved:,.0f}/mo saved")
    if warnings:
        print("\nValidation warnings (also in the report):")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
