# SonicWave Churn Analysis Pipeline

An end-to-end churn analysis of 8,500 subscribers of **SonicWave**, a fictional music/podcast streaming service.

| Phase | Status | What it does |
|---|---|---|
| 1. Data visualization | ✅ Done | Find which subscriber attributes drive churn |
| 2. Churn model | ✅ Done | Custom probability model predicting each subscriber's churn risk |
| 3. Recommendations | ✅ Built | Gemini/Gemma API turns model predictions into business recommendations |

## Dataset

`data/sonicwave_subscribers.csv` has 8,500 rows, 12 columns and no missing values. The overall churn rate is **9.7%**.

| Column | Description |
|---|---|
| `subscriber_id` | Unique ID |
| `age_group` | Under 25 / 25-34 / 35-49 / 50 plus |
| `plan_type` | Free / Basic ($7.99) / Premium ($12.99) / Family ($18.99) |
| `tenure_months` | Months subscribed (1–60) |
| `monthly_spend` | Monthly price paid |
| `avg_weekly_hours` | Average weekly listening hours |
| `content_mix` | Balanced / Music-heavy / Podcast-heavy / Low-usage |
| `payment_method` | Credit card / Debit / PayPal / Gift card |
| `support_tickets_90d` | Support tickets in the last 90 days |
| `last_ticket_topic` | Account / Billing / Content / Playback / No ticket |
| `signup_channel` | Web / iOS / Android / Partner promo |
| `churned` | 1 = churned, 0 = retained (target) |

## Phase 1 — Key findings

**1. Repeat billing tickets are the strongest churn signal.** Subscribers with 2+ tickets whose last ticket was about billing churn at 53–57%. Repeat tickets on any other topic stay at 0–10%.

![Tickets x topic](figures/03_heatmap_tickets_x_topic.png)

**2. Partner-promo signups who never start listening churn at 52%.** Partner promo with any other content mix churns at ~7%, the same as every other channel.

![Channel x content mix](figures/04_heatmap_channel_x_content_mix.png)

**3. Premium churns at 13% vs 7–8% for other plans** and accounts for most of the ~$9.1k in monthly revenue lost to churn. *(Phase 2 shows this is almost entirely the partner-promo effect — see below.)*

![Plan x topic](figures/05_heatmap_plan_x_topic.png)
![Revenue lost](figures/07_revenue_lost_by_plan.png)

**4. Tenure, listening hours, age and payment method barely move churn.** Oddly, actual weekly hours show no pattern while the `Low-usage` content-mix label churns at 23%, so the two columns measure different things.

![Segments overview](figures/01_churn_by_segment.png)

All charts: [`figures/`](figures/). Heatmap cells with fewer than 30 subscribers are marked "too few" instead of showing a rate.

## Phase 2 — Churn model and risk segments

`train_churn_model.py` compares three models with 5-fold stratified cross-validation:

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic regression (raw features) | 0.823 | 0.418 | 0.0685 |
| **Logistic regression + interaction flags** (chosen) | **0.834** | **0.465** | **0.0561** |
| Gradient boosting (raw features) | 0.837 | 0.464 | 0.0586 |

Phase 1 showed churn concentrates where two conditions meet, which a linear model can't learn on its own. Engineered flags (`features.py`) fix that:

- `billing_repeat`: 2+ support tickets in 90 days and the last one was about Billing
- `promo_premium_low_usage`: signed up via Partner promo, on Premium, Low-usage content mix
- `billing_and_promo`: has both of the above. Without it the model adds the two risks together and predicts ~98% churn for the 28 subscribers who have both, when they actually churn at ~54%. With it, their prediction drops to ~66–71%.

With them, plain logistic regression matches gradient boosting while staying fully explainable through odds ratios. The top 10% of subscribers by predicted risk churn at ~60% (6.2× lift) and contain ~62% of all churners.

![Model comparison](figures/08_model_comparison.png)
![Odds ratios](figures/09_churn_drivers_odds_ratios.png)
![Calibration](figures/10_calibration.png)

### Risk segments

A shallow surrogate decision tree fitted to the out-of-fold churn probabilities splits the base into segments you can act on:

| Segment | Subscribers | Predicted churn | Revenue at risk / mo |
|---|---|---|---|
| S1 Partner-promo Premium, low usage | 322 | 61% | $2,543 |
| S2 Repeat billing complaints | 682 | 56% | $4,071 |
| S3 Everyone else (baseline) | 7,496 | 3% | $2,546 |

![Risk segments](figures/11_risk_segments.png)

**Correction to Phase 1:** the partner-promo low-usage effect is *only* Premium. Non-Premium partner-promo low-usage subscribers churn at 2–4%. Outside segment S1, Premium churns like every other plan, so Premium's higher raw churn rate comes from who is on it rather than the plan itself.

The full segment report for Phase 3 is [`outputs/segment_risk.json`](outputs/segment_risk.json). It has each segment's rule, size, predicted vs actual churn, revenue at risk, distinguishing traits, the model's churn drivers, and caveats. It contains no subscriber IDs.

## Phase 3 — Retention recommendations with Gemini

`generate_recommendations.py` sends Gemini (`gemini-3.8-flash` by default) the Phase 2 segment report plus six key charts. Gemini is multimodal, so it reads the charts as images, not just the JSON numbers. It must answer in a fixed JSON schema (Pydantic), with the following for each segment:

- priority rank
- root-cause hypothesis
- supporting evidence
- 2–4 actions with owner, effort, timeline, assumed churn reduction, estimated savings and KPI
- an A/B test to confirm the fix works

The system prompt keeps it grounded: quote real numbers, treat drivers as hypotheses, rank by revenue at risk, don't over-invest in baseline churn, and respect the model's caveats. For example, it can't recommend changing the Premium plan, because Phase 2 showed Premium's raw churn comes from the partner-promo segment.

**Guardrails:** the script checks Gemini's answer against the model's numbers. Every segment must have a plan, savings can't exceed the revenue at risk, each estimate must equal the stated reduction × that segment's at-risk revenue, and assumed reductions must stay within 5–50%. Any violations are listed in the report.

Outputs:

- [`outputs/recommendations.md`](outputs/recommendations.md): readable retention plan
- [`outputs/recommendations.json`](outputs/recommendations.json): the same plan as structured data

## Dashboard

`build_dashboard.py` collects the outputs of all three phases into one self-contained page, [`docs/index.html`](docs/index.html). It has interactive charts (hover for details), the model comparison, risk segments and Gemini's retention plan. It needs no server: open the file in a browser, or host it free with GitHub Pages.

**GitHub Pages:** in the repo, go to *Settings → Pages*. Under *Build and deployment*, set *Source* to "Deploy from a branch", then choose `main` and `/docs`. The dashboard will be at `https://<your-username>.github.io/sonicwave-churn-pipeline/`.

## Run it

```bash
pip install -r requirements.txt
python visualize_churn.py    # Phase 1: figures 01-07
python train_churn_model.py  # Phase 2: model, outputs/segment_risk.json, figures 08-11

# Phase 3: copy .env.example to .env and add your Gemini API key, then
python generate_recommendations.py            # writes outputs/recommendations.md + .json
python generate_recommendations.py --dry-run  # preview the prompt without calling the API
python build_dashboard.py                     # writes docs/index.html
```
