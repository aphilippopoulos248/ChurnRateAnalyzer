# SonicWave Churn Analysis Pipeline

An end-to-end churn analysis of 8,500 subscribers of **SonicWave**, a fictional music/podcast streaming service.

| Phase | Status | What it does |
|---|---|---|
| 1. Data visualization | ✅ Done | Find which subscriber attributes drive churn |
| 2. Churn model | Planned | Custom probability model predicting each subscriber's churn risk |
| 3. Recommendations | Planned | Gemini/Gemma API turns model predictions into business recommendations |

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

**3. Premium churns at 13% vs 7–8% for other plans** and accounts for most of the ~$9.1k in monthly revenue lost to churn.

![Plan x topic](figures/05_heatmap_plan_x_topic.png)
![Revenue lost](figures/07_revenue_lost_by_plan.png)

**4. Tenure, listening hours, age and payment method barely move churn.** Oddly, actual weekly hours show no pattern while the `Low-usage` content-mix label churns at 23%, so the two columns measure different things.

![Segments overview](figures/01_churn_by_segment.png)

All charts: [`figures/`](figures/). Heatmap cells with fewer than 30 subscribers are marked "too few" instead of showing a rate.

## Run it

```bash
pip install -r requirements.txt
python visualize_churn.py   # writes figures/*.png
```
