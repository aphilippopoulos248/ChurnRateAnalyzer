"""
Shared data loading and feature engineering for the SonicWave churn pipeline.

Phase 2 (model training) and Phase 3 (recommendations) both import from here,
so a subscriber is always described by the same features.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sonicwave_subscribers.csv"

TARGET = "churned"

CATEGORICAL = [
    "age_group",
    "plan_type",
    "content_mix",
    "payment_method",
    "last_ticket_topic",
    "signup_channel",
]
NUMERIC = ["tenure_months", "avg_weekly_hours", "support_tickets_90d"]

# Interaction flags found in Phase 1 + 2 EDA. A linear model can't discover
# "A AND B" on its own, so we hand it these two explicitly.
ENGINEERED = {
    "billing_repeat": "2+ support tickets in 90 days and the last one was about Billing",
    "promo_premium_low_usage": "Signed up via Partner promo, on Premium, Low-usage content mix",
}

# monthly_spend is excluded: it is a deterministic function of plan_type.
# subscriber_id is an identifier, not a feature.
RAW_FEATURES = CATEGORICAL + NUMERIC
ALL_FEATURES = CATEGORICAL + NUMERIC + list(ENGINEERED)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["billing_repeat"] = (
        (df["last_ticket_topic"] == "Billing") & (df["support_tickets_90d"] >= 2)
    ).astype(int)
    df["promo_premium_low_usage"] = (
        (df["signup_channel"] == "Partner promo")
        & (df["content_mix"] == "Low-usage")
        & (df["plan_type"] == "Premium")
    ).astype(int)
    return df


def load_data(path: Path = DATA) -> pd.DataFrame:
    return add_engineered_features(pd.read_csv(path))
