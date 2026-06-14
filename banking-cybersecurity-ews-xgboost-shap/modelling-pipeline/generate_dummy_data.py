"""
generate_dummy_data.py

Generates a synthetic dataset that mirrors the 28-item survey instrument used
in the thesis:
  "An Explainable Early-Warning System for Predicting At-Risk Graduate
  Trainees in Banking Cybersecurity Onboarding Using XGBoost and SHAP:
  A Low-Resource Sub-Saharan African Validation"

Sections (28 items total):
  1. Demographics (5)
  2. Prior IT Exposure (4)
  3. Onboarding Engagement (4)
  4. Self-Determination Theory (SDT) scales (9: autonomy, competence, relatedness)
  5. Perceived Difficulty (3)
  6. Self-Reported Assessment Performance (2)
  7. Outcome label (1: at_risk)

The synthetic data includes realistic correlations (e.g., lower SDT scores,
lower engagement, and higher perceived difficulty increase the probability
of being labeled "at-risk") so that the modelling pipeline (XGBoost + SMOTE
+ SHAP) has meaningful signal to detect during smoke testing.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 350  # within the 300-400 GMT target range

rng = np.random.default_rng(RANDOM_SEED)


def generate_dummy_dataset(n=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Section 1: Demographics (5 items)
    # ------------------------------------------------------------------
    age = rng.integers(21, 30, size=n)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    highest_education = rng.choice(
        ["Bachelor's", "Master's", "Professional Cert"],
        size=n, p=[0.7, 0.25, 0.05]
    )
    years_prior_work_experience = rng.integers(0, 4, size=n)
    bank_name = rng.choice(
        ["GCB", "Ecobank", "Absa", "Stanbic", "AccessBank", "Other"],
        size=n, p=[0.2, 0.18, 0.17, 0.15, 0.15, 0.15]
    )

    # ------------------------------------------------------------------
    # Section 2: Prior IT Exposure (4 items)
    # ------------------------------------------------------------------
    it_related_degree = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    prior_cyber_certification = rng.choice(["Yes", "No"], size=n, p=[0.15, 0.85])
    prior_cyber_training = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    self_rated_it_proficiency = rng.integers(1, 6, size=n)  # 1-5 Likert

    # ------------------------------------------------------------------
    # Latent "ability/motivation" factor — drives correlations across
    # engagement, SDT, difficulty, performance, and the outcome label.
    # ------------------------------------------------------------------
    latent = rng.normal(0, 1, size=n)

    # ------------------------------------------------------------------
    # Section 3: Onboarding Engagement (4 items)
    # ------------------------------------------------------------------
    training_attendance_rate = np.clip(
        70 + 10 * latent + rng.normal(0, 8, size=n), 30, 100
    ).round(1)
    modules_completed_pct = np.clip(
        65 + 12 * latent + rng.normal(0, 10, size=n), 20, 100
    ).round(1)
    avg_time_per_module_minutes = np.clip(
        45 - 5 * latent + rng.normal(0, 10, size=n), 10, 120
    ).round(1)
    engagement_score = np.clip(
        np.round(3 + 0.8 * latent + rng.normal(0, 0.6, size=n)), 1, 5
    ).astype(int)

    # ------------------------------------------------------------------
    # Section 4: SDT scales (9 items, 1-7 Likert) — autonomy, competence,
    # relatedness (3 items each), all loading on the latent factor.
    # ------------------------------------------------------------------
    def likert_7(loading):
        vals = 4 + loading * latent + rng.normal(0, 1.0, size=n)
        return np.clip(np.round(vals), 1, 7).astype(int)

    sdt_autonomy_1 = likert_7(1.1)
    sdt_autonomy_2 = likert_7(1.0)
    sdt_autonomy_3 = likert_7(0.9)

    sdt_competence_1 = likert_7(1.3)
    sdt_competence_2 = likert_7(1.2)
    sdt_competence_3 = likert_7(1.1)

    sdt_relatedness_1 = likert_7(0.8)
    sdt_relatedness_2 = likert_7(0.7)
    sdt_relatedness_3 = likert_7(0.9)

    # ------------------------------------------------------------------
    # Section 5: Perceived Difficulty (3 items, 1-5 Likert)
    # Negatively associated with latent factor (higher latent -> lower
    # perceived difficulty / workload, higher perceived support).
    # ------------------------------------------------------------------
    perceived_content_difficulty = np.clip(
        np.round(3 - 0.7 * latent + rng.normal(0, 0.8, size=n)), 1, 5
    ).astype(int)
    perceived_workload = np.clip(
        np.round(3 - 0.5 * latent + rng.normal(0, 0.8, size=n)), 1, 5
    ).astype(int)
    perceived_supervisor_support = np.clip(
        np.round(3 + 0.6 * latent + rng.normal(0, 0.8, size=n)), 1, 5
    ).astype(int)

    # ------------------------------------------------------------------
    # Section 6: Self-Reported Assessment Performance (2 items)
    # ------------------------------------------------------------------
    self_rated_confidence = np.clip(
        np.round(3 + 0.7 * latent + rng.normal(0, 0.7, size=n)), 1, 5
    ).astype(int)
    self_reported_quiz_score = np.clip(
        70 + 8 * latent + rng.normal(0, 8, size=n), 30, 100
    ).round(1)

    # ------------------------------------------------------------------
    # Section 7: Outcome label (at_risk: 1 = at risk, 0 = not at risk)
    # Derived via a logistic function of the latent factor + key drivers,
    # producing a realistic class imbalance (~25% at-risk).
    # ------------------------------------------------------------------
    risk_logit = (
        -1.2
        - 1.4 * latent
        - 0.02 * (training_attendance_rate - 70)
        - 0.015 * (modules_completed_pct - 65)
        + 0.25 * (perceived_content_difficulty - 3)
        - 0.2 * (self_rated_confidence - 3)
        + rng.normal(0, 0.5, size=n)
    )
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    at_risk = (rng.random(n) < risk_prob).astype(int)

    df = pd.DataFrame({
        # Demographics
        "age": age,
        "gender": gender,
        "highest_education": highest_education,
        "years_prior_work_experience": years_prior_work_experience,
        "bank_name": bank_name,
        # Prior IT Exposure
        "it_related_degree": it_related_degree,
        "prior_cyber_certification": prior_cyber_certification,
        "prior_cyber_training": prior_cyber_training,
        "self_rated_it_proficiency": self_rated_it_proficiency,
        # Onboarding Engagement
        "training_attendance_rate": training_attendance_rate,
        "modules_completed_pct": modules_completed_pct,
        "avg_time_per_module_minutes": avg_time_per_module_minutes,
        "engagement_score": engagement_score,
        # SDT scales
        "sdt_autonomy_1": sdt_autonomy_1,
        "sdt_autonomy_2": sdt_autonomy_2,
        "sdt_autonomy_3": sdt_autonomy_3,
        "sdt_competence_1": sdt_competence_1,
        "sdt_competence_2": sdt_competence_2,
        "sdt_competence_3": sdt_competence_3,
        "sdt_relatedness_1": sdt_relatedness_1,
        "sdt_relatedness_2": sdt_relatedness_2,
        "sdt_relatedness_3": sdt_relatedness_3,
        # Perceived Difficulty
        "perceived_content_difficulty": perceived_content_difficulty,
        "perceived_workload": perceived_workload,
        "perceived_supervisor_support": perceived_supervisor_support,
        # Self-Reported Assessment Performance
        "self_rated_confidence": self_rated_confidence,
        "self_reported_quiz_score": self_reported_quiz_score,
        # Outcome
        "at_risk": at_risk,
    })

    # Introduce a small amount of missingness (~2%) in a few columns to
    # exercise the preprocessing notebook's missing-value handling.
    for col in ["self_reported_quiz_score", "avg_time_per_module_minutes",
                "perceived_supervisor_support"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dummy_dataset()
    out_path = "data/dummy_gmt_survey_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows x {df.shape[1]} columns -> {out_path}")
    print(df["at_risk"].value_counts(normalize=True).rename("proportion"))
    print(df.isna().sum()[df.isna().sum() > 0])
