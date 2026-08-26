# SDT-Weighted XGBoost for Explainable Early-Warning Prediction of At-Risk Staff in Banking Cybersecurity Onboarding: A Single-Institution Study in Ghana.

![Python](https://img.shields.io/badge/Python-3.8.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange) ![SHAP](https://img.shields.io/badge/SHAP-0.44.1-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

This repository contains the full implementation of an explainable machine learning early-warning system (EWS) that predicts Access Bank Ghana staff at risk of cybersecurity onboarding non-compliance, using a fabricated SDT-weighted asymmetric-loss objective for XGBoost.

The system combines:

- **XGBoost**, trained with a custom-engineered **SDT-Weighted Asymmetric Onboarding-Risk Loss** — a sample-weighted logistic loss combining a fixed false-negative penalty with a continuous Self-Determination Theory (SDT) vulnerability multiplier — for at-risk prediction.
- **SHAP (SHapley Additive exPlanations)** — for global and local model interpretability.

This work is submitted in partial fulfilment of the requirements for the **MSc Cybersecurity** degree at the **Kwame Nkrumah University of Science and Technology (KNUST), Ghana**, 2025–2026.

---

## Research Details

| Item                 | Details                                   |
| -------------------- | ----------------------------------------- |
| **Author**           | Frank Nana Asiedu Acquah                  |
| **Institution**      | KNUST, Ghana                              |
| **Programme**        | MSc Cybersecurity                         |
| **Supervisor**       | Doc. Eric Osei                            |
| **Status**           | In Progress                               |

---

## Research Objectives

1. To fabricate an SDT-weighted asymmetric-loss XGBoost classifier for predicting at-risk staff during banking cybersecurity onboarding: A Single-Institution Study in Ghana, using institutional field survey data.
2. To generate globally and locally interpretable SHAP explanations of the model's risk predictions for training coordinators.

---

## Repository Structure

```
banking-cybersecurity-ews-xgboost-shap/
│
├── modelling-pipeline/
│   ├── 01_preprocessing_fixed.py     
│   ├── 02_modelling_fixed.py           
│   ├── 03_smote_fixed.py              
│   ├── 04_shap_fixed.py                
│   ├── 05_evaluation_fixed.py         
│   ├── 06_multiseed_sensitivity.py 
│   
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Methodology

### Dataset

- **Type:** Primary, single-source — onboarding survey administered to Access Bank Ghana staff (all current staff, regardless of tenure), no public or externally sourced dataset fused in.
- **Collection window:** 20 June 2026 – 31 July 2026.
- **Response rate:** 457 of 550 eligible staff (83.1%); 418 respondents survived a censoring filter (unconcluded onboarding programmes excluded).
- **Features:** 28-item instrument covering demographic, behavioural, assessment-performance and Self-Determination Theory (autonomy, competence, relatedness) items; 43 one-hot-encoded predictors reduced to 37 by SHAP-guided pruning.
- **Label:** at_risk (binary: 1 = at-risk of onboarding non-compliance, 0 = not-at-risk).
- **Ethics:** submitted to the KNUST Humanities and Social Sciences Research Ethics Committee (HuSSREC); reference number and approval date pending at time of writing — not yet inserted anywhere in this repository or the associated manuscript.

### Models

| Model                              | Role          |
| ----------------------------------- | ------------- |
| XGBoost, SDT-weighted asymmetric loss (fabricated) | Primary/engineered model |
| XGBoost, default logistic loss, identical tuned hyperparameters | Baseline (single, tuning-matched) |

A single, tuning-matched XGBoost baseline is used by design, so that the objective function is the only difference between baseline and engineered model (see `02_modelling_fixed.py`); Logistic Regression / Random Forest / Decision Tree baselines described in an earlier proposal draft were not part of the executed study.

### Evaluation Metrics

- F1-score (primary), precision, recall, ROC-AUC
- McNemar's exact test + 2,000-resample paired bootstrap 95% CI on the F1 delta
- 5-fold cross-validation (training distribution) and 5-seed retraining sensitivity (test set)
- SHAP global (bar, beeswarm) and local (waterfall) explanations

---

## Installation

```
git clone https://github.com/frankacquah-blip/banking-cybersecurity-ews-xgboost-shap.git
cd banking-cybersecurity-ews-xgboost-shap
pip install -r requirements.txt
```

---

## Reproducibility

All experiments use `random_state=42` (model/CV/split) with `06_multiseed_sensitivity.py` additionally varying the seed across {42, 0, 1, 7, 123} to check retraining stability. To reproduce results, run in order from the project root:

```
python modelling-pipeline/01_preprocessing.py
python modelling-pipeline/02_modelling.py
python modelling-pipeline/03_smote.py
python modelling-pipeline/04_shap.py
python modelling-pipeline/05_evaluation.py
python modelling-pipeline/06_multiseed_sensitivity.py
```

Each script reads from and writes to `data/`, `models/` and `outputs/` relative to the project root. The real participant dataset (`data/survey_export.csv`) is not included in this repository (see Ethics); a schema-compatible synthetic dataset for smoke-testing is available under the legacy `modelling-pipeline/` notebooks.

---

## Ethical Statement

This study was submitted for approval to the KNUST Humanities and Social Sciences Research Ethics Committee (HuSSREC). As of the date of this commit, approval is still pending; the reference number and approval date will be inserted here once received — no placeholder or invented reference number appears anywhere in this repository. All participants provided written informed consent. Data are fully anonymised prior to analysis (numeric participant codes, coded institution name). No personally identifiable information is stored in this repository.

---

## Citation

If you use this work, please cite:

```
Asiedu Acquah, F. N. (2026). SDT-Weighted XGBoost for Explainable Early-Warning Prediction of
At-Risk Staff in Banking Cybersecurity Onboarding: A Single-Institution Study in Ghana.
MSc Thesis, KNUST, Ghana.
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Frank Nana Asiedu Acquah** — MSc Cybersecurity, KNUST Ghana
GitHub: [@frankacquah-blip](https://github.com/frankacquah-blip)
