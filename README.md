# An Explainable Early-Warning System for Predicting At-Risk Graduate Trainees in Banking Cybersecurity Onboarding Using XGBoost and SHAP

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

This repository contains the full implementation of an explainable machine learning early-warning system (EWS) designed to predict graduate management trainees at risk of cybersecurity training non-compliance during banking onboarding programmes in Ghana.

The system combines:
- **XGBoost** — a gradient boosted decision tree classifier for at-risk prediction
- **SHAP (SHapley Additive exPlanations)** — for global and local model interpretability

This work is submitted in partial fulfilment of the requirements for the **MSc Cybersecurity** degree at the **Kwame Nkrumah University of Science and Technology (KNUST), Ghana**, 2025–2026.

---

## Research Details

| Item | Details |
|------|---------|
| **Author** | Frank Acquah |
| **Institution** | KNUST, Ghana |
| **Programme** | MSc Cybersecurity |
| **Supervisor** | [Supervisor Name] |
| **Target Journal 1** | Computers & Education: AI (Elsevier) |
| **Target Journal 2** | Journal of Educational Data Mining (JEDM) |
| **Status** | In Progress |

---

## Research Objectives

1. To develop an XGBoost-based predictive model that identifies graduate management trainees at risk of cybersecurity training non-compliance during banking onboarding.
2. To apply SHAP to generate globally and locally interpretable explanations of the model's risk predictions for use by training coordinators.
3. To evaluate XGBoost against baseline classifiers (Logistic Regression, Random Forest, Decision Tree) and validate SHAP explanation utility through domain expert review.

---

## Repository Structure

```
banking-cybersecurity-ews-xgboost-shap/
│
├── data/
│   ├── raw/                  # Raw collected data (anonymised — not uploaded)
│   └── processed/            # Cleaned and preprocessed dataset
│
├── src/
│   ├── preprocessing.py      # Data cleaning, normalisation, SMOTE
│   ├── train_baselines.py    # Logistic Regression, Random Forest, Decision Tree
│   ├── train_xgboost.py      # XGBoost model training and hyperparameter tuning
│   ├── shap_analysis.py      # Global and local SHAP explanation generation
│   └── evaluate.py           # Metrics, statistical significance, visualisations
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb        # Preprocessing pipeline
│   ├── 03_Model_Training.ipynb       # Model training and CV
│   ├── 04_SHAP_Explanations.ipynb    # SHAP analysis and plots
│   └── 05_Results_Visualisation.ipynb
│
├── outputs/
│   ├── figures/              # ROC curves, SHAP plots, confusion matrices
│   └── models/               # Saved model files (.json)
│
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## Methodology

### Dataset
- **Type:** Primary — survey and knowledge assessment instrument administered to graduate management trainees in Ghanaian banking institutions
- **Target sample size:** n = 300–400 participants
- **Features:** Demographic background, prior IT knowledge, module quiz scores, training engagement rating, self-efficacy score, compliance intention scale, perceived training relevance
- **Label:** At-Risk (binary: 1 = at-risk of non-compliance, 0 = on-track)
- **Ethics:** Approved by KNUST Committee on Human Research, Publications and Ethics (CHRPE)

### Models
| Model | Role |
|-------|------|
| XGBoost + SHAP | Primary model |
| Logistic Regression (L2) | Baseline 1 |
| Random Forest | Baseline 2 |
| Decision Tree (CART) | Baseline 3 |

### Evaluation Metrics
- AUC-ROC (primary)
- AUC-PR (Precision-Recall)
- F1-Score (Macro and Weighted)
- Precision and Recall
- McNemar's test with 95% Confidence Intervals
- SHAP global (beeswarm) and local (waterfall/force) explanations

---

## Installation

```bash
# Clone the repository
git clone https://github.com/frankacquah-blip/banking-cybersecurity-ews-xgboost-shap.git
cd banking-cybersecurity-ews-xgboost-shap

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

```
python==3.10
xgboost==2.0.3
shap==0.44.1
scikit-learn==1.3.2
imbalanced-learn==0.11.0
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4
jupyter==1.0.0
```

---

## Reproducibility

All experiments use `random_state=42` throughout. To reproduce results:

```bash
python src/preprocessing.py
python src/train_baselines.py
python src/train_xgboost.py
python src/shap_analysis.py
python src/evaluate.py
```

Full experiment logs and outputs will be saved to `/outputs/`.

---

## Ethical Statement

This study was approved by the KNUST Committee on Human Research, Publications and Ethics (CHRPE), Reference No: *(to be inserted upon approval)*. All participants provided written informed consent. Data were fully anonymised prior to analysis. No personally identifiable information is stored in this repository.

---

## Citation

If you use this work, please cite:

```
Acquah, F. (2026). An Explainable Early-Warning System for Predicting At-Risk Graduate
Trainees in Banking Cybersecurity Onboarding Using XGBoost and SHAP. MSc Thesis,
KNUST, Ghana.
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Frank Acquah**
MSc Cybersecurity, KNUST Ghana
GitHub: [@frankacquah-blip](https://github.com/frankacquah-blip)
