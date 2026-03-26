# 👥 Customer Churn Predictor
### ML Classification | Telecom Churn Prediction · 86% Accuracy

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-86.2%25-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Problem Statement

Telecom companies lose millions annually when customers switch providers. Retaining an existing customer costs **5× less** than acquiring a new one. Early identification of at-risk customers enables targeted, cost-effective retention.

**Goal:** Build, compare, and deploy an ML classification model that predicts customer churn with >85% accuracy using the IBM Telco dataset.

---

## 📁 Folder Structure

```
customer-churn-predictor/
│
├── data/
│   └── raw/telco_churn.csv
│
├── src/
│   └── train.py          ← Full pipeline: EDA + Preprocessing + Training + Evaluation
│
├── outputs/
│   ├── best_model.pkl
│   ├── eda_overview.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw CSV  →  EDA  →  Preprocessing  →  Feature Engineering
         →  Train 3 Models  →  Compare  →  Save Best  →  Visualise
```

**Models compared:**
| Model | Test Accuracy | ROC-AUC | CV Score (5-fold) |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.842 | 79.8% |
| **Random Forest** | **86.2%** | **0.911** | **85.7%** |
| Gradient Boosting | 84.9% | 0.901 | 84.3% |

**🏆 Winner: Random Forest — 86.2% Accuracy, AUC 0.911**

---

## 🔑 Top Churn Drivers

| Rank | Feature | Insight |
|---|---|---|
| 1 | `Contract_Month-to-month` | Month-to-month = 3× higher churn |
| 2 | `tenure` | New customers churn most |
| 3 | `MonthlyCharges` | High bills → higher churn |
| 4 | `TechSupport_No` | No support = dissatisfied |
| 5 | `InternetService_Fiber optic` | Fibre users churn more |

**💼 Business Recommendations:**
- 🎯 Target **month-to-month customers** with annual contract upgrade offers
- 📞 Proactively offer **tech support** to high-billing customers in first 12 months
- 💰 Review **pricing for fibre optic** plans — high churn suggests cost/value gap

---

## 🚀 How to Run

```bash
git clone https://github.com/mohanigupta/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt

# Run full pipeline
python src/train.py
```

### `requirements.txt`
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 👩‍💻 Author
**Mohani Gupta** | 📧 mohanigupta279@gmail.com | 🔗 [LinkedIn](https://linkedin.com/in/mohanigupta)
