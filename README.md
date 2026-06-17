# Risk Scoring & Prioritization System

## Overview

This project was developed for the AI/ML Intelligence Hackathon under the **Risk Scoring & Prioritization** track.

The objective is to help financial crime analysts identify high-risk accounts by assigning an account-level risk score based on transaction patterns, KYC attributes, and behavioral signals. Rather than treating all accounts equally, the system prioritizes accounts that are most likely to be involved in money laundering activities.

The final solution combines machine learning, account-level aggregation, explainable AI, and risk ranking to create an analyst-friendly investigation workflow.

---

## Problem Statement

Financial institutions process thousands of transactions daily, making manual investigation of every account impractical. Analysts require a reliable risk scoring mechanism that can:

* Estimate the likelihood of suspicious activity
* Rank accounts by investigation priority
* Explain the reasoning behind each risk assessment

---

## Dataset

### Files Provided

* `ml_features.csv`

  * Transaction-level features
  * KYC attributes
  * Behavioral indicators
  * Target label: `is_suspicious_tx`

### Key Features

* Transaction Amount
* Cross-Border Activity
* Account Age
* PEP Indicators
* Sanctions Indicators
* Transaction Velocity
* Country Risk Scores
* Transaction Mode

---

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data Cleaning & Validation
3. Feature Engineering
4. Transaction-Level Risk Prediction
5. Account-Level Risk Aggregation
6. Risk Score Generation
7. Account Ranking
8. Explainability using SHAP
9. Performance Evaluation
10. Dashboard Development

---

## Models Evaluated

* Random Forest
* LightGBM

The final model was selected based on performance across highly imbalanced data, with a focus on precision and ranking quality.

---

## Account-Level Risk Scoring

The model predicts transaction-level risk probabilities.

These probabilities are aggregated at the account level to generate:

* Account Risk Score
* Investigation Priority Rank
* Supporting Risk Factors

Example:

| Account | Risk Score | Rank |
| ------- | ---------- | ---- |
| ACC001  | 96.2       | 1    |
| ACC017  | 92.8       | 2    |
| ACC045  | 91.1       | 3    |

---

## Explainability

To ensure transparency and analyst trust, SHAP explanations are used to identify the factors contributing to each risk score.

Example output:

* High transaction velocity
* Cross-border activity
* PEP association
* Large transaction amounts

---

## Evaluation Metrics

Because suspicious transactions represent a small fraction of the dataset, traditional accuracy is not sufficient.

Metrics used:

* PR-AUC
* ROC-AUC
* Recall
* F1 Score
* Precision@K
* Ranking Performance

---

## Repository Structure

```text
aml.ipynb
dashboard
app.py
Technical documentation
slides
```

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

##
