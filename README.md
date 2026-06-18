<h1>**Risk Scoring & Prioritization System**</h1>
<h2>Overview</h2>

This project is developed for the AI/ML Intelligence Hackathon under the Risk Scoring & Prioritization track.

The objective is to help financial crime analysts identify high-risk accounts by assigning an account-level risk score based on transaction patterns, KYC attributes, and behavioral signals. Rather than treating all accounts equally, the system prioritizes accounts that are most likely to be involved in money laundering activities.

The final solution combines machine learning, account-level aggregation, explainable AI, and risk ranking to create an analyst-friendly investigation workflow.

<h2>Problem Statement</h2>

Financial institutions process thousands of transactions daily, making manual investigation of every account impractical. Analysts require a reliable risk scoring mechanism that can:

Estimate the likelihood of suspicious activity
Rank accounts by investigation priority
Explain the reasoning behind each risk assessment
Dataset
Files Provided
ml_features.csv
Transaction-level features
KYC attributes
Behavioral indicators
Target label: is_suspicious_tx
Key Features
Transaction Amount
Cross-Border Activity
Account Age
PEP Indicators
Sanctions Indicators
Transaction Velocity
Country Risk Scores
Transaction Mode
Project Workflow
Exploratory Data Analysis (EDA)
Data Cleaning & Validation
Feature Engineering
Transaction-Level Risk Prediction
Account-Level Risk Aggregation
Risk Score Generation
Account Ranking
Explainability using SHAP
Performance Evaluation
Dashboard Development
Models Evaluated


<h2>Model Used</h2>
Logistic Regression
Artificial Neural Network (ANN)
Random Forest
LightGBM

The final model was selected based on performance across highly imbalanced data, with a focus on precision and ranking quality.

<h2>Explainability</h2>

To ensure transparency and analyst trust, SHAP explanations are used to identify the factors contributing to each risk score.

Example output:

High transaction velocity
Cross-border activity
PEP association
Large transaction amounts
Evaluation Metrics

Because suspicious transactions represent a small fraction of the dataset, traditional accuracy is not sufficient.

<h2>Metrics used</h2>

PR-AUC<br>
ROC-AUC<br>
Recall<br>
F1 Score<br>
Precision@K<br>
Ranking Performance<br>
<br>

<h2>Repository Structure</h2>
aml.ipynb<br>
dashboard<br>
app.py<br>
Technical documentation<br>
slides<br>
<br> 
Running the Dashboard
streamlit run dashboard/app.py
