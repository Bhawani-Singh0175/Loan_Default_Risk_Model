Loan Default Risk Model & FICO Score Quantization
Overview

This project focuses on credit risk modeling and borrower risk analysis using Python and machine learning techniques. The project was developed as part of the JPMorgan Chase & Co. Quantitative Research Virtual Experience Program on Forage.

The project consists of two major components:

Loan Default Prediction & Expected Loss Estimation
FICO Score Quantization & Rating Mapping

The goal is to estimate borrower default risk, calculate expected financial losses, and generate automated credit rating buckets from FICO scores.

Project Components
1. Loan Default Prediction Model

A Logistic Regression model was built to estimate the Probability of Default (PD) for borrowers using financial and credit-related attributes.

Features Used
Credit lines outstanding
Loan amount outstanding
Total debt outstanding
Income
Years employed
FICO score
Model Outputs
Probability of Default (PD)
Expected Loss estimation

Expected Loss Formula:

Expected Loss=PD×LoanAmount×(1−RecoveryRate)

Assumed Recovery Rate:

RecoveryRate=10%

2. FICO Score Quantization & Rating Mapping

This module creates an automated rating system by converting continuous FICO scores into discrete credit risk buckets.

Techniques Used
Quantization
Log-Likelihood Optimization
Dynamic Programming
Objective

Generate optimal FICO score boundaries that best summarize borrower default behavior.

Rating Rule
Lower rating number = better credit quality
Higher rating number = higher credit risk

Example:

FICO 800 → Rating 1
FICO 720 → Rating 2
FICO 650 → Rating 5
FICO 550 → Rating 9
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Logistic Regression
Dynamic Programming
Statistical Modeling
Project Structure
Loan_Default_Risk_Model/
│
├── loan_default_model.py
├── fico_score_rating_map.py
├── Task 3 and 4_Loan_Data.csv
├── README.md
├── .gitignore
Installation

Clone the repository:

git clone https://github.com/Bhawani-Singh0175/Loan_Default_Risk_Model.git

Move into the project folder:

cd Loan_Default_Risk_Model

Install required libraries:

pip install pandas numpy scikit-learn matplotlib
Running the Project
Run Loan Default Prediction Model
python loan_default_model.py

This outputs:

Model accuracy
Probability of Default
Expected Loss
Run FICO Score Rating Map
python fico_score_rating_map.py

This outputs:

Optimized FICO rating buckets
Default rates per bucket
Sample borrower ratings
Example Output
Loan Default Prediction
Model Accuracy: 0.9850

Prediction Result:
{
 'Probability of Default': 0.4311,
 'Expected Loss': 1940.05
}
FICO Rating Example
FICO Score: 750 | Rating: 2 | Estimated Default Rate: 0.0261
Key Concepts Applied
Credit Risk Modeling
Probability of Default (PD)
Expected Loss Estimation
Statistical Classification
Quantization
Likelihood Functions
Dynamic Programming
Financial Risk Analysis
Learning Outcomes

This project helped strengthen understanding of:

Machine Learning for Finance
Credit Risk Analytics
Borrower Risk Assessment
Statistical Modeling
Financial Data Analysis
Optimization Techniques
