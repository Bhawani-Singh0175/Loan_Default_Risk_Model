# Loan Default Risk Model & FICO Score Quantization

## Overview

This project focuses on credit risk modeling and borrower risk analysis using Python and machine learning techniques. The project was developed as part of the JPMorgan Chase & Co. Quantitative Research Virtual Experience Program on Forage.

The project consists of two major components:

1. Loan Default Prediction & Expected Loss Estimation  
2. FICO Score Quantization & Rating Mapping  

The goal is to estimate borrower default risk, calculate expected financial losses, and generate automated credit rating buckets from FICO scores.

---

# Features

## Loan Default Prediction Model

- Built a Logistic Regression model to estimate Probability of Default (PD)
- Estimated Expected Loss for borrowers
- Performed feature engineering and risk analysis
- Used borrower financial and credit-related attributes

### Features Used

- Credit lines outstanding
- Loan amount outstanding
- Total debt outstanding
- Income
- Years employed
- FICO score

### Expected Loss Formula

```python
Expected Loss = PD × Loan Amount × (1 - Recovery Rate)
```

Assumed Recovery Rate = 10%

---

## FICO Score Quantization & Rating Mapping

- Built an automated FICO rating system
- Applied Quantization techniques
- Used Log-Likelihood Optimization
- Implemented Dynamic Programming for optimal bucket boundaries

### Rating Rule

- Lower rating = Better credit quality
- Higher rating = Higher default risk

Example:

| FICO Score | Rating |
|------------|---------|
| 800        | 1       |
| 750        | 2       |
| 680        | 4       |
| 620        | 6       |
| 550        | 9       |

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Dynamic Programming
- Statistical Modeling
- Financial Risk Analysis

---

# Project Structure

```bash
Loan_Default_Risk_Model/
│
├── loan_default_model.py
├── fico_score_rating_map.py
├── Task 3 and 4_Loan_Data.csv
├── README.md
├── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Bhawani-Singh0175/Loan_Default_Risk_Model.git
```

Move into the project folder:

```bash
cd Loan_Default_Risk_Model
```

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

# Running the Project

## Run Loan Default Prediction Model

```bash
python loan_default_model.py
```

### Output

- Model Accuracy
- Probability of Default
- Expected Loss

---

## Run FICO Score Rating Map

```bash
python fico_score_rating_map.py
```

### Output

- Optimized FICO rating buckets
- Default rates per bucket
- Sample borrower ratings

---

# Example Output

## Loan Default Prediction

```python
Model Accuracy: 0.9850

Prediction Result:
{
 'Probability of Default': 0.4311,
 'Expected Loss': 1940.05
}
```

---

## FICO Rating Example

```python
FICO Score: 750 | Rating: 2 | Estimated Default Rate: 0.0261
```

---

# Key Concepts Applied

- Credit Risk Modeling
- Probability of Default (PD)
- Expected Loss Estimation
- Statistical Classification
- Quantization
- Likelihood Functions
- Dynamic Programming
- Financial Data Analysis

---

# Learning Outcomes

This project helped strengthen understanding of:

- Machine Learning for Finance
- Credit Risk Analytics
- Borrower Risk Assessment
- Statistical Modeling
- Optimization Techniques

---

