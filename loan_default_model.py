import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("Loading dataset...")

# Load dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")


# Features and target
X = df.drop(columns=["default", "customer_id"])
y = df["default"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training model...")

# Train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model.fit(X_train, y_train)

print("Model training completed.")


# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")


# Expected Loss Function
def calculate_expected_loss(
    loan_amount,
    probability_of_default,
    recovery_rate=0.10
):

    expected_loss = (
        probability_of_default
        * loan_amount
        * (1 - recovery_rate)
    )

    return round(expected_loss, 2)


# Prediction Function
def predict_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):

    borrower_data = pd.DataFrame([{
    "credit_lines_outstanding": credit_lines_outstanding,
    "loan_amt_outstanding": loan_amt_outstanding,
    "total_debt_outstanding": total_debt_outstanding,
    "income": income,
    "years_employed": years_employed,
    "fico_score": fico_score
}])

    pd_value = model.predict_proba(
        borrower_data
    )[0][1]

    expected_loss = calculate_expected_loss(
        loan_amt_outstanding,
        pd_value
    )

    return {
        "Probability of Default": round(pd_value, 4),
        "Expected Loss": round(expected_loss, 2)
    }


# Example Test
result = predict_expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=7000,
    income=40000,
    years_employed=5,
    fico_score=620
)

print("\nPrediction Result:")
print(result)