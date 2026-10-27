import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Read Dataset

df = pd.read_csv("loan_approval.csv")

print(df.head())


# Step 2: Remove Duplicates

df = df.drop_duplicates()


# Step 3: Handle Missing Values

df["Income_Lakhs"] = df["Income_Lakhs"].fillna(df["Income_Lakhs"].median())

df["Credit_Score"] = df["Credit_Score"].fillna(df["Credit_Score"].median())

df["Loan_Amount_Lakhs"] = df["Loan_Amount_Lakhs"].fillna(df["Loan_Amount_Lakhs"].median())

df["Debt_to_Income_Pct"] = df["Debt_to_Income_Pct"].fillna(df["Debt_to_Income_Pct"].median())

df["Employment_Years"] = df["Employment_Years"].fillna(df["Employment_Years"].median())

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Existing_Loan"] = df["Existing_Loan"].fillna(df["Existing_Loan"].mode()[0])

df["Loan_Purpose"] = df["Loan_Purpose"].fillna(df["Loan_Purpose"].mode()[0])


# Step 4: Convert Text into Numbers

df["Existing_Loan"] = df["Existing_Loan"].map({"Yes": 1, "No": 0})

df["Approved"] = df["Approved"].map({"Yes": 1, "No": 0})


# Step 5: One-Hot Encoding

df = pd.get_dummies(df, columns=["Loan_Purpose"], dtype=int)


# Step 6: Select Input Columns

columns = [
    "Income_Lakhs",
    "Credit_Score",
    "Existing_Loan",
    "Loan_Amount_Lakhs",
    "Debt_to_Income_Pct",
    "Employment_Years",
    "Age",
    "Loan_Purpose_Business",
    "Loan_Purpose_Education",
    "Loan_Purpose_Home",
    "Loan_Purpose_Personal",
    "Loan_Purpose_Vehicle"
]

X = df[columns]

y = df["Approved"]


# Step 7: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 8: Create Decision Tree Model

model = DecisionTreeClassifier(criterion="gini", random_state=42)


# Step 9: Train Model

model.fit(X_train, y_train)


# Step 10: Predict Test Data

prediction = model.predict(X_test)


# Step 11: Evaluate Model

print("Accuracy:", accuracy_score(y_test, prediction))

print(classification_report(y_test, prediction, labels=[0, 1],
                            target_names=["Rejected", "Approved"],
                            zero_division=0))


# Step 12: Save Model

joblib.dump(model, "loan_decision_tree_model.pkl")

print("Model Saved Successfully")