import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_decision_tree_model.pkl")

# Page title
st.title("Loan Approval Prediction")

# Customer inputs
income = st.number_input("Annual Income in Lakhs")

credit_score = st.number_input("Credit Score")

existing_loan = st.selectbox("Existing Loan", ["No", "Yes"])

loan_amount = st.number_input("Loan Amount in Lakhs")

dti = st.number_input("Debt to Income Ratio")

employment = st.number_input("Employment Years")

age = st.number_input("Age")

purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Education", "Home", "Personal", "Vehicle"]
)

# Convert existing loan into number
if existing_loan == "Yes":
    existing_loan = 1
else:
    existing_loan = 0

# Convert loan purpose into numbers
business = 0
education = 0
home = 0
personal = 0
vehicle = 0

if purpose == "Business":
    business = 1
elif purpose == "Education":
    education = 1
elif purpose == "Home":
    home = 1
elif purpose == "Personal":
    personal = 1
else:
    vehicle = 1

# Submit button
if st.button("Check Loan Approval"):

    # Prepare customer data
    data = pd.DataFrame({
        "Income_Lakhs": [income],
        "Credit_Score": [credit_score],
        "Existing_Loan": [existing_loan],
        "Loan_Amount_Lakhs": [loan_amount],
        "Debt_to_Income_Pct": [dti],
        "Employment_Years": [employment],
        "Age": [age],
        "Loan_Purpose_Business": [business],
        "Loan_Purpose_Education": [education],
        "Loan_Purpose_Home": [home],
        "Loan_Purpose_Personal": [personal],
        "Loan_Purpose_Vehicle": [vehicle]
    })

    # Predict loan approval
    prediction = model.predict(data)

    # Display result
    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")

    st.caption("This is a model prediction, not a real bank decision.")