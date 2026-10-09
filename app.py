import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to estimate their risk of leaving.")

yes_no = ["No", "Yes"]
addon = ["No", "Yes", "No internet service"]

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", yes_no)
    partner = st.selectbox("Has partner", yes_no)
    dependents = st.selectbox("Has dependents", yes_no)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", yes_no)
    multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", addon)
    backup = st.selectbox("Online backup", addon)
with c2:
    protection = st.selectbox("Device protection", addon)
    tech = st.selectbox("Tech support", addon)
    tv = st.selectbox("Streaming TV", addon)
    movies = st.selectbox("Streaming movies", addon)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", yes_no)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)

total = tenure * monthly   # approximation of total charges
st.caption(f"Estimated total charges: {total:,.2f}")

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multi,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    prob = model.predict_proba(row)[0][1]
    st.metric("Churn probability", f"{prob:.0%}")
    if prob >= 0.5:
        st.error("High risk: consider a retention offer or a contract upgrade.")
    else:
        st.success("Lower risk.")