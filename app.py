import numpy as np
import joblib
import streamlit as st
# Load the trained model
model = joblib.load('diabetes_model.pkl')

st.title("Diabetes Prediction System")
st.markdown("Enter patient details to predict the likelihood of diabetes.")

# Get user input for the features
age=st.number_input("Enter Age:", min_value=1.0, max_value=120.0)
bmi=st.number_input("Enter BMI:")
glucose=st.number_input("Enter Glucose Level:")
hypertension_options=st.selectbox("Hypertension:",["No","Yes"])
heart_disease_options=st.selectbox("Heart Disease:",["No","Yes"])
hba1c=st.number_input("Enter HbA1c Level:")
gender=st.selectbox("Enter Gender:",["Male","Other","Female"])
smoking=st.selectbox("Smoking:",["Never","Former","Current","Ever","Not Current"])

# Encode categorical variables
hypertension=1 if hypertension_options == "Yes" else 0
heart_disease=1 if heart_disease_options == "Yes" else 0

gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0 
smoking_current = 1 if smoking == "Current" else 0
smoking_ever = 1 if smoking == "Ever" else 0 
smoking_former = 1 if smoking == "Former" else 0
smoking_never = 1 if smoking == "Never" else 0
smoking_not_current = 1 if smoking == "Not Current" else 0

# Prepare input data for prediction
if st.button("Predict"):
    input_data = np.array([[age, bmi, glucose, hypertension, heart_disease, hba1c, gender_male, gender_other, smoking_current, smoking_ever, smoking_former, smoking_never, smoking_not_current]])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    labels = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
    st.subheader("Prediction Result")
    st.success(f"Diagnosis: {labels[prediction]}")
    st.subheader("Probability")
    st.write(f"No Diabetes: {probabilities[0]*100:.2f}%")
    st.write(f"Pre-Diabetes: {probabilities[1]*100:.2f}%")
    st.write(f"Diabetes: {probabilities[2]*100:.2f}%")  



st.subheader("Credit")
st.write("Somnath Banerjee")
st.write("Piyush Chakraborty")
st.write("Neelotpal Chakraborty")
st.write("Sakil Mondal")



 
