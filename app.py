# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('diabetes_model.sav', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit Interface
st.title("Diabetes Detection System")
st.write("Provide the following details to predict diabetes.")

# Input Fields for User Data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, max_value=2.5, step=0.01, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)

# Collect input into a feature list
features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# Button for Prediction
if st.button("Predict Diabetes"):
    # Reshape the input for the model
    prediction = model.predict([features])
    # Display the result
    if prediction[0] == 1:
        st.error("The model predicts that you are likely to have diabetes.")
    else:
        st.success("The model predicts that you are unlikely to have diabetes.")

# Optional: Display dataset preview
if st.checkbox("Show Dataset"):
    data = pd.read_csv('diabetes.csv')
    st.write(data.head())
