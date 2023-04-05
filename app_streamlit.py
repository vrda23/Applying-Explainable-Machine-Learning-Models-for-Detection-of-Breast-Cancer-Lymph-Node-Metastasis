import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the saved model
with open('random_forest_classifier.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

# Function to get user inputs
def get_user_input():
    with st.sidebar.form("input_form"):
        age = st.sidebar.slider("Age", 18, 100, 50)
        tumour_size = st.sidebar.number_input("Tumour size (cm)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        tumour_grade = st.sidebar.selectbox("Tumour grade", [1, 2, 3])
        er = st.sidebar.slider("ER status (%)", 0, 100, 50)
        pr = st.sidebar.slider("PR status (%)", 0, 100, 50)
        her2 = st.sidebar.selectbox("HER-2 status", [0, 1, 2, 3])
        ki67 = st.sidebar.slider("Ki67 (%)", 0, 100, 20)

        histological_type = st.sidebar.selectbox(
            "Histological type",
            [
                "Ca with medullary features",
                "Lobular invasive",
                "Metaplastic",
                "Mucinose invasive",
                "NOS invasive",
                "Other rare types",
            ],
        )
        submit_button = st.form_submit_button("Submit")

    # Encode categorical features
    histological_dummies = pd.get_dummies(pd.Series(histological_type)).reindex(
        columns=[
            'Ca with medullary features',
            'Lobular invasive',
            'Metaplastic',
            'Mucinose invasive',
            'NOS invasive',
            'Other rare types',
        ],
        fill_value=0,
    )

    user_input = pd.DataFrame(
        {
            "Age": [age],
            "Tumour size": [tumour_size],
            "Tumour grade": [tumour_grade],
            "ER": [1 if er == "Positive" else 0],
            "PR": [1 if pr == "Positive" else 0],
            "HER-2": [1 if her2 == "Positive" else 0],
            "Ki67": [ki67],
        }
    ).join(histological_dummies)

    return user_input

# Streamlit app
st.set_page_config(page_title="Lymph Node Status Classifier", page_icon=":hospital:", layout="wide", initial_sidebar_state="collapsed")
st.title("Lymph Node Status Classifier \n (for patients eligible for neoadjuvant treatment)")
st.write("Random forest model optimized and validated as described in https://pubmed.ncbi.nlm.nih.gov/36765592/")
st.write("NOTE: App is for experimental and educational purposes ONLY!")
st.write("Enter patient details in the sidebar to predict lymph node status.")

user_input = get_user_input()

prediction = rf_classifier.predict(user_input)
prediction_proba = rf_classifier.predict_proba(user_input)

st.subheader("Prediction")
st.write("Lymph node status:", "Positive" if prediction[0] == 1 else "Negative")

st.subheader("Prediction Probability")
st.write(
    f"Probability of positive lymph node status: {prediction_proba[0][1] * 100:.2f}%"
)
