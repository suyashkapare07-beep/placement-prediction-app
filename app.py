import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained model and scaler artifacts
try:
    with open("placement_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("placement_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Missing required pickle files! Please run train.py first.")

st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Student Placement Prediction System")
st.write("Input student details to predict whether they will get **PLACED** or **NOT PLACED**.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
    aptitude = st.slider("Aptitude Test Score (0-100)", 0, 100, 70)
    internships = st.number_input("Completed Internships", min_value=0, max_value=5, value=1)

with col2:
    projects = st.number_input("Completed Projects", min_value=0, max_value=15, value=2)
    certifications = st.number_input("Workshops / Certifications", min_value=0, max_value=10, value=1)

st.markdown("---")

# 2. When the user clicks the button, organize inputs and scale them
if st.button("Predict Placement Status", use_container_width=True):
    # Match the exact feature names used in your train.py line 13
    raw_features = pd.DataFrame([{
        'CGPA': cgpa,
        'Internships': internships,
        'Projects': projects,
        'Workshops/Certifications': certifications,
        'AptitudeTestScore': aptitude
    }])
    
    # Scale features using the saved scaler
    scaled_features = scaler.transform(raw_features)
    
    # Run prediction
    prediction = model.predict(scaled_features)[0]
    
    # Display the final classification result
    if prediction == 1:
        st.success("🎉 **Congratulations! The model predicts this student will be PLACED!**")
    else:
        st.error("⚠️ **The model predicts this student might NOT GET PLACED. Needs improvement!**")