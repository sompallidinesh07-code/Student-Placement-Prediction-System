import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Load model and scaler
model = joblib.load("placement_model.pkl")

scaler = joblib.load("scaler.pkl")

# Title
st.title("🎓 Student Placement Prediction System")

st.write(
    "Predict placement chances using Machine Learning"
)

# Sidebar
st.sidebar.header("Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

internships = st.sidebar.slider("Internships", 0, 10, 1)

projects = st.sidebar.slider("Projects", 0, 10, 2)

coding = st.sidebar.slider("Coding Skills", 0, 100, 70)

communication = st.sidebar.slider(
    "Communication Skills",
    0,
    100,
    70
)

aptitude = st.sidebar.slider(
    "Aptitude Score",
    0,
    100,
    70
)

dsa = st.sidebar.slider(
    "DSA Score",
    0,
    100,
    70
)

certifications = st.sidebar.slider(
    "Certifications",
    0,
    10,
    2
)

hackathons = st.sidebar.slider(
    "Hackathons",
    0,
    10,
    1
)

backlogs = st.sidebar.slider(
    "Backlogs",
    0,
    20,
    0
)

ml_knowledge = st.sidebar.slider(
    "ML Knowledge",
    0,
    100,
    70
)

system_design = st.sidebar.slider(
    "System Design",
    0,
    100,
    65
)

open_source = st.sidebar.slider(
    "Open Source Contributions",
    0,
    100,
    50
)

extracurriculars = st.sidebar.slider(
    "Extracurricular Activities",
    0,
    10,
    2
)

# Predict Button
if st.sidebar.button("Predict Placement"):

    data = np.array([[
        cgpa,
        internships,
        projects,
        coding,
        communication,
        aptitude,
        dsa,
        certifications,
        hackathons,
        backlogs,
        ml_knowledge,
        system_design,
        open_source,
        extracurriculars
    ]])

    # Scale
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    # Probability
    probability = model.predict_proba(
        data_scaled
    )

    placement_prob = probability[0][1] * 100

    # Result
    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(
            "✅ Likely to be Placed"
        )

    else:

        st.error(
            "❌ Less Placement Chance"
        )

    # Probability
    st.metric(
        "Placement Probability",
        f"{placement_prob:.2f}%"
    )

    # Progress Bar
    st.progress(int(placement_prob))

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = placement_prob,
        title = {
            'text': "Placement Probability"
        },
        gauge = {
            'axis': {'range': [0,100]},
            'bar': {'color': "green"}
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )