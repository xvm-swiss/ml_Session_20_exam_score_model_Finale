import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from preprocessing import encode_sleep_quality, encode_facility_rating, encode_exam_difficulty, generate_course_diff_column, apply_1_hot_encoding
import plotly.graph_objects as go

# 1. Setup & Daten laden
model = joblib.load('artifacts/svm_pipeline.pkl') 
st.set_page_config('Exam Score Prediction', '📚', 'wide')

#st.markdown('<h1 style="font-size:50px;color:orange;font-family:times new roman;">Exam Score Prediction using ML</h1>', unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Standardgrösse für Desktop */
        .main-title {
            font-size: 50px;
            color: orange;
            font-family: "Times New Roman", Times, serif;
            font-weight: bold;
        }

        /* Anpassung für Mobile (Bildschirmbreite unter 768px) */
        @media (max-width: 768px) {
            .main-title {
                font-size: 24px !important; /* 12px ist extrem klein, 24px ist lesbar */
            }
        }
    </style>
    <h1 class="main-title">Exam Score Prediction using ML</h1>
    """, unsafe_allow_html=True)


df = pd.read_csv('data/feature eng data/feature_eng_data.csv')

# --- NEU: PLATZHALTER FÜR TACHO GANZ OBEN ---
tacho_placeholder = st.container()

st.divider()

# 2. Eingabe-Maske (Deine Boxen)
box_11, box_12, box_13, box_14 = st.columns(4)
gender = box_11.selectbox('Gender', options=df['gender'].unique())
course = box_12.selectbox('Course', options=df['course'].unique())
study_hours = box_13.slider('Study Hours', min_value=df['study_hours'].min(), max_value=df['study_hours'].max())
class_attendance = box_14.slider('class_attendance', min_value=df['class_attendance'].min(), max_value=df['class_attendance'].max())

box_21, box_22, box_23, box_24 = st.columns(4)
internet_access = box_21.selectbox('internet_access', options=df['internet_access'].unique())
sleep_hours = box_22.slider('sleep_hours', min_value=df['sleep_hours'].min(), max_value=df['sleep_hours'].max())
sleep_quality = box_23.selectbox('sleep_quality', options=df['sleep_quality'].unique())
study_method = box_24.selectbox('study_method', options=df['study_method'].unique())

box_31, box_32 = st.columns(2)
facility_rating = box_31.selectbox('facility_rating', options=df['facility_rating'].unique())
exam_difficulty = box_32.selectbox('exam_difficulty', options=df['exam_difficulty'].unique())

# 3. Daten-Verarbeitung
data = pd.DataFrame(
    [[gender, course, study_hours, class_attendance, internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty]],
    columns=['gender', 'course', 'study_hours', 'class_attendance', 'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
)

data = generate_course_diff_column(data)
data = encode_sleep_quality(data)
data = encode_facility_rating(data)
data = encode_exam_difficulty(data)
data = apply_1_hot_encoding(data, df)

# Modell-Spalten abgleichen
model_features = model.feature_names_in_
data = data[model_features]

# 4. Berechnung
prediction = model.predict(data)[0]

# --- NEU: TACHO IN DEN PLATZHALTER SCHREIBEN ---
with tacho_placeholder:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.write("### Live Analyse")
        st.metric(label="Current score", value=f"{prediction:.2f}")
        if prediction >= 75: st.success("Very good preparation!")
        elif prediction >= 50: st.info("Passed, but there's still room for improvement.")
        else: st.error("Attention: Increase your study time!")

    with col_right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Exam Score Chance", 'font': {'color': 'orange', 'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "#f8d7da"},
                    {'range': [50, 80], 'color': "#fff3cd"},
                    {'range': [80, 100], 'color': "#d4edda"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': 90}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
