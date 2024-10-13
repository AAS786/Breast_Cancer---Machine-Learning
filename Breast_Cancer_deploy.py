import streamlit as st
import pickle as pk
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Loading the saved model
model = pk.load(open('Breast_deploy.sav', 'rb'))

# Custom CSS with Background Image and Enhanced Input Box Design
st.markdown(f"""
    <style>
    body {{
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
    }}
    .main {{
        background-image: url('https://i.postimg.cc/VLktPrb5/free-vector.jpg'); /* Replace this with your image link */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    h1 {{
        color: blue;
        text-align: center;
        text-decoration: underline;
        margin-bottom: 30px;
    }}
    h3 {{
        text-align: center;
    }}
    .stTextInput > div {{
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }}
    .stTextInput input {{
        padding: 10px;
        font-size: 16px;
        border: 2px solid #ccc;
        border-radius: 5px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s, box-shadow 0.3s;
    }}
    .stTextInput input:focus {{
        border-color: #4CAF50;
        box-shadow: 0px 4px 6px rgba(76, 175, 80, 0.5);
    }}
    .stButton {{
        display: flex;
        justify-content: center; 
        margin-top: 20px;
    }}
    .stButton button {{
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 5px;
        transition-duration: 0.4s;
    }}
    .stButton button:hover {{
        background-color: white; 
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }}
    </style>
    """, unsafe_allow_html=True)

# Title of the application
st.markdown("<h1>Breast Cancer Detection using <br>Machine Learning</h1>", unsafe_allow_html=True)

# Input layout with columns for data entry
col1, col2, col3, col4 = st.columns(4)
with col1:
    radius_mean = st.text_input('**Radius Mean**', placeholder='Enter Radius Mean')
    smoothness_mean = st.text_input('**Smoothness Mean**', placeholder='Enter Smoothness Mean')
    symmetry_mean = st.text_input('**Symmetry Mean**', placeholder='Enter Symmetry Mean')
    perimeter_se = st.text_input('**Perimeter Se**', placeholder='Enter Perimeter Se')
    concavity_se = st.text_input('**Concavity SE**', placeholder='Enter Concavity SE')
    radius_worst = st.text_input('**Radius Worst**', placeholder='Enter Radius Worst')
    smoothness_worst = st.text_input('**Smoothness Worst**', placeholder='Enter Smoothness Worst ')
with col2:
    texture_mean = st.text_input('**Texture Mean**', placeholder='Enter Texture Mean')
    compactness_mean = st.text_input('**Compactness Mean**', placeholder='Enter Compactness Mean')
    fractal_dimension_mean = st.text_input('**Fractal Dimension Mean**', placeholder='Enter Fractal Dimension Mean')
    area_se = st.text_input('**Area Se**', placeholder='Enter Area Se')
    concave_points_se = st.text_input('**Concave Points Se**', placeholder='Enter Concave Points Se')
    texture_worst = st.text_input('**Texture Worst**', placeholder='Enter Texture Worst')
    compactness_worst = st.text_input('**Compactness Worst**', placeholder='Enter Compactness Worst')
    symmetry_worst = st.text_input('**Symmetry Worst**', placeholder='Enter Symmetry Worst')
with col3:
    perimeter_mean = st.text_input('**Perimeter Mean**', placeholder='Enter Perimeter Mean')
    concavity_mean = st.text_input('**Concavity Mean**', placeholder='Enter Concavity  Mean')
    radius_se = st.text_input('**Radius SE**', placeholder='Enter Radius SE')
    smoothness_se = st.text_input('**Smoothness Se**', placeholder='Enter Smoothness Se')
    symmetry_se = st.text_input('**Symmetry Se**', placeholder='Enter Symmetry Se')
    perimeter_worst = st.text_input('**Perimeter Worst**', placeholder='Enter Perimeter Worst')
    concavity_worst = st.text_input('**Concavity Worst**', placeholder='Enter Concavity Worst')
    fractal_dimension_worst = st.text_input('**Fractal Dimension Worst**', placeholder='Enter Fractal Dimension Worst')
with col4:
    area_mean = st.text_input('**Area Mean**', placeholder='Enter Area Mean')
    concave_points_mean = st.text_input('**Concave Points Mean**', placeholder='Enter Concave Points Mean')
    texture_se = st.text_input('**Texture Se**', placeholder='Enter Texture Se')
    compactness_se = st.text_input('**Compactness Se**', placeholder='Enter Compactness Se')
    fractal_dimension_se = st.text_input('**Fractal Dimension Se**', placeholder='Enter Fractal Dimension Se')
    area_worst = st.text_input('**Area Worst**', placeholder='Enter Area Worst')
    concave_points_worst = st.text_input('**Concave Points Worst**', placeholder='Enter Concave Points Worst')

# Prediction Button
if st.button('**Breast Cancer Detection Result**'):
    try:
        # Collecting input data into a list
        input_data = [
            float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), 
            float(smoothness_mean), float(compactness_mean), float(concavity_mean), float(concave_points_mean),
            float(symmetry_mean), float(fractal_dimension_mean), float(radius_se), float(texture_se),
            float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se), 
            float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se), 
            float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), 
            float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst), 
            float(symmetry_worst), float(fractal_dimension_worst)
        ]

        # Reshaping input data for model prediction
        reshaped_input = np.array(input_data).reshape(1, -1)
        gen_prediction = model.predict(reshaped_input)

        # Prediction result
        if gen_prediction[0] == 0:
            Predict_diagnosis = 'The person has a High level of Cancer Symptoms'
            result_color = "red"
        else:
            Predict_diagnosis = 'The person has a Very Low level of Cancer Symptoms'
            result_color = "green"

        # Display prediction result
        st.markdown(f"<h3 style='color: {result_color};'>{Predict_diagnosis}</h3>", unsafe_allow_html=True)
        
        # Trigger balloon animation
        st.balloons()
        
    except ValueError:
        st.error("Please enter valid numerical inputs in all fields.")
