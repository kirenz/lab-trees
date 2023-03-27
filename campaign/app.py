"""
Streamlit App Marketing Prediction
"""

# ---------------------------------------------
# Python Setup
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier
# ---------------------------------------------

# Load the saved model
model_filename = 'gradientboosted_model'
model = load(model_filename)

# Define prediction function


def predict(age, city, income, membership_days, campaign_engagement):
    data = {'age': [age],
            'city': [city],
            'income': [income],
            'membership_days': [membership_days],
            'campaign_engagement': [campaign_engagement]}

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['city'])

    # Ensure both city columns are present
    for col in ['city_Berlin', 'city_Stuttgart']:
        if col not in df.columns:
            df[col] = 0

    # Reorder the columns to match the training data
    column_order = ['age', 'income', 'membership_days',
                    'campaign_engagement', 'city_Berlin', 'city_Stuttgart']
    df = df[column_order]

    prediction = model.predict(df)
    return prediction[0]


# -----------------------------------------------------------------------------
# Streamlit app

# Title
st.title('Online Marketing XGBoost Model')

# Slider for input data
age = st.slider('Age', min_value=18, max_value=64, value=30)
city = st.selectbox('City', ['Berlin', 'Stuttgart'])
income = st.slider('Income', min_value=20000, max_value=150000, value=50000)
membership_days = st.slider(
    'Membership days', min_value=100, max_value=3000, value=500)
campaign_engagement = st.slider(
    'Campaign engagement', min_value=0, max_value=10, value=5)

# Model prediction
if st.button('Predict'):
    result = predict(age, city, income, membership_days, campaign_engagement)
    st.write(f'The prediction is: {result}')
