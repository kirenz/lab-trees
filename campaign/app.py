"""
Streamlit App Marketing Prediction
"""

# ---------------------------------------------
# Python Setup
import streamlit as st
import pandas as pd
from joblib import load
# ---------------------------------------------

# Load the saved model
model_filename = 'gradientboosted_model.joblib'
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

    prediction = model.predict_proba(df)
    return prediction[0]


# -----------------------------------------------------------------------------
# Streamlit app

# Title
st.title('Campaign Response Prediction')

# Text
st.text('Input customer data:')

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
    class_prediction = 1 if result[1] >= 0.5 else 0

    st.header(f'The response probability is: {100 * result[1]:.2f} %')
    if class_prediction == 0:
        st.write("It is unlikely that the customer will respond.")
    else:
        st.write("The customer is likely to respond.")

    st.write(f':green[Probability of responding: {100 * result[1]:.2f} %]')
    st.markdown(f':red[Probability of not responding: {result[0]:.2f} %]')
