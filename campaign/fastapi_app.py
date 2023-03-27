"""
FastAPI with Gradient Boosted Tree Model
"""
# ---------------------------------------------
# Python Setup
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from typing import List
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier
from pydantic import BaseModel

# ---------------------------------------------
# Load the saved model
model_filename = 'gradientboosted_model'
model = load(model_filename)

# ---------------------------------------------
# Define input data


class InputData(BaseModel):
    age: int
    city: str
    income: int
    membership_days: int
    campaign_engagement: int


class InputDataList(BaseModel):
    data: List[InputData]


# ---------------------------------------------
# Start app
app = FastAPI()


# Define prediction function
def predict(age, city, income, membership_days, campaign_engagement):
    data = {'age': [age],
            'city': [city],
            'income': [income],
            'membership_days': [membership_days],
            'campaign_engagement': [campaign_engagement]}

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['city'])

    # Ensure both columns are present
    for col in ['city_Berlin', 'city_Stuttgart']:
        if col not in df.columns:
            df[col] = 0

    # Reorder the columns to match the training data
    column_order = ['age', 'income', 'membership_days',
                    'campaign_engagement', 'city_Berlin', 'city_Stuttgart']
    df = df[column_order]

    prediction = model.predict(df)
    return int(prediction[0])  # Cast the prediction result to a Python int

# Serve model prediction


@app.post('/predict')
async def get_prediction(input_data_list: InputDataList):
    predictions = [predict(record.age, record.city, record.income, record.membership_days,
                           record.campaign_engagement) for record in input_data_list.data]
    results = []
    for input_data, prediction in zip(input_data_list.data, predictions):
        result = input_data.dict()  # Convert the input data to a dictionary
        result['prediction'] = prediction
        results.append(result)
    return {'results': results}
