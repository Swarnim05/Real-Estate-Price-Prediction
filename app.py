import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading your saved model
import json

# Load model and columns data
lr_clf = joblib.load(r'C:\Users\nainw\Downloads\Real-Estate-Price-Prediction\price prediction real estate\Data3.pickle') 
with open(r'C:\Users\nainw\Downloads\Real-Estate-Price-Prediction\price prediction real estate\columns.json', 'r') as f:
    data_columns = json.load(f)['data columns']

# Function for price prediction
def predict_price(location, area, balcony, bhk):
    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = balcony
    x[2] = bhk
    if location.lower() in data_columns:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

# Streamlit app
st.title("Real Estate Price Prediction")

st.header("Enter the property details:")

# Location selection
location = st.selectbox("Location", options=data_columns[3:])  # Assuming the first 3 columns are not locations

# Area input (sqft)
area = st.number_input("Area (in sqft)", min_value=100, max_value=10000, step=100)

# Balcony input (Yes=1, No=0)
balcony = st.radio("Does the property have a balcony?", ("No", "Yes"))
balcony = 1 if balcony == "Yes" else 0

# BHK input
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# Prediction button
if st.button("Predict Price"):
    # Get predicted price
    predicted_price = predict_price(location, area, balcony, bhk)

    # Display price in Crores or Lakhs
    if predicted_price > 100:
        predicted_price_in_crores = predicted_price / 100
        st.success(f"The predicted price is: ₹ {predicted_price_in_crores:,.2f} Crores")
    else:
        st.success(f"The predicted price is: ₹ {predicted_price:,.2f} Lakhs")
