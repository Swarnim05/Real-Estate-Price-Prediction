import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# Set page configuration with a premium real estate icon
st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="centered")

# Load model and columns metadata
@st.cache_resource
def load_assets():
    model = joblib.load('Data3.pickle')
    with open('columns.json', 'r') as f:
        cols = json.load(f)['data columns']
    return model, cols

try:
    lr_clf, data_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Function for price prediction
def predict_price(location, area, balcony, bhk, prop_type):
    # Construct a DataFrame matching the model's training columns
    df_input = pd.DataFrame(np.zeros((1, len(data_columns))), columns=data_columns)
    
    # Map lowercase column names to their exact training casing
    col_map = {col.lower(): col for col in data_columns}
    
    # Set continuous and numeric features
    df_input[col_map['area']] = float(area)
    df_input[col_map['bhk']] = float(bhk)
    df_input[col_map['balcony']] = float(balcony)
    
    # Set one-hot encoded property type
    type_col = f"type_{prop_type.lower()}"
    if type_col in col_map:
        df_input[col_map[type_col]] = 1.0
        
    # Set one-hot encoded location
    loc_col = location.lower()
    if loc_col in col_map:
        df_input[col_map[loc_col]] = 1.0
        
    # Run prediction
    return lr_clf.predict(df_input)[0]

# Streamlit App UI Design
st.title("🏠 Real Estate Price Prediction")
st.markdown("Enter property details below to predict the estimated price in Lakhs/Crores.")

st.write("---")

st.header("Property Details")

# Layout inputs using columns for a cleaner interface
col1, col2 = st.columns(2)

with col1:
    # Location selection (starts after area, bhk, balcony, and 3 property type columns)
    location = st.selectbox("Location / Locality", options=data_columns[6:])
    
    # Area input (sqft)
    area = st.number_input("Area (in sqft)", min_value=200, max_value=4000, value=1200, step=50)

with col2:
    # Property type selection
    prop_type = st.selectbox("Property Type", options=["Flat", "House", "Villa"])
    
    # BHK input
    bhk = st.number_input("Number of BHK (Rooms)", min_value=1, max_value=10, value=2, step=1)

# Balcony input (Yes=1, No=0)
balcony = st.radio("Does the property have a balcony?", ("No", "Yes"), horizontal=True)
balcony_val = 1 if balcony == "Yes" else 0

st.write("---")

# Prediction action
if st.button("Predict Estimated Price", type="primary", use_container_width=True):
    with st.spinner("Calculating price..."):
        try:
            predicted_price = predict_price(location, area, balcony_val, bhk, prop_type)
            
            # Display price formatted beautifully in Crores or Lakhs
            if predicted_price >= 100:
                predicted_price_in_crores = predicted_price / 100
                st.success(f"### The predicted price is: **₹ {predicted_price_in_crores:,.2f} Crores**")
            else:
                st.success(f"### The predicted price is: **₹ {predicted_price:,.2f} Lakhs**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
