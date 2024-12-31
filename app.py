import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load the trained model

model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('oneHotEncoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# streamlit app
st.title('Churn Prediction App')

# User input
geograpgy = st.selectbox('Geography', onehot_encoder_geo.categories_[0])  
gender = st.selectbox('Gender',label_encoder_gender.classes_) 
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10) 
num_of_products = st.slider('Number of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-Hot encode 'Geography
geo_enabled = onehot_encoder_geo.transform([[geograpgy]]).toarray()
geo_enabled_df = pd.DataFrame(geo_enabled, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combone one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_enabled_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict CHurn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'The probability of the customer churning is {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is Not likely to churn')
