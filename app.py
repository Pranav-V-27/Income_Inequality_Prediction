import streamlit as st
import pandas as pd
import joblib

# Load model
model = 'Model/logistic_regression_model.pkl'

# Load label encoder 
encoder = 'Model/label_encoder.pkl'

st.title('Income Inequality Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 25)
    workclass = st.sidebar.selectbox('Work Class', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    fnlwgt = st.sidebar.slider('FNLWGT', min_value=0)
    education = st.sidebar.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    education_num = st.sidebar.slider('Education Num', 0,16, 10)
    marital_status = st.sidebar.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.sidebar.text_input('Occupation')
    relationship = st.sidebar.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.sidebar.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
    capital_gain = st.sidebar.slider('Capital Gain', min_value=0)
    capital_loss = st.sidebar.slider('Capital Loss', min_value=0)
    hours_per_week = st.sidebar.slider('Hours Per Week', min_value=0)
    native_country = st.sidebar.text_input('Native Country')
    
    data = {'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'education_num': education_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss,
            'hours_per_week': hours_per_week,
            'native_country': native_country}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Encoding categorical data
for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']:
    df[col] = encoder.transform(df[col]) 

# Predict    
prediction = model.predict(df)

# Display
if st.button('Predict'):
    if prediction == 1:
        st.subheader('Income will likely be above the limit')
    else:
        st.subheader('Income will likely not be above the limit')
        
st.subheader('Input Parameters')
st.write(df)
