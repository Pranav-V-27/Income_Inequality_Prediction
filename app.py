import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = load('Model/logistic_regression_model.pkl')  # Change to the correct model file name

# Create title and intro
st.title('Income Inequality Prediction App')
st.write("""
This app predicts whether a person's income will be above the limit based on census data.
""")

# Get user input 
st.subheader('User Input Parameters')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
#class = st.selectbox('Work Class', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
#fnlwgt = st.number_input('FNLWGT', min_value=0) 
education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
#education_num = st.number_input('Education Num', min_value=0, max_value=16, value=10)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
# occupation = st.text_input('Occupation')
#relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])  
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.selectbox('Gender', ['Female', 'Male'])
gains = st.number_input('Capital Gain', min_value=0)
losses = st.number_input('Capital Loss', min_value=0) 
#hours_per_week = st.number_input('Hours Per Week', min_value=0)
#citizenship = st.text_input('citizenship')

# Create feature dictionary    
user_data = {
    'age': age,
    #'workclass': class,
    'education': education,
    #'education_num': education_num,
    'marital_status': marital_status,
    #'relationship': relationship,
    'race': race,
    'sex': gender,
    'capital_gain': gains,
    'capital_loss': losses,
    #'hours_per_week': hours_per_week,
    #'citizenship': citizenship
}

# Transform into DataFrame
features = pd.DataFrame(user_data, index=[0])

# Make prediction
prediction = model.predict(features)

# Output prediction
st.subheader('prediction')
if prediction[0] == 1:
  st.write('Income will likely be above the limit')
else:
  st.write('Income will likely not be above the limit')
