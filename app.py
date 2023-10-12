import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load(r'Model/logistic_regression_model.pkl')

def label_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value
   
# Create title and intro
st.title('Income Inequality Prediction App')
st.write("""
This app predicts whether a person's income will be above the limit based on census data.
""")

# Get user input
st.subheader('User Input Parameters')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
#workclass = st.selectbox('Work Class', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
education_num = st.number_input('Education Num', min_value=0, max_value=16, value=10)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.text_input('Occupation')
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.selectbox('Sex', ['Female', 'Male'])
capital_gain = st.number_input('Capital Gain', min_value=0)
capital_loss = st.number_input('Capital Loss', min_value=0)
hours_per_week = st.number_input('Hours Per Week', min_value=0)
native_country = st.text_input('Native Country')

# Apply LabelEncoder to categorical features
label_encoder = ('Model/label_encoder.pkl')
user_data = {
   'age': age,
    'education': label_encoder.transform([education])[0],  # Corrected this line
    'education_num': education_num,
    'marital_status': label_encoder.transform([marital_status])[0],
    'occupation': label_encoder.transform([occupation])[0],
    'race': label_encoder.transform([race])[0],
    'sex': label_encoder.transform([sex])[0],
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'native_country': native_country
}

# Transform into DataFrame
features = pd.DataFrame(user_data, index=[0])

# Make prediction
prediction = model.predict(features)

# Output prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Income will likely be above the limit')
else:
    st.write('Income will likely not be above the limit')
