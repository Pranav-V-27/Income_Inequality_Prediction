import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.impute import SimpleImputer
# import zipfile

# with zipfile.ZipFile("./Model/exc.zip") as zip:
#     with zip.open("content/ExtraTreesClassifier.joblib") as myZip:
#         model = joblib.load(myZip)
# model = joblib.load(r'Model/ExtraTreesClassifier.joblib')
def label_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value
model = '.Model/logistic_regression_model.pkl'

train =  pd.read_csv(".Dataset/project_2_data.csv")
train_copy = train.copy()
#SI = SimpleImputer(strategy="most_frequent")
#train_copy = pd.DataFrame(SI.fit_transform(train_copy))

train_copy.columns = train_copy.columns

for col in train_copy.columns:
  if col in 'ID', 'gender', 'education', 'class', 'education_institute',
       'marital_status', 'race', 'is_hispanic', 'employment_commitment',
       'unemployment_reason', 'is_labor_union', 'industry_code_main',
       'occupation_code_main', 'household_stat', 'household_summary',
       'under_18_family', 'veterans_admin_questionnaire', 'tax_status',
       'citizenship', 'country_of_birth_own', 'country_of_birth_father',
       'country_of_birth_mother', 'migration_code_change_in_msa',
       'migration_prev_sunbelt', 'migration_code_move_within_reg',
       'migration_code_change_in_reg', 'residence_1_year_ago',
       'old_residence_reg', 'old_residence_state', 'income_above_limit']:
    continue
  le = LabelEncoder()
  train_copy[col] = le.fit_transform(train_copy[col])

model.fit(train_copy.iloc[:,:-1], train_copy.iloc[:,-1])

st.set_page_config(page_title="Income Inequality Prediction App",
                   page_icon="$$", layout="wide")

#creating option list for dropdown menu
age = st.number_input('Age', min_value=0, max_value=100, value=25)
workclass = st.selectbox('Work Class', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
#fnlwgt = st.number_input('FNLWGT', min_value=0)
education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
education_num = st.number_input('Education Num', min_value=0, max_value=16, value=10)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation_code_main = st.text_input('occupation_code_main')
#relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.selectbox('Sex', ['Female', 'Male'])
gains = st.number_input('Capital Gain', min_value=0)
losses = st.number_input('Capital Loss', min_value=0)
#hours_per_week = st.number_input('Hours Per Week', min_value=0)
#native_country = st.text_input('Native Country')

# Create feature dictionary
user_data = {
    'age': age,
    'workclass': LabelEncoder(workclass),
    #'fnlwgt': fnlwgt,
    'education': LabelEncoder(education),
    'education_num': education_num,
    'marital_status': LabelEncoder(marital_status),
    'occupation': LabelEncoder(occupation_code_main),
    #'relationship': relationship,
    'race': LabelEncoder(race),
    'sex': LabelEncoder(gender),
    'capital_gain': gains,
    'capital_loss': losses,
    #'hours_per_week': hours_per_week,
    #'native_country': native_country
}

# Transform into DataFrame
features = pd.DataFrame(user_data, index=[0])

# Make prediction
prediction = model.predict(features)

# Output prediction
       

            if __name__ == '__main__':
    main()
