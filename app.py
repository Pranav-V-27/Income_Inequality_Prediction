# Import streamlit
import streamlit as st

# Import the libraries used in the code snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns',None)

# Import the autoviz library
!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class

# Define the title of the app
st.title("Data Analysis and Machine Learning App")

# Define a sidebar for user inputs
st.sidebar.header("User Inputs")

# Define a file uploader for the data file
data_file = st.sidebar.file_uploader("Upload your data file", type=["csv"])

# Define a text input for the target variable name
target_variable = st.sidebar.text_input("Enter the name of the target variable")

# Define a checkbox for running the autoviz report
run_autoviz = st.sidebar.checkbox("Run AutoViz report")

# Define a checkbox for running the logistic regression model
run_logistic = st.sidebar.checkbox("Run Logistic Regression model")

# Define a function to load and preprocess the data file
def load_data(data_file):
    # Read the csv file into a pandas dataframe
    data = pd.read_csv(data_file)

    # Define the columns with missing values that you want to impute
    columns_with_missing = ["class", "education_institute", "unemployment_reason", "is_labor_union",
                            "occupation_code_main", "under_18_family", "veterans_admin_questionnaire",
                            "migration_code_change_in_msa", "migration_prev_sunbelt",
                            "migration_code_move_within_reg", "migration_code_change_in_reg",
                            "residence_1_year_ago", "old_residence_reg", "old_residence_state"]

    # Initialize SimpleImputer with the most frequent strategy for specified columns
    SI = SimpleImputer(strategy="most_frequent")
    data[columns_with_missing] = SI.fit_transform(data[columns_with_missing])

    # Encode categorical variables using Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Return the processed dataframe
    return data

# Define a function to run the autoviz report on the data file
def run_autoviz(data_file, target_variable):
    # Initialize AutoViz class object
    AV = AutoViz_Class()

    # Perform EDA with AutoViz and save the report as html file
    report = AV.AutoViz(
        filename= data_file,   # Use the uploaded data file path
        sep=",",
        depVar=target_variable,
        dfte=None,
        header=0,
        verbose=2,
        lowess=False,
        chart_format="svg",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir="autoviz_report"  # Specify a directory to save the report files
        )

    # Display the report html file in the app using streamlit components library (st.beta_container)
    with st.beta_container():
        st.components.v1.html(report)

# Define a function to run the logistic regression model on the data file and target variable
def run_logistic(data_file, target_variable):
    # Load and preprocess the data file using the load_data function defined earlier
    data = load_data(data_file)

    # Split the data into features (X) and target (y)
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training using Logistic Regression from sklearn.linear_model library 
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Model Prediction on the test set
    y_pred = model.predict(X_test)

    # Model Evaluation using f1_score and classification_report from sklearn.metrics library
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Display the model evaluation results in the app using streamlit markdown and write functions
    st.markdown("## Model Evaluation")
    st.write(f"F1 Score: {f1}")
    st.write("\nClassification Report:\n", classification_rep)

# Check if the user has uploaded a data file
if data_file is not None:
    # Display a message that the data file is loaded
    st.success("Data file loaded successfully")

    # Check if the user has entered a target variable name
    if target_variable != "":
        # Display a message that the target variable name is valid
        st.success("Target variable name is valid")

        # Check if the user has checked the run_autoviz checkbox
        if run_autoviz:
            # Display a message that the autoviz report is running
            st.info("Running AutoViz report...")

            # Run the autoviz report using the run_autoviz function defined earlier
            run_autoviz(data_file, target_variable)

        # Check if the user has checked the run_logistic checkbox
        if run_logistic:
            # Display a message that the logistic regression model is running
            st.info("Running Logistic Regression model...")

            # Run the logistic regression model using the run_logistic function defined earlier
            run_logistic(data_file, target_variable)

    else:
        # Display a warning message that the target variable name is missing
        st.warning("Please enter the name of the target variable")

else:
    # Display a warning message that no data file is uploaded
    st.warning("Please upload a data file")
