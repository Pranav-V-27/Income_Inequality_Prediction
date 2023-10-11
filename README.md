# Income Inequality Prediction

## Description

Income inequality is a growing concern in developing nations worldwide. This project aims to address income inequality by creating a machine learning model to predict whether an individual's income is above or below a certain threshold. The model's accuracy can help policymakers manage and reduce income inequality.

## Problem Statement

The target feature for prediction is **income_above_limit**, a binary-class variable. The primary evaluation metric is the **F1-score**.

## Dataset

- The dataset contains various features, including both numeric and categorical ones. Please refer to the data for a detailed description of the columns.

## Solution Approach

### Exploratory Data Analysis (EDA)

- The project begins with an extensive exploratory data analysis to understand the dataset's characteristics.
- The EDA includes data visualization, statistics, and data cleaning.

### Data Preprocessing

- Handling missing values: Missing values are imputed using the most frequent strategy for specific columns.
- Encoding categorical variables: Categorical variables are encoded using label encoding.
- Data splitting: The dataset is split into features (X) and the target (y).

### Baseline Modeling

- Logistic Regression is used as a baseline model for income prediction.
- The model is trained on the training set and evaluated on the test set.

### Classification Models

- Additional classification models, such as XGBoost, Random Forest, and ExtraTreesClassifier, are explored.
- The models are trained and evaluated, and their performance is compared based on accuracy and F1-score.

## Dependencies

The project relies on the following Python dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- autoviz (for exploratory data analysis)

Ensure that you have these libraries installed in your Python environment.

## How to Run

To replicate this project's results:

1. Download the dataset from the provided path or use your own dataset.
2. Execute the Jupyter Notebook containing the project code.
3. Follow the code cells to perform EDA, data preprocessing, model training, and evaluation.

The project code is self-contained in the provided Jupyter Notebook.


