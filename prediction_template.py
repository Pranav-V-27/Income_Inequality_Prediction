import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_filename = 'logistic_regression_model.pkl' 
model = joblib.load(model_filename)

# Function to predict income above limit
def predict_income(data):
    prediction = model.predict(data)
    return prediction

# Sample input data for prediction (customize as needed)
sample_data = {
    'pickup_hour': 10,
    'day_of_week': 2,  # 2 corresponds to 'Tuesday'
    'hour_of_accident': 15,
    'accident_cause': 'Reckless Driving',
    'num_vehicles_involved': 2,
    'vehicle_type': 'Sedan',
    'driver_age': 30,
    'accident_area': 'Urban',
    'driving_experience': 8,
    'lanes': 2
}

# Convert the sample input data to a DataFrame
sample_df = pd.DataFrame([sample_data])

# Map day_of_week to numeric values
day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
sample_df['day_of_week'] = sample_df['day_of_week'].map(day_mapping).fillna(0).astype(int)

# Make predictions
predictions = predict_income(sample_df)

# Display predictions
for idx, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Sample {idx + 1}: Predicted Income Status: Above Limit")
    else:
        print(f"Sample {idx + 1}: Predicted Income Status: Below Limit")
