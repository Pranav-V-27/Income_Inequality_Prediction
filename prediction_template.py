import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def label_encoder(input_val, encoder):
    encoded_val = encoder.transform([input_val])[0]
    return encoded_val

def get_prediction(data, model, label_encoders):
    """
    Predict the class of a given data point.
    """
    # Apply label encoding to the data
    for feature in label_encoders:
        data[feature] = label_encoder(data[feature], label_encoders[feature])
    
    return model.predict(data)
