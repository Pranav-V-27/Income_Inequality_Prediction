import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import LabelEncoder

def label_encoder(input_val, encoder):
    encoded_val = encoder.transform([input_val])[0]
    return encoded_val



def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
