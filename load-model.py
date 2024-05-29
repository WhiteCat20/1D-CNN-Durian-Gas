import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import easygui
import joblib


def load_csv_data(file_path):
    csv_data = pd.read_csv(file_path).values.flatten()
    return np.array([csv_data])

def preprocess_data(data, scaler):
    data = scaler.transform(data)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    return data

model_path = '29052024-192640.keras'  
model = load_model(model_path)

file_path = easygui.fileopenbox(title='Select The File') 
data = load_csv_data(file_path)

scaler_path = 'scaler.pkl' 
scaler = joblib.load(scaler_path)

# Melakukan preprocessing pada data input
data = preprocess_data(data, scaler)

# Melakukan prediksi
predictions = model.predict(data)
predictions = np.round(predictions).flatten()

print(f'Predictions: {predictions}')
