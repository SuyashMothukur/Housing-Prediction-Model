import pandas as pd 
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def train_housing_model(data_path='housing_data_cleaned.csv'):
    # Loading data
    df = pd.read_csv(data_path)
    target = 'price' # Target variable
    
    feature_cols = [col for col in df.columns if col not in [target, 'log_price']]
    X = df[feature_cols].copy()
    Y = df[target].copy()


if __name__ == "__main__":
    model, encoders, metadata = train_housing_model('housing_data_cleaned.csv')
    print("Done.")