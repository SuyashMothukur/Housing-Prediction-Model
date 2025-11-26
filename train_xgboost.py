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
    y = df[target].copy()
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}' - {len(le.classes_)} unique values")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training XGBOOST
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    params = {
        'objective': 'reg:squarederror',  # Regression task
        'max_depth': 4,                    # Tree depth (prevent overfitting)
        'learning_rate': 0.05,              # Step size
        'n_estimators': 500,               # Number of trees
        'subsample': 0.8,                  # Sample 80% of data per tree
        'colsample_bytree': 0.8,           # Sample 80% of features per tree
        'min_child_weight': 5,             # Minimum samples in leaf
        'gamma': 0.2,                      # Regularization
        'random_state': 42,
        'n_jobs': -1,                      # Use all CPU cores
        'tree_method': 'hist'              # Faster training for large datasets
    }
    
    model = xgb.XGBRegressor(**params)
    print(f"\nTraining started")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50  # Print progress every 50 iterations
    )
    print(f"\nCompleted.")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"Generated predictions for {len(y_test):,} test samples")
    
    # Calculate metrics for training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTRAINING SET PERFORMANCE:")
    print(f"   MAE (Mean Absolute Error):  ${train_mae:,.0f}")
    print(f"   RMSE (Root Mean Sq Error):  ${train_rmse:,.0f}")
    print(f"   R² Score:                    {train_r2:.4f}")
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"   MAE (Mean Absolute Error):  ${test_mae:,.0f}")
    print(f"   RMSE (Root Mean Sq Error):  ${test_rmse:,.0f}")
    print(f"   R² Score:                    {test_r2:.4f}")
    
    print("\n" + "=" * 70)
    print("STEP 10: SAVING MODEL AND ENCODERS")
    print("=" * 70)
    
    # Save the trained model
    model_filename = 'xgboost_housing_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_filename}")
    
    # Save label encoders
    encoders_filename = 'label_encoders.pkl'
    with open(encoders_filename, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders: {encoders_filename}")
    
    # Save feature names
    feature_names_filename = 'feature_names.pkl'
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Saved feature names: {feature_names_filename}")
    
    # Save training metadata
    metadata = {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols
    }
    
    metadata_filename = 'model_metadata.pkl'
    with open(metadata_filename, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata: {metadata_filename}")
    
    return model, label_encoders, metadata
    


if __name__ == "__main__":
    model, encoders, metadata = train_housing_model('housing_data_cleaned.csv')
    print("Done.")