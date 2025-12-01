import pandas as pd 
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_housing_model(data_path='housing_data_cleaned.csv'):
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} rows")
    
    target = 'price'
    
    # ========================================
    # FIX: DROP CATEGORICAL COLUMNS, KEEP TARGET ENCODINGS
    # ========================================
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)
    
    # Columns to EXCLUDE from training
    exclude_cols = [
        target,          # Target variable
        'log_price',     # Alternative target
        'city',          # 🔥 DROP - using city_target_encoded instead
        'state',         # 🔥 DROP - using state_target_encoded instead
        'region',        # 🔥 DROP - using region_target_encoded instead
        'zip_code'       # Keep as numeric feature (it's already an int)
    ]
    
    # Get all feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    print(f"✓ Target variable: {target}")
    print(f"✓ Number of features: {len(feature_cols)}")
    print(f"\n📊 Features being used:")
    for col in feature_cols:
        print(f"   - {col}")
    
    # Verify no categorical columns remain
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"\n⚠️  WARNING: Found categorical columns: {categorical_cols}")
        print("These should have been target-encoded in clean_data.py!")
    else:
        print(f"\n✅ All features are numeric - ready for XGBoost!")
    
    # ========================================
    # TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set:     {len(X_test):,} samples")
    
    # ========================================
    # TRAIN XGBOOST
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,                    # Increased from 4 - more complex geography
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.2,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    
    print(f"\nTraining started...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    print(f"✓ Training completed!")
    
    # ========================================
    # MAKE PREDICTIONS
    # ========================================
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n📊 TRAINING SET PERFORMANCE:")
    print(f"   MAE:  ${train_mae:,.0f}")
    print(f"   RMSE: ${train_rmse:,.0f}")
    print(f"   R²:   {train_r2:.4f}")
    
    print(f"\n📊 TEST SET PERFORMANCE:")
    print(f"   MAE:  ${test_mae:,.0f}")
    print(f"   RMSE: ${test_rmse:,.0f}")
    print(f"   R²:   {test_r2:.4f}")
    
    # ========================================
    # FEATURE IMPORTANCE
    # ========================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # Check if geographic features are important
    geo_features = ['city_target_encoded', 'state_target_encoded', 'zip_target_encoded', 
                    'region_target_encoded', 'state_median_price']
    geo_importance = importance_df[importance_df['feature'].isin(geo_features)]
    
    print(f"\n🗺️  GEOGRAPHIC FEATURE IMPORTANCE:")
    for idx, row in geo_importance.iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    total_geo_importance = geo_importance['importance'].sum()
    print(f"\n   Total geographic importance: {total_geo_importance:.4f} ({total_geo_importance*100:.1f}%)")
    
    if total_geo_importance < 0.15:
        print(f"   ⚠️  WARNING: Geographic features have low importance!")
        print(f"   This may indicate the model isn't learning location properly.")
    else:
        print(f"   ✅ Geographic features are influential!")
    
    # ========================================
    # SAVE MODEL
    # ========================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    # Save model
    model_filename = 'xgboost_housing_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model: {model_filename}")
    
    # Save feature names
    feature_names_filename = 'feature_names.pkl'
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Saved feature names: {feature_names_filename}")
    
    # Save metadata
    metadata = {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_cols': feature_cols,
        'feature_importance': importance_df.to_dict('records')
    }
    
    metadata_filename = 'model_metadata.pkl'
    with open(metadata_filename, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_filename}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"🎯 Model Performance:")
    print(f"   Test R²: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
    print(f"   Test MAE: ${test_mae:,.0f}")
    print(f"\n🗺️  Geographic Learning:")
    print(f"   Geographic features account for {total_geo_importance*100:.1f}% of importance")
    
    return model, metadata


if __name__ == "__main__":
    model, metadata = train_housing_model('housing_data_cleaned.csv')
    print("\n✨ Model ready for predictions!")