from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load model and encoders at startup
print("Loading model...")
with open('xgboost_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load training data for reference values
training_data = pd.read_csv('housing_data_cleaned.csv')

print("✓ Model loaded successfully!")


def prepare_features(property_data):
    """Prepare property data for prediction"""
    
    # Create DataFrame
    df = pd.DataFrame([property_data])
    
    # Feature engineering (same as training)
    df['price_per_sqft'] = 0  # Placeholder
    df['total_rooms'] = df['bed'] + df['bath']
    df['bath_bed_ratio'] = df['bath'] / df['bed']
    df['size_per_bedroom'] = df['house_size'] / df['bed']
    
    # Get city median price
    city_data = training_data[training_data['city'] == property_data['city']]
    
    if len(city_data) > 0:
        city_median = city_data['price'].median()
    else:
        state_data = training_data[training_data['state'] == property_data['state']]
        if len(state_data) > 0:
            city_median = state_data['price'].median()
        else:
            city_median = training_data['price'].median()
    
    df['city_median_price'] = city_median
    
    # Get zip median price per sqft
    zip_data = training_data[training_data['zip_code'] == property_data['zip_code']]
    
    if len(zip_data) > 0:
        zip_median_ppsf = zip_data['price_per_sqft'].median()
    else:
        if len(city_data) > 0:
            zip_median_ppsf = city_data['price_per_sqft'].median()
        else:
            zip_median_ppsf = training_data['price_per_sqft'].median()
    
    df['zip_median_ppsf'] = zip_median_ppsf
    df['price_per_sqft'] = zip_median_ppsf
    df['price_vs_city'] = 1.0
    
    # Encode categorical variables
    for col in ['city', 'state']:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
    
    # Ensure all features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    return X


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for price predictions
    
    Expected JSON:
    {
        "bed": 3,
        "bath": 2.0,
        "house_size": 2000,
        "city": "Seattle",
        "state": "Washington",
        "zip_code": 98101,
        "acre_lot": 0.15
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['bed', 'bath', 'house_size', 'city', 'state', 'zip_code']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare property data
        property_data = {
            'bed': float(data['bed']),
            'bath': float(data['bath']),
            'house_size': float(data['house_size']),
            'city': str(data['city']),
            'state': str(data['state']),
            'zip_code': int(data['zip_code']),
            'acre_lot': float(data.get('acre_lot', 0))
        }
        
        # Prepare features
        X = prepare_features(property_data)
        
        # Make prediction
        predicted_price = float(model.predict(X)[0])
        
        # Calculate confidence interval
        mae = metadata['test_mae']
        
        # Prepare response
        response = {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price - mae,
                'upper': predicted_price + mae
            },
            'property_details': property_data,
            'model_stats': {
                'r2_score': metadata['test_r2'],
                'mae': mae,
                'confidence': metadata['test_r2'] * 100
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_r2': metadata['test_r2'],
        'training_samples': metadata['train_samples']
    }), 200


@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of available states"""
    states = sorted(training_data['state'].unique().tolist())
    return jsonify({'states': states}), 200


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get cities for a specific state"""
    state = request.args.get('state')
    if not state:
        return jsonify({'error': 'State parameter required'}), 400
    
    cities = sorted(
        training_data[training_data['state'] == state]['city'].unique().tolist()
    )
    return jsonify({'cities': cities}), 200


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 HOUSING PRICE PREDICTOR API")
    print("=" * 70)
    print(f"Model R² Score: {metadata['test_r2']:.4f}")
    print(f"Average Error: ${metadata['test_mae']:,.0f}")
    print(f"Training Samples: {metadata['train_samples']:,}")
    print("\nAPI Endpoints:")
    print("  POST /api/predict - Get price prediction")
    print("  GET  /api/health  - Check API status")
    print("  GET  /api/states  - Get available states")
    print("  GET  /api/cities?state=X - Get cities in state")
    print("\nStarting server on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5000)