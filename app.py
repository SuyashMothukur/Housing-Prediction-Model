from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load model at startup
print("Loading model...")
with open('xgboost_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load training data for target encoding lookups
training_data = pd.read_csv('housing_data_cleaned.csv')

# Pre-calculate target encoding mappings
city_encoding = training_data.groupby('city')['price'].mean().to_dict()
state_encoding = training_data.groupby('state')['price'].mean().to_dict()
zip_encoding = training_data.groupby('zip_code')['price'].mean().to_dict()

# Region mapping
region_mapping = {
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
    'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest',
    'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest',
    'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South',
    'Maryland': 'South', 'North Carolina': 'South', 'South Carolina': 'South',
    'Virginia': 'South', 'District of Columbia': 'South', 'West Virginia': 'South',
    'Alabama': 'South', 'Kentucky': 'South', 'Mississippi': 'South',
    'Tennessee': 'South', 'Arkansas': 'South', 'Louisiana': 'South',
    'Oklahoma': 'South', 'Texas': 'South',
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West',
    'Montana': 'West', 'Nevada': 'West', 'New Mexico': 'West',
    'Utah': 'West', 'Wyoming': 'West', 'Alaska': 'West',
    'California': 'West', 'Hawaii': 'West', 'Oregon': 'West',
    'Washington': 'West',
    'Puerto Rico': 'South', 'Virgin Islands': 'South'
}

region_encoding = training_data.groupby('region')['price'].mean().to_dict()

print("✓ Model loaded successfully!")
print(f"✓ Loaded {len(city_encoding)} cities")
print(f"✓ Loaded {len(state_encoding)} states")
print(f"✓ Loaded {len(zip_encoding)} zip codes")


def prepare_features(property_data):
    """
    Prepare property data for prediction with TARGET ENCODING
    """
    df = pd.DataFrame([property_data])
    
    city = property_data['city']
    state = property_data['state']
    zip_code = property_data['zip_code']
    
    # ========================================
    # BASIC FEATURES
    # ========================================
    df['price_per_sqft'] = 0  # Placeholder
    df['total_rooms'] = df['bed'] + df['bath']
    df['bath_bed_ratio'] = df['bath'] / df['bed']
    df['size_per_bedroom'] = df['house_size'] / df['bed']
    
    # ========================================
    # TARGET ENCODING (THE FIX!)
    # ========================================
    
    # City target encoding
    df['city_target_encoded'] = city_encoding.get(city, None)
    if df['city_target_encoded'].iloc[0] is None:
        # City not found, use state average
        df['city_target_encoded'] = state_encoding.get(state, training_data['price'].mean())
    
    # State target encoding
    df['state_target_encoded'] = state_encoding.get(state, training_data['price'].mean())
    
    # Zip target encoding
    df['zip_target_encoded'] = zip_encoding.get(zip_code, None)
    if df['zip_target_encoded'].iloc[0] is None:
        # Zip not found, use city encoding
        df['zip_target_encoded'] = df['city_target_encoded']
    
    # Region encoding
    region = region_mapping.get(state, 'Other')
    df['region_target_encoded'] = region_encoding.get(region, training_data['price'].mean())
    
    # State median price
    state_data = training_data[training_data['state'] == state]
    if len(state_data) > 0:
        df['state_median_price'] = state_data['price'].median()
    else:
        df['state_median_price'] = training_data['price'].median()
    
    # ========================================
    # LEGACY FEATURES (for compatibility)
    # ========================================
    city_data = training_data[training_data['city'] == city]
    
    if len(city_data) > 0:
        df['city_median_price'] = city_data['price'].median()
        df['zip_median_ppsf'] = city_data['price_per_sqft'].median()
    else:
        df['city_median_price'] = df['state_median_price']
        df['zip_median_ppsf'] = training_data['price_per_sqft'].median()
    
    df['price_per_sqft'] = df['zip_median_ppsf']
    df['price_vs_city'] = 1.0
    df['price_vs_state'] = 1.0
    
    # ========================================
    # ENSURE ALL FEATURES EXIST
    # ========================================
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
        
        # Prepare features with TARGET ENCODING
        X = prepare_features(property_data)
        
        # Make prediction
        predicted_price = float(model.predict(X)[0])
        
        # Calculate confidence interval
        mae = metadata['test_mae']
        
        # Get market context for response
        city_avg = city_encoding.get(property_data['city'], state_encoding.get(property_data['state'], 0))
        state_avg = state_encoding.get(property_data['state'], 0)
        
        # Prepare response
        response = {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price - mae,
                'upper': predicted_price + mae
            },
            'property_details': property_data,
            'market_context': {
                'city_average': city_avg,
                'state_average': state_avg
            },
            'model_stats': {
                'r2_score': metadata['test_r2'],
                'mae': mae,
                'confidence': metadata['test_r2'] * 100
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in prediction: {e}")
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
    print(f"\n🗺️  Geographic Coverage:")
    print(f"   States: {len(state_encoding)}")
    print(f"   Cities: {len(city_encoding)}")
    print(f"   Zip Codes: {len(zip_encoding)}")
    print("\nAPI Endpoints:")
    print("  POST /api/predict - Get price prediction")
    print("  GET  /api/health  - Check API status")
    print("  GET  /api/states  - Get available states")
    print("  GET  /api/cities?state=X - Get cities in state")
    print("\nStarting server on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5000)