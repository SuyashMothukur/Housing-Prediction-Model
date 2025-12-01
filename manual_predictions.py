import pandas as pd
import numpy as np
import pickle

class HousingPricePredictor:
    """Predictor with proper geographic target encoding"""
    
    def __init__(self):
        """Load the trained model"""
        print("Loading model...")
        
        with open('xgboost_housing_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        with open('model_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load training data for target encoding lookups
        self.training_data = pd.read_csv('housing_data_cleaned.csv')
        
        # Pre-calculate target encoding mappings
        self.city_encoding = self.training_data.groupby('city')['price'].mean().to_dict()
        self.state_encoding = self.training_data.groupby('state')['price'].mean().to_dict()
        self.zip_encoding = self.training_data.groupby('zip_code')['price'].mean().to_dict()
        
        # Region mapping
        self.region_mapping = {
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
        
        self.region_encoding = self.training_data.groupby('region')['price'].mean().to_dict()
        
        print("✓ Model loaded successfully!")
        print(f"  Model R² Score: {self.metadata['test_r2']:.4f}")
        print(f"  Average Error: ${self.metadata['test_mae']:,.0f}")
    
    
    def predict(self, bed, bath, house_size, city, state, zip_code, acre_lot=0):
        """Predict housing price with proper target encoding"""
        
        property_data = {
            'bed': float(bed),
            'bath': float(bath),
            'house_size': float(house_size),
            'acre_lot': float(acre_lot),
            'city': str(city),
            'state': str(state),
            'zip_code': int(zip_code)
        }
        
        df = pd.DataFrame([property_data])
        
        # ========================================
        # FEATURE ENGINEERING (same as training)
        # ========================================
        
        # Basic features
        df['price_per_sqft'] = 0  # Placeholder
        df['total_rooms'] = df['bed'] + df['bath']
        df['bath_bed_ratio'] = df['bath'] / df['bed']
        df['size_per_bedroom'] = df['house_size'] / df['bed']
        
        # ========================================
        # TARGET ENCODING (THE FIX!)
        # ========================================
        
        # City target encoding
        df['city_target_encoded'] = df['city'].map(self.city_encoding)
        if pd.isna(df['city_target_encoded'].iloc[0]):
            # City not in training data, use state average
            df['city_target_encoded'] = df['state'].map(self.state_encoding)
            print(f"  ℹ️  {city} not in training data, using state average")
        
        # State target encoding
        df['state_target_encoded'] = df['state'].map(self.state_encoding)
        if pd.isna(df['state_target_encoded'].iloc[0]):
            # State not in training data, use national average
            df['state_target_encoded'] = self.training_data['price'].mean()
            print(f"  ⚠️  {state} not in training data, using national average")
        
        # Zip target encoding
        df['zip_target_encoded'] = df['zip_code'].map(self.zip_encoding)
        if pd.isna(df['zip_target_encoded'].iloc[0]):
            # Zip not found, use city encoding
            df['zip_target_encoded'] = df['city_target_encoded']
        
        # Region encoding
        region = self.region_mapping.get(state, 'Other')
        df['region_target_encoded'] = self.region_encoding.get(region, self.training_data['price'].mean())
        
        # State median price
        state_data = self.training_data[self.training_data['state'] == state]
        if len(state_data) > 0:
            df['state_median_price'] = state_data['price'].median()
        else:
            df['state_median_price'] = self.training_data['price'].median()
        
        # Legacy features (for compatibility)
        city_data = self.training_data[self.training_data['city'] == city]
        if len(city_data) > 0:
            df['city_median_price'] = city_data['price'].median()
            df['zip_median_ppsf'] = city_data['price_per_sqft'].median()
        else:
            df['city_median_price'] = df['state_median_price']
            df['zip_median_ppsf'] = self.training_data['price_per_sqft'].median()
        
        df['price_per_sqft'] = df['zip_median_ppsf']
        df['price_vs_city'] = 1.0
        df['price_vs_state'] = 1.0
        
        # Ensure all features exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # ========================================
        # MAKE PREDICTION
        # ========================================
        
        predicted_price = self.model.predict(X)[0]
        mae = self.metadata['test_mae']
        
        result = {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price - mae,
                'upper': predicted_price + mae
            },
            'property_details': property_data,
            'market_context': {
                'city_avg': df['city_target_encoded'].iloc[0],
                'state_avg': df['state_target_encoded'].iloc[0],
                'zip_avg': df['zip_target_encoded'].iloc[0],
                'region': region
            }
        }
        
        return result
    
    
    def print_prediction(self, result):
        """Pretty print results"""
        print("\n" + "=" * 70)
        print("🏠 PRICE PREDICTION")
        print("=" * 70)
        
        prop = result['property_details']
        print(f"\n📍 Property Details:")
        print(f"   Location: {prop['city']}, {prop['state']} {prop['zip_code']}")
        print(f"   Bedrooms: {prop['bed']}")
        print(f"   Bathrooms: {prop['bath']}")
        print(f"   Square Feet: {prop['house_size']:,.0f}")
        print(f"   Lot Size: {prop['acre_lot']:.2f} acres")
        
        print(f"\n💰 Predicted Price:")
        print(f"   ${result['predicted_price']:,.0f}")
        
        print(f"\n📊 Confidence Interval (±${self.metadata['test_mae']:,.0f}):")
        print(f"   Low:  ${result['confidence_interval']['lower']:,.0f}")
        print(f"   High: ${result['confidence_interval']['upper']:,.0f}")
        
        market = result['market_context']
        print(f"\n📈 Market Context:")
        print(f"   City Average:   ${market['city_avg']:,.0f}")
        print(f"   State Average:  ${market['state_avg']:,.0f}")
        print(f"   Zip Average:    ${market['zip_avg']:,.0f}")
        print(f"   Region: {market['region']}")
        
        print("\n" + "=" * 70)


# ========================================
# TEST PREDICTIONS
# ========================================
if __name__ == "__main__":
    predictor = HousingPricePredictor()
    
    print("\n" + "🎯" * 35)
    print("TESTING GEOGRAPHIC PREDICTIONS")
    print("🎯" * 35)
    
    # TEST: Phoenix, Arizona (Should be ~$380-450k)
    print("\n\n" + "=" * 70)
    print("TEST 1: PHOENIX, ARIZONA (Mid-tier market)")
    print("=" * 70)
    
    result1 = predictor.predict(
        bed=3,
        bath=2.0,
        house_size=1800,
        city='Phoenix',
        state='Arizona',
        zip_code=85001,
        acre_lot=0.15
    )
    predictor.print_prediction(result1)
    print(f"\n💡 Expected: ~$380,000-$450,000")
    print(f"   Actual:   ${result1['predicted_price']:,.0f}")
    
    # TEST: Seattle, Washington (Should be ~$750-900k)
    print("\n\n" + "=" * 70)
    print("TEST 2: SEATTLE, WASHINGTON (High-cost market)")
    print("=" * 70)
    
    result2 = predictor.predict(
        bed=3,
        bath=2.0,
        house_size=1800,
        city='Seattle',
        state='Washington',
        zip_code=98101,
        acre_lot=0.10
    )
    predictor.print_prediction(result2)
    print(f"\n💡 Expected: ~$750,000-$900,000")
    print(f"   Actual:   ${result2['predicted_price']:,.0f}")
    
    # TEST: Dallas, Texas (Should be ~$320-400k)
    print("\n\n" + "=" * 70)
    print("TEST 3: DALLAS, TEXAS (Mid-tier market)")
    print("=" * 70)
    
    result3 = predictor.predict(
        bed=3,
        bath=2.0,
        house_size=1800,
        city='Dallas',
        state='Texas',
        zip_code=75201,
        acre_lot=0.15
    )
    predictor.print_prediction(result3)
    print(f"\n💡 Expected: ~$320,000-$400,000")
    print(f"   Actual:   ${result3['predicted_price']:,.0f}")
    
    print("\n\n" + "=" * 70)
    print("TEST 4: HARTFORD, CONNECTICUT (Mid-tier market)")
    print("=" * 70)
    
    result4 = predictor.predict(
        bed=3, 
        bath=3.0, 
        house_size=1380, 
        city='Chicago',
        state='Illinois',
        zip_code=60651, 
        acre_lot=0.08
    )
    predictor.print_prediction(result4)
    print(f"\n💡 Expected: ~$395,000-$425,000")
    print(f"   Actual:   ${result3['predicted_price']:,.0f}")