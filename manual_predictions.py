import pandas as pd
import numpy as np
import pickle

class HousingPricePredictor:
    """Simple predictor for testing manual inputs"""
    
    def __init__(self):
        """Load the trained model and encoders"""
        print("Loading model...")
        
        with open('xgboost_housing_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        with open('model_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load training data for reference values
        self.training_data = pd.read_csv('housing_data_cleaned.csv')
        
        print("✓ Model loaded successfully!")
        print(f"  Model R² Score: {self.metadata['test_r2']:.4f}")
        print(f"  Average Error: ${self.metadata['test_mae']:,.0f}")
    
    
    def predict(self, bed, bath, house_size, city, state, zip_code, acre_lot=0):
        """
        Predict housing price with manual input
        
        Args:
            bed: int - number of bedrooms (1-15)
            bath: float - number of bathrooms (0.5-10)
            house_size: int - square footage (200-20000)
            city: str - city name (e.g., 'Seattle', 'Los Angeles')
            state: str - state name (e.g., 'Washington', 'California')
            zip_code: int - zip code
            acre_lot: float - lot size in acres (optional, default 0)
        
        Returns:
            dict with prediction and details
        """
        
        # Create property data dictionary
        property_data = {
            'bed': float(bed),
            'bath': float(bath),
            'house_size': float(house_size),
            'acre_lot': float(acre_lot),
            'city': str(city),
            'state': str(state),
            'zip_code': int(zip_code)
        }
        
        # Create DataFrame
        df = pd.DataFrame([property_data])
        
        # === FEATURE ENGINEERING (same as training) ===
        
        # Basic features
        df['price_per_sqft'] = 0  # Placeholder, will use zip median
        df['total_rooms'] = df['bed'] + df['bath']
        df['bath_bed_ratio'] = df['bath'] / df['bed']
        df['size_per_bedroom'] = df['house_size'] / df['bed']
        
        # Get city median price from training data
        city_data = self.training_data[self.training_data['city'] == city]
        
        if len(city_data) > 0:
            city_median = city_data['price'].median()
        else:
            # City not found, use state median
            state_data = self.training_data[self.training_data['state'] == state]
            if len(state_data) > 0:
                city_median = state_data['price'].median()
            else:
                # State not found, use overall median
                city_median = self.training_data['price'].median()
        
        df['city_median_price'] = city_median
        
        # Get zip median price per sqft
        zip_data = self.training_data[self.training_data['zip_code'] == zip_code]
        
        if len(zip_data) > 0:
            zip_median_ppsf = zip_data['price_per_sqft'].median()
        else:
            # Zip not found, use city median
            if len(city_data) > 0:
                zip_median_ppsf = city_data['price_per_sqft'].median()
            else:
                # Use overall median
                zip_median_ppsf = self.training_data['price_per_sqft'].median()
        
        df['zip_median_ppsf'] = zip_median_ppsf
        df['price_per_sqft'] = zip_median_ppsf
        df['price_vs_city'] = 1.0  # Neutral starting point
        
        # === ENCODE CATEGORICAL VARIABLES ===
        
        for col in ['city', 'state']:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Value not seen during training, use mode (most common)
                    df[col] = 0
                    print(f"  ⚠️  Warning: {col} '{property_data[col]}' not in training data, using fallback")
        
        # === SELECT FEATURES IN CORRECT ORDER ===
        
        # Ensure all features exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select only training features in correct order
        X = df[self.feature_names]
        
        # === MAKE PREDICTION ===
        
        predicted_price = self.model.predict(X)[0]
        
        # Calculate confidence interval (±1 MAE)
        mae = self.metadata['test_mae']
        
        result = {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price - mae,
                'upper': predicted_price + mae
            },
            'property_details': property_data,
            'market_context': {
                'city_median': city_median,
                'zip_median_ppsf': zip_median_ppsf
            }
        }
        
        return result
    
    
    def print_prediction(self, result):
        """Pretty print prediction results"""
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
        print(f"   City Median Price: ${market['city_median']:,.0f}")
        print(f"   Zip Median $/sqft: ${market['zip_median_ppsf']:,.0f}")
        
        print("\n" + "=" * 70)


# ========================================
# TEST PREDICTIONS
# ========================================
if __name__ == "__main__":
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    print("\n" + "🎯" * 35)
    print("TESTING PREDICTIONS")
    print("🎯" * 35)
    
    # ========================================
    # TEST 1: Everett, Washington (Your location!)
    # ========================================
    print("\n\n" + "=" * 70)
    print("TEST 1: REDMOND, WASHINGTON")
    print("=" * 70)
    
    result1 = predictor.predict(
        bed=4,
        bath=2.5,
        house_size=2540,
        city='Redmond',
        state='Washington',
        zip_code=98052,
        acre_lot=0.17
    )
    predictor.print_prediction(result1)
    
    # ========================================
    # TEST 2: Seattle, Washington
    # ========================================
    print("\n\n" + "=" * 70)
    print("TEST 2: SEATTLE, WASHINGTON")
    print("=" * 70)
    
    result2 = predictor.predict(
        bed=4,
        bath=2.5,
        house_size=2710,
        city='Seattle',
        state='Washington',
        zip_code=98101,
        acre_lot=0.048
    )
    predictor.print_prediction(result2)
    
    # ========================================
    # TEST 3: Los Angeles, California
    # ========================================
    print("\n\n" + "=" * 70)
    print("TEST 3: LOS ANGELES, CALIFORNIA")
    print("=" * 70)
    
    result3 = predictor.predict(
        bed=3,
        bath=2.0,
        house_size=1500,
        city='Los Angeles',
        state='California',
        zip_code=90210,
        acre_lot=0.2
    )
    predictor.print_prediction(result3)
    
    # ========================================
    # TEST 4: New York, New York
    # ========================================
    print("\n\n" + "=" * 70)
    print("TEST 4: NEW YORK, NEW YORK")
    print("=" * 70)
    
    result4 = predictor.predict(
        bed=2,
        bath=2.0,
        house_size=1200,
        city='New York',
        state='New York',
        zip_code=10001,
        acre_lot=0
    )
    predictor.print_prediction(result4)
    
    # ========================================
    # TEST 5: Miami, Florida
    # ========================================
    print("\n\n" + "=" * 70)
    print("TEST 5: MIAMI, FLORIDA")
    print("=" * 70)
    
    result5 = predictor.predict(
        bed=4,
        bath=3.0,
        house_size=2500,
        city='Miami',
        state='Florida',
        zip_code=33101,
        acre_lot=0.25
    )
    predictor.print_prediction(result5)