import pandas as pd
import numpy as np

def clean_housing_data(filepath='realtor-data.zip.csv'):
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Remove rows with Na or 0 prices
    df = df[df['price'].notna() & (df['price'] > 0)]
    print(f"Removed invalid prices: {len(df):,} rows remaining")
    
    # Remove outlier prices
    df = df[df['price'] >= 5000]
    df = df[df['price'] <= 20000000]
    print(f"Removed outliers prices: {len(df):,} rows remaning")
    print(f"New price range: ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")
    
    # Clean the bedrooms 
    
    # Remove null bedrooms
    df = df[df['bed'].notna()]
    print(f"Removed null bedrooms: {len(df):,}, rows remaining")
    
    # Remove bedrooms outside of 1-15
    df = df[df['bed'] >= 1]
    df = df[df['bed'] <= 15]
    print(f"Kept bedroom sizes of 1-15: {len(df):,}, rows remaining")
    
    # Clean the bathrooms 
    df = df[df['bath'].notna()]
    print(f"Removed null bathrooms: {len(df):,}, rows remaining")

    df = df[df['bath'] >= 0.5]
    df = df[df['bath'] <= 10]
    print(f"Kept 0.5 to 10 bathrooms: {len(df):,}, rows remaining")

    # Clean the size of houses
    df = df[df['house_size'].notna()]
    print(f"Removed null houses: {len(df):,}, rows remaining")

    df = df[df['house_size'] >= 200]
    df = df[df['house_size'] <= 20000]
    print(f"Restricted housing size to 200 to 20000 sqaure feet: {len(df):,}, rows remaining")
    
    # Clean the location data 
    df = df[df['state'].notna()]
    df = df[df['city'].notna()]
    df = df[df['zip_code'].notna()]
    print(f"Removed null addresses: {len(df):,}, rows remaining")
    
    df['zip_code'] = df['zip_code'].astype(int)
    df = df[(df['zip_code'] >= 501) & (df['zip_code'] <= 99950)]
    print(f"Kept valid zip codes only: {len(df):,}, rows remaining")
    
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['price', 'bed', 'bath', 'house_size', 'city', 'state'])
    print(f"Removed {original_len - len(df):,} duplicates")
    print(f"Remaining: {len(df):,} rows")
    
    # Feature Engineering 
    
    df['price_per_sqft'] = df['price'] / df['house_size']
    df = df[df['price_per_sqft'] > 0]
    print(f"Created the price per sqaure foot feature.")
    
    df['total_rooms'] = df['bath'] + df['bed']
    print(f"Created total rooms feature.")
    
    df['bath_bed_ratio'] = df['bath'] / df['bed']
    print(f"Created bath to bed ratio per home.")

    df['size_per_bedroom'] = df['house_size'] / df['bed']
    print(f"Created the size per bedroom feature.")
    
    # ========================================
    # FIX #1: TARGET ENCODING FOR GEOGRAPHY
    # ========================================
    print("\n🔧 GEOGRAPHIC TARGET ENCODING:")
    
    # City target encoding - MEAN price per city
    df['city_target_encoded'] = df.groupby('city')['price'].transform('mean')
    print(f"✓ Created 'city_target_encoded' - captures actual city price levels")
    
    # State target encoding - MEAN price per state
    df['state_target_encoded'] = df.groupby('state')['price'].transform('mean')
    print(f"✓ Created 'state_target_encoded' - captures state-level pricing")
    
    # Zip target encoding - MEAN price per zip
    df['zip_target_encoded'] = df.groupby('zip_code')['price'].transform('mean')
    print(f"✓ Created 'zip_target_encoded' - hyper-local pricing")
    
    # ========================================
    # FIX #2: ADD REGION ENCODING
    # ========================================
    print("\n🗺️  REGIONAL GROUPING:")
    
    # US Census regions
    region_mapping = {
        # Northeast
        'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
        'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
        'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
        
        # Midwest
        'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
        'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest',
        'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest',
        'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
        
        # South
        'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South',
        'Maryland': 'South', 'North Carolina': 'South', 'South Carolina': 'South',
        'Virginia': 'South', 'District of Columbia': 'South', 'West Virginia': 'South',
        'Alabama': 'South', 'Kentucky': 'South', 'Mississippi': 'South',
        'Tennessee': 'South', 'Arkansas': 'South', 'Louisiana': 'South',
        'Oklahoma': 'South', 'Texas': 'South',
        
        # West
        'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West',
        'Montana': 'West', 'Nevada': 'West', 'New Mexico': 'West',
        'Utah': 'West', 'Wyoming': 'West', 'Alaska': 'West',
        'California': 'West', 'Hawaii': 'West', 'Oregon': 'West',
        'Washington': 'West',
        
        # Territories
        'Puerto Rico': 'South', 'Virgin Islands': 'South'
    }
    
    df['region'] = df['state'].map(region_mapping).fillna('Other')
    print(f"✓ Created 'region' - grouped states into Census regions")
    print(f"  Regions: {df['region'].unique().tolist()}")
    
    # Region target encoding
    df['region_target_encoded'] = df.groupby('region')['price'].transform('mean')
    print(f"✓ Created 'region_target_encoded'")
    
    # ========================================
    # FIX #3: STATE MEDIAN PRICE
    # ========================================
    df['state_median_price'] = df.groupby('state')['price'].transform('median')
    print(f"✓ Created 'state_median_price' - state-level price anchor")
    
    # Keep old features for compatibility
    df['city_median_price'] = df.groupby('city')['price'].transform('median')
    df['zip_median_ppsf'] = df.groupby('zip_code')['price_per_sqft'].transform('median')
    print(f"✓ Kept legacy features for compatibility")
    
    # ========================================
    # ADDITIONAL FEATURES
    # ========================================
    df['price_vs_city'] = df['price'] / df['city_median_price']
    df['price_vs_state'] = df['price'] / df['state_median_price']
    df['log_price'] = np.log(df['price'])
    print(f"✓ Created price ratio features")
    
    final_cols = [
        'price',  # TARGET
        
        # Property features
        'bed', 'bath', 'house_size', 'acre_lot',
        
        # Engineered property features
        'price_per_sqft', 'total_rooms', 'bath_bed_ratio', 'size_per_bedroom',
        
        # 🔥 NEW: Geographic target encodings (THE FIX!)
        'city_target_encoded',
        'state_target_encoded', 
        'zip_target_encoded',
        'region_target_encoded',
        'state_median_price',
        
        # Legacy geographic features (keep for compatibility)
        'city_median_price',
        'zip_median_ppsf',
        'price_vs_city',
        'price_vs_state',
        
        # Original categorical columns (will be dropped after target encoding)
        'city', 'state', 'zip_code', 'region',
        
        # Other
        'log_price'
    ]

    final_cols = [col for col in final_cols if col in df.columns]
    df = df[final_cols]
    
    print(f"Selected {len(final_cols)} columns:")
    for col in final_cols:
        print(f"    - {col}")

    if 'acre_lot' in df.columns:
        df['acre_lot'] = df['acre_lot'].fillna(0)
        print(f"Filled acre_lot nulls with 0")
       
    df = df.replace([np.inf, -np.inf], np.nan) 
    before_null_drop = len(df)
    df = df.dropna()
    print(f"Dropped {before_null_drop - len(df):,} rows with remaining nulls")
    
    print("\n" + "=" * 60)
    print("FINAL DATASET STATS")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Min Price: ${df['price'].min():,.0f}")
    print(f"Max Price: ${df['price'].max():,.0f}")
    print(f"Mean: ${df['price'].mean():,.0f}")
    print(f"Median: ${df['price'].median():,.0f}")
    
    print(f"\n📍 Geographic Features Summary:")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Cities: {df['city'].nunique():,}")
    print(f"  Zip Codes: {df['zip_code'].nunique():,}")
    print(f"  Regions: {df['region'].nunique()}")
    
    # Show some target encoding examples
    print(f"\n💰 Target Encoding Examples:")
    state_prices = df.groupby('state')['state_target_encoded'].first().sort_values(ascending=False)
    print(f"  Most expensive states:")
    for state, price in state_prices.head(5).items():
        print(f"    {state:20s}: ${price:,.0f}")
    print(f"  Least expensive states:")
    for state, price in state_prices.tail(5).items():
        print(f"    {state:20s}: ${price:,.0f}")
    
    return df

# Run the cleaning
if __name__ == "__main__":
    # Clean the data
    df_clean = clean_housing_data('realtor-data.zip.csv')
    
    # Save cleaned data
    output_file = 'housing_data_cleaned.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\nSaved cleaned data to: {output_file}")
    
    # Also save compressed version (saves space)
    compressed_file = 'housing_data_cleaned.csv.gz'
    df_clean.to_csv(compressed_file, index=False, compression='gzip')
    print(f"Saved compressed version to: {compressed_file}")
    
    print("Ready for XGBoost training!")    