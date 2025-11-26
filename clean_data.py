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