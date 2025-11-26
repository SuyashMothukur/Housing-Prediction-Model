import pandas as pd
import numpy as np

def clean_housing_data(filepath='realto-data.zip.csv'):
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Remove rows with Na or 0 prices
    df = df[df['price'].notna() & (df['price'] > 0)]
    print(f"Removed invalid prices: {len(df):,} rows remaining")
    
    # Remove outlier prices
    df = df[df['price'] >= 5000]
    df = df[df['price'] <= 20000000]
    print(f"Removed outliers prices: {len(df):,} rows remaning")
    
    return df

# Run the cleaning

if __name__ == "__main__":
    # Clean the data
    df_clean = clean_housing_data('realtor-data.zip.csv')
    
    # Save cleaned data
    output_file = 'housing_data_cleaned.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n💾 Saved cleaned data to: {output_file}")
    
    # Also save compressed version (saves space)
    compressed_file = 'housing_data_cleaned.csv.gz'
    df_clean.to_csv(compressed_file, index=False, compression='gzip')
    print(f"💾 Saved compressed version to: {compressed_file}")
    
    print("\n✨ Ready for XGBoost training!")    