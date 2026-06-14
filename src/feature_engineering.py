import pandas as pd
import numpy as np

# Define sensor columns
SENSOR_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

def sort_and_reset(df):
    """
    Sort DataFrame by index and reset index.
    
    Parameters: df (pd.DataFrame): Input DataFrame
    Returns: pd.DataFrame: Sorted and reset DataFrame
    """
    df = df.sort_index()
    df = df.reset_index(drop=True)
    return df

def generate_rolling_features(df, window=5):
    """
    Generate rolling mean, standard deviation and variance
    for all sensor columns in a single pipeline.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        window (int): Rolling window size (default=5)
    Returns:
        pd.DataFrame: DataFrame with all rolling features added
    """
    for col in SENSOR_COLUMNS:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
        df[f'{col}_rolling_var'] = df[col].rolling(window=window).var()

    # Drop NaN rows created by rolling
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Print final feature list
    rolling_features = [col for col in df.columns if 'rolling' in col]
    print(f"Output shape after rolling features (window={window}): {df.shape}")
    print(f"Total rolling features added: {len(rolling_features)}")
    print(f"Rolling feature list: {rolling_features}")

    return df

# Test function end to end
if __name__ == "__main__":
    df = pd.read_csv('../data/ai4i2020.csv')
    df = sort_and_reset(df)
    df = generate_rolling_features(df, window=5)
    print("Feature engineering pipeline completed successfully!")

# ============================================
# MODULE SUMMARY
# ============================================
# 1. SENSOR_COLUMNS — standardized list of all 5 sensor features
# 2. sort_and_reset() — sorts and resets DataFrame index
# 3. generate_rolling_features() — generates rolling mean, std
#    and variance for all sensor columns in single pipeline
# ============================================

print("Feature engineering module loaded successfully")