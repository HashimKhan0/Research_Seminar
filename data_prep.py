import pandas as pd
import numpy as np

# --- Define Annualization Factor ---
# We need to scale our 5-minute volatility to be annualized
# 1. Periods in a day: 6.5 hours (9:30-16:00) * 60 mins/hr = 390 mins
#    Number of 5-min periods per day = 390 / 5 = 78
# 2. Trading days in a year: ~252
# Annualization Factor = sqrt(Periods per Day * Days per Year)
# If your data is 24h (like FX/Crypto), change 78 to (60*24)/5 = 288
# For US equities, 78 * 252 is standard.
ANNUALIZATION_FACTOR = np.sqrt(78 * 252)
# This factor is approx 140.2

def calculate_realized_volatility(log_returns, window=5):
    """
    Calculates 5-minute realized volatility by summing squared log returns.
    `raw=True` ensures the rolling window has at least `window` periods.
    """
    # Use raw=True for speed, it passes a numpy array
    return log_returns.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=True
    )

def create_sequences(data, target, sequence_length=10):
    """
    Creates sequences of past data (X) and corresponding targets (y).
    X will be shape [samples, sequence_length]
    y will be shape [samples]
    """
    X = []
    y = []
    # We stop `sequence_length` steps before the end to ensure a target exists
    for i in range(len(data) - sequence_length):
        # The sequence of 10 past values
        X.append(data[i:(i + sequence_length)])
        # The target value that occurs *after* the sequence
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data(csv_path, sequence_length=10, train_split=0.8):
    """
    Main function to load CSV, filter, process, and save data.
    """
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        print("Please make sure your CSV file is in the same directory.")
        return

    print(f"Loaded {len(df)} rows from {csv_path}...")
    
    # Ensure 'timestamp' and 'market_status' columns exist
    if 'timestamp' not in df.columns or 'market_status' not in df.columns or 'close' not in df.columns:
        print("Error: CSV must contain 'timestamp', 'market_status', and 'close' columns.")
        return

    # Convert timestamp to datetime
    # Use errors='coerce' to turn any bad timestamps into NaT (Not a Time)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Drop any rows where timestamp conversion failed
    df = df.dropna(subset=['timestamp'])
    
    # Set as index
    df = df.set_index('timestamp').sort_index()

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: Index could not be converted to DatetimeIndex. Aborting.")
        print(f"Index type is: {type(df.index)}")
        return

    # 2. Filter for regular market hours
    # *** UPDATED ***: Based on your log, using 'Trading'
    market_status_string = 'Trading' 
    df_market = df[df['market_status'] == market_status_string].copy()
    
    if df_market.empty:
        print(f"Error: No rows found with market_status == '{market_status_string}'.")
        print("Please check the unique values in your 'market_status' column and update the script.")
        print(f"Found statuses in your file: {df['market_status'].unique()}")
        return
        
    print(f"Filtered down to {len(df_market)} '{market_status_string}' market rows...")

    # 3. Calculate 1-minute log returns
    # We group by date to prevent calculating returns across midnight/gaps
    # Use `df_market.index.floor('D')` for robust date grouping
    df_market['log_return'] = df_market.groupby(df_market.index.floor('D'))['close'].apply(
        lambda x: np.log(x) - np.log(x.shift(1))
    )
    
    # --- THIS IS THE FIX ---
    # The first log_return of each day is NaN. Fill it with 0.
    df_market['log_return'] = df_market['log_return'].fillna(0)
    # --- END FIX ---
    
    # 4. Calculate 5-minute Realized Volatility (RV) and Annualize it
    # We must also group by date here for the rolling window
    rv_5min_unscaled = df_market.groupby(df_market.index.floor('D'))['log_return'].apply(
        lambda x: calculate_realized_volatility(x, window=5)
    )
    # Re-align the grouped data back to the main dataframe
    df_market['rv_5min_annualized'] = rv_5min_unscaled.reset_index(level=0, drop=True) * ANNUALIZATION_FACTOR
    
    print(f"Calculated 5-min RV and annualized it (Factor={ANNUALIZATION_FACTOR:.2f}).")
    # This should now print a valid number, not 'nan'
    print(f"Example mean annualized vol: {df_market['rv_5min_annualized'].mean():.4f}")

    # 5. Create the target variable: next 5-min RV (also annualized)
    # This shifts the data *up* by 5 rows (5 minutes)
    df_market['target_rv_5min'] = df_market['rv_5min_annualized'].shift(-5)

    # 6. Clean data
    # We now have the feature ('rv_5min_annualized') and the target ('target_rv_5min')
    feature_col = 'rv_5min_annualized'
    target_col = 'target_rv_5min'
    df_processed = df_market[[feature_col, target_col]].dropna()
    
    if len(df_processed) < sequence_length * 2:
        print("Error: Not enough data after processing and cleaning NaNs to create sequences.")
        print(f"Total rows after dropna: {len(df_processed)}")
        return

    print(f"Processed data has {len(df_processed)} rows after cleaning NaNs.")

    # 7. Create sequences
    #    X will now have only 1 feature: [rv_5min_annualized]
    features_data = df_processed[feature_col].values
    target_data = df_processed[target_col].values
    
    X, y = create_sequences(features_data, target_data, sequence_length)
    
    # Reshape X for LSTM: [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f"Created {X.shape[0]} sequences of length {X.shape[1]} with {X.shape[2]} feature.")

    # 8. Split data
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 9. Save processed data
    output_file = 'processed_vol_data_pure_dl.npz'
    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # *** UPDATE THIS to match your CSV file name ***
    # I'll assume your file is 'final.csv' based on your log
    your_csv_filename = 'final.csv'
    prepare_data(csv_path=your_csv_filename)