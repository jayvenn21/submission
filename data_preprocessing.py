# data_preprocessing.py
import pandas as pd
import numpy as np

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def engineer_features(df):
    # Basic price and volume features
    df['price_diff'] = df['close'] - df['open']
    df['high_low_diff'] = df['high'] - df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Lag features for past minute data
    df['price_diff_lag'] = df['price_diff'].shift(1)
    df['volume_lag'] = df['volume'].shift(1)
    
    # Rolling statistics for capturing trends
    df['rolling_mean_close'] = df['close'].rolling(window=3).mean()
    df['rolling_std_close'] = df['close'].rolling(window=3).std()
    df['rolling_mean_volume'] = df['volume'].rolling(window=3).mean()
    df['rolling_std_volume'] = df['volume'].rolling(window=3).std()
    
    # Technical Indicators
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['macd'] = compute_macd(df['close'])
    df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger_bands(df['close'])
    
    # Time-based features
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Drop NaNs introduced by feature engineering
    df = df.dropna()
    return df

def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, slow=26, fast=12, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

train_df, test_df = load_data()
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)
