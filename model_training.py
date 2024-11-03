# model_training.py
import pandas as pd
import numpy as np  # <-- Add this line to import numpy
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load processed data
train_df = pd.read_csv('processed_train.csv')

# Select features and target
features = [
    'price_diff', 'high_low_diff', 'close_open_ratio', 'price_diff_lag', 'volume_lag',
    'rolling_mean_close', 'rolling_std_close', 'rolling_mean_volume', 'rolling_std_volume',
    'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'minute', 'hour', 'dayofweek'
]
X = train_df[features]
y = train_df['target']

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

# Initialize base models
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=100, subsample=0.8, colsample_bytree=0.8)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5)

# Define neural network model
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train XGBoost
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    
    # Train Random Forest
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    
    # Train Neural Network
    nn_model = create_nn_model(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    nn_pred = (nn_model.predict(X_val) > 0.5).astype(int).flatten()
    
    # Ensemble predictions (simple majority vote)
    final_pred = (xgb_pred + rf_pred + nn_pred) >= 2
    f1 = f1_score(y_val, final_pred, average='macro')
    f1_scores.append(f1)

print(f"Average Macro-Averaged F1 Score: {np.mean(f1_scores)}")

# Save models
xgb_model.save_model('xgboost_model.json')
rf_model.fit(X, y)  # Train on full dataset
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
nn_model.save('nn_model.h5')
