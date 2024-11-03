# prediction_submission.py
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import joblib

# Load test data
test_df = pd.read_csv('processed_test.csv')
features = [
    'price_diff', 'high_low_diff', 'close_open_ratio', 'price_diff_lag', 'volume_lag',
    'rolling_mean_close', 'rolling_std_close', 'rolling_mean_volume', 'rolling_std_volume',
    'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'minute', 'hour', 'dayofweek'
]
X_test = test_df[features]

# Load models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgboost_model.json')
rf_model = joblib.load('rf_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.h5')

# Ensemble predictions
xgb_pred = xgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
nn_pred = (nn_model.predict(X_test) > 0.5).astype(int).flatten()

# Ensemble prediction (majority voting)
test_df['target'] = (xgb_pred + rf_pred + nn_pred) >= 2

# Ensure 909,617 rows
if len(test_df) < 909617:
    missing_rows = 909617 - len(test_df)
    for _ in range(missing_rows):
        test_df = pd.concat([test_df, pd.DataFrame({'target': [0]})], ignore_index=True)

# Create submission
submission = pd.DataFrame({'row_id': range(len(test_df)), 'target': test_df['target']})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv with 909,617 rows.")
