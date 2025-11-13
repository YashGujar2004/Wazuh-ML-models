# send_request.py

import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.metrics import classification_report, confusion_matrix

TEST_CSV_FILE = 'test_V3.csv'

API_URL = 'http://127.0.0.1:5000/predict'

# print(f"Loading test data from {TEST_CSV_FILE}...")
# try:
#     features = joblib.load('features.pkl')
    
#     df_test = pd.read_csv(TEST_CSV_FILE)
#     # --- START OF FIX ---
#     # Convert all feature columns to numeric, coercing errors
#     # This will turn any non-numeric values (like text) into NaN (Not a Number)
#     for col in features:
#         df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
#     # --- END OF FIX ---
#     # Perform the EXACT SAME cleaning as in the training script
#     df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df_test.dropna(inplace=True)

#     # For a quick test, let's take a random sample of 1000 rows
#     # You can comment this out to test the entire file
#     if len(df_test) > 1000:
#         df_test = df_test.sample(n=10000, random_state=42)
        
#     y_true = df_test['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
#     X_test = df_test[features]

# except FileNotFoundError:
#     print(f"Error: Make sure '{TEST_CSV_FILE}', 'features.pkl' are in the same directory.")
#     exit()

# # --- 3. SEND DATA TO API FOR PREDICTION ---
# json_data = X_test.to_json(orient='records')
# headers = {'Content-Type': 'application/json'}

# print(f"Sending {len(X_test)} records to the API for prediction...")
# response = requests.post(API_URL, data=json_data, headers=headers)

# if response.status_code == 200:
#     response_json = response.json()
#     predictions = response_json['predictions']
    
#     print("\n--- Evaluation on New Test Data ---")
    
#     # Compare the API's predictions with the true labels from the file
#     print(classification_report(y_true, predictions, target_names=['Benign', 'Attack']))
    
#     print("--- Confusion Matrix ---")
#     print(confusion_matrix(y_true, predictions))
    
# else:
#     print(f"\nError: Failed to get a response. Status code: {response.status_code}")
#     print(f"Response text: {response.text}")


BATCH_SIZE = 1000 

# --- 2. LOAD AND PREPARE THE ENTIRE TEST DATA ---
print(f"Loading and cleaning test data from {TEST_CSV_FILE}...")
try:
    features = joblib.load('features_v2.pkl')
    df_test_full = pd.read_csv(TEST_CSV_FILE, low_memory=False)
    
    for col in features:
        df_test_full[col] = pd.to_numeric(df_test_full[col], errors='coerce')
    

    df_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_full.dropna(inplace=True)
    
    if len(df_test_full) > 1000:
        df_test_full = df_test_full.sample(n=1000, random_state=42)
    
    y_true = df_test_full['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    X_test = df_test_full[features]

except FileNotFoundError:
    print(f"Error: Make sure '{TEST_CSV_FILE}', 'features_v2.pkl' are in the same directory.")
    exit()

# --- 3. SEND DATA TO API IN BATCHES ---
all_predictions = []
print(f"Sending {len(X_test)} records to the API in batches of {BATCH_SIZE}...")

for i in range(0, len(X_test), BATCH_SIZE):
    # Get the current batch of data
    batch_df = X_test.iloc[i:i+BATCH_SIZE]
    
    json_data = batch_df.to_json(orient='records')
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=json_data, headers=headers)
    
    if response.status_code == 200:
        batch_predictions = response.json()['predictions']
        all_predictions.extend(batch_predictions)
    else:
        print(f"Error on batch {i//BATCH_SIZE + 1}. Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        break

# --- 4. EVALUATE THE COMBINED RESULTS ---
if len(all_predictions) == len(y_true):
    print("\n--- Evaluation on New Test Data ---")
    print(classification_report(y_true, all_predictions, target_names=['Benign', 'Attack']))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_true, all_predictions))
else:
    print("\nCould not perform evaluation due to an error during prediction.")
