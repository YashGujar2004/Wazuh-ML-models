# train_model_V4_incremental_fixed.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight 
import joblib

print("Starting incremental training process with pre-calculated class weights...")

# --- 1. SETUP ---
csv_file_path = 'C:/Users/yashg_t6wet39/Desktop/IDS/training/biggest.csv'
chunk_size = 50000
features_to_use = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Header Len', 'Bwd Header Len',
    'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'Init Fwd Win Byts', 'Init Bwd Win Byts'
]

# --- 2. FIRST PASS: CALCULATE CLASS DISTRIBUTION ---
print("First pass: Calculating class distribution from the entire dataset...")
class_counts = pd.Series([0, 0], index=[0, 1])
chunk_iterator_for_counts = pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False, usecols=['Label'])

for chunk in chunk_iterator_for_counts:
    # Convert labels and count them
    labels = chunk['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    class_counts = class_counts.add(labels.value_counts(), fill_value=0)

# Create a dummy array of labels representing the full dataset distribution
y_full_dataset_dummy = np.concatenate([
    np.zeros(int(class_counts[0])), 
    np.ones(int(class_counts[1]))
])

# Compute the weights
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_full_dataset_dummy)
# Convert weights to a dictionary format that SGDClassifier understands
calculated_weights = {0: weights[0], 1: weights[1]}
print(f"Calculated class weights: {calculated_weights}")


# --- 3. INITIALIZE SCALER AND MODEL WITH CUSTOM WEIGHTS ---
scaler = StandardScaler()
# Pass the pre-calculated dictionary to the class_weight parameter
model = SGDClassifier(loss='log_loss', random_state=42, class_weight=calculated_weights)


# --- 4. SECOND PASS: LOOP THROUGH DATA AND TRAIN INCREMENTALLY ---
print("\nSecond pass: Reading CSV and training in chunks...")
chunk_iterator_for_training = pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False)
total_chunks = (len(y_full_dataset_dummy) // chunk_size) + 1
chunk_count = 0

for chunk in chunk_iterator_for_training:
    chunk_count += 1
    print(f"  - Processing chunk {chunk_count}/{total_chunks}...")
    
    # (The rest of the cleaning and training loop is the same as before)
    for col in features_to_use:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    if chunk.empty:
        continue
        
    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    X_chunk = chunk[features_to_use]
    y_chunk = chunk['Label']
    
    scaler.partial_fit(X_chunk)
    X_chunk_scaled = scaler.transform(X_chunk)
    
    model.partial_fit(X_chunk_scaled, y_chunk, classes=np.array([0, 1]))

# --- 5. SAVE THE MODEL ---
print("\nTraining complete!")
print("Saving model and scaler to disk...")
joblib.dump(model, 'ids_model_v3.pkl')
joblib.dump(scaler, 'scaler_v3.pkl')
joblib.dump(features_to_use, 'features_v3.pkl')
print("Model and scaler saved successfully!")