import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Loading and cleaning data...")
df = pd.read_csv('C:/Users/yashg_t6wet39/Desktop/IDS/training/balanced_training.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)


features_to_use = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Flow Byts/s', 'Flow Pkts/s',
    'Init Fwd Win Byts', 'Init Bwd Win Byts'
]
X = df[features_to_use]
y = df['Label']


# We fit the scaler on the entire dataset so it learns the global min/max of our data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
print(f"Model Accuracy on Test Set: {model.score(X_test, y_test):.4f}")

print("Saving model and scaler to disk...")
joblib.dump(model, 'ids_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features_to_use, 'features.pkl')
print("Model and scaler saved successfully!")
print((df.columns))