# predict_app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model, scaler, and feature list
model = joblib.load('ids_model_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')
features = joblib.load('features_v2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    df = pd.DataFrame(json_data)
    df = df[features]

    # Scale the new data using the loaded scaler
    scaled_data = scaler.transform(df)

    # Make predictions using the loaded model
    predictions = model.predict(scaled_data)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
