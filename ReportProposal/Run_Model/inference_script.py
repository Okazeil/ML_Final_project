from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('my_model.h5')

# Load the pre-fitted scaler
scaler = joblib.load('scaler.pkl')

# Function to map numeric predictions to labels
def map_label(prediction):
    return "working" if prediction == 1 else "not working"

# Function to preprocess input data
def preprocess_input(data):
    return scaler.transform(data)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array(data['input'])
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Preprocess features using the scaler
        features_scaled = preprocess_input(features)

        # Make prediction
        numeric_predictions = model.predict(features_scaled)

        # Convert numeric predictions to string labels
        predictions = [map_label(int(pred[0])) for pred in numeric_predictions]

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)