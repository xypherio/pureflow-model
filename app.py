import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load model artifacts
model = load('water_quality_model.joblib')
scaler = load('feature_scaler.joblib')
model_info = load('model_info.joblib')
feature_cols = model_info['feature_cols']
target_cols = model_info['target_cols']

api = Flask(__name__)
CORS(api)

@api.route('/api/water_quality_prediction', methods=['POST'])
def water_quality_prediction():
    """

    Expects a JSON payload with the following structure (sample values provided):
    {
        "inputs": [
            {
                "month_sin": 0.0,
                "month_cos": -1.0,
                "hour_sin": 0.0,
                "hour_cos": -1.0,
                "season": 2,
                "rain_binary": 1,
                "rain_intensity": 2,
                "rain_last_6h": 3,
                "rain_last_24h": 8,
                "pH_lag1": 7.2,
                "temperature_lag1": 25.5,
                "salinity_lag1": 35.1,
                "turbidity_lag1": 12.5,
                "pH_lag24": 7.1,
                "temperature_lag24": 24.8,
                "salinity_lag24": 35.0,
                "turbidity_lag24": 11.9
            }
        ]
    }

    On success, returns a JSON response with the following structure:
    {
        "status": "success",
        "message": "Prediction generated successfully.",
        "data": {
            "predictions": [
                {
                    "pH": <predicted_value>,
                    "temperature": <predicted_value>,
                    "salinity": <predicted_value>,
                    "turbidity": <predicted_value>
                }
            ]
        }
    }

    On error, returns a JSON response with the following structure:
    {
        "status": "error",
        "message": "<error_message>",
        "error": "<detailed_error_string>"
    }
    """
    try:
        data = request.json['inputs']
        input_df = pd.DataFrame(data)

        # Ensure all required feature columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required feature: {col}'
                }), 400

        # Preprocess the input data
        # The Random Forest model was trained on unscaled data, so we predict directly.
        
        # Make prediction
        prediction = model.predict(input_df[feature_cols])
        
        # Format the response
        response = []
        for i, pred in enumerate(prediction):
            pred_dict = {target: round(float(val), 4) for target, val in zip(target_cols, pred)}
            response.append(pred_dict)
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction generated successfully.',
            'data': {
                'predictions': response
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing the request.',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    api.run(port=8000, debug=True)
