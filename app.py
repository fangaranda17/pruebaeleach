from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('random_forest_iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
