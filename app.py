from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('random_forest_iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
   try:
        data = request.get_json()
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Log the error and return a custom error message
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
