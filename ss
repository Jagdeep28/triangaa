import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

def prepare_data(sequence, window_size=10):
    X, y = [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size])
    return np.array(X), np.array(y)

def predict_next_values(sequence):
    if len(sequence) != 100:
        return {"error": "Please provide exactly 100 values!"}
    
    X, y = prepare_data(sequence, window_size=10)
    model = LogisticRegression()
    model.fit(X, y)
    
    last_window = np.array(sequence[-10:]).reshape(1, -1)
    predictions = [model.predict(last_window)[0] for _ in range(10)]
    return ['B' if p == 1 else 'S' for p in predictions]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("sequence", [])
    if not isinstance(data, list) or len(data) != 100:
        return jsonify({"error": "Invalid input. Provide a list of exactly 100 values (B/S)."}), 400
    
    binary_sequence = [1 if v == 'B' else 0 for v in data]
    predictions = predict_next_values(binary_sequence)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
