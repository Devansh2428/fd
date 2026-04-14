from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Farmer Friend API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [
            data["N"],
            data["P"],
            data["K"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]

        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)[0]

        # Fake scores (for UI demo)
        crops = [
            {"name": prediction, "score": 95},
            {"name": "Rice", "score": 88},
            {"name": "Maize", "score": 80}
        ]

        return jsonify({"crops": crops})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)