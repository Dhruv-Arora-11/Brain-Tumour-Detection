from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from PIL import Image
from flask_cors import CORS
from preprocessing import custom_preprocessor

app = Flask(__name__)

# Force CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# --- THE FIX ---
# Create a custom wrapper for the Dense layer that strips out the bad keyword
class PatchedDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None) # Remove the troublemaker
        super().__init__(**kwargs)

# Load model using the custom object to intercept the layer building
try:
    model = load_model(
        "trained_model.keras", 
        custom_objects={'Dense': PatchedDense}, 
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load trained_model.keras")


def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = custom_preprocessor(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # (1,224,224,3)
    
    return image

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file)

    processed = preprocess(image)
    prediction = model.predict(processed)
    class_index = int(np.argmax(prediction))

    return jsonify({"prediction": class_index})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)