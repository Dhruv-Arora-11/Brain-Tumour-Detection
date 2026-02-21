from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

model = load_model("trained_model.h5")

def preprocess(image):
    image = image.convert("RGB")   # IMPORTANT
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    processed = preprocess(image)

    prediction = model.predict(processed)
    result = np.argmax(prediction)

    return jsonify({"prediction": int(result)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)