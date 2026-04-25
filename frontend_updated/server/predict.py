import sys
import json
import os
import random

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print(json.dumps({"error": "Image file not found"}))
    sys.exit(1)

try:
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_model.h5')

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        raise FileNotFoundError("No valid model file")

    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))

    result = {
        "className": CLASS_NAMES[predicted_index],
        "classIndex": predicted_index,
        "confidence": float(prediction[0][predicted_index]),
        "probabilities": [float(p) for p in prediction[0]],
        "mock": False
    }

except Exception:
    probs = [random.uniform(0.01, 0.15) for _ in range(4)]
    winner = random.randint(0, 3)
    probs[winner] = random.uniform(0.7, 0.95)
    total = sum(probs)
    probs = [p / total for p in probs]
    predicted_index = winner

    result = {
        "className": CLASS_NAMES[predicted_index],
        "classIndex": predicted_index,
        "confidence": probs[predicted_index],
        "probabilities": probs,
        "mock": True
    }

print(json.dumps(result))
