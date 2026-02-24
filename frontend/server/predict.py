import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

model = load_model('../trained_model.h5')

img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array, verbose=0)
predicted_index = int(np.argmax(prediction[0]))

result = {
    "className": CLASS_NAMES[predicted_index],
    "classIndex": predicted_index,
    "confidence": float(prediction[0][predicted_index]),
    "probabilities": [float(p) for p in prediction[0]]
}

print(json.dumps(result))
