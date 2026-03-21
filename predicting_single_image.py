from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from preprocessing import custom_preprocessor

import numpy as np

img_path = "t1.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = custom_preprocessor(img_array)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

model = load_model("trained_model.keras")
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)

print(predicted_index)