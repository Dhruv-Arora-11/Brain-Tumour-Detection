import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 1. SETTINGS & PATHS
# We point to the parent folder "Training" which contains the pituitary folder and others
data_path = r"C:\Users\maina\Downloads\archive (5)\Training"
batch_size = 32
img_height = 180
img_width = 180

# 2. LOAD DATA
# This automatically labels your images based on the folder names
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Identify class names (pituitary, glioma, etc.)
class_names = train_ds.class_names
print(f"Detected Classes: {class_names}")

# 3. CONFIGURE FOR PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. BUILD THE MODEL (CNN)
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), # Normalization
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names)) # Output layer matching number of tumor types
])

# 5. COMPILE
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 6. TRAIN
epochs = 2
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 7. SAVE THE MODEL
model.save("brain_tumor_model.h5")
print("Model saved successfully!")