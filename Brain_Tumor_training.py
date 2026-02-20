import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- SETTINGS ---
TRAIN_DIR = r"C:\Users\maina\OneDrive\Desktop\Capstone _Project\archive (5)\Training"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("🚀 Loading Training Dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"✅ Found classes: {class_names}")

# Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- ARCHITECTURE: U-Net inspired CNN ---
def build_model():
    inputs = layers.Input(shape=(128, 128, 3))
    x = layers.Rescaling(1./255)(inputs)
    
    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Bottleneck
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name="bottleneck")(x)
    
    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    return models.Model(inputs, outputs)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n🔥 Starting Training...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save the model for the testing script
model.save("brain_tumor_model.h5")
print("✅ Model saved as brain_tumor_model.h5")