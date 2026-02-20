import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# ==========================================
# 1. CONNECT TO THE DATASET
# ==========================================
# Point this to your Training folder (containing glioma, pituitary, etc.)
DATASET_PATH = r"C:\Users\maina\Downloads\archive (5)\Training"

print(f"🚀 Connecting to Brain MRI dataset at: {DATASET_PATH}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names
print(f"✅ Classes found: {class_names}")

# Store file paths for the random dashboard generator
val_file_paths = val_ds.file_paths 

# Optimization for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. BUILD THE U-NET MODEL
# ==========================================
def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)

    # Encoder (Contracting Path)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="bottleneck_conv")(p2)

    # Simple Decoder for classification/detection
    f = layers.Flatten()(b)
    d = layers.Dense(64, activation='relu')(f)
    outputs = layers.Dense(num_classes, activation='softmax')(d)

    return models.Model(inputs, outputs)

model = build_unet((128, 128, 3), len(class_names))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. START TRAINING
# ==========================================
print("\n🔥 TRAINING BRAIN TUMOR MODEL...")
history = model.fit(train_ds, validation_data=val_ds, epochs=2)

# ==========================================
# 4. GRAD-CAM DASHBOARD (EXPLAINABLE AI)
# ==========================================
print("\n🔍 Generating MRI Focus Dashboard...")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Dashboard Plotting
num_samples = 4
plt.figure(figsize=(12, 12))

for i in range(num_samples):
    # Select random image from validation
    img_path = random.choice(val_file_paths)
    img = cv2.imread(img_path)
    img_res = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_res, axis=0)
    
    # AI Prediction
    prediction = model.predict(img_array / 255.0)
    pred_label = class_names[np.argmax(prediction)]

    # Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, "bottleneck_conv")
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (128, 128))
    
    overlay = cv2.addWeighted(img_res, 0.6, heatmap, 0.4, 0)

    # Subplots
    plt.subplot(num_samples, 2, i*2 + 1)
    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.title(f"Original MRI {i+1}")
    plt.axis('off')

    plt.subplot(num_samples, 2, i*2 + 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"AI Detection: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
print("✅ Project Execution Complete!")






from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 5. PROFESSIONAL PERFORMANCE EVALUATION
# ==========================================
print("\n📊 Generating Professional Performance Report...")

# 1. Get all true labels and predictions from the validation set
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# 2. Print the Professional Classification Report
# This shows Precision, Recall, and F1-Score for EACH tumor type
print("\n" + " " * 10 + "MEDICAL DIAGNOSIS REPORT")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=class_names))
print("=" * 50)

# 3. Plot a Professional Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('AI Prediction')
plt.ylabel('Actual Doctor Diagnosis')
plt.title('Brain Tumor Confusion Matrix')
plt.show()