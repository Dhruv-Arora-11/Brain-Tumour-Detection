import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# --- SETTINGS ---
TEST_DIR = r"C:\Users\maina\OneDrive\Desktop\Capstone _Project\archive (5)\Testing"
model = load_model("brain_tumor_model.h5")
print("🧠 Model Loaded Successfully!")

# Load Test Data
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=(128, 128), batch_size=32, shuffle=False
)
class_names = test_ds.class_names
test_file_paths = test_ds.file_paths

# --- GRAD-CAM ENGINE ---
def get_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==========================================
# 1. GENERATE HEATMAP DASHBOARD
# ==========================================
plt.figure(figsize=(12, 8))
for i in range(4):
    path = random.choice(test_file_paths)
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (128, 128))
    
    # Prediction
    img_batch = np.expand_dims(img_res, axis=0) / 255.0
    pred = model.predict(img_batch, verbose=0)
    label = class_names[np.argmax(pred)]
    
    # Heatmap
    hm = get_gradcam(img_batch, model, "bottleneck")
    hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(hm, (128, 128))), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_res, 0.6, hm, 0.4, 0)
    
    plt.subplot(2, 2, i+1)
    plt.imshow(overlay)
    plt.title(f"AI Prediction: {label}")
    plt.axis('off')

plt.suptitle("Explainable AI: Tumor Localization Heatmaps")
plt.show()

# ==========================================
# 2. CONFUSION MATRIX & ACCURACY
# ==========================================
y_true, y_pred = [], []
for imgs, lbls in test_ds:
    p = model.predict(imgs, verbose=0)
    y_true.extend(lbls.numpy())
    y_pred.extend(np.argmax(p, axis=1))

print("\n" + "="*30 + "\nMEDICAL REPORT\n" + "="*30)
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Industry Standard Confusion Matrix")
plt.ylabel('Doctor Diagnosis')
plt.xlabel('AI Prediction')
plt.show()