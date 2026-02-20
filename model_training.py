import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Create output directory for saving model
os.makedirs("output", exist_ok=True)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')  # 4 classes
])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2
)
train_data = datagen.flow_from_directory(
    "Training",
    target_size = (224,224),
    batch_size = 32,
    class_mode = "categorical",
    subset = "training"
)

val_data = datagen.flow_from_directory(
    "Training",
    target_size = (224,224),
    batch_size = 32,
    class_mode = "categorical",
    subset = "validation"
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=20,
    validation_data = val_data
)


model.save("output/trained_model.h5")

from tensorflow.keras.models import load_model
model = load_model("output/trained_model.h5") 


datagen_for_testing = ImageDataGenerator(
    rescale = 1./255
)

test_data = datagen_for_testing.flow_from_directory(
    "Testing",
    target_size = (224,224),
    batch_size = 32,
    class_mode ="categorical",
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Get predictions
pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)

# True labels
y_true = test_data.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Detailed report
print(classification_report(y_true, y_pred))
