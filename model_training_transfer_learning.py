from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)


for layer in base_model.layers:
    layer.trainable = False
    
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2,
    rotation_range=30,        # rotation
    width_shift_range=0.1,    # horizontal shift
    height_shift_range=0.1,   # vertical shift
    shear_range=0.1,          # shear transform
    zoom_range=0.2,           # zoom
    fill_mode='nearest'
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
    validation_data=val_data
)


for layer in base_model.layers[-20:]:
    layer.trainable = True