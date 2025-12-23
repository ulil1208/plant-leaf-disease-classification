import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from cnn_model import build_mobilenetv2_model

# =============================
# CONFIG
# =============================
DATASET_DIR = "../data/raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# =============================
# DATA GENERATOR
# =============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_generator.num_classes

# =============================
# BUILD MODEL
# =============================
model = build_mobilenetv2_model(num_classes)
model.summary()

# =============================
# TRAINING
# =============================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# =============================
# EVALUATION
# =============================
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = predictions.argmax(axis=1)

print("\nClassification Report:")
print(
    classification_report(
        test_generator.classes,
        y_pred,
        target_names=test_generator.class_indices.keys()
    )
)

# =============================
# SAVE MODEL
# =============================
model.save("model_mobilenetv2.h5")
print("Model saved as model_mobilenetv2.h5")

