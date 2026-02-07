import tensorflow as tf
import os
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    RandomFlip, RandomRotation
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# ===============================
# SPEED SETTINGS
# ===============================
mixed_precision.set_global_policy("mixed_float16")

IMG_SIZE = 192          # smaller = faster
BATCH_SIZE = 64         # increase if memory allows
EPOCHS_HEAD = 8         # classifier training
EPOCHS_FINE = 5         # fine-tuning
AUTOTUNE = tf.data.AUTOTUNE

train_dir = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/train"
val_dir   = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/val"

# ===============================
# LOAD DATA (FAST)
# ===============================
print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ===============================
# PREPROCESSING
# ===============================
def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ===============================
# CLASS WEIGHTS (SAFE)
# ===============================
print("Computing class weights...")

y_train = np.concatenate([
    np.argmax(y.numpy(), axis=1)
    for _, y in train_ds.unbatch().batch(1024)
])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ===============================
# MODEL
# ===============================
print("Building model...")

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
], name="augmentation")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

# IMPORTANT: force float32 output for mixed precision
outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    ModelCheckpoint(
        "rice_disease_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ===============================
# PHASE 1 – TRAIN HEAD
# ===============================
print("\n🚀 Training classifier head...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===============================
# PHASE 2 – FINE TUNING (FAST)
# ===============================
print("\n🔥 Fine-tuning last layers...")

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===============================
# SAVE FINAL MODEL
# ===============================
model.save("rice_disease_final_2.keras")
print("\n✅ Training completed successfully!")
