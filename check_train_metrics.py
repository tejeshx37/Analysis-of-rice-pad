import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# Paths
model_path = "/Users/mtejeshx37/Analysis-of-rice-pad/rice_disease_final_2.keras"
train_dir = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/train"
IMG_SIZE = 192
BATCH = 32

print(f"Loading model from {model_path}...")
model = load_model(model_path)

print("Preparing partial training data...")
# Use a validation split to just grab a small chunk (10%) of training data for a quick check
train_datagen = ImageDataGenerator(validation_split=0.1) 

# We use 'subset="validation"' here just to get that 10% chunk of the training folder
# This acts as our "Train Subset" for evaluation
train_subset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation", 
    seed=42,
    shuffle=False
)

print("Evaluating on training subset (approx 10% of train data)...")
results = model.evaluate(train_subset, verbose=1)
print(f"\nTraining Accuracy (Subset): {results[1]*100:.2f}%")
print(f"Training Loss (Subset): {results[0]:.4f}")
