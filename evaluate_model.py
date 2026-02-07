import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# Paths
model_path = "/Users/mtejeshx37/Analysis-of-rice-pad/rice_disease_final_2.keras"
val_dir = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/val"
IMG_SIZE = 192
BATCH = 32

# Load Model
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Prepare Validation Data (Test Set)
print("Preparing validation data...")
# Model trained on [0, 255] range (EfficientNet default), so no rescaling needed if preprocess_input is identity.
val_gen = ImageDataGenerator()

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False # Important for evaluation to match predictions with labels
)

# Predict
print("Running predictions...")
predictions = model.predict(val_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Evaluation Metrics
print("\n--- Evaluation Report ---")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Overall Accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
