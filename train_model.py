import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# 2. Config
# ---------------------------
DATA_DIR = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset_Split"
MODEL_SAVE_PATH = "best_resnet50_rice.pth"

BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
NUM_CLASSES = 20

# Training/runtime tweaks
LOG_EVERY = 50  # print progress every N batches
USE_PRETRAINED = True  # set to False to avoid any weight download delays
WARMUP_ON_DEVICE = True  # run a quick warmup pass on the selected device

# Apple Silicon (MPS) optimization for MacBook Air
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------------------
# 3. Image Transforms
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# 4. Datasets & Loaders
# ---------------------------
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=val_test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_dataset.classes)
print(f"Train/Val/Test batches per epoch: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

# ---------------------------
# 5. Model (ResNet50)
# ---------------------------
# Using the standard ResNet50 with optional pre-trained weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if USE_PRETRAINED else None)

# Freeze backbone initially
for param in model.parameters():
    param.requires_grad = False

# Replace the fully connected layer for our 20 classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)
if WARMUP_ON_DEVICE:
    print("Warming up model on device...")
    model.train()
    # Forward-only warmup
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224, device=DEVICE)
        _ = model(dummy)
    # Temporary backward warmup across the network
    was_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = True
    dummy = torch.randn(2, 3, 224, 224, device=DEVICE)
    out = model(dummy)
    out.sum().backward()
    model.zero_grad()
    for p, flag in zip(model.parameters(), was_flags):
        p.requires_grad = flag
    print("Warmup complete.")

# ---------------------------
# 6. Loss & Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
# Only train the newly added head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ---------------------------
# 7. Train & Validate
# ---------------------------
def train_one_epoch(model, loader, epoch_idx=0):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == 0 or (batch_idx + 1) == len(loader):
            avg = running_loss / (batch_idx + 1)
            print(f"  epoch {epoch_idx+1} batch {batch_idx+1}/{len(loader)} - avg_loss {avg:.4f}")

    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    return accuracy_score(targets, preds)

# ---------------------------
# 8. Training Loop
# ---------------------------
best_val_acc = 0

print("\nStarting Training...")
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, epoch)
    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Accuracy: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nBest Validation Accuracy:", best_val_acc)
print("Model saved at:", MODEL_SAVE_PATH)

# ---------------------------
# 9. Final Test Accuracy
# ---------------------------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_acc = evaluate(model, test_loader)
print("Final Test Accuracy:", test_acc)

