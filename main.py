import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# --- NEW IMPORTS FOR PLOTTING & METRICS ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# =====================================================================
# 1. HELPER FUNCTIONS
# =====================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, (correct / total) * 100


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, (correct / total) * 100


# --- NEW FUNCTION: Plot Progression ---
def plot_training_history(history, model_name):
    print(f"Generating training history graphs for {model_name}...")
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()


# --- NEW FUNCTION: Confusion Matrix ---
def generate_confusion_matrix(model, dataloader, device, class_names, model_name):
    print(f"Generating confusion matrix for {model_name}...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating for CM"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Breed')
    plt.ylabel('True Breed')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()


# =====================================================================
# 2. MAIN EXECUTION FUNCTION
# =====================================================================
def main():
    # --- Configuration ---
    DATA_DIR = "images/structured"
    NUM_CLASSES = 12
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = 'efficientnet_b0'

    print(f"Using device: {DEVICE}")
    print(f"Selected model architecture: {MODEL_NAME}")

    # --- Data Loading ---
    IMG_SIZE = 288 if MODEL_NAME == 'efficientnet_b2' else 224

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Extract class names for the confusion matrix later
    CLASS_NAMES = train_dataset.classes

    # --- Model Setup ---
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Metrics Tracking Setup ---
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] ({time.time() - start_time:.1f}s) -> "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_{MODEL_NAME}.pth")
            print(f" *** Saved new best model! ***")

    # =====================================================================
    # 3. FINAL REPORTING & VISUALIZATION
    # =====================================================================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best Model Achieved at Epoch {best_epoch}:")
    print(f" - Validation Accuracy: {best_val_acc:.2f}%")
    print(f" - Validation Loss: {best_val_loss:.4f}")

    # Generate Training Graph
    plot_training_history(history, MODEL_NAME)

    # Generate Confusion Matrix (Using best saved weights)
    print("\nLoading best weights for final evaluation...")
    model.load_state_dict(torch.load(f"best_{MODEL_NAME}.pth"))
    generate_confusion_matrix(model, val_loader, DEVICE, CLASS_NAMES, MODEL_NAME)

    print("\nAll tasks finished. Check your directory for:")
    print(f"1. best_{MODEL_NAME}.pth")
    print(f"2. {MODEL_NAME}_training_history.png")
    print(f"3. {MODEL_NAME}_confusion_matrix.png")


if __name__ == "__main__":
    main()