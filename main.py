import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


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


def plot_training_history(history, model_name, output_dir):
    print(f"Generating training history graphs...")
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
    # Route the save to the designated folder
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'))
    plt.close()


def evaluate_and_generate_metrics(model, dataloader, device, class_names, model_name, output_dir):
    print(f"Generating confusion matrix and advanced metrics...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating Setup"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 1. Generate Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Breed')
    plt.ylabel('True Breed')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # --- 2. Calculate Precision, Recall, F1 ---
    # We use average='weighted' to account for any slight imbalances in your cat breed datasets
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return precision, recall, f1


# =====================================================================
# 2. MAIN EXECUTION FUNCTION
# =====================================================================
def main():
    # --- Configuration ---
    DATA_DIR = "images/structured"
    NUM_CLASSES = 12
    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = 'efficientnet_b0'

    # --- Create Directory Structure ---
    # Creates a folder like "efficientnet_b0_10" in your current working directory
    OUTPUT_DIR = f"{MODEL_NAME}_{EPOCHS}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Selected model architecture: {MODEL_NAME}")
    print(f"All outputs will be saved to: folder '{OUTPUT_DIR}/'")

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
    epoch_times = []  # List to track seconds per epoch

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # Track Time
        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] ({epoch_duration:.1f}s) -> "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Save the model directly into our new dynamic folder
            save_path = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
            torch.save(model.state_dict(), save_path)
            print(f" *** Saved new best model to {OUTPUT_DIR}! ***")

    # =====================================================================
    # 3. FINAL REPORTING, EXPORT, & METRICS TEXT FILE
    # =====================================================================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - EXPORTING RESULTS")
    print("=" * 50)

    # 1. Generate Training Graph in folder
    plot_training_history(history, MODEL_NAME, OUTPUT_DIR)

    # 2. Generate Confusion Matrix & Extended Metrics (Using best saved weights)
    model.load_state_dict(torch.load(save_path))
    precision, recall, f1 = evaluate_and_generate_metrics(model, val_loader, DEVICE, CLASS_NAMES, MODEL_NAME,
                                                          OUTPUT_DIR)

    # 3. Calculate Time Average
    avg_sec_per_epoch = sum(epoch_times) / len(epoch_times)

    # 4. Generate the structured metrics .txt file
    txt_file_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_metrics.txt")
    with open(txt_file_path, "w") as text_file:
        text_file.write(f"--- Experiment Details ---\n")
        text_file.write(f"Model Architecture: {MODEL_NAME}\n")
        text_file.write(f"Total Epochs Run: {EPOCHS}\n")
        text_file.write(f"Best Weights Achieved at Epoch: {best_epoch}\n\n")

        text_file.write(f"--- Core Validation Metrics ---\n")
        text_file.write(f"Validation Accuracy: {best_val_acc:.2f}%\n")
        text_file.write(f"Validation Loss: {best_val_loss:.4f}\n\n")

        text_file.write(f"--- Advanced Classification Metrics (Weighted) ---\n")
        text_file.write(f"F1 Score: {f1:.4f}\n")
        text_file.write(f"Precision: {precision:.4f}\n")
        text_file.write(f"Recall: {recall:.4f}\n\n")

        text_file.write(f"--- Performance Metrics ---\n")
        text_file.write(f"Average sec/epoch: {avg_sec_per_epoch:.2f} seconds\n")

    print(f"\nAll tasks finished successfully. Check the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()