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

# --- NEW IMPORTS FOR GRAD-CAM ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# HELPER FUNCTIONS

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

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return precision, recall, f1


# GRAD-CAM HELPER FUNCTIONS
def get_target_layer(model, model_name):
    """Automatically finds the correct final CNN layer based on the timm architecture."""
    if 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'efficientnet' in model_name:
        return [model.conv_head]
    elif 'convnext' in model_name:
        return [model.stages[-1].blocks[-1]]
    else:
        # Fallback
        return [list(model.children())[-2]]


def generate_gradcam_samples(model, dataloader, device, class_names, model_name, output_dir, num_samples=5):
    print(f"Generating Explainability (Grad-CAM) Visualizations...")
    model.eval()
    target_layers = get_target_layer(model, model_name)

    # Initialize the CAM object
    # use_cuda must be boolean, checking if device is cuda
    cam = GradCAM(model=model, target_layers=target_layers)

    # Get a single batch of validation images
    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))

    for i in range(num_samples):
        input_tensor = images[i].unsqueeze(0)
        true_label_idx = labels[i].item()

        # Generate the CAM mask (returns a NumPy array)
        # targets=None automatically uses the highest scoring category
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        # Un-normalize the image so we can display it correctly
        img_show = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_show = std * img_show + mean
        img_show = np.clip(img_show, 0, 1)

        # Overlay the heatmap on the image
        visualization = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

        # Plot Original
        axes[i, 0].imshow(img_show)
        axes[i, 0].set_title(f"True: {class_names[true_label_idx]}")
        axes[i, 0].axis('off')

        # Plot Grad-CAM
        with torch.no_grad():
            pred_idx = model(input_tensor).argmax(dim=1).item()

        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Grad-CAM (Pred: {class_names[pred_idx]})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_gradcam_samples.png'))
    plt.close()

def main():
    # --- Configuration ---
    DATA_DIR = "images/structured"
    NUM_CLASSES = 12
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = 'efficientnet_b2'

    # --- Create Directory Structure ---
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
    epoch_times = []

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)

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

            save_path = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
            torch.save(model.state_dict(), save_path)
            print(f" *** Saved new best model to {OUTPUT_DIR}! ***")

    # FINAL REPORTING, EXPORT, & METRICS
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - EXPORTING RESULTS")
    print("=" * 50)

    # Generate Training Graph
    plot_training_history(history, MODEL_NAME, OUTPUT_DIR)

    # Load best weights for final evaluations
    model.load_state_dict(torch.load(save_path))

    # Generate Confusion Matrix & Extended Metrics
    precision, recall, f1 = evaluate_and_generate_metrics(model, val_loader, DEVICE, CLASS_NAMES, MODEL_NAME,
                                                          OUTPUT_DIR)

    # Generate Grad-CAM Explainability Samples
    generate_gradcam_samples(model, val_loader, DEVICE, CLASS_NAMES, MODEL_NAME, OUTPUT_DIR, num_samples=5)

    # Calculate Time Average & Save TXT File
    avg_sec_per_epoch = sum(epoch_times) / len(epoch_times)
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