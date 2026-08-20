import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import numpy as np
import time

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_PATH = "img.png"
MODEL_WEIGHTS = "best_convnext_tiny.pth"
MODEL_NAME = "convnext_tiny"
#MODEL_WEIGHTS = "best_resnet50.pth"
#MODEL_NAME = "resnet50"
NUM_CLASSES = 12

CLASS_NAMES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_target_layer(model, model_name):
    """Automatically finds the correct final CNN layer for Grad-CAM."""
    if 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'efficientnet' in model_name:
        return [model.conv_head]
    elif 'convnext' in model_name:
        return [model.stages[-1].blocks[-1]]
    else:
        return [list(model.children())[-2]]


def predict_and_explain():
    print(f"Loading {MODEL_NAME} onto {DEVICE}...")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    IMG_SIZE = 288 if MODEL_NAME == 'efficientnet_b2' else 224

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at '{IMAGE_PATH}'. Please check the path.")
        return

    original_image = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    print(f"Analyzing image...")
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_breed = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print("\n" + "=" * 40)
    print("PREDICTION RESULT:")
    print(f"Breed: {predicted_breed}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print("=" * 40)

    print("Generating Grad-CAM visualization...")
    target_layers = get_target_layer(model, MODEL_NAME)

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Un-normalize the image so it looks normal when we overlay the heatmap
    img_show = input_tensor[0].cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_show = std * img_show + mean
    img_show = np.clip(img_show, 0, 1)

    # Create the overlay
    visualization = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Original Image
    axes[0].imshow(img_show)
    axes[0].set_title("Original Input")
    axes[0].axis('off')

    # Plot Grad-CAM Image
    axes[1].imshow(visualization)
    axes[1].set_title(f"Prediction: {predicted_breed} ({confidence_pct:.1f}%)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    startTime = time.time()
    predict_and_explain()
    endTime = time.time()
    print(f"Total execution time: {endTime - startTime:.2f} seconds")