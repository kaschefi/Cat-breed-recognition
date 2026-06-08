import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt

# 1. CONFIGURATION
IMAGE_PATH = "img.png"
MODEL_WEIGHTS = "best_efficientnet_b0.pth"
MODEL_NAME = "efficientnet_b0"
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


def predict_image():
    print(f"Loading {MODEL_NAME} onto {DEVICE}...")

    # MODEL SETUP
    # We set pretrained=False because we are loading custom trained weights,
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode (turns off dropout)

    IMG_SIZE = 288 if MODEL_NAME == 'efficientnet_b2' else 224

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    # Load image and add a "batch" dimension (from [3, 224, 224] to [1, 3, 224, 224])
    image = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)


    print(f"Analyzing image...")
    with torch.no_grad():
        output = model(input_tensor)

        # Convert raw output to percentages (probabilities)
        probabilities = F.softmax(output[0], dim=0)

        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_breed = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print("\n" + "=" * 40)
    print("PREDICTION RESULT:")
    print(f"Breed: {predicted_breed}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print("=" * 40)

    plt.imshow(image)
    plt.title(f"Predicted: {predicted_breed} ({confidence_pct:.1f}%)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    predict_image()