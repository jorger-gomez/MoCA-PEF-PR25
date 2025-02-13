import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

# ------------------------------
# Device Configuration
# ------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Input/Output Configuration
# ------------------------------
input_folder = "/Users/diegotovar/Pictures/MoCA/testing_images"
base_output_folder = "/Users/diegotovar/Pictures/MoCA/testing_images/PyTorch/output_segmentations_pytorch"
log_file = "/Users/diegotovar/Pictures/MoCA/testing_images/PyTorch/logs/models_log.csv"

# Create output folder
i = 1
output_folder = f"{base_output_folder}_{i}"
while os.path.exists(output_folder):
    i += 1
    output_folder = f"{base_output_folder}_{i}"
os.makedirs(output_folder, exist_ok=True)

# Retrieve list of images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print("No images found in the input folder.")
    exit()

# ------------------------------
# Helper Functions
# ------------------------------
def get_decoder_channels(model_name):
    """Returns the decoder channels based on the model type."""
    default_channels = {
        "Unet": (256, 128, 64, 32, 16),
        "UnetPlusPlus": (256, 128, 64, 32, 16),
        "DeepLabV3Plus": 256,
        "Linknet": 256,
        "PSPNet": 512,
        "FPN": 256,
        "PAN": 32
    }
    return default_channels.get(model_name.split(" ")[0], 256)

def get_input_size(model_name):
    """Returns the expected input size for a given model."""
    default_sizes = {
        "Unet": (256, 256),
        "UnetPlusPlus": (256, 256),
        "DeepLabV3Plus": (512, 512),
        "Linknet": (256, 256),
        "PSPNet": (512, 512),
        "FPN": (256, 256),
        "PAN": (256, 256)
    }
    return default_sizes.get(model_name.split(" ")[0], (256, 256))

def save_model_results(models_to_test):
    """Saves model details into the log file after processing all images."""
    final_results = []
    for model_name, info in models_to_test.items():
        final_results.append({
            "Model": model_name,
            "Encoder": info["encoder_name"],
            "Input Size": f"{get_input_size(model_name)[0]}x{get_input_size(model_name)[1]}",
            "Decoder Channels": get_decoder_channels(model_name),
            "Favorite": "Pending",
            "Comment": "Acceptable segmentation"
        })
    df_new = pd.DataFrame(final_results)
    df_new.to_csv(log_file, index=False)
    print(f"Models saved in {log_file}")

# ------------------------------
# Define Models to Test
# ------------------------------
models_to_test = {
    "Unet (ResNet101)": {
        "model": smp.Unet(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1),
        "encoder_name": "resnet101",
        "force_cpu": False
    },
    "Unet++ (DenseNet169)": {
        "model": smp.UnetPlusPlus(encoder_name="densenet169", encoder_weights="imagenet", in_channels=3, classes=1),
        "encoder_name": "densenet169",
        "force_cpu": False
    },
    "Unet (SE-ResNeXt101)": {
        "model": smp.Unet(encoder_name="se_resnext101_32x4d", encoder_weights="imagenet", in_channels=3, classes=1),
        "encoder_name": "se_resnext101_32x4d",
        "force_cpu": False
    }
}


# Assign models to the appropriate device
for model_name, info in models_to_test.items():
    pretrained_weights = "imagenet"
    info["preprocess_fn"] = smp.encoders.get_preprocessing_fn(info["encoder_name"], pretrained_weights)
    model_device = torch.device("cpu") if info.get("force_cpu", False) else device
    info["model"].to(model_device).eval()

# ------------------------------
# Process images with each model and generate a single visualization
# ------------------------------
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Unable to load image {image_file}")
        continue

    fig, axes = plt.subplots(1, len(models_to_test) + 1, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for idx, (model_name, info) in enumerate(models_to_test.items()):
        print(f"Processing {image_file} with {model_name}...")
        input_size = get_input_size(model_name)
        image_resized = cv2.cvtColor(cv2.resize(image, input_size), cv2.COLOR_BGR2RGB)
        preprocess_fn = info["preprocess_fn"]
        image_preprocessed = preprocess_fn(image_resized.astype(np.uint8))
        image_tensor = torch.tensor(image_preprocessed, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        model_device = torch.device("cpu") if info.get("force_cpu", False) else device
        image_tensor = image_tensor.to(model_device)

        start_time = time.perf_counter()
        with torch.no_grad():
            prediction = info["model"](image_tensor)
        elapsed_time = time.perf_counter() - start_time
        mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        axes[idx + 1].imshow(mask, cmap="gray")
        axes[idx + 1].set_title(f"{model_name}\n{info['encoder_name']}\n{elapsed_time:.4f}s")
        axes[idx + 1].axis("off")
    
    plt.tight_layout()
    output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_comparison.png")
    plt.savefig(output_image_path)
    plt.close()
    print(f"Saved: {output_image_path}")

# Save models in the log only after all images are processed
save_model_results(models_to_test)

print("Processing completed!")
