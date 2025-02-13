import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm

# Configure TensorFlow Keras backend
sm.set_framework('tf.keras')
sm.framework()

# ------------------------------
# Input/Output Configuration
# ------------------------------
input_folder = "/Users/diegotovar/Pictures/MoCA/testing_images"
base_output_folder = "/Users/diegotovar/Pictures/MoCA/testing_images/Keras/output_segmentations"

# Create output folder (increment index if folder exists)
i = 1
output_folder = f"{base_output_folder}_{i}"
while os.path.exists(output_folder):
    i += 1
    output_folder = f"{base_output_folder}_{i}"
os.makedirs(output_folder, exist_ok=True)

# Retrieve list of images (allowed extensions: png, jpg, jpeg)
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print("No images found in the input folder.")
    exit()

# ------------------------------
# Define Backbone to Use
# ------------------------------
BACKBONE = "efficientnetb7"

# ------------------------------
# Define Models to Test
# ------------------------------
models_to_test = {
    "U-Net": {'model': sm.Unet(BACKBONE, classes=1, activation='sigmoid'), 'input_size': (224, 224)},
    "Linknet": {'model': sm.Linknet(BACKBONE, classes=1, activation='sigmoid'), 'input_size': (224, 224)},
    "PSPNet": {'model': sm.PSPNet(BACKBONE, classes=1, activation='sigmoid'), 'input_size': (384, 384)},
    "FPN": {'model': sm.FPN(BACKBONE, classes=1, activation='sigmoid'), 'input_size': (224, 224)}
}

# ------------------------------
# Process All Images in the Folder
# ------------------------------
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nProcessing image: {image_file}")

    # Dictionary to store results per image
    results = {}

    for model_name, model_info in models_to_test.items():
        model = model_info['model']
        input_size = model_info['input_size']

        print(f" - Testing {model_name} ({BACKBONE})...")

        # Resize and normalize the image according to model input size
        image_resized = cv2.resize(image, input_size)
        image_normalized = image_resized / 255.0
        image_tensor = np.expand_dims(image_normalized, axis=0)

        # Measure inference time
        start_time = time.time()
        prediction = model.predict(image_tensor)
        elapsed_time = time.time() - start_time

        # Convert prediction to binary mask
        mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255

        # Save result
        results[model_name] = {
            "Backbone": BACKBONE,
            "Time": elapsed_time,
            "Prediction": mask
        }
        print(f"   -> {model_name} ({BACKBONE}) completed in {elapsed_time:.4f}s")

    # ------------------------------
    # Save Results as a Unified Image
    # ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (model_name, result) in zip(axes.flatten(), results.items()):
        ax.imshow(result["Prediction"], cmap="gray")
        ax.set_title(f"{model_name} ({result['Backbone']})\nTime: {result['Time']:.4f}s")
        ax.axis("off")
    
    plt.tight_layout()
    plot_output_path = os.path.join(output_folder, f"{image_file}_segmentation_results.png")
    plt.savefig(plot_output_path)
    plt.close()
    print(f"   -> Results saved in: {plot_output_path}")

print("\nProcessing complete! Segmentations are in:", output_folder)
