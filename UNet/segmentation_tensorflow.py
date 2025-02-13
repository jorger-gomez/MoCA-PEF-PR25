import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm

def configure_backend():
    """
    Configures TensorFlow Keras as the backend for segmentation models.
    """
    sm.set_framework('tf.keras')
    sm.framework()

def create_output_folder(base_output_folder):
    """
    Creates an output folder for storing segmentation results.
    If a folder with the same name exists, increments the index.
    
    Args:
        base_output_folder (str): Base path for the output folder.
    
    Returns:
        str: Path to the created output folder.
    """
    i = 1
    output_folder = f"{base_output_folder}_{i}"
    while os.path.exists(output_folder):
        i += 1
        output_folder = f"{base_output_folder}_{i}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def load_images(input_folder):
    """
    Retrieves a list of image file paths from the specified folder.
    
    Args:
        input_folder (str): Path to the folder containing images.
    
    Returns:
        list: List of image file paths.
    """
    return [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def process_images(image_files, input_folder, output_folder, models_to_test, backbone):
    """
    Processes all images using different segmentation models and saves results.
    
    Args:
        image_files (list): List of image filenames.
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        models_to_test (dict): Dictionary containing model configurations.
        backbone (str): Backbone architecture used for segmentation models.
    """
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"\nProcessing image: {image_file}")
        results = {}

        for model_name, model_info in models_to_test.items():
            model = model_info['model']
            input_size = model_info['input_size']

            print(f" - Testing {model_name} ({backbone})...")

            # Resize and normalize the image
            image_resized = cv2.resize(image, input_size)
            image_normalized = image_resized / 255.0
            image_tensor = np.expand_dims(image_normalized, axis=0)

            # Measure inference time
            start_time = time.time()
            prediction = model.predict(image_tensor)
            elapsed_time = time.time() - start_time

            # Convert prediction to binary mask
            mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255

            # Store result
            results[model_name] = {
                "Backbone": backbone,
                "Time": elapsed_time,
                "Prediction": mask
            }
            print(f"   -> {model_name} ({backbone}) completed in {elapsed_time:.4f}s")

        save_results(image_file, results, output_folder)

def save_results(image_file, results, output_folder):
    """
    Saves segmentation results as a unified image.
    
    Args:
        image_file (str): Name of the original image file.
        results (dict): Dictionary containing segmentation results.
        output_folder (str): Path to the output folder.
    """
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

def main():
    """
    Main function to execute image segmentation using multiple models.
    """
    configure_backend()
    input_folder = "/Users/diegotovar/Pictures/MoCA/testing_images"
    base_output_folder = "/Users/diegotovar/Pictures/MoCA/testing_images/Keras/output_segmentations"
    output_folder = create_output_folder(base_output_folder)
    image_files = load_images(input_folder)
    
    if not image_files:
        print("No images found in the input folder.")
        exit()
    
    backbone = "efficientnetb7"
    models_to_test = {
        "U-Net": {'model': sm.Unet(backbone, classes=1, activation='sigmoid'), 'input_size': (224, 224)},
        "Linknet": {'model': sm.Linknet(backbone, classes=1, activation='sigmoid'), 'input_size': (224, 224)},
        "PSPNet": {'model': sm.PSPNet(backbone, classes=1, activation='sigmoid'), 'input_size': (384, 384)},
        "FPN": {'model': sm.FPN(backbone, classes=1, activation='sigmoid'), 'input_size': (224, 224)}
    }
    
    process_images(image_files, input_folder, output_folder, models_to_test, backbone)
    print("\nProcessing complete! Segmentations are in:", output_folder)

if __name__ == "__main__":
    main()
