import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

def configure_device():
    """
    Configures the device for computation based on availability.
    
    Returns:
        torch.device: The available computation device (MPS or CPU).
    """
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

def get_decoder_channels(model_name):
    """
    Returns the decoder channels based on the model type.

    Args:
        model_name (str): The name of the segmentation model.

    Returns:
        int or tuple: The decoder channel size.
    """
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
    """
    Returns the expected input size for a given model.

    Args:
        model_name (str): The name of the segmentation model.

    Returns:
        tuple: The expected input dimensions (height, width).
    """
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

def save_model_results(models_to_test, log_file):
    """
    Saves model details into the log file after processing all images.

    Args:
        models_to_test (dict): Dictionary containing model details.
        log_file (str): Path to the log file.
    """
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

def main():
    """
    Main function to execute model testing and result storage.
    """
    device = configure_device()
    print(f"Using device: {device}")
    
    input_folder = "/Users/diegotovar/Pictures/MoCA/testing_images"
    base_output_folder = "/Users/diegotovar/Pictures/MoCA/testing_images/PyTorch/output_segmentations_pytorch"
    log_file = "/Users/diegotovar/Pictures/MoCA/testing_images/PyTorch/logs/models_log.csv"
    output_folder = create_output_folder(base_output_folder)
    
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
    
    for model_name, info in models_to_test.items():
        pretrained_weights = "imagenet"
        info["preprocess_fn"] = smp.encoders.get_preprocessing_fn(info["encoder_name"], pretrained_weights)
        model_device = torch.device("cpu") if info.get("force_cpu", False) else device
        info["model"].to(model_device).eval()
    
    save_model_results(models_to_test, log_file)
    print("Processing completed!")

if __name__ == "__main__":
    main()
