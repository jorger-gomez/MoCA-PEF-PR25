"""
Clock Drawing Segmentation Inference

This script implements a segmentation system for clock drawings.
It loads a pre-trained U-Net++ model with 
SE-ResNet50 backbone to segment four key components of clock drawings: 
entire clock, numbers, hands, and contour.

Usage:
    python clock_inference.py --input path/to/image.png --model path/to/model.pth --output path/to/output_dir

Arguments:
    --input     : Path to input image or directory. Required.
                  The input image should preferably have a "_Background" suffix.
    
    --model     : Path to trained model (.pth file). Required.
                  The model should be a PyTorch model with U-Net++ architecture.
    
    --output    : Output directory. Optional.
                  Default: Creates a directory with the image name (minus "_Background")
                  in the same location as the input image.
    
    --img_size  : Image size for model input. Optional. Default: 256.
                  Higher values may improve quality but require more memory.
    
    --threshold : Threshold for binary mask. Optional. Default: 0.5.
                  Values range from 0.0 to 1.0. Higher values create more selective masks.
    
    --batch     : Process all images in the input directory. Optional flag.
                  When used, --input should be a directory path.

Outputs:
    For each processed image, the script creates:
    1. A directory with the same name as the input image (without "_Background" suffix)
    2. The original image saved as "{name}_Background.png"
    3. Four PNG masks with transparency for each component:
       - "{name}_entire.png"  : The entire clock
       - "{name}_numbers.png" : Numbers on the clock face
       - "{name}_hands.png"   : Clock hands (hour and minute)
       - "{name}_contour.png" : Clock contour/outline

Examples:
    # Process a single image:
    python clock_inference.py --input "image_Background.png" --model "model.pth"
    
    # Process all images in a directory:
    python clock_inference.py --input "images_dir/" --model "model.pth" --batch
    
    # Specify output directory and custom parameters:
    python clock_inference.py --input "image.png" --model "model.pth" --output "results" --img_size 320 --threshold 0.6

Authors:
    Diego Aldahir Tovar Ledesma - diego.tovar@udem.edu
    Jorge Rodrigo Gómez Mayo - jorger.gomez@udem.edu

Organization: Universidad de Monterrey
"""

import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Clock Drawing Segmentation Inference')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--output', default=None, help='Output directory (default: creates a directory with the image name)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for model input (default: 256)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary mask (default: 0.5)')
    parser.add_argument('--batch', action='store_true', help='Process all images in the input directory')
    
    return parser.parse_args()


class Config:
    """Configuration class with parameters for the segmentation model"""
    def __init__(self, args):
        # Device for inference
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                               "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Image size for the model
        self.IMG_SIZE = args.img_size
        
        # Threshold for binary masks
        self.THRESHOLD = args.threshold
        
        # Model parameters
        self.ENCODER = 'se_resnet50'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = ['entire', 'numbers', 'hands', 'contour']
        self.NUM_CLASSES = len(self.CLASSES)


def build_model(config):
    """
    Builds the segmentation model.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        nn.Module: The constructed segmentation model
    """
    # Using U-Net++ architecture with SE-ResNet50 backbone
    model = smp.UnetPlusPlus(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.NUM_CLASSES,
        activation='sigmoid'
    )
    
    print(f"Model built: UnetPlusPlus with encoder {config.ENCODER}")
    return model


def get_output_folder(image_path, output_dir=None):
    """
    Determine the output folder for saving results
    
    Args:
        image_path: Path to the input image
        output_dir: Optional user-specified output directory
        
    Returns:
        str: Path to the output folder
    """
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Remove '_Background' suffix if present
    if base_name.endswith("_Background"):
        base_name = base_name[:-11]
    elif base_name.endswith("_background"):
        base_name = base_name[:-11]
    
    # If output directory is specified, create subfolder with the base name
    if output_dir:
        folder_path = os.path.join(output_dir, base_name)
    else:
        # Otherwise, create folder in the same directory as the image
        folder_path = os.path.join(os.path.dirname(image_path), base_name)
    
    # Create the output folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path, base_name


def process_image(config, model, image_path, output_dir=None):
    """
    Process a single image to generate segmentation masks
    
    Args:
        config: Configuration object
        model: Loaded segmentation model
        image_path: Path to the input image
        output_dir: Optional output directory
    """
    print(f"Processing image: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return
    
    # Determine output folder and base name
    output_folder, base_name = get_output_folder(image_path, output_dir)
    
    # Save a copy of the original image in the output folder
    background_path = os.path.join(output_folder, f"{base_name}_Background.png")
    cv2.imwrite(background_path, image)
    print(f"Saved background image: {background_path}")
    
    # Convert image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations for the model
    transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)
    
    # Generate predictions
    with torch.no_grad():
        pred = model(image_tensor)
    
    pred_np = pred.squeeze().cpu().numpy()
    
    # Get original image dimensions
    h, w = image.shape[:2]
    
    # Set up visualization figure
    fig, axes = plt.subplots(1, config.NUM_CLASSES + 1, figsize=(5*(config.NUM_CLASSES + 1), 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Process each class
    for i, class_name in enumerate(config.CLASSES):
        # Resize prediction to original image dimensions
        pred_class = cv2.resize(pred_np[i], (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply threshold to get binary mask
        pred_binary = (pred_class > config.THRESHOLD).astype(np.uint8) * 255
        
        # Apply post-processing based on class type
        if class_name == 'contour':
            # For contours, apply morphological closing
            kernel = np.ones((3, 3), np.uint8)
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Remove small components (noise)
            num_labels, labels = cv2.connectedComponents(pred_binary)
            min_size = 20  # Threshold to consider noise
            
            # Keep only components larger than threshold
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8)
                if np.sum(component_mask) < min_size:
                    pred_binary[component_mask == 1] = 0
                    
        elif class_name == 'hands':
            # For hands, emphasize thin lines
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel_line)
            
            # Also apply median filter to smooth
            pred_binary = cv2.medianBlur(pred_binary, 3)
            
        elif class_name == 'numbers':
            # For numbers, preserve more details
            kernel = np.ones((2, 2), np.uint8)
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
        else:  # 'entire'
            # For the entire figure, use opening to remove noise
            kernel = np.ones((3, 3), np.uint8)
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Ensure binary values (0 or 255)
        _, pred_binary = cv2.threshold(pred_binary, 127, 255, cv2.THRESH_BINARY)
        
        # Visualization
        axes[i+1].imshow(pred_binary, cmap='gray')
        axes[i+1].set_title(f'Mask: {class_name}')
        axes[i+1].axis('off')
        
        # Create transparent mask (RGBA)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0:3] = 255  # White foreground
        rgba[..., 3] = pred_binary  # Alpha channel from binary mask
        
        # Save the mask
        output_path = os.path.join(output_folder, f"{base_name}_{class_name}.png")
        cv2.imwrite(output_path, rgba)
        print(f"Saved mask: {output_path}")

def process_directory(config, model, input_dir, output_dir):
    """
    Process all images in a directory
    
    Args:
        config: Configuration object
        model: Loaded segmentation model
        input_dir: Directory containing input images
        output_dir: Output directory for results
    """
    # Look for images with common extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_paths = []
    
    for ext in extensions:
        # Try both lowercase and uppercase extensions
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in directory: {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for image_path in image_paths:
        process_image(config, model, image_path, output_dir)


def main():
    """Main function to run the inference pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize configuration
    config = Config(args)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Threshold: {config.THRESHOLD}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file does not exist: {args.model}")
        sys.exit(1)
    
    # Load the model
    print("Loading model...")
    model = build_model(config)
    model.load_state_dict(torch.load(args.model, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Process input (either single image or directory)
    if args.batch:
        # Check if input directory exists
        if not os.path.isdir(args.input):
            print(f"ERROR: Input directory does not exist: {args.input}")
            sys.exit(1)
        
        # Process all images in directory
        process_directory(config, model, args.input, args.output)
    else:
        # Check if input file exists
        if not os.path.isfile(args.input):
            print(f"ERROR: Input file does not exist: {args.input}")
            sys.exit(1)
        
        # Process single image
        process_image(config, model, args.input, args.output)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()