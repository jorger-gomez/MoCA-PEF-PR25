""" Data Augmenter

    This script applies data augmentation to images and their corresponding segmentation masks.
    The same transformations are applied to both the original image and all masks to maintain consistency.

    The script performs the following steps:
    1. Loads images and their associated masks from individual subdirectories.
    2. Applies a set of transformations such as rotation, scaling, shearing, elastic deformation, noise addition, blur, and contrast adjustments.
    3. Saves the augmented images and masks with a unique prefix "A###-" to distinguish them from the original files.
    4. Stores the generated dataset in the specified output directory.

    Authors: Diego Tovar & Rodrigo Gomez
    Contact: diego.tovar@udem.edu & jorger.gomez@udem.edu
    Organization: Universidad de Monterrey (UDEM)
    First created on February 25, 2025

    Usage Examples:
        Basic usage:
            python data_augmenter.py --input_path /path/to/images --output_path /path/to/output --num_images 20

        Arguments:
            --input_path (-i): Path to the folder containing subfolders with images.
            --output_path (-o): Path to the folder where augmented images will be saved.
            --num_images (-n): Number of augmented images to generate per original image (default: 20).
"""

import argparse
import os
import numpy as np
from tqdm import tqdm  
from PIL import Image, ImageEnhance, ImageFilter
from skimage.transform import AffineTransform, warp
from skimage.util import random_noise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to augment images by applying the same transformations to each subfolder.")
    parser.add_argument('--input_path', '-i', type=str, required=True, help="Path to the folder containing subfolders with images.")
    parser.add_argument('--output_path', '-o', type=str, required=True, help="Path to the folder where augmented images will be saved.")
    parser.add_argument('--num_images', '-n', type=int, default=20, help="Number of augmented images to generate per original image (default: 20).")
    return parser.parse_args()

def generate_random_parameters():
    """Generates a set of random transformation parameters."""
    parameters = {
        'theta': np.random.uniform(-90, 90),  # Rotation in degrees
        'tx': np.random.uniform(-2, 2),  # Translation in pixels
        'ty': np.random.uniform(-2, 2),
        'zx': np.random.uniform(0.5, 1.75),  # Zoom in X (distortion)
        'zy': np.random.uniform(0.5, 1.75),  # Zoom in Y (distortion)
        'flip_horizontal': np.random.choice([True, False]),
        'flip_vertical': np.random.choice([True, False]),
        'brightness': np.random.uniform(0.5, 1.5),  # Brightness adjustment
        'contrast': np.random.uniform(0.5, 1.5),  # Contrast adjustment
        'blur': np.random.choice([True, False]),  # Apply blur effect
        'blur_radius': np.random.uniform(0.5, 3) if np.random.choice([True, False]) else 0,
        'add_noise': np.random.choice([True, False]),
        'cutout': np.random.choice([True, False]),
        'cutout_x': np.random.randint(0, 320),
        'cutout_y': np.random.randint(0, 320),
        'cutout_w': np.random.randint(0, 320),
        'cutout_h': np.random.randint(0, 320),
        'noise_variance': np.random.uniform(0.01, 0.05) if np.random.choice([True, False]) else 0
    }
    return parameters

def apply_transformation(image, parameters):
    """Applies transformations to an image based on given parameters."""
    transform = AffineTransform(
        rotation=np.deg2rad(parameters['theta']),
        translation=(parameters['tx'], parameters['ty']),
        scale=(parameters['zx'], parameters['zy'])
    )
    transformed_image = warp(np.array(image), transform.inverse, mode='edge')
    transformed_image = Image.fromarray((transformed_image * 255).astype(np.uint8))
    
    # Horizontal flip
    if parameters['flip_horizontal']:
        transformed_image = transformed_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Vertical flip
    if parameters['flip_vertical']:
        transformed_image = transformed_image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Apply brightness and contrast
    enhancer = ImageEnhance.Brightness(transformed_image)
    transformed_image = enhancer.enhance(parameters['brightness'])
    
    enhancer = ImageEnhance.Contrast(transformed_image)
    transformed_image = enhancer.enhance(parameters['contrast'])
    
    # Apply blur effect
    if parameters['blur']:
        transformed_image = transformed_image.filter(ImageFilter.GaussianBlur(parameters['blur_radius']))
    
    # Apply noise
    if parameters['add_noise'] and parameters['noise_variance'] > 0:
        transformed_image = Image.fromarray((random_noise(np.array(transformed_image), mode='gaussian', var=parameters['noise_variance']) * 255).astype(np.uint8))
    
    return transformed_image

def augment_images(input_path, output_path, num_images=20):
    os.makedirs(output_path, exist_ok=True)
    subfolders = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    total_images = sum(len(os.listdir(os.path.join(input_path, sub))) for sub in subfolders)
    print(f"{total_images} images found in {len(subfolders)} folders.")

    total_to_generate = total_images * num_images
    print(f"{total_to_generate} images to be generated.")
    counter = 0
    
    with tqdm(total=total_to_generate, desc="Generating augmented images", unit="img") as pbar:
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_path, subfolder)
            images = [img for img in os.listdir(subfolder_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
            
            for i in range(num_images):
                parameters = generate_random_parameters()
                batch_prefix = f"A{i+1}_"
                
                for img_name in images:
                    img_path = os.path.join(subfolder_path, img_name)
                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGBA' if img.mode == 'RGBA' else 'RGB')
                        transformed_img = apply_transformation(img, parameters)
                        batch_filename = f"{batch_prefix}{os.path.splitext(img_name)[0]}.png"
                        transformed_img.save(os.path.join(output_path, batch_filename))
                        counter += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {img_name}: {e}")
    
    print(f"Process completed. {counter} augmented images have been saved.")

def main():
    print("Starting Please Wait...", end="\r")
    args = parse_arguments()
    augment_images(args.input_path, args.output_path, args.num_images)

if __name__ == "__main__":
    main()