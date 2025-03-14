""" Data Augmenter

    This script applies data augmentation to images and their corresponding segmentation masks.
    The same transformations are applied to both the original image and all masks to maintain consistency.
    It performs various transformations including affine transformations, elastic deformations, noise addition,
    blur, brightness/contrast adjustments, and cutouts. Elastic deformation is implemented as described in reference [1].

    The script performs the following steps:
    1. Loads images and their associated masks from individual subdirectories.
    2. Applies a set of transformations.
    3. Saves the augmented images and masks with a unique prefix "A###_" to distinguish them from the original files.
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
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from skimage.transform import AffineTransform, warp
from skimage.util import random_noise
from scipy.ndimage import map_coordinates, gaussian_filter

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Script to augment images by applying the same transformations to each subfolder.")
    parser.add_argument('--input_path', '-i', type=str, required=True, help="Path to the folder containing subfolders with images.")
    parser.add_argument('--output_path', '-o', type=str, required=True, help="Path to the folder where augmented images will be saved.")
    parser.add_argument('--num_images', '-n', type=int, default=20, help="Number of augmented images to generate per original image (default: 20).")
    return parser.parse_args()

def generate_random_parameters():
    """Generates a set of random transformation parameters with ranges simulating different camera parameters."""
    parameters = {
        # Affine transformation parameters
        'theta': np.random.uniform(-45, 45),       # Rotation in degrees (limited to avoid cutting too much of the clock)
        'tx': np.random.uniform(-10, 10),            # Translation in pixels (x)
        'ty': np.random.uniform(-10, 10),            # Translation in pixels (y)
        'zx': np.random.uniform(0.8, 1.2),           # Zoom factor in X
        'zy': np.random.uniform(0.8, 1.2),           # Zoom factor in Y
        'flip_horizontal': np.random.choice([True, False]),
        'flip_vertical': np.random.choice([True, False]),
        'brightness': np.random.uniform(0.8, 1.2),     # Brightness adjustment (simulate realistic camera settings)
        'contrast': np.random.uniform(0.8, 1.2),       # Contrast adjustment (simulate realistic camera settings)
        'blur': np.random.choice([True, False]),       # Apply blur effect
        'blur_radius': np.random.uniform(1, 3) if np.random.choice([True, False]) else 0,  # Blur radius between 1 and 3
        'add_noise': np.random.choice([True, False]),
        'noise_variance': np.random.uniform(0.001, 0.01) if np.random.choice([True, False]) else 0,  # Lower noise variance
        # Elastic transformation parameters
        'elastic': np.random.choice([True, False]),
        'elastic_alpha': np.random.uniform(30, 50),   # Scaling factor for elastic transformation
        'elastic_sigma': np.random.uniform(4, 6),       # Smoothing factor for elastic transformation
        # Cutout parameters
        'cutout': np.random.choice([True, False]),
        'num_cutouts': np.random.randint(0, 5)  # number of cutouts (from 0 to 4)
    }
    return parameters

def elastic_transform_multi(image, alpha, sigma, random_state=None):
    """
    Applies elastic deformation on an image as described in reference [1].
    Based on the code from reference [2].
    This function supports multi-channel images (e.g., RGB, RGBA).

    References:
        [1] P. Y. Simard, D. Steinkraus, and J. C. Platt, “Best practices for convolutional neural networks applied to visual document analysis,” 
            in Proc. of the International Conference on Document Analysis and Recognition, 2003, pp. 958–962, doi:10.1109/ICDAR.2003.1227801.
        [2] GitHub Gist by chsasank, available at https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape[:2]
    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    # Create meshgrid coordinates (using 'ij' indexing)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1,)), np.reshape(y + dy, (-1,))
    
    if image.ndim == 3:
        transformed = np.empty_like(image)
        # Apply the same transformation to each channel
        for channel in range(image.shape[2]):
            transformed[:, :, channel] = map_coordinates(image[:, :, channel], indices, order=1, mode='reflect').reshape(shape)
        return transformed
    else:
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def apply_transformation(image, parameters, not_mask):
    """Applies a series of transformations to an image based on given parameters."""
    # Apply affine transformation
    transform = AffineTransform(
        rotation=np.deg2rad(parameters['theta']),
        translation=(parameters['tx'], parameters['ty']),
        scale=(parameters['zx'], parameters['zy'])
    )
    transformed_image = warp(np.array(image), transform.inverse, mode='edge')
    transformed_image = Image.fromarray((transformed_image * 255).astype(np.uint8))
    
    # Apply horizontal flip
    if parameters['flip_horizontal']:
        transformed_image = transformed_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Apply vertical flip
    if parameters['flip_vertical']:
        transformed_image = transformed_image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(transformed_image)
    transformed_image = enhancer.enhance(parameters['brightness'])
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(transformed_image)
    transformed_image = enhancer.enhance(parameters['contrast'])
    
    # Apply blur effect if enabled
    if parameters['blur']:
        transformed_image = transformed_image.filter(ImageFilter.GaussianBlur(parameters['blur_radius']))
    
    # Add noise if enabled
    if parameters['add_noise'] and parameters['noise_variance'] > 0:
        np_img = np.array(transformed_image)
        # If image has transparency (4 channels), apply noise only to the RGB channels and keep alpha unchanged.
        if np_img.ndim == 3 and np_img.shape[2] == 4:
            noisy_rgb = random_noise(np_img[:, :, :3], mode='gaussian', var=parameters['noise_variance'])
            noisy_rgb = (noisy_rgb * 255).astype(np.uint8)
            alpha_channel = np_img[:, :, 3]
            np_noisy = np.dstack((noisy_rgb, alpha_channel))
            transformed_image = Image.fromarray(np_noisy)
        else:
            noisy = random_noise(np_img, mode='gaussian', var=parameters['noise_variance'])
            transformed_image = Image.fromarray((noisy * 255).astype(np.uint8))
    
    # Apply elastic transformation if enabled
    if parameters.get('elastic', False):
        np_img = np.array(transformed_image)
        transformed_np = elastic_transform_multi(np_img, parameters['elastic_alpha'], parameters['elastic_sigma'])
        transformed_image = Image.fromarray(transformed_np.astype(np.uint8))
    
    # Apply cutout if enabled
    if parameters['cutout']:
        width, height = transformed_image.size
        num_cutouts = parameters.get('num_cutouts', np.random.randint(0, 5))
        for _ in range(num_cutouts):
            # Define the cutout size as a random percentage of the image dimensions
            min_cut_w = max(1, int(width * 0.1))
            max_cut_w = max(1, int(width * 0.3))
            min_cut_h = max(1, int(height * 0.1))
            max_cut_h = max(1, int(height * 0.3))
            cutout_w = np.random.randint(min_cut_w, max_cut_w + 1)
            cutout_h = np.random.randint(min_cut_h, max_cut_h + 1)
            # Choose a random position ensuring the rectangle is fully within the image
            x = np.random.randint(0, width - cutout_w + 1)
            y = np.random.randint(0, height - cutout_h + 1)
            draw = ImageDraw.Draw(transformed_image)
            # If not_mask is True (background image), fill with black; otherwise (mask), fill with transparency.
            fill_color = (0, 0, 0) if not_mask else (0, 0, 0, 0)
            draw.rectangle([x, y, x + cutout_w, y + cutout_h], fill=fill_color)
    
    return transformed_image

def augment_images(input_path, output_path, num_images=20):
    """Generates augmented images applying the same transformations to images and their masks."""
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
                        # Determine if the image is not a mask based on its filename (contains "background", case-insensitive)
                        not_mask = 'background' in img_name.lower()
                        # Convert image: if not_mask, use RGB; otherwise, use RGBA (to support transparency)
                        img = img.convert('RGB' if not_mask else 'RGBA')
                        transformed_img = apply_transformation(img, parameters, not_mask)
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

"""
References:
[1] P. Y. Simard, D. Steinkraus, and J. C. Platt, “Best practices for convolutional neural networks applied to visual document analysis,” 
    in Proc. of the International Conference on Document Analysis and Recognition, 2003, pp. 958–962, doi:10.1109/ICDAR.2003.1227801.
[2] GitHub Gist by chsasank, available at https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a.
"""