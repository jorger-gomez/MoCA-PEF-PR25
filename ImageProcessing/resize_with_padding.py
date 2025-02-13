import os
from tkinter import Tk, filedialog
from PIL import Image, ImageOps, ImageChops

def process_images():
    """
    Processes images within a selected folder by cropping white borders, adding padding to make them square, 
    and resizing them to 128x128 pixels. The processed images are saved in separate subdirectories.
    
    Steps:
    1. Prompts the user to select a folder containing images.
    2. Creates subdirectories for padded and resized images.
    3. Iterates through images in the folder, applying processing steps.
    4. Saves the processed images in the appropriate folders.
    
    Returns:
        None
    """
    # Open a file dialog to select the folder
    Tk().withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select a folder")
    
    if not folder_path:
        print("No folder was selected.")
        return

    # Define output directories
    padded_folder = os.path.join(folder_path, "padded")  # Folder for images with padding
    resized_folder = os.path.join(folder_path, "resized")  # Folder for resized images
    os.makedirs(padded_folder, exist_ok=True)
    os.makedirs(resized_folder, exist_ok=True)

    # Iterate over the images in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    # Ensure the image is in RGB mode
                    img = img.convert("RGB")

                    # Detect and crop white borders
                    bg = Image.new(img.mode, img.size, (255, 255, 255))  # White background
                    diff = ImageChops.difference(img, bg)
                    bbox = diff.getbbox()
                    if bbox:
                        img = img.crop(bbox)

                    # Determine the maximum side for square padding
                    max_side = max(img.size)
                    new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))  # White background

                    # Center the cropped image on the square canvas
                    offset = ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2)
                    new_img.paste(img, offset)

                    # Save the padded image
                    padded_output_path = os.path.join(padded_folder, filename)
                    new_img.save(padded_output_path)
                    print(f"Padded image saved: {padded_output_path}")

                    # Resize the image to 128x128 pixels using high-quality resampling
                    resized_img = new_img.resize((128, 128), Image.Resampling.LANCZOS)

                    # Save the resized image
                    resized_output_path = os.path.join(resized_folder, filename)
                    resized_img.save(resized_output_path)
                    print(f"Resized image saved: {resized_output_path}")
            except Exception as e:
                print(f"Could not process the file {filename}: {e}")

if __name__ == "__main__":
    process_images()
