'''
python image_trim_white_space.py 3d-sketches-processing/intersections_png
'''

import os
import glob
import argparse
from PIL import Image
import numpy as np

def trim_whitespace(image_path, output_path=None, threshold=240):
    """
    Trim whitespace from a PNG image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save trimmed image (if None, overwrites original)
        threshold (int): RGB threshold for considering pixels as "white" (0-255)
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Find non-white pixels (all RGB values below threshold)
        non_white = np.any(img_array < threshold, axis=2)
        
        # Find bounding box of non-white pixels
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print(f"Warning: {image_path} appears to be entirely white - skipping")
            return False
        
        # Get crop coordinates
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
        
        # Add small padding (optional)
        padding = 10
        height, width = img_array.shape[:2]
        top = max(0, top - padding)
        bottom = min(height - 1, bottom + padding)
        left = max(0, left - padding)
        right = min(width - 1, right + padding)
        
        # Crop image
        cropped_img = img.crop((left, top, right + 1, bottom + 1))
        
        # Save
        if output_path is None:
            output_path = image_path
        
        cropped_img.save(output_path, 'PNG', optimize=True)
        
        original_size = f"{img.width}x{img.height}"
        new_size = f"{cropped_img.width}x{cropped_img.height}"
        print(f"Trimmed: {os.path.basename(image_path)} ({original_size} -> {new_size})")
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def trim_folder(folder_path, output_folder=None, threshold=240, pattern="*.png"):
    """
    Trim whitespace from all PNG files in a folder.
    
    Args:
        folder_path (str): Input folder containing PNG files
        output_folder (str): Output folder (if None, overwrites originals)
        threshold (int): RGB threshold for white pixels
        pattern (str): File pattern to match
    """
    # Get all PNG files
    png_files = glob.glob(os.path.join(folder_path, pattern))
    
    if not png_files:
        print(f"No PNG files found in {folder_path}")
        return
    
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"Found {len(png_files)} PNG files to process...")
    
    success_count = 0
    for png_file in png_files:
        if output_folder:
            output_path = os.path.join(output_folder, os.path.basename(png_file))
        else:
            output_path = None
        
        if trim_whitespace(png_file, output_path, threshold):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(png_files)} files processed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trim whitespace from PNG files')
    parser.add_argument('input_folder', help='Folder containing PNG files')
    parser.add_argument('--output', '-o', help='Output folder (optional, overwrites if not specified)')
    parser.add_argument('--threshold', '-t', type=int, default=240, 
                       help='RGB threshold for white pixels (0-255, default: 240)')
    parser.add_argument('--pattern', '-p', default='*.png', 
                       help='File pattern to match (default: *.png)')
    parser.add_argument('--backup', '-b', action='store_true', 
                       help='Create backup of original files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        exit(1)
    
    # Create backup if requested
    if args.backup and not args.output:
        backup_folder = args.input_folder + "_backup"
        os.makedirs(backup_folder, exist_ok=True)
        png_files = glob.glob(os.path.join(args.input_folder, args.pattern))
        for png_file in png_files:
            backup_path = os.path.join(backup_folder, os.path.basename(png_file))
            Image.open(png_file).save(backup_path)
        print(f"Backup created in: {backup_folder}")
    
    # Process files
    trim_folder(args.input_folder, args.output, args.threshold, args.pattern)