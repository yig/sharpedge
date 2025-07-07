'''
python image_merge.py 3d-sketches-processing/intersections_png --width 2100 --height 1500 --output sketches_latex.png 

'''
import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont
import math

def calculate_grid_layout(num_images, aspect_ratio=0.5625):
    """
    Calculate optimal grid layout for given number of images on HD canvas.
    
    Args:
        num_images (int): Number of images to arrange
        aspect_ratio (float): HD aspect ratio (height/width = 1080/1920 ≈ 0.5625)
    
    Returns:
        tuple: (rows, cols) for optimal layout
    """
    # Try different grid configurations
    best_ratio_diff = float('inf')
    best_layout = (1, num_images)
    
    for cols in range(1, num_images + 1):
        rows = math.ceil(num_images / cols)
        grid_aspect = rows / cols
        ratio_diff = abs(grid_aspect - aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_layout = (rows, cols)
    
    return best_layout

def merge_pngs_to_hd(input_folder, output_path, width=1920, height=1080, margin=20, 
                     title=None, pattern="*.png", max_images=None):
    """
    Merge all PNG files in a folder into a single HD-sized image.
    
    Args:
        input_folder (str): Folder containing PNG files
        output_path (str): Path for output merged image
        width (int): Output image width in pixels (default: 1920)
        height (int): Output image height in pixels (default: 1080)
        margin (int): Margin around images and page edges in pixels
        title (str): Optional title for the merged image
        pattern (str): File pattern to match
        max_images (int): Maximum number of images to include
    """
    # HD dimensions
    canvas_width = width
    canvas_height = height
    
    print(f"Canvas dimensions: {canvas_width}x{canvas_height} pixels")
    
    # Get all PNG files
    png_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return False
    
    if max_images:
        png_files = png_files[:max_images]
    
    print(f"Found {len(png_files)} PNG files to merge")
    
    # Load and resize images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            images.append((img, os.path.basename(png_file)))
        except Exception as e:
            print(f"Error loading {png_file}: {e}")
    
    if not images:
        print("No valid images found")
        return False
    
    # Calculate grid layout (16:9 aspect ratio for HD)
    rows, cols = calculate_grid_layout(len(images), aspect_ratio=9/16)
    print(f"Using {rows}x{cols} grid layout")
    
    # Calculate available space
    title_height = 60 if title else 0
    available_width = canvas_width - 2 * margin
    available_height = canvas_height - 2 * margin - title_height
    
    # Calculate cell dimensions
    cell_width = available_width // cols
    cell_height = available_height // rows
    
    # Calculate image dimensions (with padding between images)
    img_padding = 10
    img_width = cell_width - img_padding
    img_height = cell_height - img_padding
    
    print(f"Each image will be resized to: {img_width}x{img_height} pixels")
    
    # Create HD canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Add title if specified
    if title:
        try:
            # Try to use a reasonable font
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                font = ImageFont.load_default()
        
        # Center title
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        draw.text((title_x, margin), title, fill='black', font=font)
    
    # Place images in grid
    for i, (img, filename) in enumerate(images):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = margin + col * cell_width + img_padding // 2
        y = margin + title_height + row * cell_height + img_padding // 2
        
        # Resize image while maintaining aspect ratio
        img_resized = resize_image_aspect_ratio(img, img_width, img_height)
        
        # Center image in cell
        img_x = x + (img_width - img_resized.width) // 2
        img_y = y + (img_height - img_resized.height) // 2
        
        # Paste image (handle transparency)
        if img_resized.mode == 'RGBA':
            canvas.paste(img_resized, (img_x, img_y), img_resized)
        else:
            canvas.paste(img_resized, (img_x, img_y))
        
        # Add filename label (optional)
        label_font_size = max(12, min(24, img_width // 20))
        try:
            label_font = ImageFont.truetype("arial.ttf", label_font_size)
        except:
            label_font = ImageFont.load_default()
        
        # Truncate filename if too long
        display_name = filename
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."
        
        label_bbox = draw.textbbox((0, 0), display_name, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = img_x + (img_resized.width - label_width) // 2
        label_y = img_y + img_resized.height + 5
        
        # Add background for text
        draw.rectangle([label_x - 2, label_y - 2, label_x + label_width + 2, label_y + 15], 
                      fill='white', outline='lightgray')
        draw.text((label_x, label_y), display_name, fill='black', font=label_font)
    
    # Save merged image
    canvas.save(output_path, 'PNG')
    print(f"Merged image saved to: {output_path}")
    print(f"Final dimensions: {canvas.width}x{canvas.height} pixels")
    
    return True

def resize_image_aspect_ratio(img, max_width, max_height):
    """
    Resize image while maintaining aspect ratio to fit within max dimensions.
    """
    img_width, img_height = img.size
    
    # Calculate scaling factors
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    
    # Use the smaller ratio to ensure image fits
    scale_ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(img_width * scale_ratio)
    new_height = int(img_height * scale_ratio)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge PNG files into HD-sized image (1920x1080)')
    parser.add_argument('input_folder', help='Folder containing PNG files')
    parser.add_argument('--output', '-o', default='merged_hd.png', 
                       help='Output filename (default: merged_hd.png)')
    parser.add_argument('--width', type=int, default=1920, 
                       help='Output width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, 
                       help='Output height in pixels (default: 1080)')
    parser.add_argument('--title', '-t', help='Title for the merged image')
    parser.add_argument('--pattern', '-p', default='*.png', 
                       help='File pattern to match (default: *.png)')
    parser.add_argument('--max-images', type=int, 
                       help='Maximum number of images to include')
    parser.add_argument('--margin', type=int, default=20, 
                       help='Margin around page edges in pixels (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        exit(1)
    
    success = merge_pngs_to_hd(
        input_folder=args.input_folder,
        output_path=args.output,
        width=args.width,
        height=args.height,
        margin=args.margin,
        title=args.title,
        pattern=args.pattern,
        max_images=args.max_images
    )
    
    if not success:
        exit(1)