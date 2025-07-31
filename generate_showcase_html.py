"""
Generate HTML for 3D GLTF models viewing - Modified to only process 'good' folder.

This script scans the 'good' subfolder in a given GLTF directory and finds sets of models
that include the following five files for each object:
  - {name}_sketch.gltf
  - {name}_n0.gltf
  - {name}_n1.gltf
  - {name}_2n.gltf       (2 normals)
  - {name}_surface.gltf

For each valid object, an interactive HTML viewer is generated with 5 columns
(using <model-viewer>) for side-by-side comparison.

Output: gltf_viewer.html
"""

import os
from pathlib import Path
import re

def natural_sort_key(s):
    """Helper function to sort strings naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def find_gltf_objects(gltf_path):
    """Find all base object names that have the required 5 GLTF files in the 'good' folder"""
    gltf_dir = Path(gltf_path)
    if not gltf_dir.exists():
        print(f"Warning: {gltf_dir} does not exist")
        return []
    
    # Only look in the 'good' subfolder
    good_folder = gltf_dir / 'good'
    if not good_folder.exists():
        print(f"Warning: 'good' folder not found in {gltf_dir}")
        return []
    
    objects = []
    
    # Find all .gltf files in the good folder
    gltf_files = list(good_folder.glob('*.gltf'))
    
    # Group files by base name
    base_names = set()
    for gltf_file in gltf_files:
        name = gltf_file.stem
        # Remove suffixes to get base name
        if name.endswith('_n0') or name.endswith('_n1') or name.endswith('_sketch') or name.endswith('_surface') or name.endswith('_2n'):
            base_name = '_'.join(name.split('_')[:-1])
        else:
            base_name = name
        base_names.add(base_name)
    
    # Check which base names have all 5 required files
    for base_name in base_names:
        required_files = [
            f"{base_name}_2n.gltf",
            f"{base_name}_n0.gltf", 
            f"{base_name}_n1.gltf",
            f"{base_name}_sketch.gltf",
            f"{base_name}_surface.gltf"
        ]
        
        if all((good_folder / req_file).exists() for req_file in required_files):
            objects.append(base_name)
    
    # Sort objects naturally
    objects = sorted(objects, key=natural_sort_key)
    print(f"Found {len(objects)} complete objects in 'good' folder")
    
    return objects

def generate_gltf_viewer(gltf_path):
    """Generate HTML viewer for GLTF files with 5 columns - only from 'good' folder"""
    
    objects = find_gltf_objects(gltf_path)
    
    if not objects:
        print("No valid GLTF object sets found in 'good' folder!")
        return None
    
    print('Found objects:', objects)
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLTF Model Viewer - Good Models</title>
    <script type="module" src="https://cdnjs.cloudflare.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background-color: #333333;
            color: white;
            font-family: Arial, sans-serif;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
            margin-bottom: 40px;
        }}
        .model-cell {{
            background: #444444;
            padding: 15px;
            border-radius: 8px;
            height: 350px;
            display: flex;
            flex-direction: column;
        }}
        .model-title {{
            text-align: center;
            margin-bottom: 10px;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
        }}
        model-viewer {{
            width: 100%;
            height: 300px;
            background-color: #555555;
            --poster-color: #555555;
            flex-grow: 1;
            border-radius: 5px;
        }}
        .object-title {{
            color: #cccccc;
            margin: 30px 0 15px 0;
            font-size: 1.2em;
            text-align: center;
            font-weight: normal;
        }}
        @media (max-width: 1400px) {{
            .model-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
        @media (max-width: 1000px) {{
            .model-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        @media (max-width: 768px) {{
            .model-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GLTF Model Viewer - Good Models</h1>
        </div>
        
        {object_sections}
    </div>
</body>
</html>'''

    # Generate sections for each object
    object_sections = ""
    for obj_name in objects:
        # Use the original object name without formatting
        display_name = obj_name
        
        object_sections += f'''
        <div class="object-title">{display_name}</div>
        <div class="model-grid">'''
        
        # Define the 5 file variants and their titles
        variants = [
            ('_sketch', 'Sketch'),
            ('_n0', 'Normal 0'),
            ('_n1', 'Normal 1'),
            ('_2n', '2 Normals'),
            ('_surface', 'Surface')
        ]
        
        for suffix, title in variants:
            model_path = f"gltf/good/{obj_name}{suffix}.gltf"
            
            object_sections += f'''
            <div class="model-cell">
                <h3 class="model-title">{title}</h3>
                <model-viewer src="{model_path}" camera-controls auto-rotate shadow-intensity="1" 
                              camera-orbit="45deg 75deg 2m" field-of-view="30deg"></model-viewer>
            </div>'''
        
        object_sections += '''
        </div>'''

    # Generate final HTML
    final_html = html_template.format(
        object_sections=object_sections
    )
    
    # Write to file
    output_path = Path(gltf_path).parent / 'models_showcase.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    return output_path

# Example usage
if __name__ == "__main__":
    # Set the path to your gltf folder
    gltf_folder_path = "gltf"  # Change this to your actual path
    
    output_file = generate_gltf_viewer(gltf_folder_path)
    
    if output_file:
        print(f"HTML file generated at: {output_file}")
        print("Open the HTML file in a web browser to view your GLTF models!")
    else:
        print("Failed to generate HTML file - no valid GLTF objects found in 'good' folder.")