"""
Generate HTML for 3D GLTF models viewing.

This script scans subfolders in a given GLTF directory and finds sets of models
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
    """Find all base object names that have the required 5 GLTF files"""
    gltf_dir = Path(gltf_path)
    if not gltf_dir.exists():
        print(f"Warning: {gltf_dir} does not exist")
        return {}
    
    # Find all subfolders
    subfolders = [d for d in gltf_dir.iterdir() if d.is_dir()]
    
    objects_by_folder = {}
    
    for subfolder in subfolders:
        folder_name = subfolder.name
        objects_by_folder[folder_name] = []
        
        # Find all .gltf files
        gltf_files = list(subfolder.glob('*.gltf'))
        
        # Group files by base name
        base_names = set()
        for gltf_file in gltf_files:
            name = gltf_file.stem
            # Remove suffixes to get base name
            if name.endswith('_n0') or name.endswith('_n1') or name.endswith('_sketch') or name.endswith('_surface'):
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
            
            if all((subfolder / req_file).exists() for req_file in required_files):
                objects_by_folder[folder_name].append(base_name)
        
        # Sort objects naturally
        objects_by_folder[folder_name] = sorted(objects_by_folder[folder_name], key=natural_sort_key)
        print(f"Found {len(objects_by_folder[folder_name])} complete objects in {folder_name}")
    
    # Remove empty folders
    objects_by_folder = {k: v for k, v in objects_by_folder.items() if v}
    
    return objects_by_folder

def generate_gltf_viewer(gltf_path):
    """Generate HTML viewer for GLTF files with 5 columns"""
    
    objects_by_folder = find_gltf_objects(gltf_path)
    
    if not objects_by_folder:
        print("No valid GLTF object sets found!")
        return None
    
    print('Found folders:', list(objects_by_folder.keys()))
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLTF Model Viewer (5 Columns)</title>
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
        .folder-nav {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .folder-button {{
            padding: 12px 24px;
            background-color: #444444;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-size: 16px;
            text-transform: capitalize;
            transition: background-color 0.3s;
        }}
        .folder-button:hover {{
            background-color: #555555;
        }}
        .folder-button.active {{
            background-color: #666666;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .folder-section {{
            display: none;
        }}
        .folder-section.active {{
            display: block;
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
            <h1>GLTF Model Viewer (5 Columns)</h1>
            <div class="folder-nav">
                {folder_buttons}
            </div>
        </div>
        
        {folder_sections}
    </div>

    <script>
        function showFolder(folder) {{
            // Hide all sections
            document.querySelectorAll('.folder-section').forEach(section => {{
                section.classList.remove('active');
            }});
            // Show selected section
            document.getElementById(folder + '-section').classList.add('active');
            // Update button states
            document.querySelectorAll('.folder-button').forEach(button => {{
                button.classList.remove('active');
            }});
            document.querySelector(`[onclick="showFolder('${{folder}}')"]`).classList.add('active');
        }}

        // Show first folder section by default
        document.addEventListener('DOMContentLoaded', () => {{
            const firstButton = document.querySelector('.folder-button');
            if (firstButton) {{
                firstButton.click();
            }}
        }});
    </script>
</body>
</html>'''

    # Generate folder buttons
    folder_buttons = ""
    for folder in objects_by_folder.keys():
        folder_buttons += f'''
            <button class="folder-button" onclick="showFolder('{folder}')">{folder}</button>'''

    # Generate sections for each folder
    folder_sections = ""
    for folder_name, objects in objects_by_folder.items():
        folder_sections += f'''
        <div id="{folder_name}-section" class="folder-section">'''
        
        for obj_name in objects:
            # Use the original object name without formatting
            display_name = obj_name
            
            folder_sections += f'''
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
                model_path = f"gltf/{folder_name}/{obj_name}{suffix}.gltf"
                
                folder_sections += f'''
                <div class="model-cell">
                    <h3 class="model-title">{title}</h3>
                    <model-viewer src="{model_path}" camera-controls auto-rotate shadow-intensity="1" 
                                  camera-orbit="45deg 75deg 2m" field-of-view="30deg"></model-viewer>
                </div>'''
            
            folder_sections += '''
            </div>'''
        
        folder_sections += "\n        </div>"

    # Generate final HTML
    final_html = html_template.format(
        folder_buttons=folder_buttons,
        folder_sections=folder_sections
    )
    
    # Write to file
    output_path = Path(gltf_path).parent / 'gltf_viewer.html'
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
        print("Failed to generate HTML file - no valid GLTF objects found.")