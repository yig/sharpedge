import os
from pathlib import Path
import re

def natural_sort_key(s):
    """Helper function to sort strings naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_common_model_names(folders, base_path):
    """Get model names that exist across all folders"""
    model_names = set()
    first_folder = True
    
    for folder in folders:
        folder_path = Path(base_path) / 'gltfs' / folder

        print(folder_path)
        if not folder_path.exists():
            continue
            
        current_models = {f.name for f in folder_path.glob('*.gltf')}
        print(current_models)
        if first_folder:
            model_names = current_models
            first_folder = False
        else:
            model_names &= current_models
            
    return sorted(list(model_names), key=natural_sort_key)

def group_by_prefix(model_names):
    """Group model names by their prefixes (t2f, ils)"""
    groups = {}
    for name in model_names:
        if name.startswith('t2f_'):
            prefix = 't2f'
        elif name.startswith('ils_'):
            prefix = 'ils'
        elif name.startswith('onshape_'):
            prefix = 'onshape'
        elif name.startswith('flowrep_'):
            prefix = 'flowrep'
        elif name.startswith('cassie_'):
            prefix = 'cassie'
        elif name.startswith('author'):
            prefix = 'author_vr'
        else:
            prefix = 'other'
        
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(name)
    
    return {k: sorted(v, key=natural_sort_key) for k, v in groups.items()}

def generate_3d_viewer_html(base_path):
    """Generate HTML with 3D model viewers for each model type"""
    folders = ['sketches', 'normals', 'surfaces', 'pi_surface_gltf', 'wn_1_surface']
    model_names = get_common_model_names(folders, base_path)
    grouped_models = group_by_prefix(model_names)
    
    print('Found prefixes:', list(grouped_models.keys()))
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Model Viewer</title>
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
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .models-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .model-group {{
            background: #444444;
            padding: 15px;
            border-radius: 8px;
        }}
        .model-title {{
            text-align: center;
            margin-bottom: 10px;
            color: #ffffff;
        }}
        model-viewer {{
            width: 100%;
            height: 300px;
            background-color: #555555;
            --poster-color: #555555;
        }}
        .category-title {{
            color: #ffffff;
            margin: 30px 0 20px 0;
            font-size: 1.5em;
            text-align: center;
        }}
        .prefix-nav {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
        }}
        .prefix-button {{
            padding: 10px 20px;
            background-color: #444444;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 16px;
            text-transform: uppercase;
        }}
        .prefix-button:hover {{
            background-color: #555555;
        }}
        .prefix-button.active {{
            background-color: #666666;
        }}
        .prefix-section {{
            display: none;
        }}
        .prefix-section.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Model Viewer</h1>
            <div class="prefix-nav">
                {prefix_buttons}
            </div>
        </div>
        {prefix_sections}
    </div>

    <script>
        function showPrefix(prefix) {{
            // Hide all sections
            document.querySelectorAll('.prefix-section').forEach(section => {{
                section.classList.remove('active');
            }});
            // Show selected section
            document.getElementById(prefix + '-section').classList.add('active');
            // Update button states
            document.querySelectorAll('.prefix-button').forEach(button => {{
                button.classList.remove('active');
            }});
            document.querySelector(`[onclick="showPrefix('${{prefix}}')"]`).classList.add('active');
        }}

        // Show first prefix section by default
        document.addEventListener('DOMContentLoaded', () => {{
            const firstButton = document.querySelector('.prefix-button');
            if (firstButton) {{
                firstButton.click();
            }}
        }});
    </script>
</body>
</html>'''

    # Generate prefix buttons
    prefix_buttons = ""
    for prefix in grouped_models.keys():
        prefix_buttons += f'''
            <button class="prefix-button" onclick="showPrefix('{prefix}')">{prefix.upper()}</button>'''

    # Generate sections for each prefix
    prefix_sections = ""
    for prefix, models in grouped_models.items():
        prefix_sections += f'''
        <div id="{prefix}-section" class="prefix-section">'''
        
        for model_name in models:
            # Get display name by removing prefix and file extension
            display_name = model_name.replace(f'{prefix}_', '').replace('.gltf', '').replace('_', ' ').title()
            
            prefix_sections += f"""
            <div class="category-title">{display_name}</div>
            <div class="models-grid">"""
            
            for folder in folders:
                folder_display_name = folder.replace('_', ' ').title()
                prefix_sections += f"""
                <div class="model-group">
                    <h3 class="model-title">{folder_display_name}</h3>
                    <model-viewer src="gltfs/{folder}/{model_name}"
                        camera-controls
                        auto-rotate
                        shadow-intensity="1">
                    </model-viewer>
                </div>"""
                
            prefix_sections += "\n            </div>"
        
        prefix_sections += "\n        </div>"

    # Generate final HTML
    final_html = html_template.format(
        prefix_buttons=prefix_buttons,
        prefix_sections=prefix_sections
    )
    
    # Write to file
    output_path = Path(base_path) / 'index.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    return output_path

# Example usage
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_file = generate_3d_viewer_html(base_path)
    print(f"HTML file generated at: {output_file}")