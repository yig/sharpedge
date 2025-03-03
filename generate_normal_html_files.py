import os
from pathlib import Path
import re

def natural_sort_key(s):
    """Helper function to sort strings naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_common_model_names(base_path):
    """Get model names that exist across required folders"""
    # Look for model names based on text files
    text_path = Path(base_path) / 'debug_normals' / 'normal_info'
    
    if not text_path.exists():
        print(f"Warning: {text_path} does not exist")
        return []
    
    # Get all text files
    model_names = {f.name.replace('.txt', '.gltf') for f in text_path.glob('*.txt')}
    print(f"Found {len(model_names)} models based on text files")
    
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

def read_text_file(file_path):
    """Read text file contents safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def generate_hybrid_viewer(base_path):
    """Generate HTML with GLTF models and text information"""
    # Define the display titles and corresponding folders in debug_normals
    cell_info = [
        # First row
        {'folder': 'sketches', 'title': 'initial sketch'},
        {'folder': 'convex_hull', 'title': 'convex hull normal'},
        {'folder': 'edge_normals', 'title': 'edge normal from convex hull'},
        {'folder': 'normal_info', 'title': 'information txt', 'type': 'text'},
        
        # Second row
        {'folder': 'initial_most_perpendicular', 'title': 'most perpendicular normal'},
        {'folder': 'initial_parallel_transport', 'title': 'parallel on most perpendicular'},
        {'folder': 'borrowed_normal', 'title': 'borrowed normal'},
        {'folder': 'borrowed_parallel_transport', 'title': 'parallel on borrowed'},
        
        # Third row
        {'folder': 'initial_estimate', 'title': 'initial estimate'},
        {'folder': 'final_optimize', 'title': 'optimization'}
    ]
    
    model_names = get_common_model_names(base_path)
    grouped_models = group_by_prefix(model_names)
    
    print('Found prefixes:', list(grouped_models.keys()))
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Normal Information Viewer</title>
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
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 40px;
        }}
        .model-cell {{
            background: #444444;
            padding: 10px;
            border-radius: 8px;
            height: 300px;
            display: flex;
            flex-direction: column;
        }}
        .model-title {{
            text-align: center;
            margin-bottom: 8px;
            color: #ffffff;
            font-size: 14px;
        }}
        model-viewer {{
            width: 100%;
            height: 260px;
            background-color: #555555;
            --poster-color: #555555;
            flex-grow: 1;
        }}
        .text-content {{
            background-color: #555555;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            overflow: auto;
            height: 260px;
            flex-grow: 1;
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
            flex-wrap: wrap;
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
            <h1>Normal Information Viewer</h1>
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
            
            # Create grid for the model
            prefix_sections += f'''
            <div class="category-title">{display_name}</div>
            <div class="model-grid">'''
            
            # First row - 4 cells
            for i in range(4):
                cell = cell_info[i]
                
                if cell.get('type') == 'text':
                    # Handle text content
                    txt_file_path = Path(base_path) / 'debug_normals' / cell['folder'] / model_name.replace('.gltf', '.txt')
                    text_content = read_text_file(txt_file_path)
                    
                    prefix_sections += f'''
                <div class="model-cell">
                    <h3 class="model-title">{cell['title']}</h3>
                    <div class="text-content">{text_content}</div>
                </div>'''
                else:
                    # Handle GLTF model
                    model_path = f"debug_normals/{cell['folder']}/{model_name}"
                    
                    prefix_sections += f'''
                <div class="model-cell">
                    <h3 class="model-title">{cell['title']}</h3>
                    <model-viewer src="{model_path}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
                </div>'''
            
            # Second row - 4 cells
            prefix_sections += '''
            </div>
            <div class="model-grid">'''
            
            for i in range(4, 8):
                cell = cell_info[i]
                model_path = f"debug_normals/{cell['folder']}/{model_name}"
                
                prefix_sections += f'''
                <div class="model-cell">
                    <h3 class="model-title">{cell['title']}</h3>
                    <model-viewer src="{model_path}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
                </div>'''
            
            # Third row - 2 cells plus 2 empty
            prefix_sections += '''
            </div>
            <div class="model-grid">'''
            
            for i in range(8, 10):
                cell = cell_info[i]
                model_path = f"debug_normals/{cell['folder']}/{model_name}"
                
                prefix_sections += f'''
                <div class="model-cell">
                    <h3 class="model-title">{cell['title']}</h3>
                    <model-viewer src="{model_path}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
                </div>'''
            
            # Add empty cells to complete the grid
            prefix_sections += '''
                <div class="model-cell" style="visibility: hidden;"></div>
                <div class="model-cell" style="visibility: hidden;"></div>
            </div>'''
        
        prefix_sections += "\n        </div>"

    # Generate final HTML
    final_html = html_template.format(
        prefix_buttons=prefix_buttons,
        prefix_sections=prefix_sections
    )
    
    # Write to file
    output_path = Path(base_path) / 'normal_info_viewer.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    return output_path

# Example usage
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_file = generate_hybrid_viewer(base_path)
    print(f"HTML file generated at: {output_file}")