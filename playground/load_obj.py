#!/usr/bin/env python3

def load_obj(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse vertex
                v = [float(x) for x in line.split()[1:4]]
                vertices.append(v)
            elif line.startswith('f '):
                # Parse face
                # Convert to 0-based indexing by subtracting 1
                f = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                faces.append(f)
                
    return vertices, faces

# Example usage:
vertices, faces = load_obj('bunny.obj')

