import numpy as np 

def load_sketch_polyline_data(filename):
    """
    Parse OBJ file to extract vertex coordinates and polyline data.
    
    Args:
        filename (str): File name
        
    Returns:
        tuple: (V, E, P) where:
            - V: nx3 array of vertex coordinates 
            - E: mx2 array of edge vertex indices (no duplicates)
            - P: list of arrays containing vertex indices for each polyline
    """
    vertices = []
    polylines = []
    
    # Read and parse file line by line
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if not parts:
                continue
                
            # Parse vertex coordinates
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
                
            # Parse polyline data
            elif parts[0] == 'l':
                # Convert to 0-based indexing and store vertex indices
                polyline = [int(idx) - 1 for idx in parts[1:]]
                polylines.append(np.array(polyline))
    
    # Convert vertices to numpy array
    V = np.array(vertices)
    
    # Extract unique edges from polylines
    edges = set()
    for poly in polylines:
        # Create edges from consecutive vertices in polyline
        for i in range(len(poly) - 1):
            # Sort vertex indices to avoid duplicate edges
            v1, v2 = sorted([poly[i], poly[i + 1]])
            edges.add((v1, v2))
    
    # Convert edges to numpy array
    E = np.array(list(edges))
    
    # Store polylines as list of numpy arrays
    P = polylines



    
    print(f"\nRead from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} unique edges")
    print(f"- {len(polylines)} polylines")
    print()

    return V, E, P

def load_cdt_obj(filename):
    """
    Parse an OBJ file and extract vertices and lines.
    Returns vertices as a numpy array and lines as a list of index pairs.
    """
    vertices = []
    lines = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):  # vertex line
                # Split the line and convert coordinates to float
                coords = line.strip().split()[1:]
                vertex = [float(x) for x in coords]
                vertices.append(vertex)
            elif line.startswith('l '):  # line element
                # Get the two vertex indices (subtracting 1 because OBJ indices start at 1)
                indices = line.strip().split()[1:]
                assert len(indices) == 2, f"CDT file line must have exactly 2 indices, found {len(indices)}"
                idx0, idx1 = int(indices[0]) - 1, int(indices[1]) - 1
                lines.append((idx0, idx1))
    

    print(f"\nRead from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(lines)} lines")


    return np.array(vertices), lines


def load_normal_data(filename):
    """
    Read vertices, edges, and edge normal data from an OBJ file.
    
    Args:
        filename (str): Path to input OBJ file containing:
            - v x y z: vertex coordinates
            - e v1 v2: edge between vertices v1 and v2 (1-based indices)
            - vn nx ny nz: normal vector components
        
    Returns:
        tuple: (V, E, N) where:
            - V: nx3 array of vertex coordinates
            - E: mx2 array of edge vertex indices (0-based)
            - N: mx3 array of normal vectors corresponding to edges
    """
    # Initialize lists to store file data
    vertices = []
    edges = []
    normals = []
    
    # Read and parse file line by line
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                # Parse vertex coordinates (x, y, z)
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'l':
                # Parse edge vertex indices, converting from 1-based to 0-based indexing
                edges.append([int(x)-1 for x in parts[1:3]])
            elif parts[0] == 'vn':
                # Parse normal vector components (nx, ny, nz)
                normals.append([float(x) for x in parts[1:4]])
    
    # Convert lists to numpy arrays for efficient processing
    V = np.array(vertices)
    E = np.array(edges)
    N = np.array(normals)
    
    # Print summary statistics
    print(f"\nRead from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} edges")
    print(f"- {len(normals)} normal vectors")
    print()
    
    return V, E, N

def write_normal_data(V, E, N, filename):
    """
    Write vertices, edges, and edge normal data to an OBJ file.
    
    Args:
        V (ndarray): nx3 array of vertex coordinates (x, y, z)
        E (ndarray): mx2 array of edge vertex pairs (0-based indices)
        N (ndarray or dict): 
            - If ndarray: mx3 array of normal vectors for edges (nx, ny, nz)
            - If dict: Dictionary with edge indices as keys and normal vectors as values
        filename (str): Path to output OBJ file
        
    File format:
        v x y z     # vertex coordinates
        e i j       # edge between vertices i and j (1-based indices)
        vn nx ny nz # normal vector for edge
        
    Note:
        Edge indices are automatically converted from 0-based (input)
        to 1-based (OBJ file format) during writing.
    """  
    # Write data to file
    with open(filename, 'w') as f:
        # Write vertices with 6 decimal precision
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Write edges, converting from 0-based to 1-based indexing
        for i, e in enumerate(E):
            f.write(f"l {e[0]+1} {e[1]+1}\n")
        
        # Handle normal vectors based on type
        normal_count = 0
        if isinstance(N, dict):
            # If N is a dictionary, write normals by their keys
            for idx, normal in N.items():
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                normal_count += 1
        else:
            # If N is an array, write normals in order
            for normal in N:
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                normal_count += 1
    
    # Print summary of written data
    print(f"Wrote to {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {normal_count} normal vectors")

def write_two_normal(V, E, normals, filename):
    """
    Write vertices, edges, and dual normal data to an OBJ file.
    Each edge has exactly two normal vectors.
    
    Args:
        V (ndarray): nx3 array of vertex coordinates (x, y, z)
        E (ndarray): mx2 array of edge vertex pairs (0-based indices)
        normals (dict): Dictionary with keys (edge_idx, which_edge) where:
                        - edge_idx is the edge index
                        - which_edge is 0 or 1 for the first or second normal
                        Values are 3D normal vectors (nx, ny, nz)
        filename (str): Path to output OBJ file
        
    File format:
        v x y z     # vertex coordinates
        l i j       # edge between vertices i and j (1-based indices)
        vn nx ny nz # normal vector for edge
        
    Note:
        Edge indices are automatically converted from 0-based (input)
        to 1-based (OBJ file format) during writing.
        For each edge, both normal vectors are written consecutively.
    """
    # Write data to file
    with open(filename, 'w') as f:
        # Write vertices with 6 decimal precision
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Write edges, converting from 0-based to 1-based indexing
        for i, e in enumerate(E):
            f.write(f"l {e[0]+1} {e[1]+1}\n")
        
        # Get all unique edge indices in sorted order
        edge_indices = sorted(set(edge_idx for edge_idx, _ in normals.keys()))
        
        # Write both normals for each edge
        for edge_idx in edge_indices:
            # Write first normal (0)
            normal = normals[(edge_idx, 0)]
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            # Write second normal (1)
            normal = normals[(edge_idx, 1)]
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
    
    # Print summary of written data
    print(f"Wrote to {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {len(edge_indices) * 2} normal vectors (2 per edge)")

def load_mesh(filename):
    """
    Read a tetrahedral mesh from either .tet or .node/.ele files.
    
    Args:
        filename: Path to either a .tet file or .node file
        
    Returns:
        vertices: List of [x, y, z] coordinates
        tets: List of tetrahedral indices (0-based)
        
    Raises:
        ValueError: If file format is not supported or data is invalid
        FileNotFoundError: If file(s) don't exist
    """
    from pathlib import Path
    
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filename}")
        
    # Helper function to remove comments and empty lines
    def clean_lines(file):
        return [line.split('#')[0].strip() for line in file if line.split('#')[0].strip()]
        
    if filepath.suffix == '.tet':
        # Read .tet format
        with open(filepath, 'r') as f:
            lines = clean_lines(f)
            
            # Parse header
            n_vertices = int(lines[0].split()[0])
            n_inner_tets = int(lines[1].split()[0])
            n_outer_tets = int(lines[2].split()[0])
            
            # Read vertices
            vertices = []
            current_line = 3
            for i in range(n_vertices):
                x, y, z = map(float, lines[current_line + i].split())
                vertices.append([x, y, z])
                
            # Read tetrahedra
            tets = []
            current_line += n_vertices
            for i in range(n_inner_tets + n_outer_tets):
                indices = list(map(int, lines[current_line + i].split()))
                tets.append(indices[1:])  # Skip the first number (4)
                
    elif filepath.suffix == '.node':
        # Check for corresponding .ele file
        ele_path = filepath.with_suffix('.ele')
        if not ele_path.exists():
            raise FileNotFoundError(f"Required .ele file not found: {ele_path}")
            
        # Read .node file
        with open(filepath, 'r') as f:
            lines = clean_lines(f)
            
            # Parse header
            n_vertices, dim, n_attr, n_boundary = map(int, lines[0].split())
            if dim != 3:
                raise ValueError(f"Expected 3D data, got dimension {dim}")
            
            # Read vertices
            vertices = []
            for i in range(n_vertices):
                parts = lines[i + 1].split()
                # Skip index, take x,y,z
                vertices.append(list(map(float, parts[1:4])))
        
        # Read .ele file
        with open(ele_path, 'r') as f:
            lines = clean_lines(f)
            
            # Parse header
            n_tets, nodes_per_tet, n_attr = map(int, lines[0].split())
            if nodes_per_tet != 4:
                raise ValueError(f"Expected 4 nodes per tet, got {nodes_per_tet}")
            
            # Read tetrahedra
            tets = []
            for i in range(n_tets):
                parts = lines[i + 1].split()
                # Convert from 1-based to 0-based indexing, skip tet index
                tet = [int(idx) - 1 for idx in parts[1:5]]
                tets.append(tet)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Must be .tet or .node")
    
    print(f"\nRead from {filename}:")
    print(f"Mesh elements:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(tets)} tetrahedra")
    print()


    return vertices, tets

    
def export_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
        # Write faces (add 1 because OBJ is 1-indexed)
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    
    # Print summary of written data
    print(f"Wrote to {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(faces)} faces")

def load_obj(filename):
    """
    Load vertices and faces from an OBJ file.
    
    Args:
        filename (str): Path to the OBJ file
        
    Returns:
        tuple: (vertices, faces) where:
            vertices is a list of [x, y, z] coordinates
            faces is a list of [v1, v2, v3] vertex indices (0-indexed)
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skip comments
                continue
                
            values = line.split()
            if not values:  # Skip empty lines
                continue
                
            if values[0] == 'v':  # Vertex
                v = [float(x) for x in values[1:4]]  # Only take x, y, z coordinates
                vertices.append(v)
                
            elif values[0] == 'f':  # Face
                # Convert face indices to 0-based indexing
                # Handle both formats: 'f 1 2 3' and 'f 1/1/1 2/2/2 3/3/3'
                face = []
                for v in values[1:4]:  # Only take first three vertices for triangular faces
                    # Split on '/' and take the first index (vertex index)
                    vertex_idx = int(v.split('/')[0]) - 1  # Convert to 0-based indexing
                    face.append(vertex_idx)
                faces.append(face)
    
    # Print summary of read data
    print(f"Read from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(faces)} faces")
    
    return vertices, faces

def write_string_to_file(content, filename, encoding='utf-8'):
    """
    Write a string to a file.
    
    Parameters
    ----------
    content : str
        The string content to write to the file
    filename : str
        Path to the output file
    encoding : str, default='utf-8'
        File encoding to use
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        with open(filename, 'w', encoding=encoding) as file:
            file.write(content)
        print(f"Successfully wrote to file: {filename}")
        return True
    except Exception as e:
        print(f"Error writing to file {filename}: {str(e)}")
        return False