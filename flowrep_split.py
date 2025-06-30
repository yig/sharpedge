#!/usr/bin/env python3
"""
OBJ Polyline Processor
Reads an OBJ file with vertices (v) and lines (l), calculates vertex valence,
and splits polylines at high valence vertices.
"""

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import sys 

def parse_obj_file(filename: str) -> Tuple[List[List[float]], List[List[int]]]:
    """Parse OBJ file and extract vertices and line segments."""
    vertices = []
    edges = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                coords = [float(x) for x in line.split()[1:]]
                vertices.append(coords)
            elif line.startswith('l '):
                indices = [int(x) for x in line.split()[1:]]
                edges.append(indices)
    
    return vertices, edges

def calculate_valence(edges: List[List[int]], num_vertices: int) -> Dict[int, int]:
    """Calculate the valence (degree) of each vertex."""
    valence = defaultdict(int)
    
    for edge in edges:
        if len(edge) == 2:
            v1, v2 = edge
            valence[v1] += 1
            valence[v2] += 1
    
    return valence

def build_adjacency_list(edges: List[List[int]]) -> Dict[int, Set[int]]:
    """Build adjacency list representation of the graph."""
    adj = defaultdict(set)
    
    for edge in edges:
        if len(edge) == 2:
            v1, v2 = edge
            adj[v1].add(v2)
            adj[v2].add(v1)
    
    return adj

def find_polylines(adj: Dict[int, Set[int]], valence: Dict[int, int], 
                  valence_threshold: int = 3) -> List[List[int]]:
    """
    Find polylines by tracing paths and splitting at high valence vertices.
    
    Args:
        adj: Adjacency list
        valence: Vertex valence dictionary
        valence_threshold: Vertices with valence >= this will be split points
    
    Returns:
        List of polylines, each polyline is a list of vertex indices
    """
    visited_edges = set()
    polylines = []
    
    # Find all high valence vertices (junction points)
    high_valence_vertices = {v for v, val in valence.items() if val >= valence_threshold}
    
    def trace_polyline(start_vertex: int, next_vertex: int) -> List[int]:
        """Trace a polyline from start_vertex through next_vertex until hitting a junction."""
        polyline = [start_vertex, next_vertex]
        current = next_vertex
        prev = start_vertex
        
        while True:
            # Mark this edge as visited
            edge = tuple(sorted([prev, current]))
            visited_edges.add(edge)
            
            # Find next vertex (should be exactly one unvisited neighbor, unless we hit a junction)
            neighbors = adj[current] - {prev}
            unvisited_neighbors = []
            
            for neighbor in neighbors:
                edge = tuple(sorted([current, neighbor]))
                if edge not in visited_edges:
                    unvisited_neighbors.append(neighbor)
            
            # Stop if we hit a high valence vertex or end of chain
            if current in high_valence_vertices or len(unvisited_neighbors) != 1:
                break
                
            # Continue to next vertex
            next_vertex = unvisited_neighbors[0]
            polyline.append(next_vertex)
            prev = current
            current = next_vertex
        
        return polyline
    
    # Start from high valence vertices and trace outgoing polylines
    for vertex in high_valence_vertices:
        for neighbor in adj[vertex]:
            edge = tuple(sorted([vertex, neighbor]))
            if edge not in visited_edges:
                polyline = trace_polyline(vertex, neighbor)
                if len(polyline) >= 2:
                    polylines.append(polyline)
    
    # Handle any remaining unvisited edges (isolated chains)
    for vertex in adj:
        for neighbor in adj[vertex]:
            edge = tuple(sorted([vertex, neighbor]))
            if edge not in visited_edges:
                polyline = trace_polyline(vertex, neighbor)
                if len(polyline) >= 2:
                    polylines.append(polyline)
    
    return polylines

def write_polylines_obj(vertices: List[List[float]], polylines: List[List[int]], 
                       output_filename: str):
    """Write vertices and polylines to a new OBJ file."""
    with open(output_filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, vertex))}\n")
        
        f.write("\n")
        
        # Write polylines
        for i, polyline in enumerate(polylines):
            f.write(f"l {' '.join(map(str, polyline))}\n")

def print_analysis(vertices: List[List[float]], edges: List[List[int]], 
                  valence: Dict[int, int], polylines: List[List[int]]):
    """Print analysis of the mesh."""
    print(f"Mesh Analysis:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Polylines found: {len(polylines)}")
    print()
    
    # Valence distribution
    valence_dist = defaultdict(int)
    for val in valence.values():
        valence_dist[val] += 1
    
    print("Valence distribution:")
    for val in sorted(valence_dist.keys()):
        print(f"  Valence {val}: {valence_dist[val]} vertices")
    print()
    
    # High valence vertices
    high_valence = [(v, val) for v, val in valence.items() if val >= 3]
    high_valence.sort(key=lambda x: x[1], reverse=True)
    
    print("High valence vertices (>= 3):")
    for vertex, val in high_valence[:10]:  # Show top 10
        print(f"  Vertex {vertex}: valence {val}")
    if len(high_valence) > 10:
        print(f"  ... and {len(high_valence) - 10} more")
    print()
    
    # Polyline statistics
    polyline_lengths = [len(p) for p in polylines]
    print("Polyline statistics:")
    print(f"  Average length: {sum(polyline_lengths) / len(polyline_lengths):.2f}")
    print(f"  Min length: {min(polyline_lengths)}")
    print(f"  Max length: {max(polyline_lengths)}")
    print()
    
    # Show first few polylines
    print("First 5 polylines:")
    for i, polyline in enumerate(polylines[:5]):
        print(f"  Polyline {i+1}: {polyline}")

def main():

    if len(sys.argv) != 3:
        print("Usage: python normalize_obj.py input.obj output.obj")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]


    valence_threshold = 3  # Split at vertices with 3+ connections
    
    try:
        # Parse input file
        vertices, edges = parse_obj_file(input_filename)
        
        # Calculate vertex valence
        valence = calculate_valence(edges, len(vertices))
        
        # Build adjacency list
        adj = build_adjacency_list(edges)
        
        # Find polylines
        polylines = find_polylines(adj, valence, valence_threshold)
        
        # Print analysis
        print_analysis(vertices, edges, valence, polylines)
        
        # Write output
        write_polylines_obj(vertices, polylines, output_filename)
        print(f"Polylines written to: {output_filename}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_filename}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()