Usage:
    ./cut_mesh <normal_file.normal> <surface_file.obj> [-t targetEdgeLength]

Cut the mesh using normal_file.normal and surface_file.obj.

- surface_file.obj is generated from the normal file.
- targetEdgeLength (optional): the edge length used to generate the surface file.
  Default = 0.04. If a different target length was used when generating the surface,
  it should match here as well.