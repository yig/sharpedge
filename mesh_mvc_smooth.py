# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "libigl",
#     "numpy",
#     "scipy",
#     "polyscope",
# ]
# ///
import igl
import numpy as np
import scipy as sp
import polyscope as ps
from pprint import pprint
from collections import defaultdict

def solve_system_with_constraints_hard( A, b, constraint_indices, constraint_values ):
    '''
    Given:
        A: A system matrix
        b: A right hand side
        constraint_indices: A sequence of indices to constrain in x
        constraint_values: A sequence of values to constrain x to corresponding to `constraint_indices`
    Returns:
        The solution x to Ax=b such that x[ constraint_indices ] = constraint_values.
    '''

    ## Outline:
    '''
    If solving a system of equations:
    Ax=b
    Assume A is symmetric. Let A_*,k be the column corresponding to a fixed value x_k=b_k. If we update the matrix to zero row k and place 1 at A_k,k, we can keep A symmetric by also zeroing column k. Column k are the coefficients of the known value that should be added to each entry to equal b. We can update the right hand side to subtract the coefficients times the known values.

    b' = b - x_k A_*,k

    For many constraints, we can write:

    b' = b - Ar, where r_k = x_k if x_k is constrained and 0 otherwise.
    Then assign b'_k = r_k for constraints k, zero A's k-th row and column, and set A_k,k = 1.
    '''

    r = np.zeros_like(b)
    r[ constraint_indices ] = constraint_values

    b_prime = b - A @ r
    b_prime[ constraint_indices ] = r[ constraint_indices ]

    A = A.copy()
    A[:,constraint_indices] = 0
    A[constraint_indices,:] = 0
    A[constraint_indices,constraint_indices] = 1.0

    x = sp.sparse.linalg.spsolve( A, b_prime )
    assert np.abs( x[constraint_indices] - constraint_values ).max() < 1e-9
    return x

def cotmatrix( V, F ):
    # return igl.cotmatrix(V, F)
    l = igl.edge_lengths(V,F)
    l_intrinsic, F_intrinsic = igl.intrinsic_delaunay_triangulation(l,F)[:2]

    ## Mollify the intrinsic lengths to avoid numerical issues [Sharp and Crane 2020].
    # Every non-boundary edge is in l_intrinsic twice. Taking the mean double-counts non-boundary edges.
    # That won't matter for a closed mesh, but will slightly undercount boundary edges for a mesh with boundary.
    delta = 1e-4 * np.mean(l_intrinsic)
    
    '''
    eps2 = 0.
    for T in l_intrinsic:
        for i in range(3):
            eps2 = np.maximum( eps2, np.maximum( 0, delta - T[i] - T[(i+1)%3] + T[(i+2)%3] ) )
    '''

    eps = 0.
    for i in range(3):
        eps = np.maximum( eps, np.maximum( 0, delta - l_intrinsic[:,i] - l_intrinsic[:,(i+1)%3] + l_intrinsic[:,(i+2)%3] ).max() )
    
    # assert eps == eps2

    l_intrinsic += eps

    L = igl.cotmatrix_intrinsic(l_intrinsic, F_intrinsic)
    return L

def smooth_laplacian_magnitude_rivers(V, F, constrained = None, iterations=10, step_size = 1e-4, threshold=1e-5, plot_every=1, recompute_matrices = True, recompute_normals = True, laplacian = 'graph', fix_errata = True):
    print( "smooth_laplacian_magnitude_rivers() with parameters:" )
    pprint( locals() )

    V_new = V.copy()

    import pdb
    pdb.set_trace()

    for iter in range(iterations):
        if iter == 0 or recompute_matrices:
            if laplacian == 'cotangent':
                ## Cotangent laplacian
                L = cotmatrix(V, F)
            elif laplacian == 'graph':
                ## Graph laplacian
                A = igl.adjacency_matrix(F)
                L = A - sp.sparse.diags(np.asarray(A.sum(axis=1)).squeeze())
            else:
                raise NotImplementedError(f"Unknown Laplacian type: {laplacian}")
            
            ## The inverse mass matrix doesn't make the diagonal -1.
            ## We need diagonal values to be -1 for the weighted averaging operator.
            #M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
            # negative because L's diagonal is negative and should stay negative
            Minv = sp.sparse.diags(1 / -L.diagonal())
            ## Build the weighted averaging operator.
            wavg = Minv @ L
            ## Zero wavg's diagonal.
            wavg = wavg.tocoo()
            wavg.setdiag(0)
            wavg = wavg.tocsr()

        if iter == 0 or recompute_normals:
            N = igl.per_vertex_normals(V, F)

        # Step 1: compute Laplacian magnitude
        c = np.linalg.norm(-Minv @ L @ V, axis=1)

        if iter == 0: plot(v, f, c, title="Initial Laplacian magnitude")

        # Step 2: average Laplacian magnitude
        c_avg = wavg @ c
        # pdb.set_trace()

        # Step 3: compute vertex normals using current V_new
        vertex_normals = igl.per_vertex_normals(V_new, F)
        # plot_normal(V_new, F, vertex_normals)
        
        # Step 4: update vertex positions along normal direction
        displacement = np.zeros_like(V_new)
        max_disp = 0.0

        for i in range(len(V_new)):
            # Skip constrained vertices
            if i in constrained: continue
            
            direction = vertex_normals[i]
            d = c_avg[i] * direction
            displacement[i] = d
            # max_disp = max(max_disp, np.linalg.norm(d))

        if fix_errata:
            V_new_new = wavg @ V_new + displacement
        else:
            V_new_new = V_new + displacement
        
        # Keep constrained vertices fixed
        V_new_new[constrained] = V_new[constrained]

        max_disp = np.max( np.linalg.norm( V_new_new - V_new, axis = 1 ) )

        print(f"[iter {iter+1}] max displacement: {max_disp:.6f}")
        if max_disp < threshold:
            ## Break after finishing this loop.
            iter = iterations - 1

        V_new = V_new_new

        if iter % plot_every == 0 or iter == iterations - 1:
            plot(V_new, f, c)
        
        ## The loop iterator doesn't respect us setting iter = iterations - 1, which
        ## we do to trigger the final saving and plotting.
        if iter == iterations - 1: break
    
    return V_new

def smooth_fibermesh(V, F,
                     constrained = None,
                     iterations=100, threshold=1e-5,
                     recompute_normals = True,
                     recompute_matrices = False,
                     plot_every=1,
                     save_every=0, save_base='fibermesh_'
                     ):
    print( "smooth_fibermesh() with parameters:" )
    pprint( locals() )

    V_orig = V
    V = V.copy()

    import pdb
    pdb.set_trace()
    
    for iter in range(iterations):
        if iter == 0 or recompute_matrices:
            L = cotmatrix(V, F)
            M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
            Minv = sp.sparse.diags(1 / M.diagonal())
        
        # Step 1: Smooth Laplacian magnitudes
        if iter == 0 or recompute_normals:
            N = igl.per_vertex_normals(V, F)
        # Negate and halve because the laplacian computes -2Hn
        Hn = -.5 * Minv @ L @ V
        H = np.linalg.norm(Hn, axis=1) * np.sign( ( Hn * N ).sum(axis=1) )
        
        A = (L.T@Minv@L).tocsc()
        B = np.zeros(V.shape[0])
        # known = np.asarray(constrained)
        # Y = H[constrained]
        # Aeq = sp.sparse.coo_matrix( (0,V_new.shape[0]) ).tocsc()
        # Beq = np.zeros((0,V_new.shape[1]))
        # H_smoothed = igl.min_quad_with_fixed( A, B, known, Y, Aeq, Beq, True )
        H_smoothed = solve_system_with_constraints_hard( A, B, constrained, H[constrained] )

        # Step 2: Update positions
        target_LM = -2 * ( L.T @ H_smoothed[:,None] * N )
        V_new = solve_system_with_constraints_hard( A, target_LM, constrained, V[constrained] )

        max_disp = np.max( np.linalg.norm( np.abs( V_new - V ), axis = 1 ) )
        V = V_new
        
        print(f"[iter {iter+1}] max displacement: {max_disp:.6f}")
        if max_disp < threshold:
            ## Break after finishing this loop.
            iter = iterations - 1
        
        ## Save the first, last, and every `save_every` iterations.
        if save_every != 0 and (iter % save_every == 0 or iter == iterations - 1):
            igl.write_triangle_mesh( save_base + f'{iter:05d}.obj', V, F )
            print( "Saved:", save_base + f'{iter:05d}.obj' )

        ## Plot on the first, last, and every `plot_every` iterations.
        if plot_every != 0 and (iter % plot_every == 0 or iter == iterations - 1):
            plot(V, F, H_smoothed)
        
        ## The loop iterator doesn't respect us setting iter = iterations - 1, which
        ## we do to trigger the final saving and plotting.
        if iter == iterations - 1: break

    return V

def compute_curvature_cot( V, F, N ):
    L = cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())

    # Negate and halve because the laplacian computes -2Hn
    Hn = -.5 * (Minv @ L @ V)
    H = np.linalg.norm(Hn, axis=1) * np.sign( ( Hn * N ).sum(axis=1) )
    
    return Hn, H

def compute_curvature_stanko( V, F, N, method = 'EG' ):
    Hn = np.zeros( N.shape )
    ## Use the method from Stanko et al. 2016 ("C&G" or "EG" short paper)
    if method == 'C&G':
        for i,j,k in F:
            n = N[i] + N[j] + N[k]
            n /= np.linalg.norm(n)

            Hn[i] += np.cross( n, V[k] - V[j] )
            Hn[j] += np.cross( n, V[i] - V[k] )
            Hn[k] += np.cross( n, V[j] - V[i] )
    elif method == 'EG':
        for i,j,k in F:
            n_jk = N[j] + N[k]
            n_ij = N[i] + N[j]
            n_ki = N[k] + N[i]
            n_jk /= np.linalg.norm(n_jk)
            n_ij /= np.linalg.norm(n_ij)
            n_ki /= np.linalg.norm(n_ki)
            
            Hn[i] += np.cross( n_jk, V[k] - V[j] )
            Hn[j] += np.cross( n_ij, V[i] - V[k] )
            Hn[k] += np.cross( n_ki, V[j] - V[i] )
    else:
        raise NotImplementedError(f"Unknown curvature computation method: {method}")

    # These H are scaled by the mass (voronoi area) of each vertex
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())

    # Divide by 4 because this computes 4Hn
    Hn = .25 * Minv @ Hn
    H = np.linalg.norm(Hn, axis=1) * np.sign( ( Hn * N ).sum(axis=1) )

    # Return Hn and H
    return Hn, H

def smooth_stanko(V, F, N = None, constrained = None, target_curvature_paper = True, curvature_method = 'C&G' ):
    print( "smooth_stanko() called with parameters:" )
    pprint( locals() )

    # import pdb
    # pdb.set_trace()
    
    L = cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())
    
    # Step 1: Smooth normals
    if N is None: N = igl.per_vertex_normals(V, F)
    
    # A = (L.T@Minv@L).tocsc()
    A = (L @ Minv @ L @ Minv @ L).tocsc() 

    B = np.zeros(N.shape)
    N_star = solve_system_with_constraints_hard( A, B, constrained, N[constrained] )
    # Normalize the computed normals
    N_star /= np.linalg.norm( N_star, axis = 1 )[:,None]

    # Step 2: Smooth positions
    V_star = solve_system_with_constraints_hard( A, np.zeros(V.shape), constrained, V[constrained] )

    # Step 3: Compute target curvatures
    Hn, H = compute_curvature_stanko( V_star, F, N_star, method = curvature_method )
    # plot(V, F, H)

    if target_curvature_paper:
        target_LM = -2 * ( L.T @ (np.abs(H[:,None]) * N_star) )
    else:
        target_LM = -2 * ( L.T @ Hn )
    V_new = solve_system_with_constraints_hard( A, target_LM, constrained, V[constrained] )

    max_disp = np.max( np.linalg.norm( np.abs( V_new - V ), axis = 1 ) )
    V = V_new
    
    print(f"max displacement: {max_disp:.6f}")

    plot(V, F, H)

    return V

def laplace_order( V, F, order ):
    '''
    Given:
        V: #Vx3 array of vertex positions
        F: #Fx3 array of triangle indices
        order: The order of the Laplacian to compute
    Returns:
        A sparse matrix representing the `order`-th power of the Laplacian operator.
    '''

    ## This follows igl::harmonic().
    ## There is a comment that gptoolbox kharmonic is better.
    L = cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())

    laplace = -Minv @ L
    for i in range(1,order):
        laplace = laplace @ (-Minv @ L)
    
    return laplace

def smooth_flow(V, F,
                constrained = None,
                order = 1,
                step_size = 1e-2,
                step_mode = 'explicit',
                iterations=100, threshold=1e-5,
                recompute_matrices = True,
                plot_every=1,
                save_every=0, save_base='flow_'
                ):
    print( "smooth_flow() called with parameters:" )
    pprint( locals() )
    
    V_orig = V
    V = V.copy()

    import pdb
    pdb.set_trace()
    
    for iter in range(iterations):
        if iter == 0 or recompute_matrices:
            laplace = laplace_order( V, F, order )

        if step_mode == 'explicit':
            V_new = V - .5 * step_size * laplace @ V
            V_new[ constrained ] = V_orig[ constrained ]
        elif step_mode == 'implicit':
            update_operator = sp.sparse.eye(V.shape[0]) + .5 * step_size * laplace
            # V_new = sp.sparse.linalg.spsolve( update_operator, V )
            V_new = solve_system_with_constraints_hard( update_operator, V, constrained, V_orig[constrained] )
        else:
            raise NotImplementedError(f"Unknown step_mode: {step_mode}")

        max_disp = np.max( np.linalg.norm( np.abs( V_new - V ), axis = 1 ) )
        V = V_new
        
        print(f"[iter {iter+1}] max displacement: {max_disp:.6f}")
        if max_disp < threshold:
            ## Break after finishing this loop.
            iter = iterations - 1
        
        ## Save the first, last, and every `save_every` iterations.
        if save_every != 0 and (iter % save_every == 0 or iter == iterations - 1):
            igl.write_triangle_mesh( save_base + f'{iter:05d}.obj', V, F )
            print( "Saved:", save_base + f'{iter:05d}.obj' )

        ## Plot on the first, last, and every `plot_every` iterations.
        if plot_every != 0 and (iter % plot_every == 0 or iter == iterations - 1):
            plot(V, F, compute_curvature_cot(V,F,igl.per_vertex_normals(V, F))[1])
        
        ## The loop iterator doesn't respect us setting iter = iterations - 1, which
        ## we do to trigger the final saving and plotting.
        if iter == iterations - 1: break

    return V

def smooth_flow_projected(V, F,
                constrained = None,
                order = 1,
                step_size = 1e-2,
                step_mode = 'explicit',
                iterations=100, threshold=1e-5,
                recompute_matrices = True,
                recompute_normals = True,
                plot_every=1,
                save_every=0, save_base='flow_'
                ):
    print( "smooth_flow_projected() called with parameters:" )
    pprint( locals() )

    V_orig = V
    V = V.copy()

    import pdb
    pdb.set_trace()
    
    for iter in range(iterations):
        if iter == 0 or recompute_matrices:
            laplace = laplace_order( V, F, order )
            
            # Now broadcast for our projected system
            laplace = sp.sparse.block_diag( [laplace, laplace, laplace] )
        
        ## Build the normal projection matrix that takes a vector of n normal displacement values d,
        ## one per vertex, and products a 3n vector d1x1 d2x2 d3x3 ... d1y1 d2y2 d3y3 ... d1z1 d2z2 d3z3 ...
        if iter == 0 or recompute_normals:
            N = igl.per_vertex_normals(V, F)
            Nproj = sp.sparse.block_array( [ [sp.sparse.diags(N[:,c])] for c in range(3) ] )

        if step_mode == 'explicit':
            V_new = V.ravel('F') - .5 * step_size * ( Nproj @ Nproj.T @ ( laplace @ V.ravel('F') - V_orig.ravel('F') ) + V_orig.ravel('F') )
            V_new = V_new.reshape(-1,3,order='F')
            V_new[ constrained ] = V_orig[ constrained ]
        elif step_mode == 'implicit':
            update_operator = sp.sparse.eye(laplace.shape[0]) + .5 * step_size * laplace
            update_operator = Nproj.T @ update_operator @ Nproj
            rhs = -step_size * Nproj.T @ laplace @ V.ravel('F')
            # V_new = sp.sparse.linalg.spsolve( update_operator, V )
            z = solve_system_with_constraints_hard( update_operator, rhs, constrained, np.zeros(len(constrained)) )
            V_new = Nproj @ z + V.ravel('F')
            V_new = V_new.reshape(-1,3,order='F')
            assert np.abs( V_new[ constrained ] - V[ constrained ] ).max() < 1e-7
        else:
            raise NotImplementedError(f"Unknown step_mode: {step_mode}")
        
        max_disp = np.max( np.linalg.norm( np.abs( V_new - V ), axis = 1 ) )
        V = V_new
        
        print(f"[iter {iter+1}] max displacement: {max_disp:.6f}")
        if max_disp < threshold:
            ## Break after finishing this loop.
            iter = iterations - 1
        
        ## Save the first, last, and every `save_every` iterations.
        if save_every != 0 and (iter % save_every == 0 or iter == iterations - 1):
            igl.write_triangle_mesh( save_base + f'{iter:05d}.obj', V, F )
            print( "Saved:", save_base + f'{iter:05d}.obj' )

        ## Plot on the first, last, and every `plot_every` iterations.
        if plot_every != 0 and (iter % plot_every == 0 or iter == iterations - 1):
            plot(V, F, compute_curvature_cot(V,F,igl.per_vertex_normals(V, F))[1])
        
        ## The loop iterator doesn't respect us setting iter = iterations - 1, which
        ## we do to trigger the final saving and plotting.
        if iter == iterations - 1: break

    return V

def plot(v, f, k, title="Curvature"):
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.add_scalar_quantity(title, k, defined_on='vertices', cmap='coolwarm', enabled=True)
    ps_mesh.set_edge_width(1)
    ps.set_ground_plane_mode('none')
    ps.show()

def plot_normal(v, f, n):
    '''
    '''
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.add_vector_quantity("vertex normals", n, defined_on="vertices", enabled=True)
    ps.set_ground_plane_mode('none')
    ps.show()

def extract_boundary_vertices(vertices, faces):
    """直接提取网格的边界顶点"""
    # 统计每条边被多少个面使用
    edge_count = defaultdict(int)
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edge_count[edge] += 1
    
    # 收集边界顶点
    boundary_vertices = set()
    for edge, count in edge_count.items():
        if count == 1:  # 边界边
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])
    
    return sorted(list(boundary_vertices))

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Laplacian magnitude smoothing based on FiberMesh')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('--output', '-o', type=str, help='Output file path for smoothed mesh')
    args = parser.parse_args()

    v, f = igl.read_triangle_mesh(args.mesh_file)

    constrained = extract_boundary_vertices(v, f)

    print('constrained', constrained)

    # constrained = []
    # if os.path.split( args.mesh_file )[-1] == 'spherecylinder_2n_smoothed.obj':
    #     #import pdb
    #     #pdb.set_trace()
    #     # Constrain the vertices on the x axis curve
    #     constrained.extend( np.where(np.abs(v[:,0]) < 1e-5)[0] )
    #     # Constrain the vertices on the z axis curve
    #     constrained.extend( np.where(np.abs(v[:,2]) < 1e-5)[0] )
    #     # Constrain the top vertices
    #     constrained.extend( np.where(v[:,1] > .38)[0] )
    # else:
    #     ## Constrain the top and bottom vertices
    #     constrained.append( np.argmin( v[:,1], axis = 0 ) )
    #     constrained.append( np.argmax( v[:,1], axis = 0 ) )
    #     ## Constrain the front and back vertices
    #     constrained.append( np.argmin( v[:,2], axis = 0 ) )
    #     constrained.append( np.argmax( v[:,2], axis = 0 ) )

    constrained = np.unique(constrained)

    print( f"Constrained vertices ({len(constrained)}):", constrained )

    # 执行 FiberMesh smoothing
    # v_smoothed = smooth_laplacian_magnitude_rivers(v, f, constrained = constrained, iterations=4000, plot_every=100, threshold=1e-5)
    # v_smoothed = smooth_laplacian_magnitude_rivers(v, f, constrained = constrained, iterations=4000, plot_every=100, threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    # import sys
    # sys.exit(0)
    os.makedirs( 'steps', exist_ok = True )
    # v_smoothed = smooth_fibermesh(v, f, constrained = constrained, iterations=4000, plot_every=1, save_every=1, save_base='steps/fibermesh_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    ## This one works well:
    v_smoothed = smooth_stanko(v, f, constrained = constrained, target_curvature_paper = True)
    
    if args.output:
        igl.write_triangle_mesh(args.output, v_smoothed, f)

    ## Flows, explicit:
    ## They all get spiky because of poor mesh elements.
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 1, step_size = 0.0001, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 2, step_size = 0.000000000001, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 3, step_size = 1e-19, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)
    ## Without recomputing matrices, this fails completely:
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 2, step_size = 0.000000000001, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = False)

    ## Flows, implicit:
    ## This works, but makes a minimal surface (which we don't want).
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 1, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)
    ## These surfaces are good, but they eventually bunch vertices along constraints (tangential drift). We need normal projection.
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 2, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)
    # v_smoothed = smooth_flow(v, f, constrained = constrained, iterations=4000, order = 3, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True)

    ## Flow projected, explicit:
    ## They all get spiky because of poor mesh elements.
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 1, step_size = 1e-5, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 2, step_size = 1e-10, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 3, step_size = 1e-19, step_mode = 'explicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)

    ## Flows projected, implicit:
    ## This is lumpy. The normals are bad because of poor mesh elements.
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 1, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    ## This one works ok, but I think still gets lumpy because of poor mesh elements:
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 2, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)
    # v_smoothed = smooth_flow_projected(v, f, constrained = constrained, iterations=4000, order = 3, step_size = 0.1, step_mode = 'implicit', plot_every=1, save_every=1, save_base='steps/flow_', threshold=1e-5, recompute_matrices = True, recompute_normals = True)

    # 平滑后的 curvature
    # c_smooth = compute_laplacian_magnitude(v_smoothed, f)
    # plot(v_smoothed, f, c_smooth, title="Smoothed Laplacian magnitude")
