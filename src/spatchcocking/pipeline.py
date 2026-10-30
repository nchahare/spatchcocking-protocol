"""The spatchcocking projection, byte-exact from the published module.

Provenance
----------
Every function here is copied CHARACTER FOR CHARACTER from
`spatchcocking_utils.py` in nchahare/spatchcocking @ 3306f17, the code behind
Chahare, Imamura & Nerurkar (2026), Biophysical Reports 6:100280. Only module
headers and imports are new.

Why the published module is split rather than kept whole
--------------------------------------------------------
It is 1664 lines and 31 functions, and a reader following the protocol has to
find the method inside it. The split is by role, not by preference:

* `pipeline.py` (this file) - the projection itself, plus the two routines that
  attach a property to a mesh before it is projected.
* `meshing.py` - segmentation mask to surface mesh. The protocol's entry point.

The rest of the published module is left out: first versions superseded by the
`*2` variants, the old plotting and the old picker, and `getProperCurvature`,
which fails on any current vedo. They remain in nchahare/spatchcocking.

Deviations from the original, all non-numerical
-----------------------------------------------
* `from vedo import *` becomes explicit imports. The wildcard pulled roughly 400
  names into any namespace doing `from spatchcocking import *`, `Mesh`, `Plane`
  and `show` among them, shadowing callers' own variables.
* `project_point_on_variety` comes from `._vendor`, not from a vedo development
  branch that has never been released.

Nothing inside a function body is changed, so known defects are still present:
`getAxis` opens a blocking Plotter whether or not `check` is set, and
`getDefaultname` silently writes files called "noname-*". Both are left as
published, so that these functions stay identical to the published ones;
`api.py` works around them.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from vedo import (
    Arrow,
    Line,
    LinearTransform,
    Mesh,
    Plane,
    Plotter,
    Points,
    Spline,
    Text2D,
    progressbar,
    show,
)

from ._vendor import project_point_on_variety

__all__ = [
    "compute_and_save_curvatures",
    "transfer_points_to_mesh",
    "getAxis",
    "getPlanes",
    "getDeformedmesh2",
    "get_flatdata2",
    "transform_to_radial",
    "transform_to_cartesian",
    "normalize_values2",
]


# ==========================================================================
# helpers
# ==========================================================================


def getDefaultname(namefile):
    """
    Return a default name derived from files in the current directory when no name is provided.

    Args:
        namefile (str or None):
            If a string is provided, it is returned unchanged. If None, the function
            searches the current working directory for files ending with the suffix
            '.tif' and returns the filename with that suffix removed.

    Returns:
        namefile (str)
    """
    if namefile==None:
        names = [f[:-4] for f in os.listdir('.') if f.endswith('.tif')]
        if names:
            namefile = names[0]
        else:
            # Case: No tif files found
            namefile = f"noname"
    return namefile




# ==========================================================================
# attaching properties to the mesh
# ==========================================================================


def getTightercmap(values, sigma=3):
    """
    For highlighting contrast
    
    Args:
        values (np.array):
        sigma (float): to clip the standard deviation
    
    Returns:
        valuemin, valuemax (float)

    """
    valuemin = np.median(values) -sigma*np.std(values)
    valuemax = np.median(values) +sigma*np.std(values)
    return valuemin, valuemax


def compute_and_save_curvatures(msh, depth=5, degree=2, save_name=None, check=False):
    """
    Calculate Gaussian, Mean, and Principal curvatures (k1, k2), 
    optionally save to .npy, and visualize.
    """
    # 1. Prepare the mesh and calculate reference normals
    msh1 = msh.clone()
    msh1.compute_normals() 
    adlist = msh1.compute_adjacency()

    # 2. Compute Curvatures
    gauss_vals, mean_vals = [], []
    pts = msh1.points
    normals = msh1.vertex_normals

    print("Calculating surface curvatures...")
    for i in progressbar(range(msh1.npoints)):
        ids = msh1.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        
        # Calculate local polynomial fit using the vertex normal to prevent flipping
        _, poly, _ = project_point_on_variety(
            pts[i], 
            pts[ids],
            degree=degree,
            normal=normals[i] 
        )
        gauss_vals.append(poly[3])
        
        # You applied *-1 here, which is perfectly fine to flip the H convention
        mean_vals.append(poly[4] * -1)

    gauss_vals = np.array(gauss_vals)
    mean_vals  = np.array(mean_vals)

    # --- NEW: Calculate Principal Curvatures k1 and k2 ---
    # We use np.maximum to prevent negative values inside the sqrt 
    # caused by tiny floating-point precision errors where H^2 ~= K
    discriminant = np.maximum(mean_vals**2 - gauss_vals, 0)
    k1_vals = mean_vals + np.sqrt(discriminant)
    k2_vals = mean_vals - np.sqrt(discriminant)

    # 3. Add data to the mesh pointdata
    msh1.pointdata["Gauss_Curvature"] = gauss_vals
    msh1.pointdata["Mean_Curvature"] = mean_vals
    msh1.pointdata["K1"] = k1_vals
    msh1.pointdata["K2"] = k2_vals

    # 4. Save to .npy (X, Y, Z, Gauss, Mean, K1, K2)
    if save_name:
        combined = np.hstack((
            pts, 
            gauss_vals.reshape(-1, 1), 
            mean_vals.reshape(-1, 1),
            k1_vals.reshape(-1, 1),
            k2_vals.reshape(-1, 1)
        ))
        file_path = f"{save_name}-curvatures.npy"
        np.save(file_path, combined)
        print(f"Saved curvature data to: {file_path}")
        

    # 5. Simple Check Visualization
    if check:
        # Use percentiles to prevent extreme outliers from washing out the colors
       
        vmin_g, vmax_g = getTightercmap(gauss_vals) # just to visualize the curvatures
        vmin_m, vmax_m = getTightercmap(mean_vals) # just to visualize the curvatures
        vmin_k1, vmax_k1 = getTightercmap(k1_vals) # just to visualize the curvatures
        vmin_k2, vmax_k2 = getTightercmap(k2_vals) # just to visualize the curvatures
        
        
        # Fallbacks in case the mesh is completely flat
        vmax_g = vmax_g if vmax_g > 0 else 1.0
        vmax_m = vmax_m if vmax_m > 0 else 1.0
        vmax_k1 = vmax_k1 if vmax_k1 > 0 else 1.0
        vmax_k2 = vmax_k2 if vmax_k2 > 0 else 1.0

        msh_m = msh1.clone().lighting("glossy")
        msh_m.cmap("PuOr_r", "Mean_Curvature", vmin=-vmax_m, vmax=vmax_m).add_scalarbar()

        msh_g = msh1.clone().lighting("glossy")
        msh_g.cmap("coolwarm", "Gauss_Curvature", vmin=-vmax_g, vmax=vmax_g).add_scalarbar()
        
        msh_k1 = msh1.clone().lighting("glossy")
        msh_k1.cmap("viridis", "K1", vmin=-vmax_k1, vmax=vmax_k1).add_scalarbar()

        msh_k2 = msh1.clone().lighting("glossy")
        msh_k2.cmap("plasma", "K2", vmin=-vmax_k2, vmax=vmax_k2).add_scalarbar()

        # Show side-by-side in a 2x2 grid (N=4)
        plt = Plotter(N=4, axes=4, size=(1600, 1600))
        plt.at(0).show("Mean Curvature (H)", msh_m)
        plt.at(1).show("Gaussian Curvature (K)", msh_g)
        plt.at(2).show("Max Principal Curvature (K1)", msh_k1)
        plt.at(3).show("Min Principal Curvature (K2)", msh_k2)
        plt.interactive().close()

    return msh1


def transfer_points_to_mesh(mesh, pts_data, scalar_name, show_result=True):
    """Interpolates scalar values from a point cloud onto a mesh surface.

    This function takes a 2D array containing spatial coordinates and scalar 
    values, converts it into a vedo Points object, and uses it to interpolate 
    data onto the target mesh using a nearest-neighbors approach.

    Args:
        mesh: A vedo.Mesh object representing the target surface.
        pts_data: A numpy.ndarray of shape (N, 4). The first three columns 
            must be XYZ coordinates, and the fourth column must be the 
            scalar value (e.g., thickness).
        scalar_name: A string representing the name of the data field to 
            be created on the mesh. Defaults to "thickness".
        n_neighbors: An integer specifying the number of nearest points 
            to consider for the interpolation. Defaults to 5.
        show_result: A boolean. If True, opens a vedo window to visualize 
            the result. Defaults to True.

    Returns:
        A vedo.Mesh object containing the interpolated scalar data.

    Raises:
        ValueError: If pts_data does not have exactly 4 columns.
    """
    if pts_data.shape[1] != 4:
        raise ValueError(f"Expected array shape (N, 4), got {pts_data.shape}")
    
    n_neighbors = 5

    # Create a clone to prevent modifying the original mesh object
    mesh_output = mesh.clone()

    # Split coordinates from scalar values
    coords = pts_data[:, :3]
    scalars = pts_data[:, 3]

    # Initialize vedo Points object for interpolation source
    v_points = Points(coords, r=10)
    v_points.pointdata[scalar_name] = scalars
    
    # Perform the spatial interpolation onto the mesh
    mesh_output.interpolate_data_from(v_points, n=n_neighbors)

    # Visualization and color mapping
    if show_result:
        mesh_output.cmap("viridis", scalar_name).add_scalarbar()
        print(f"Interpolation complete. Field '{scalar_name}' added.")
        show(mesh_output).close()

    return mesh_output




# ==========================================================================
# medial axis
# ==========================================================================


def nearest_neighbor_order(points, start=0):
    """
    Compute a visit order of 3D (or n-D) points using a greedy nearest-neighbor heuristic.

    Parameters
    - points: array-like of shape (N, D). Sequence of N points in D-dimensional space.
    - start: int, optional. Index of the starting point in `points`. Default 0.

    Returns
    - order: list of int, length N. Indices into `points` giving the visitation order,
    where consecutive indices are chosen greedily as the closest unvisited point.

    Notes
    - This is a greedy heuristic for constructing a single polyline through all points.
    - It does NOT guarantee the globally shortest path (the Travelling Salesman Problem).
    - Time complexity: O(N^2 * D) in this implementation (each step searches remaining points).
    - Space complexity: O(N) for bookkeeping.
    - For large N, consider spatial acceleration (KDTree) or heuristic improvement (2-opt).
    """

    # Convert input to a NumPy array to ensure vectorized numeric operations.
    # If `points` is already an ndarray, this is cheap (returns view); otherwise it creates one.
    pts = np.asarray(points)

    # N is the number of points (rows). This determines how many iterations we will do.
    N = len(pts)

    # `remaining` holds indices of points not yet visited.
    # Using a set gives O(1) average-time membership and removal.
    remaining = set(range(N))

    # `order` stores the visiting sequence of indices. Start with the given index.
    order = [start]

    # Remove the start index from the set of remaining (it's now visited).
    remaining.remove(start)

    # `cur` is the index of the current point (we start there).
    cur = start

    # Loop until all points have been visited.
    while remaining:
        # `cur_pt` is the coordinates of the current point.
        cur_pt = pts[cur]

        # Find the index in `remaining` whose point has the minimum squared Euclidean distance
        # to `cur_pt`. We use squared distance (sum of squared differences) because square
        # root is monotonic and unnecessary for comparisons; this is slightly faster.
        nearest = min(remaining, key=lambda i: np.sum((cur_pt - pts[i])**2))

        # Append the found index to the order (we will visit it next).
        order.append(nearest)

        # Mark it as visited by removing from `remaining`.
        remaining.remove(nearest)

        # Update current index to the newly visited point and repeat.
        cur = nearest

    # Return the full visiting order (list of indices).
    return order


def getAxis(mesh, 
            endpts,
            namefile=None,
            num_points = 12,
            N=15,
            check=None):
    """
    do the mls smoothing iterations, for getting the skeleton axis
    
    Args:
        msh (mesh_object): Mesh
        endpts (array): end points arrays
        namefile (str): Name to store the axis points
        num_points (int): number of final axis points
        N (int): Number of iterations for smoothening (10-30)
        check (bool, optional): If True, opens a Vedo Plotter window to 
            show the spline. Defaults to None (No plotting).
    Returns:
        decimated_points (array): decimated axis points 
    """
    
    pcl = Points(mesh.vertices).subsample(0.02)

    plt = Plotter(N=N, axes=1)
    for i in range(N):
        pcl = pcl.clone().smooth_mls_1d(f=1).color(i) 
        # f goes from 0-2, 0.5 works well. 1 if you get strange curves
        plt.at(i).show(f"iteration {i+1}", pcl, elevation=-8)
    plt.interactive().close()

    pcl.subsample(0.1)

    # show(pcl.color('k'),mesh, axes=1)

    ## closest point to the selected point, because we need to know where are the pts
    idx = pcl.closest_point(endpts[0], return_point_id=True)
    # print("closest point index:", idx) 

    ## get the points and then order them and connect them with a spline
    pclpts = pcl.vertices
    # print(pclpts)   

    # points are randomly is ordered, we have to order them posterior to anterior
    order = nearest_neighbor_order(pclpts, start=idx)
    ordered_pts = pclpts[order]

    # ISSUE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # add the end points to the medial axis (Lets keep it as it is for now)
    # ordered_pts = np.vstack([endpts[0], ordered_pts, endpts[1]])

    ## add two extra points at the ends, one in the direction of the first segment
    ## one in the direction of the last segment
    
    first_dir = ordered_pts[1] - ordered_pts[0]
    last_dir = ordered_pts[-1] - ordered_pts[-2]    

    delta = 2
    new_start = ordered_pts[0] - first_dir*delta
    new_end = ordered_pts[-1] + last_dir*delta

    new_ordered_pts = np.vstack([new_start, ordered_pts, new_end])
    pspline = Spline(new_ordered_pts, closed=False).c('green')

    psplinepts = pspline.vertices
    
    # Calculate the step size to get exactly num_points
    indices = np.linspace(0, len(psplinepts) - 1, num_points).astype(int)

    decimated_points = psplinepts[indices]
    
    if check:
        ptsd = Points(decimated_points).c('red').ps(8)
        new_pspline1 = Spline(decimated_points, closed=False).c('green')
        show(new_pspline1, ptsd, mesh.alpha(0.5), axes=1)

    
    # save the new pspline pts to numpy file
    namefile = getDefaultname(namefile)
    np.save(namefile + "-axis.npy", decimated_points)
    
    return decimated_points




# ==========================================================================
# dorsal frame and cross-sectional planes
# ==========================================================================


def find_closest_dorsal_points(axispts, dorsalpts):
    """
    Finds the index of the closest dorsalpt for every axispt.
    
    Args:
        axispts (np.array): Shape (N, 3)
        dorsalpts (np.array): Shape (M, 3)
        
    Returns:
        closest_indices (np.array): Indices of dorsalpts closest to each axispt.
        closest_points (np.array): The actual coordinates from dorsalpts.
    """
    
    # Use broadcasting to calculate the difference between every pair of points
    # axispts[:, np.newaxis, :] shape: (N, 1, 3)
    # dorsalpts[np.newaxis, :, :] shape: (1, M, 3)
    # Difference shape: (N, M, 3)
    diff = axispts[:, np.newaxis, :] - dorsalpts[np.newaxis, :, :]
    
    # Calculate squared Euclidean distance: (N, M)
    # We sum along the last axis (the coordinates)
    dist_sq = np.sum(diff**2, axis=2)
    
    # For each axispt (rows), find the index of the minimum distance in dorsalpts (cols)
    closest_indices = np.argmin(dist_sq, axis=1)
    
    # Map indices back to the actual coordinates
    closest_points = dorsalpts[closest_indices]
    
    return closest_indices, closest_points


def getPlanes(mesh, axispts, endpts,
                  skip_index = np.array([]),
                  check=None):
    """
    Get planes along the axis
    
    Args:
        mesh (mesh_object): Mesh
        axispts (array): axis points arrays
        skip_index (array): number of planes which look wrong
        check (bool):  to see the details of process
        
    Returns:
        axis_info = {
        'axis': axispts,
        'axis normals': normals,
        'dorsal normals': dnormals
        }
    """
    axis = Points(axispts).c('red').ps(8)


    # Compute Tangents (Direction of the path)
    # This returns the unit vectors pointing along the curve
    normals = np.gradient(axispts, axis=0)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    
    ## Points closer to dorsal line    
    dorsal_line = Spline(endpts, 100, closed=False)
    dorsal_pts = dorsal_line.vertices
    indices, subset_dorsal_pts = find_closest_dorsal_points(axispts, dorsal_pts)
    
    # Vector from axispts to dorsal_pts (v = P - O)
    v = subset_dorsal_pts - axispts

    # Calculate the projection of v onto the plane
    # dist is (v dot n). We subtract the component parallel to the normal.
    dist = np.sum(v * normals, axis=1, keepdims=True)
    dnormal_vec = v - (dist * normals)
    propt = dnormal_vec + axispts # projected point. we dont need this
    
    # Normalize the resulting vector to get unit dnormals
    dnormals = dnormal_vec / np.linalg.norm(dnormal_vec, axis=1, keepdims=True)


    # Initialize an empty list to hold flag posts
    fss = []
    for idx, p in enumerate(axispts):
        # Create a flag post at point p with the index as the label
        fs = axis.flagpost(f"{idx}", p, c='orange',vspacing = 0.1 )
        fss.append(fs)

    if check:
        plt = Plotter(axes=1)
        plt += mesh.alpha(0.3)
        plt += axis
        plt += fss
        # plt += Points(subset_dorsal_pts, r=20, c='red')
        # plt += Points(propt, r=20, c='blue')


    for i in range(0, len(axispts)):
        if i in skip_index:
            continue
        normal = normals[i, :]
        othernormal = dnormals[i,:]
        pt1 = axispts[i,:]
        plane = Plane(pos=pt1, normal=normal, s=(1000, 1000)).alpha(0.1).c('green')

        if check:
            plt += Arrow(pt1, pt1+normal*50)
            plt += Arrow(pt1, pt1+othernormal*50)
            plt += plane

    
    if check:
        doc = Text2D("REMEMBER INTERSECTING PLANES \n Planes along the axis points, \n see if the planes intersect, \n also plane size = 1000, \n adjust dist threshold if planes are too small")
        plt += doc
        plt.show().close() 
    
    axis_info = {
    'axis': axispts,
    'axis normals': normals,
    'dorsal normals': dnormals
    }
    
    return axis_info




# ==========================================================================
# radial coordinates
# ==========================================================================


def transform_to_radial(points, center, normal, dnormal):
    
    """Transform 3D points to radial coordinates around a center point and normal vector.

        Args:
            points: array-like of shape (N, 3). Sequence of N 3D points.
            center: array-like of shape (3,). The center point for the radial transformation.
            normal: array-like of shape (3,). The normal vector defining the axis of rotation.
            dnormal: array-like of shape (3,). The normal vector defining the axis of rotation in dorsal direction.

        Returns
            radial_coords: array of shape (N, 3). Each row contains (radius, angle, height) for each point. where radius is the distance from the axis, angle is the angle around the axis in radians, and height is the projection along the normal vector.
    """
    
    # Ensure inputs are arrays
    pts = np.asarray(points)
    n = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, n) # Strictly orthogonal X
    x_axis /= np.linalg.norm(x_axis)

    # Vector from center to points
    vec = pts - center
    
    # Projection and math (Vectorized)
    height = np.dot(vec, n)
    # Reshape height for broadcasting: (N, 1)
    proj = vec - height[:, np.newaxis] * n
    radius = np.linalg.norm(proj, axis=1)
    
    # Angle relative to our specific x_axis and y_axis
    angle = np.arctan2(np.dot(proj, y_axis), np.dot(proj, x_axis))
    
    # Stack into (N, 3)
    return np.column_stack((radius, angle, height))


def transform_to_cartesian(radial_coords, center, normal, dnormal):
    """Transform 3D cartesian points from radial coordinates around a center point and normal vector.

        Args:
            points: array-like of shape (N, 3). Sequence of N 3D points.
            center: array-like of shape (3,). The center point for the radial transformation.
            normal: array-like of shape (3,). The normal vector defining the axis of rotation.
            dnormal: array-like of shape (3,). The normal vector defining the axis of rotation in dorsal direction.

        Returns
           points : array of shape (N, 3). cartesian coordinates.
    """
    radial_coords = np.asarray(radial_coords)
    radius = radial_coords[:, 0]
    angle = radial_coords[:, 1]
    height = radial_coords[:, 2]

    # Reconstruct the EXACT same coordinate basis logic
    n = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, n)
    x_axis /= np.linalg.norm(x_axis)

    # Vectorized reconstruction
    # (N, 1) * (3,) results in (N, 3)
    proj = (radius * np.cos(angle))[:, np.newaxis] * x_axis + (radius * np.sin(angle))[:, np.newaxis] * y_axis
           
    points = center + proj + (height[:, np.newaxis] * n)
    
    return points




# ==========================================================================
# straightening
# ==========================================================================


def getDeformedmesh2(mesh, axis_info,
                    namefile=None,
                    skip_index =np.array([]),
                    dists_threshold=300,
                    check=None):
    """
    Straightening the mesh
    
    Args:
        mesh (mesh_object): Mesh
        axis_info (dictionary): axis points arrays and normals
            axis_info = {
                'axis': axispts,
                'axis normals': normals,
                'dorsal normals': dnormals
                }
        skip_index (array): number of planes which look wrong
        namefile (str): Name to store the curvature values
        dists_threshold (int): distance to exclude points when mesh is very curved
        check (bool): to see the details of process
    
    Returns:
        deformed_mesh (mesh_object): Deformed mesh
    """
    normals = axis_info['axis normals']
    dnormals = axis_info['dorsal normals']
    axispts = axis_info['axis']
    
    
    namefile = getDefaultname(namefile)    # get name
        
    mesh.c('lightblue').decimate(n=1000) # its important to decimate

    # get new normal from first two axis points
    newNormal = np.array([0, 0, 1]) #normals[0]
    newdnormal = np.array([0, 1, 0])#dnormals[0]
    # %% straighten the axis points
    # start from the first point and add distance along newNormal
    straightened_pts = [axispts[0]]
    for i in range(1, len(axispts)):
        dist = np.linalg.norm(axispts[i] - axispts[i-1])
        new_pt = straightened_pts[-1] + newNormal * dist
        straightened_pts.append(new_pt) 
    straightened_pts = np.array(straightened_pts)   

    if check:
        # convert to vedo Points for visualization
        axis = Points(axispts).c('red').ps(8)
        straightened_axis = Points(straightened_pts).c('blue').ps(8)
        plt = Plotter(axes=1)
        plt += mesh.alpha(0.3)
        plt += axis
        plt += straightened_axis

    # empty list of points in orginal mesh corresponding to each axis point and their transformed positions
    original_points = []
    transformed_points = []

    for i in range(0, len(axispts)):
    
        # if i is in skip_index, skip this iteration
        if i in skip_index:
            continue

        pt1 = axispts[i]
        normal = normals[i]
        dnormal = dnormals[i]

            
        cut = mesh.clone().intersect_with_plane(pt1, normal) 
        # cut the mesh with the plane at pt1 with normal
        propoints = cut.vertices # get the points of the cut mesh

        # if no points found, skip
        if len(propoints) == 0:
            # skip this iteration
            continue

        # check if there are two clusters of points in the propoints, by checking their distance from pt1 and then plotting a histogram
        dists = np.linalg.norm(propoints - pt1, axis=1)
        propoints = propoints[dists < dists_threshold]     # delete points that are too far away
        original_points.append(propoints)     # append original points

        # transform the propoints to radial cordinations around pt1 and normal
        radial_coords = transform_to_radial(propoints, pt1, normal, dnormal)

        # then back to cartesian at straightened position and newnormal
        new_propoints = transform_to_cartesian(radial_coords, straightened_pts[i], newNormal, newdnormal)
        
        # append transformed points
        transformed_points.append(new_propoints)
        
        if check:
            plt += mesh.alpha(0.3)
            plt += Points(propoints).ps(5).c('red')
            plt += Points(new_propoints).ps(5).c('green')

    if check:
        plt.show().close()

    # using warp function to deform the mesh based on the original and transformed points
    original_points = np.vstack(original_points)
    transformed_points = np.vstack(transformed_points)

    ## deform the mesh and show
    print(mesh)
    print("i am warpping now --")
    deformed_mesh = mesh.clone().warp(original_points, transformed_points, sigma=1)
    print("i am down warpping --")
    print(deformed_mesh)

    # move mesh to origin  
    Aligntransform = LinearTransform()
    Aligntransform.translate(-axispts[0])
    Aligntransform.move(deformed_mesh)


    # show([mesh,deformed_mesh],N=2,axes=1)
    
    return deformed_mesh




# ==========================================================================
# flattening
# ==========================================================================


def get_flatdata2(deformed_mesh, namefile=None, check=None):
    """
    Flat projection from cylinderical object
    
    Args:
        mesh (mesh_object): Mesh
        axispts (array): axis points arrays
        namefile (str): Name to store the values
        check (bool): to see the plot
    
    Returns:
        radius, angle, height, demesh2values
    """

    namefile = getDefaultname(namefile)    # get name

    dmesh2 = deformed_mesh.clone().decimate(n=500).smooth(boundary=True).subdivide(n=2)
    # show([deformed_mesh, dmesh2], N=2) # check

    dmesh2pts = dmesh2.vertices
    
    center = np.mean(dmesh2pts, axis=0)
    normal = np.array([0, 0, 1])   # Axis of the cylinder/object
    dnormal = np.array([0, 1, 0])  # Direction of 90 degrees

    # --- Build the Rotation Matrix ---
    # Align: normal -> Z, dnormal -> Y
    z_axis = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # R maps World -> Local
    R = np.vstack([x_axis, y_axis, z_axis])

    # --- Transform to Local Coordinates ---
    pts_centered = dmesh2pts - center
    local_pts = pts_centered @ R.T 

    # --- Convert to Polar ---
    # In our local system: 
    # local_pts[:, 0] is X, local_pts[:, 1] is Y
    radius = np.linalg.norm(local_pts[:, :2], axis=1)
    angle = np.arctan2(local_pts[:, 1], local_pts[:, 0])
    height = local_pts[:, 2]
    
    print('we got the flat mesh')
    

    return radius, angle, height, dmesh2




# ==========================================================================
# normalizing the map
# ==========================================================================


def fix_angles(angles, shift_deg=0):
    """For the times when dorsal line doesnt align with 0 degree

    Args:
        angles (array): angles in degrees
        shift_deg (int, optional): angle to displace in degrees. Defaults to 0.

    Returns:
        fixed_angles (array)
    """
    fixed_angles = angles + shift_deg
    fixed_angles = (fixed_angles + 180) % 360 - 180
    return fixed_angles


def normalize_values2(height, angle, shift_deg=0):
    """Normalize all the values 

    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        norm_height, angle_degrees, scaled_curvature

    """
        
    # normalize height from 0 to 1
    norm_height = (height - np.min(height)) / (np.max(height) - np.min(height))

    # convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    angle_degrees = fix_angles(angle_degrees, shift_deg=shift_deg)

    return norm_height, angle_degrees
