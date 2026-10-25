"""Superseded or dead code from the published module. Staged for deletion.

Nothing here is called by this repo, and nothing here is expected to appear in
the protocol. It is kept rather than deleted because deciding what a release
should contain is a judgement, and this file makes that judgement reversible:
delete the file, or promote a function out of it, as a deliberate act.

Byte-exact from `spatchcocking_utils.py`; see `pipeline.py` for provenance.

What is in here and why
-----------------------
* `getDeformedmesh`, `get_flatdata`, `normalize_values` - first versions,
  superseded by the `*2` variants the paper actually used.
* `visualize_flatmesh`, `visualize_flatmesh2` - plotting. This repo renders maps
  through `spatchcocking.plotting.grid_map`, which reproduces what
  `visualize_flatmesh2` got right (interpolate in normalized (s, theta), then
  restore the extent) without the rest.
* `selectPointsonMesh` - the published interactive picker, roughly twelve clicks
  along the dorsal midline with no undo beyond clear-all and no validation.
  Superseded by `scripts/pick_anchors.py`.
* `query4skip_index` - collects bad plane indices through `input()`. Commented
  out in every published notebook.
* `getProperCurvature` - DEAD. It raises on any vedo currently obtainable, yet
  is listed in the public API reference on jexpnimesh.com. Kept here only so the
  decision to drop it from the docs is recorded somewhere.
* `getnormal` - a two-line normalized difference, inlined everywhere it is used.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from vedo import (
    Line,
    LinearTransform,
    Mesh,
    Plane,
    Plotter,
    Points,
    Sphere,
    Text2D,
    color_map,
    progressbar,
    show,
)

from ._vendor import project_point_on_variety
from .pipeline import (
    find_closest_dorsal_points,
    fix_angles,
    getDefaultname,
    getTightercmap,
    transform_to_cartesian,
    transform_to_radial,
)


# ==========================================================================
# superseded projection
# ==========================================================================


def getnormal(pt1, pt2):
    """
    Get normal between two points
    
    Args:
        pt1 (3d nparray): start pt
        pt2 (3d nparray): end pt
        
    Returns:
        normalized_projection_axis
    """    
    # pt1 is start, pt2 is end
    projection_axis = pt2- pt1
    # Calculate the norm of the resulting vector
    norm = np.linalg.norm(projection_axis)
    # Divide by the norm
    normalized_projection_axis = projection_axis / norm
    return normalized_projection_axis


def getDeformedmesh(mesh, axis_info,
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
    curvaturevals = np.load(namefile+"-curvature.npy") # load curvature
        
    mesh.c('lightblue').decimate(n=1000) # its important to decimate

    # load mesh pts and assign curvature
    meshpts = Points(curvaturevals[:,:3]).ps(8) 
    meshpts.pointdata["curvature"] = curvaturevals[:,3]

    # apply pt data on the mesh
    mesh.interpolate_data_from(meshpts, n=2)

    curvature0 = mesh.pointdata['curvature']

    vmin, vmax = getTightercmap(curvaturevals[:,3], sigma=1) # just to visualize the curvatures
    mesh.cmap('RdBu',mesh.pointdata['curvature'],vmin=vmin, vmax=vmax).add_scalarbar()


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
   
    print("i am warpping now --")
    deformed_mesh = mesh.clone().warp(original_points, transformed_points, sigma=1)
    print("i am down warpping --")

    deformed_mesh.pointdata['curvature'] = curvature0 # get curvature data applied to the mesh
    vmin, vmax = getTightercmap(curvature0, sigma=1) # just to visualize the curvatures
    deformed_mesh.cmap('RdBu',deformed_mesh.pointdata['curvature'],vmin=vmin, vmax=vmax).add_scalarbar() 

    # move mesh to origin  
    Aligntransform = LinearTransform()
    Aligntransform.translate(-axispts[0])
    Aligntransform.move(deformed_mesh)


    # show([mesh,deformed_mesh],N=2,axes=1)

    deformed_mesh.write(namefile+"-straight-mesh.stl")

    dmeshpts = deformed_mesh.vertices
    dcurvs = deformed_mesh.pointdata['curvature']
    dcurvs = dcurvs.reshape(-1, 1) 
    deformedmeshvalues = np.hstack((dmeshpts, dcurvs))  

    np.save(namefile + '-deformedvalues.npy', deformedmeshvalues) # save the files
    
    return deformed_mesh


def get_flatdata(deformed_mesh, namefile=None, check=None, densitydata=None):
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
    if densitydata:
        demesh2values = dmesh2.pointdata['density']
    else:
        demesh2values = dmesh2.pointdata['curvature']

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
    
    combined_data = np.column_stack((radius, angle, height, dmesh2pts, demesh2values))
    np.save(namefile + '-flatdata.npy', combined_data) # save the files
    
    if check:
        visualize_flatmesh(height, angle, demesh2values)


    return radius, angle, height, dmesh2pts, demesh2values


def normalize_values(height, angle, curvature, shift_deg=0, namefile=None, densitydata=None):
    """Normalize all the values 

    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        norm_height, angle_degrees, scaled_curvature

    """
    namefile = getDefaultname(namefile)    # get name

        
    # normalize height from 0 to 1
    norm_height = (height - np.min(height)) / (np.max(height) - np.min(height))

    # convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    angle_degrees = fix_angles(angle_degrees, shift_deg=shift_deg)

    # scale curvature
    if densitydata:
        scaled_curvature = curvature
    else:
        scaled_curvature = curvature*1e6
    
    
    # save the normalized values
    combined_normalized = np.column_stack((norm_height, angle_degrees, scaled_curvature))
    
    np.save(namefile+"-normalized-flatdata.npy", combined_normalized) # save the files
    

    return norm_height, angle_degrees, scaled_curvature




# ==========================================================================
# superseded plotting
# ==========================================================================


def visualize_flatmesh(height, angle, curvature, sigma=3, namefile=None, colorstr='RdBu', densitydata=None):
    """
    Visualize Flat projection
    
    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        
    """
    
    namefile = getDefaultname(namefile)    # get name
    # 1. Your data
    x = height 
    y = angle
    z = curvature

    # 2. NORMALIZE coordinates to [0, 1] range
    # This prevents the "vertical streaking" by making the axes look equal to the math
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 3. Create a dense grid in the normalized space [0, 1]
    grid_size = 500
    xi_norm = np.linspace(0, 1, grid_size)
    yi_norm = np.linspace(0, 1, grid_size)
    xi_norm, yi_norm = np.meshgrid(xi_norm, yi_norm)

    # 4. Interpolate using the normalized coordinates
    # 'linear' is often cleaner for large scale differences than 'cubic'
    zi = griddata((x_norm, y_norm), z, (xi_norm, yi_norm), method='linear')

    # 5. Plotting
    plt.figure(figsize=(12, 4))

    # We use the original x.min/max for the 'extent' so the axes show real units
    vmin, vmax = getTightercmap(curvature, sigma=sigma) # just to visualize the curvatures

    if densitydata:
        colorstr='Blues'
        vmin = 0
        vmax = np.max(curvature)


    # This takes your RGB array and turns it into a 'cmap' object
    cols = color_map(range(256), colorstr)
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_name", cols)

    im = plt.imshow(zi, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    origin='lower', 
                    aspect='auto', 
                    cmap=my_cmap,
                    vmin=vmin, vmax= vmax)

    plt.colorbar(im, label='Curvature Value')
    plt.xlabel('Height (Z-axis)')
    plt.ylabel('theta (Ventral - Dorsal - Ventral)')
    plt.title('')
        
    plt.savefig(
    namefile + "-flat.png", 
    dpi=300,                # High resolution for printing/reports
    bbox_inches='tight',    # Removes extra whitespace around the edges
    transparent=False,      # Set to True if you want a transparent background
    facecolor='white'       # Ensures background is solid white
    )
    plt.show()


def visualize_flatmesh2(height, angle, data,
    sigma=3, namefile=None,
    colorstr='RdBu'):
    """
    Visualize Flat projection
    
    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        
    """
    
    namefile = getDefaultname(namefile)    # get name
    # 1. Your data
    x = height 
    y = angle
    z = data

    # 2. NORMALIZE coordinates to [0, 1] range
    # This prevents the "vertical streaking" by making the axes look equal to the math
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 3. Create a dense grid in the normalized space [0, 1]
    grid_size = 500
    xi_norm = np.linspace(0, 1, grid_size)
    yi_norm = np.linspace(0, 1, grid_size)
    xi_norm, yi_norm = np.meshgrid(xi_norm, yi_norm)

    # 4. Interpolate using the normalized coordinates
    # 'linear' is often cleaner for large scale differences than 'cubic'
    zi = griddata((x_norm, y_norm), z, (xi_norm, yi_norm), method='linear')

    # 5. Plotting
    plt.figure(figsize=(12, 4))

    # We use the original x.min/max for the 'extent' so the axes show real units
    vmin, vmax = getTightercmap(data, sigma=sigma) # just to visualize the curvatures


    # This takes your RGB array and turns it into a 'cmap' object
    cols = color_map(range(256), colorstr)
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_name", cols)

    im = plt.imshow(zi, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    origin='lower', 
                    aspect='auto', 
                    cmap=my_cmap,
                    vmin=vmin, vmax= vmax)

    plt.colorbar(im, label='Value')
    plt.xlabel('Normalized Rostral-Caudal Distance')
    plt.ylabel('Ventral - Dorsal - Ventral')
    plt.title('')
        
    # plt.savefig(
    # namefile + "-flat.png", 
    # dpi=300,                # High resolution for printing/reports
    # bbox_inches='tight',    # Removes extra whitespace around the edges
    # transparent=False,      # Set to True if you want a transparent background
    # facecolor='white'       # Ensures background is solid white
    # )
    plt.show()




# ==========================================================================
# superseded interaction
# ==========================================================================


def selectPointsonMesh(mesh,
                       namefile=None):
    """
    Select points on the dorsal surface of the tube. starting from posterior end

    Args:
        mesh (mesh_object): mesh
        namefile (str): label for saved file

    Returns:
        selected_points (array): selected points on the dorsal surface
    """


    # mesh = mesh.alpha(0.5)
    mesh.pickable(True)

    # Variables to store selected points and their marker objects
    selected_points = []
    selected_markers = []

    # Create the plotter
    plt = Plotter(title="Click to select points. Press 'c' to clear.", axes=9)

    # Callback function for clicking
    def on_left_click(evt):
        if evt.actor and evt.picked3d is not None:
            point_coords = evt.picked3d
            print("Selected point:", point_coords)

            # Store point and add a red marker
            selected_points.append(point_coords)
            marker = Sphere(pos=point_coords, r=5, c="red")
            selected_markers.append(marker)
            plt.add(marker)  # Use plt here, not plotter
            plt.render()

    # Keypress callback: clear markers
    def on_keypress(evt):
        if evt.keypress  == "c":
            print("Clearing all selected points...")
            for marker in selected_markers:
                plt.remove(marker)
            selected_markers.clear()
            selected_points.clear()
            plt.render()
            
            
    # Add callbacks
    plt.add_callback("left_click", on_left_click)
    plt.add_callback("key_press", on_keypress)


    # Show mesh and interact
    doc = Text2D("Select points on the dorsal surface of the tube.\n Starting from posterior end \n In that order \n Press 'c' to clear")
    plt += doc
    plt.show(mesh, interactive=True).close()
    
    namefile = getDefaultname(namefile)

    np.save(namefile + "-endpts.npy", selected_points)

    return selected_points


def query4skip_index():
    # Ask for input for planes to be skiped
    user_input = input("Enter a list of indexes to be skipped separated by spaces (or press Enter for empty): ")

    try:
        # .split() on an empty string "" returns []
        # np.array([]) creates an empty array
        items = user_input.split()
        
        if not items:
            skip_index = np.array([], dtype=int)
            print("Input was empty. Stored an empty NumPy array.")
        else:
            skip_index = np.array(items, dtype=int)
            print("skip index Array:", skip_index)
            
    except ValueError:
        print("Error: Please ensure you only enter integers.")
    return skip_index




# ==========================================================================
# dead
# ==========================================================================


def getProperCurvature(msh, depth, 
                       namefile=None,
                       type="Gaussian",
                       check=None):
    """
    Get Curvature by fitting Quadratic surface on the a local neighborhood.
    
    Args:
        msh (mesh_object): Mesh
        depth (int): Count of adjacent points to fit the quadratic surface
        namefile (str): Name to store the curvature values
        type (str): Type of curvature to store/return. 
            Options: "Gaussian" or "Mean".
        check (bool, optional): If True, opens a Vedo Plotter window to 
            show the curvature. Defaults to None (No plotting).
        
    Returns:
        msh1 (mesh_object): Mesh with Curvature
        saves curvature values in a curvature.npy file
    
    Thanks Marco Musy. https://github.com/marcomusy/vedo/discussions/1272
    """
    msh1 = msh.clone()
    msh1.compute_normals()

    # Compute adjacency list for the mesh vertices
    adlist = msh1.compute_adjacency()

    # Compute curvature at all points by fitting a quadratic surface

    curvs_g = []
    curvs_m = []
    pts1 = msh1.points
    for i in progressbar(range(msh1.npoints)):
        ids = msh1.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        bpts = msh1.points[ids]
        _, res = project_point_on_variety(pts1[i], bpts, degree=2)
        curvs_g.append(res[4])
        curvs_m.append(res[5])
    

    # Use () to call the method
    if type.lower() == 'gaussian':
        msh1.pointdata["Gauss_Curvature"] = curvs_g
        curvs_val = msh1.pointdata["Gauss_Curvature"]
    elif type.lower() == 'mean':
        msh1.pointdata["Mean_Curvature"] = curvs_m
        curvs_val = msh1.pointdata["Mean_Curvature"]
    else:
        # Optional: Fallback or error if someone types something else
        raise ValueError(f"Unknown curvature type: {type}. Choose 'Gaussian' or 'Mean'.")
    
    namefile = getDefaultname(namefile)
    
    meshpts = msh1.vertices
    curv_col = curvs_val.reshape(-1, 1) 
    combinedcurvaturepos = np.hstack((meshpts, curv_col))
    np.save(namefile + '-curvature.npy', combinedcurvaturepos) # save the files
    
    if check:
        vmin, vmax = getTightercmap(curvs_val) # just to visualize the curvatures
        __doc__ = "Curvature Distribution"
        show(msh1.cmap('RdBu',curvs_val,vmin=vmin, vmax=vmax).add_scalarbar(),  __doc__)   

        
    return msh1
