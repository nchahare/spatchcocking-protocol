"""Segmentation mask to surface mesh - the protocol's entry point.

Byte-exact from the published `spatchcocking_utils.py`; see `pipeline.py` for
the provenance and the terms of the split.

These are here because the protocol starts at a mask, not at a mesh: a mask is
a more universal interface than "closed manifold PLY", which many tools export
badly, and mesh quality then becomes something the protocol controls.

Known defect of the published `get_mesh`, carried over unchanged: on a
non-manifold surface marching cubes yields nothing and an EMPTY STL is written
silently - VTK logs at ERR level, nothing raises, and the caller prints
"Saved:". `mask_to_mesh`, below, is the replacement, and raises instead.
"""
import os

import numpy as np
import tifffile
import trimesh
from scipy import ndimage
from skimage import filters, measure, morphology
from vedo import Mesh, Plotter, Points, Text2D, show

__all__ = [
    "load_tiff_stack",
    "preprocess_mask",
    "extract_mesh_marching_cubes",
    "extract_mesh_dual_contouring",
    "tiff_stack_to_mesh",
    "get_mesh",
    "save_mesh",
    "mask_to_mesh",
]


# ==========================================================================
# helpers
# ==========================================================================


def strippathname(fullpath):
    """
    Strip path and return only the filename without extension.
    
    Parameters:
    fullpath (str): Full file path
    
    Returns:
    str: Filename without extension
    """
    base = os.path.basename(fullpath)
    return os.path.splitext(base)[0]




# ==========================================================================
# loading and cleaning the mask
# ==========================================================================


def load_tiff_stack(file_path):
    """
    Load 3D TIFF stack (Mask of lumen)
    Note: Lumen = 1, Background = 0
    
    Parameters:
    file_path (str): Path to TIFF stack file
    
    Returns:
    numpy.ndarray: 3D array of the stack
    """
    try:
        # Load TIFF stack
        stack = tifffile.imread(file_path)
        
        # Handle different TIFF formats
        if stack.ndim == 4:
            # Multi-channel, take first channel
            stack = stack[0] if stack.shape[0] < stack.shape[-1] else stack[:, :, :, 0]
        elif stack.ndim == 2:
            # Single slice, add dimension
            stack = stack[np.newaxis, :, :]
        
        print(f"Loaded TIFF stack with shape: {stack.shape}")
        print(f"Data type: {stack.dtype}")
        print(f"Value range: [{stack.min()}, {stack.max()}]")
        
        return stack
        
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None


def preprocess_mask(mask, 
                   threshold=0.5, 
                   remove_small_objects=True,
                   min_size=100,
                   smooth=True,
                   fill_holes=True):
    """
    Preprocess the 3D mask before mesh extraction
    
    Parameters:
    mask (numpy.ndarray): 3D mask array
    threshold (float): Threshold for binarization
    remove_small_objects (bool): Remove small connected components
    min_size (int): Minimum size for connected components
    smooth (bool): Apply smoothing
    fill_holes (bool): Fill holes in the mask
    
    Returns:
    numpy.ndarray: Preprocessed binary mask
    """
    print("Preprocessing mask...")
    
    # Convert to binary if not already
    if mask.dtype != bool:
        binary_mask = mask > threshold * mask.max()
    else:
        binary_mask = mask.copy()
    
    print(f"Binary mask has {np.sum(binary_mask)} voxels")
    
    # Remove small objects
    if remove_small_objects:
        print("Removing small objects...")
        # max_size removes components <= its value; min_size removed only those
        # < its value, so min_size - 1 keeps the original threshold.
        binary_mask = morphology.remove_small_objects(binary_mask, max_size=min_size - 1)
        print(f"After cleaning: {np.sum(binary_mask)} voxels")
    
    # Fill holes
    if fill_holes:
        print("Filling holes...")
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        print(f"After filling holes: {np.sum(binary_mask)} voxels")
    
    # Smooth the mask
    if smooth:
        print("Smoothing mask...")
        # Apply slight gaussian smoothing
        smoothed = filters.gaussian(binary_mask.astype(float), sigma=0.5)
        binary_mask = smoothed > 0.5
        print(f"After smoothing: {np.sum(binary_mask)} voxels")
    
    return binary_mask




# ==========================================================================
# surface extraction
# ==========================================================================


def extract_mesh_marching_cubes(mask, spacing=(1.0, 1.0, 1.0), level=0.5):
    """
    Extract mesh using marching cubes algorithm
    
    Parameters:
    mask (numpy.ndarray): 3D binary mask
    spacing (tuple): Voxel spacing (z, y, x)
    level (float): Iso-surface level
    
    Returns:
    tuple: (vertices, faces, normals, values)
    """
    print("Extracting mesh using marching cubes...")
    
    # Run marching cubes
    vertices, faces, normals, values = measure.marching_cubes(
        mask.astype(float), 
        level=level, 
        spacing=spacing,
        method='lewiner'  # More robust than 'classic'
    )
    
    print(f"Marching cubes result:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Normals: {len(normals)}")
    
    return vertices, faces, normals, values


def extract_mesh_dual_contouring(mask, spacing=(1.0, 1.0, 1.0)):
    """
    Extract mesh using dual contouring (via trimesh)
    
    Parameters:
    mask (numpy.ndarray): 3D binary mask
    spacing (tuple): Voxel spacing (z, y, x)
    
    Returns:
    tuple: (vertices, faces)
    """
    print("Extracting mesh using dual contouring...")
    
    try:
        # Create voxel grid
        voxel_grid = trimesh.voxel.VoxelGrid(mask, spacing=spacing)
        
        # Extract mesh using dual contouring
        mesh = voxel_grid.marching_cubes
        
        if mesh is None:
            print("Dual contouring failed, trying alternative method...")
            # Alternative: create mesh from voxel centers
            mesh = voxel_grid.as_boxes()
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"Dual contouring result:")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(faces)}")
        
        return vertices, faces
        
    except Exception as e:
        print(f"Dual contouring failed: {e}")
        return None, None




# ==========================================================================
# mask to mesh
# ==========================================================================


def tiff_stack_to_mesh(tiff_path=None,
                      output_path=None,
                      spacing=(1.0, 1.0, 1.0),
                      method='marching_cubes',
                      preprocess=True,
                      ):
    """
    Complete pipeline to convert 3D TIFF stack to mesh
    
    Parameters:
    tiff_path (str): Path to TIFF stack file 
    output_path (str): Output mesh file path
    spacing (tuple): Voxel spacing (z, y, x)
    method (str): Extraction method ('marching_cubes', 'dual_contouring')
    preprocess (bool): Apply preprocessing
    simplify (bool): Simplify mesh
    smooth (bool): Smooth mesh
    visualize (bool): Show visualization
    
    Returns:
    tuple: (vertices, faces) of the extracted mesh
    """
    
    # Step 1: Load TIFF stack
    if tiff_path is None:
        print("put correct tiff path...")
        return None, None
    else:
        mask = load_tiff_stack(tiff_path)
        if mask is None:
            return None, None
    
    # Step 2: Preprocess mask
    if preprocess:
        mask = preprocess_mask(mask)
    
    # Step 3: Extract mesh
    if method == 'marching_cubes':
        vertices, faces, normals, values = extract_mesh_marching_cubes(mask, spacing)
    elif method == 'dual_contouring':
        vertices, faces = extract_mesh_dual_contouring(mask, spacing)
        if vertices is None:
            print("Dual contouring failed, falling back to marching cubes...")
            vertices, faces, normals, values = extract_mesh_marching_cubes(mask, spacing)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    
    # Step 4: Save mesh
    if output_path:
        save_mesh(vertices, faces, output_path)
    
    return vertices, faces


def save_mesh(vertices, faces, output_path, format='ply'):
    """
    Save mesh to file
    
    Parameters:
    vertices (numpy.ndarray): Mesh vertices
    faces (numpy.ndarray): Mesh faces
    output_path (str): Output file path
    format (str): Output format ('ply', 'obj', 'stl')
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save mesh
    mesh.export(output_path, file_type=format)
    print(f"Mesh saved to: {output_path}")


def get_mesh(mask_path, 
             px2umz, px2umxy,
             check=None):
    """
    Converts a 3D binary TIFF mask into an STL mesh.    
    
    Args:
        mask_path (str): Path to TIFF stack file 
        px2umz (float): z step size in micron
        px2umxy (float): xy pixel length in micron
        check (bool, optional): If True, opens a Vedo Plotter window to compare 
            the raw mesh vs. the processed mesh. Defaults to None (No plotting).
        
    Returns:
        mesh1 (mesh_object): vedo mesh object            
    """
    namefile = strippathname(mask_path)

    # Extract vertices and faces from the TIFF volume
    vertices, faces = tiff_stack_to_mesh(
        tiff_path=mask_path,  
        output_path= namefile + '.ply',
        spacing=(px2umz, px2umxy, px2umxy),  # Adjust spacing based on your data
        method= 'marching_cubes',  # or 'dual_contouring'
        preprocess=True,
    )
    # Smoothing a mesh, use vedo to smooth and save the mesh
    mesh0 = Mesh(namefile + '.ply').lw(1)

    mesh1 = mesh0.clone()
    mesh1.decimate(n=1000).subdivide(3).smooth().compute_normals()
    # mesh1.decimate(n=200).subdivide(3).smooth().compute_normals()
    mesh1.c('light blue').lw(0).lighting('glossy').phong()
    # other useful filters to combine are
    # mesh.decimate(), clean(), smooth()


    if check:
        plt = Plotter(shape=(1, 2), axes=7)
        plt.at(0).show(mesh0.wireframe()) # "Raw Mesh" 
        plt.at(1).show(mesh1.wireframe()) # "Processed Mesh"
        doc = Text2D("Comparing the raw vs Proceesed mesh")
        plt += doc
        plt.interactive().close()
    
    # Export and Return
    mesh1.write(namefile + '-mesh.stl')
    print("Saved:", namefile + '-mesh.stl')
    return mesh1


# ==========================================================================
# NOT FROM THE PUBLISHED MODULE
# ==========================================================================


def mask_to_mesh(mask_path,
                 px2umz,
                 px2umxy,
                 smooth_iter=100,
                 decimate_fraction=0.05,
                 final_fraction=0.1,
                 final_smooth_iter=50,
                 out_path=None,
                 raw_path=None,
                 blur_um=None):
    """Segmentation mask to finished surface mesh, in one call.

    Replaces `get_mesh` as the protocol's entry point. `get_mesh` is kept above,
    byte-exact, because the parity test needs it - but it should not be used,
    and here is why.

    On both masks of the published example, marching cubes produces a surface
    with non-manifold edges (an edge shared by four faces). `get_mesh` then runs
    `decimate(n=1000).subdivide(3)`, and vtkLoopSubdivisionFilter cannot
    subdivide a non-manifold mesh. It logs at ERR level and returns nothing;
    the smoothing that follows has nothing to smooth; the STL writer has nothing
    to write. The function then prints "Saved:" and returns a mesh with ZERO
    POINTS. Nothing raises.

    The published pipeline survives this only by accident: notebook 01 ignores
    what `get_mesh` returns and re-reads the raw `.ply` that `tiff_stack_to_mesh`
    wrote as a side effect. Re-run the notebook a second time, though, and its
    `if os.path.exists(mesh_stl)` branch loads the empty STL and carries on.

    This function does what notebook 01 actually does - marching cubes, then
    smooth, decimate, decimate, smooth - with the failure made loud.

    The float32 cast is deliberate and load-bearing. `get_mesh` routes the mesh
    through a trimesh PLY, which stores vertices as float32; keeping float64 in
    memory would change the result very slightly and so would NOT reproduce the
    published meshes. Casting explicitly reproduces them bit-for-bit while
    making the intermediate file optional rather than required.

    The smoothing and decimation defaults are notebook 01's, and they are tuned
    for a full-resolution marching-cubes surface with hundreds of thousands of
    vertices. They are far too aggressive for a small mesh, where they collapse
    the surface to a small fraction of its true size. Scale them
    with the input rather than assuming them.

    Args:
        mask_path (str): TIFF stack, binary, one object.
        px2umz (float): z step size in micron.
        px2umxy (float): xy pixel length in micron.
        smooth_iter (int): Laplacian passes before decimation.
        decimate_fraction (float): fraction of faces kept in the first pass.
        final_fraction (float): fraction kept in the second pass.
        final_smooth_iter (int): boundary-preserving passes at the end.
        out_path (str): where to write the result; None writes nothing.
        raw_path (str): where to write the undecimated marching-cubes surface;
            None writes nothing.
        blur_um (float): None (the default) is the published route, which
            blurs the mask by 0.5 voxel and thresholds it back to 0 and 1, so
            marching cubes runs on a blocky mask and the smoothing that follows
            has to remove the voxel steps. With a value, the cleaned mask is
            blurred by a Gaussian of that sigma in micron (the same in z and xy,
            whatever the voxel size) and marching cubes runs on the blurred
            values, so the surface falls between voxels. Use it when the z step
            is much coarser than the xy pixel size, with a value of a few
            micrometers, about the xy pixel size.

    Returns:
        vedo.Mesh: the finished surface, in micron.

    Raises:
        ValueError: if any stage produces an empty mesh, naming the stage.
    """
    try:
        if blur_um is None:
            vertices, faces = tiff_stack_to_mesh(
                tiff_path=mask_path,
                output_path=raw_path,
                spacing=(px2umz, px2umxy, px2umxy),
                method='marching_cubes',
                preprocess=True,
            )
        else:
            spacing = np.array([px2umz, px2umxy, px2umxy], dtype=float)
            mask = preprocess_mask(load_tiff_stack(mask_path), smooth=False)
            soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=blur_um / spacing)
            vertices, faces, _, _ = measure.marching_cubes(
                soft, level=0.5, spacing=tuple(spacing), method='lewiner')
            if raw_path:
                save_mesh(vertices, faces, raw_path)
    except ValueError as exc:
        # skimage says "Surface level must be within volume data range", which
        # is true and unhelpful: it means the mask is all background or all
        # foreground, so there is no boundary to put a surface on.
        raise ValueError(
            f"marching cubes found no surface in {mask_path}: {exc}. The mask "
            f"is empty or entirely filled, or preprocess_mask removed every "
            f"object in it.") from exc
    if vertices is None or len(vertices) == 0:
        raise ValueError(
            f"marching cubes found no surface in {mask_path}. The mask is "
            f"empty, or every object in it was removed by preprocess_mask.")

    # float32 to match what a trimesh PLY would have stored; see above.
    mesh = Mesh([np.asarray(vertices, dtype=np.float32), np.asarray(faces)])

    def _check(m, stage):
        if m.npoints == 0 or m.ncells == 0:
            raise ValueError(
                f"{stage} produced an empty mesh from {mask_path}. This is "
                f"usually a non-manifold surface - marching cubes on a mask "
                f"with thin or touching structures gives edges shared by more "
                f"than two faces. Clean the mask (fill holes, remove specks, "
                f"check the two labels do not touch) and try again.")
        return m

    _check(mesh, "marching cubes")
    _check(mesh.smooth(niter=smooth_iter), "smoothing")
    _check(mesh.decimate(fraction=decimate_fraction), "first decimation")
    _check(mesh.decimate(fraction=final_fraction), "second decimation")
    _check(mesh.smooth(niter=final_smooth_iter, boundary=True), "final smoothing")
    mesh.compute_normals()

    if out_path:
        mesh.write(out_path)
        print(f"Mesh saved to: {out_path}")
    return mesh
