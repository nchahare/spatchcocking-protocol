"""Measured fields on a surface: curvature, thickness, point density, intensity.

These are the quantities a reader attaches to the surface before projecting it.
Three of the four existed only as notebook cells in the published analysis, which
meant the package could project a surface but could not measure one. This module
is that layer, written so each field is one call, headless, and returns its values
rather than writing files as a side effect.

Behavior is deliberately identical to the published notebooks, so the published
fields reproduce. Where a notebook did something load-bearing that
was easy to miss -- the face-area normalization before curvature fitting, the
`n=5` neighbor count, the 5 smoothing iterations after density interpolation --
it is carried here with the reason attached.

`intensity_on_surface` is new. Everything else is consolidation.
"""

import numpy as np

from .pipeline import compute_and_save_curvatures

__all__ = [
    "curvature",
    "thickness",
    "density_from_points",
    "intensity_on_surface",
    "attach",
    "decimate_to_face_area",
]

# Curvature is fitted on a mesh normalized to this median face area, in um^2.
# Without it the estimate tracks triangle size rather than shape, so two meshes of
# the same embryo at different resolutions disagree.
GOOD_FACE_AREA = 1000.0

# vedo's face-quality metric id for triangle area.
_AREA_METRIC = 28


def decimate_to_face_area(mesh, good_size=GOOD_FACE_AREA):
    """Decimate until the median face area is about `good_size` um^2.

    From notebook 02. `compute_quality(metric=28)` puts the face area in
    `celldata['Quality']`; the scale factor is the ratio of the current median to
    the target. A mesh already coarser than the target is left essentially alone
    (vedo will not decimate by a factor above 1).
    """
    m = mesh.clone()
    m.compute_quality(metric=_AREA_METRIC)
    original = float(np.median(m.celldata["Quality"]))
    scale = original / good_size
    if scale > 1:
        scale = 0.999
    return m.decimate(fraction=scale, regularization=1), original


def curvature(mesh, depth=5, degree=2, good_size=GOOD_FACE_AREA, normalise=True):
    """Gaussian and mean curvature, plus principal curvatures k1 and k2.

    A quadric is fitted to each vertex's neighborhood out to `depth` rings and
    the curvatures are read off it. Units are um^-2 for Gaussian and um^-1 for
    mean; the sign convention has mean curvature positive where the surface bulges
    outwards.

    Parameters
    ----------
    mesh : vedo.Mesh
    depth, degree : int
        Neighborhood ring depth and polynomial degree for the local fit.
    good_size : float
        Target median face area in um^2 for the normalization described in
        `decimate_to_face_area`.
    normalise : bool
        Set False to fit on the mesh exactly as given. With normalization on,
        the mesh is first simplified, so the result can have fewer rows than the
        mesh has vertices.

    Returns
    -------
    vedo.Mesh
        A clone carrying `Gauss_Curvature`, `Mean_Curvature`, `K1` and `K2` in
        its point data. Note this may have fewer vertices than the input when
        `normalise` is True.
    """
    target = mesh
    if normalise:
        target, _ = decimate_to_face_area(mesh, good_size)
    return compute_and_save_curvatures(
        target, depth=depth, degree=degree, save_name=None, check=False
    )


def thickness(outer, inner):
    """Wall thickness at each vertex of `outer`, in micrometers.

    The unsigned Euclidean distance from every vertex of the outer (basal) surface
    to the nearest face of the inner (lumen) surface. This is the measurement the
    second mask buys: with one mask there is a surface, with two there is a wall.
    """
    m = outer.clone()
    result = m.distance_to(inner, signed=False)
    # vedo has returned either the distance array or the mesh itself across
    # versions; the array also lands in pointdata["Distance"] either way.
    values = result if isinstance(result, np.ndarray) else m.pointdata["Distance"]
    return np.asarray(values, dtype=np.float64)


def density_from_points(mesh, points, radius=100.0, n_neighbors=5, smooth_iter=5):
    """Local density of a point cloud, sampled onto the surface.

    Points become a field in three moves, which is how the published pHH3 route
    works and is worth stating plainly because it is not a point-to-vertex
    transfer: a kernel density estimate counts how many points fall within a
    sphere of `radius` at each voxel of a regular grid; that grid is sampled at
    each mesh vertex; the result is smoothed on the surface.

    `radius` should span the full tissue thickness, so that a nucleus anywhere
    through the wall contributes to the surface point above it.

    Parameters
    ----------
    mesh : vedo.Mesh
    points : (N, 3) array
        Coordinates in micrometers, in the same frame as the mesh.
    radius : float
        Kernel radius in micrometers.
    n_neighbors : int
        Neighbors used when sampling the density grid onto the surface.
    smooth_iter : int
        Passes of surface smoothing applied to the sampled field.

    Returns
    -------
    (N,) array
        One density value per mesh vertex.
    """
    from vedo import Points

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points has shape {points.shape}; expected (N, 3) coordinates in um."
        )

    cloud = Points(points)
    volume = cloud.density(radius=radius)

    sampled = mesh.clone().interpolate_data_from(volume, n=n_neighbors)
    sampled.smooth_data(niter=smooth_iter, relaxation_factor=1)
    # vedo names the array after the volume's own scalars.
    name = "ImageScalars" if "ImageScalars" in sampled.pointdata.keys() else None
    if name is None:
        raise RuntimeError(
            "density sampling produced no scalar array on the mesh; the point "
            "cloud and the mesh are probably in different coordinate frames."
        )
    return np.asarray(sampled.pointdata[name], dtype=np.float64)


def outward_normals(mesh):
    """Vertex normals that point out of the enclosed volume, whatever the winding.

    VTK computes normals from each triangle's vertex order, as if coordinates
    were (x, y, z). Ours are (z, y, x), a mirror image of that, so on every mesh
    made here the normals VTK gives point *into* the tissue. Rather than trust
    either convention, orient by the signed volume: the sum over triangles of
    a . (b x c) / 6 is positive when the triangles' own normals face out, and
    negative when they face in. VTK's own auto-orientation is not used: it can
    flip one mesh and not another, for example when the surface has a small
    hole.
    """
    m = mesh.clone()
    m.compute_normals()
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.cells)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    normals = np.asarray(m.vertex_normals, dtype=np.float64)
    return normals if signed_volume > 0 else -normals


INTENSITY_DIRECTIONS = ("in", "out", "both")


def intensity_on_surface(
    mesh, volume, spacing, depth_um=15.0, n_samples=4, direction="both",
    origin=(0.0, 0.0, 0.0),
):
    """Mean image intensity in a thin layer next to the surface.

    For each vertex, step along the surface normal and average the image over
    `n_samples` depths spanning `depth_um`:

    * ``"in"``: into the volume the surface encloses;
    * ``"out"``: out of it;
    * ``"both"``: `n_samples` depths each way, all averaged together. This is the
      default, because it does not depend on which side of the boundary the
      segmentation put the surface.

    "In" and "out" come from `outward_normals`, not from the normals' raw
    direction, which is mirrored on (z, y, x) meshes.

    Any channel works: an actin stain, a reporter, an in-situ signal. What the
    result means depends on where through the wall the signal lives, which is
    the thing to think about before choosing `depth_um`. Say which direction and
    depth were used: they are different measurements.

    Sampling is done with `scipy.ndimage.map_coordinates` on the array directly
    rather than through a `vedo.Volume`. The reason is axis order: mesh vertices
    are (z, y, x) in micrometers because marching cubes returns them in the
    array's own order, and indexing the array directly keeps that correspondence
    visible instead of relying on a library's internal convention. This dataset
    has already been bitten once by a silent axis/scale mismatch.

    Parameters
    ----------
    mesh : vedo.Mesh
        Vertices in micrometers, ordered (z, y, x).
    volume : (nz, ny, nx) array
        A single channel. Use `io.load_stack(path, spacing, channel=c)`.
    spacing : tuple of float
        (z, y, x) micrometers per voxel of `volume`.
    depth_um : float
        How far from the surface to sample, on each side sampled. Keep it well
        under the wall thickness.
    n_samples : int
        Number of depths per side, evenly spaced over (0, depth_um]. The surface
        itself is not sampled.
    direction : {"in", "out", "both"}
        Which side of the surface to sample; see above.
    origin : tuple of float
        (z, y, x) micrometers of the volume's first voxel, if it is not at 0.

    Returns
    -------
    (N,) array
        One mean intensity per mesh vertex, as float.

    Raises
    ------
    ValueError
        If the arguments are malformed.
    RuntimeError
        If the mesh does not overlap the volume, which almost always means the
        spacing is wrong or the two are in different frames.
    """
    from scipy.ndimage import map_coordinates

    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(
            f"volume has shape {volume.shape}; expected a single channel "
            "(z, y, x). Pass channel= to io.load_stack."
        )
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(
            f"spacing must be three positive numbers (z, y, x) in um, got {spacing}."
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")
    if depth_um <= 0:
        raise ValueError(f"depth_um must be positive, got {depth_um}.")
    if direction not in INTENSITY_DIRECTIONS:
        raise ValueError(f"direction must be 'in', 'out' or 'both', got {direction!r}.")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    outward = outward_normals(mesh)
    origin = np.asarray(origin, dtype=np.float64)

    one_side = np.linspace(depth_um / n_samples, depth_um, n_samples)
    offsets = {"in": -one_side, "out": one_side,
               "both": np.concatenate([-one_side, one_side])}[direction]
    total = np.zeros(len(verts))
    inside_any = np.zeros(len(verts), dtype=bool)

    for d in offsets:
        probe = verts + outward * d
        idx = (probe - origin) / spacing
        # map_coordinates clamps out-of-bounds reads with mode="nearest"; track
        # them separately so a mesh that misses the volume is an error, not zeros.
        within = np.all((idx >= 0) & (idx <= np.array(volume.shape) - 1), axis=1)
        inside_any |= within
        total += map_coordinates(volume, idx.T, order=1, mode="nearest")

    if not inside_any.any():
        raise RuntimeError(
            "no mesh vertex falls inside the image volume. The mesh spans "
            f"z {verts[:, 0].min():.0f}-{verts[:, 0].max():.0f}, "
            f"y {verts[:, 1].min():.0f}-{verts[:, 1].max():.0f}, "
            f"x {verts[:, 2].min():.0f}-{verts[:, 2].max():.0f} um, while the "
            f"volume spans {[round(float(s * n), 0) for s, n in zip(spacing, volume.shape)]} um "
            "from its origin. Check that spacing matches the image you loaded."
        )

    return total / len(offsets)


def attach(mesh, name, values):
    """Put a field on a mesh under a named point-data array, in place.

    Raises rather than broadcasting silently when the lengths disagree, because a
    length mismatch here means the field was measured on a different mesh, for
    example on a simplified copy of it.
    """
    values = np.asarray(values)
    if len(values) != mesh.npoints:
        raise ValueError(
            f"field '{name}' has {len(values)} values but the mesh has "
            f"{mesh.npoints} vertices. They were measured on different meshes; "
            "interpolate the field onto this one first."
        )
    mesh.pointdata[name] = values
    return mesh
