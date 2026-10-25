"""See the surface sitting inside the image it came from.

This is a look-and-confirm step, not a measurement. Its whole job is to answer one
question a number cannot: does the mesh lie where the tissue actually is? A mask
that drifted, a spacing that is wrong by a constant, a stack read in the wrong axis
order -- all of these produce a perfectly plausible 2D map and are obvious the
moment the surface is drawn inside the volume.

napari is an optional dependency. Nothing else in this package imports it, and no
step of the projection needs it. Install with::

    pip install "spatchcocking[napari]"

**The registration needs no transform, and this is the part not to "fix".**
Marching cubes returns vertices in the array's own axis order, already multiplied
by the voxel spacing, so mesh vertices are (z, y, x) in micrometers -- which is
exactly napari's order. The image layer therefore gets ``scale=spacing`` while the
surface and any points go in as raw micrometers with scale left at 1. No axis swap,
no flip, anywhere. The nonzero bounding box of the mask and the bounds of the mesh
marched from it agree to half a voxel on every axis, exactly where an isosurface
belongs relative to the voxel centers it was built from.
"""

import numpy as np

__all__ = ["build_scene", "show_in_context"]

# The contrast floor is a high percentile, not a low one. With a low black point a
# 3D maximum-intensity projection degenerates into vertical streaks: every ray
# picks up enough background to saturate.
IMAGE_FLOOR_PCT = 95.0
IMAGE_CEIL_PCT = 99.9


def _contrast_limits(volume, n_slices=8):
    """Percentile limits sampled from a few z-slices rather than the whole stack."""
    idx = np.linspace(0, volume.shape[0] - 1, min(n_slices, volume.shape[0]))
    sample = np.concatenate([np.asarray(volume[int(i)]).ravel() for i in idx])
    lo, hi = np.percentile(sample, [IMAGE_FLOOR_PCT, IMAGE_CEIL_PCT])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def _mesh_arrays(mesh):
    """vedo mesh -> (vertices (z, y, x) um, triangular faces) as plain numpy."""
    m = mesh.triangulate()
    return (
        np.asarray(m.vertices, dtype=np.float32),
        np.asarray(m.cells, dtype=np.int32),
    )


def build_scene(
    image,
    spacing,
    mesh,
    points=None,
    channel_names=None,
    colormaps=("red", "gray"),
    mesh_opacity=0.35,
    point_size=12.0,
    point_color="cyan",
    rendering="attenuated_mip",
    zoom=0.13,
):
    """Build the viewer and return it, without entering the event loop.

    Separate from `show_in_context` so the scene can be inspected and tested.
    `show_in_context` blocks in `napari.run()` until a human closes the window, so
    anything that calls it without a screenshot and without a human waits forever
    -- which is no way to write a test, and was how this split came about.

    Parameters
    ----------
    image : (z, y, x) or (c, z, y, x) array
        The raw stack. Load it with `io.load_stack`, which makes you state the
        spacing rather than trusting the file.
    spacing : tuple of float
        (z, y, x) micrometers per voxel of `image`.
    mesh : vedo.Mesh
        Vertices in micrometers. Drawn translucent so the volume shows through.
    points : (N, 3) array, optional
        A point cloud in micrometers, (z, y, x) -- the shipped pHH3 spots, for
        instance. Drawn in a color that contrasts with the first channel, purely
        as a visual check that the cloud lands on the surface.
    channel_names : sequence of str, optional
        Layer names per channel. Defaults to "channel 0", "channel 1", ...

    Returns
    -------
    napari.Viewer
    """
    try:
        import napari
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "viewing needs napari, which is an optional extra. Install it with:\n"
            '    pip install "spatchcocking[napari]"\n'
            "Nothing else in this package requires it, and the projection does not."
        ) from exc

    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3:
        raise ValueError(
            f"spacing must be (z, y, x) in um, got {spacing}. It is not read from "
            "the file on purpose -- see spatchcocking.io."
        )
    if image.ndim == 3:
        channels = [image]
    elif image.ndim == 4:
        channels = list(image)
    else:
        raise ValueError(
            f"image has shape {image.shape}; expected (z, y, x) or (c, z, y, x)."
        )

    names = list(channel_names) if channel_names else [
        f"channel {i}" for i in range(len(channels))
    ]
    if len(names) != len(channels):
        raise ValueError(
            f"{len(names)} channel names given for {len(channels)} channels."
        )

    viewer = napari.Viewer(ndisplay=3, title="spatchcocking - mesh in image")

    for i, (vol, name) in enumerate(zip(channels, names)):
        viewer.add_image(
            vol,
            name=name,
            scale=spacing,
            # Per-layer units, not the scale bar's own unit setting: this is what
            # makes the bar read "1 mm" rather than "1 kpixel".
            units="um",
            colormap=colormaps[i % len(colormaps)],
            blending="additive",
            contrast_limits=_contrast_limits(vol),
            rendering=rendering,
            attenuation=0.25,
            interpolation3d="linear",
            visible=(i == 0),
        )

    verts, faces = _mesh_arrays(mesh)
    viewer.add_surface(
        (verts, faces),
        name="mesh",
        colormap="gray",
        units="um",
        opacity=mesh_opacity,
        shading="smooth",
    )

    if points is not None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points has shape {points.shape}; expected (N, 3) in um."
            )
        viewer.add_points(
            points,
            name=f"points (n={len(points)})",
            size=point_size,
            units="um",
            face_color=point_color,
            border_color=point_color,
            border_width=0,
            blending="translucent",
        )

    # napari 0.9 moved the camera onto viewer.scene and the scale bar into
    # viewer.canvas.overlays; keep working on either side of that change.
    camera = getattr(viewer, "scene", viewer).camera
    camera.center = tuple(float(v) for v in verts.mean(axis=0))
    camera.zoom = zoom
    camera.angles = (0, 0, 90)
    try:
        viewer.canvas.overlays.scale_bar.visible = True
    except AttributeError:
        viewer.scale_bar.visible = True

    return viewer


def show_in_context(*args, screenshot=None, **kwargs):
    """Build the scene and show it.

    With `screenshot` set, render once to that path, close, and return. Without
    it, hand control to napari and block until the window is closed -- so do not
    call this from a script that has nobody in front of it.

    All other arguments go to `build_scene`.
    """
    viewer = build_scene(*args, **kwargs)

    if screenshot:
        viewer.window._qt_window.resize(1400, 1000)
        viewer.screenshot(screenshot, canvas_only=True, flash=False)
        viewer.close()
        return viewer

    import napari

    napari.run()
    return viewer
