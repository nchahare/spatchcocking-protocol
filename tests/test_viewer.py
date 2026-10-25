"""The napari viewer, including the case where napari is not installed.

That case is the one that matters most here, because it is what most readers meet:
napari is an optional extra, the test environment does not have it, and a reader
who has not installed it must get a sentence telling them what to type rather than
an ImportError traceback.

The parts that do not need napari -- the contrast percentiles and the vedo-to-numpy
conversion -- are tested directly. The full scene is tested only where napari is
importable, and skipped otherwise rather than silently not running.
"""

import os

import numpy as np
import pytest
from vedo import Mesh

from spatchcocking import viewer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(HERE, "data", "example_inputs")
OUTPUTS = os.path.join(HERE, "data", "example_outputs")
STEM = "2025-10-22-12-30"
IMAGE_SPACING = (4.5, 22.14, 22.14)

try:  # pragma: no cover - depends on the install
    import napari  # noqa: F401

    has_napari = True
except ImportError:
    has_napari = False

needs_napari = pytest.mark.skipif(not has_napari, reason="napari is an optional extra")


@pytest.fixture(scope="module")
def basal():
    return Mesh(os.path.join(OUTPUTS, "reference_surface.ply"))


# ------------------------------------------------- without napari installed

@pytest.mark.skipif(has_napari, reason="only meaningful when napari is absent")
def test_missing_napari_says_what_to_install(basal):
    """The message is the interface for anyone who has not installed the extra."""
    with pytest.raises(ImportError) as excinfo:
        viewer.build_scene(np.zeros((4, 4, 4), np.uint8), IMAGE_SPACING, basal)
    message = str(excinfo.value)
    assert 'pip install "spatchcocking[napari]"' in message
    assert "optional" in message
    # And it should say the projection does not need it, so nobody concludes the
    # pipeline is blocked on a GUI library.
    assert "projection does not" in message


def test_importing_the_package_does_not_need_napari():
    """`import spatchcocking` must stay clean so CI runs headless."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, spatchcocking, spatchcocking.api, spatchcocking.cli; "
         "print('napari' in sys.modules)"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing the package pulled in napari; it must stay lazy"
    )


# ------------------------------------------------------------- pure helpers

def test_contrast_floor_is_a_high_percentile():
    """A low black point turns the 3D MIP into vertical streaks.

    So the floor is the 95th percentile, not the 1st. Pinned because the
    difference is invisible in code review and obvious only in a render.
    """
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 40, size=(20, 32, 32), dtype=np.uint8)
    volume[5:8, 10:14, 10:14] = 250  # a bright object in a dim field
    lo, hi = viewer._contrast_limits(volume)
    assert lo > np.median(volume), "the floor must sit above the background"
    assert hi > lo


def test_contrast_limits_never_collapse():
    flat = np.full((6, 8, 8), 7, np.uint8)
    lo, hi = viewer._contrast_limits(flat)
    assert hi > lo, "a uniform volume must still give a usable range"


def test_mesh_arrays_are_triangles_in_zyx_micrometres(basal):
    verts, faces = viewer._mesh_arrays(basal)
    assert verts.shape == (basal.npoints, 3)
    assert faces.shape[1] == 3, "napari's surface layer needs triangles"
    assert faces.max() < len(verts)
    # Vertices are (z, y, x) in um, straight from marching cubes with spacing
    # applied -- napari's own order, so no transform is needed anywhere.
    assert np.allclose(verts, np.asarray(basal.vertices), atol=1e-3)


# ------------------------------------------------------- with napari present

@needs_napari
def test_scene_has_the_image_and_the_mesh(basal):
    from spatchcocking import io

    image, spacing = io.load_stack(
        os.path.join(INPUTS, f"{STEM}-image.tif"), IMAGE_SPACING
    )
    spots, _ = io.load_point_table(os.path.join(INPUTS, f"{STEM}-points_phh3.csv"))
    # build_scene, not show_in_context: the latter blocks in napari.run() until
    # a human closes the window, which in a test means forever.
    v = viewer.build_scene(
        image, spacing, basal, points=spots, channel_names=["pHH3", "DAPI"],
    )
    names = [layer.name for layer in v.layers]
    assert "mesh" in names
    assert "pHH3" in names and "DAPI" in names
    assert any(n.startswith("points") for n in names)
    # The image is scaled; the surface and points are raw micrometers.
    image_layer = v.layers["pHH3"]
    assert tuple(image_layer.scale) == spacing
    assert tuple(v.layers["mesh"].scale) == (1.0, 1.0, 1.0)
    v.close()


@needs_napari
def test_channel_count_must_match_the_names(basal):
    with pytest.raises(ValueError, match="channel names"):
        viewer.build_scene(
            np.zeros((2, 4, 4, 4), np.uint8), IMAGE_SPACING, basal,
            channel_names=["only-one"],
        )


@needs_napari
def test_rejects_a_stack_that_is_not_3d_or_4d(basal):
    with pytest.raises(ValueError, match="expected"):
        viewer.build_scene(np.zeros((4, 4), np.uint8), IMAGE_SPACING, basal)
