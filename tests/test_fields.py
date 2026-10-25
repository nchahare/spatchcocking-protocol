"""The measurement layer, checked against the fields shipped with the example.

These are reproduction tests, not smoke tests: each field is recomputed and
compared with the one the protocol wrote to data/example_outputs/. The reference
surface there has the same vertices as the published example mesh, and the pHH3
point table holds the same 22,430 cells, so these are also the published numbers.
"""

import os

import numpy as np
import pytest
from vedo import Mesh

from spatchcocking import fields, io

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(HERE, "data", "example_inputs")
OUTPUTS = os.path.join(HERE, "data", "example_outputs")
STEM = "2025-10-22-12-30"

# xy is binned 4x from the acquisition's 5.535 um; z is the RI-corrected step.
IMAGE_SPACING = (4.5, 22.14, 22.14)


@pytest.fixture(scope="module")
def basal():
    """The example's reference surface, the outer (basal) surface of the tube."""
    return Mesh(os.path.join(OUTPUTS, "reference_surface.ply"))


@pytest.fixture(scope="module")
def shipped():
    return Mesh(os.path.join(OUTPUTS, "reference_surface_fields.vtk"))


def test_density_reproduces_the_shipped_field(basal, shipped):
    spots, _ = io.load_point_table(os.path.join(INPUTS, f"{STEM}-points_phh3.csv"))
    assert spots.shape == (22430, 3)
    ours = fields.density_from_points(basal, spots, radius=100.0)
    theirs = np.asarray(shipped.pointdata["pHH3_density"])
    assert np.allclose(ours, theirs, rtol=0, atol=1e-6)


# Tolerances are absolute, not relative: both curvatures cross zero, so a
# relative tolerance is meaningless near the sign change.
#
# Measured drift between vedo 2026.6.2.dev8 and 2026.6.1, with identical
# vertices, normals, adjacency and MKL: 4.0e-11 on K, 2.8e-8 on H. The cause is
# neighbor ordering changing the row order of the least-squares fit, so the
# rounding differs. Thresholds sit an order of magnitude above that, and are
# still ~1e-5 of each quantity's full range (K spans ~7e-5, H spans ~1.2e-2).
ATOL_GAUSS = 1e-9
ATOL_MEAN = 1e-6


def test_curvature_reproduces_the_shipped_field(basal, shipped):
    """Recomputed with the vendored fit, as protocol/measure.py does."""
    assert np.array_equal(basal.vertices, shipped.vertices)
    ours = fields.curvature(basal, depth=5, degree=2, normalise=False)
    for name, atol in (("Gauss_Curvature", ATOL_GAUSS), ("Mean_Curvature", ATOL_MEAN)):
        np.testing.assert_allclose(np.asarray(ours.pointdata[name]),
                                   np.asarray(shipped.pointdata[name]),
                                   rtol=0, atol=atol, err_msg=name)


def test_curvature_matches_the_published_fit(basal):
    """Normalized, this is the fit that produced the original 3744-row file."""
    curv = fields.curvature(basal, normalise=True)
    assert curv.npoints == 3744, "face-area normalization should drop exactly five vertices"
    # The mesh as it stands is the default, and carries all 3749.
    full = fields.curvature(basal, normalise=False)
    assert full.npoints == basal.npoints == 3749

    k_norm = np.asarray(curv.pointdata["Gauss_Curvature"])
    k_full = np.asarray(full.pointdata["Gauss_Curvature"])
    # Dropping five vertices must not move the distribution.
    assert np.mean(k_full) == pytest.approx(np.mean(k_norm), rel=0.02)


def test_curvature_signs_and_units(basal):
    curv = fields.curvature(basal, normalise=False)
    k = np.asarray(curv.pointdata["Gauss_Curvature"])
    h = np.asarray(curv.pointdata["Mean_Curvature"])
    k1 = np.asarray(curv.pointdata["K1"])
    k2 = np.asarray(curv.pointdata["K2"])
    # K = k1 k2 and H = (k1 + k2)/2 must hold by construction.
    assert np.allclose(k1 * k2, k, atol=1e-12)
    assert np.allclose((k1 + k2) / 2, h, atol=1e-12)
    assert k1.min() >= k2.min()


def test_attach_refuses_a_field_measured_on_another_mesh(basal):
    with pytest.raises(ValueError, match="different meshes"):
        fields.attach(basal.clone(), "bogus", np.zeros(basal.npoints - 1))


# ---------------------------------------------------------------- intensity

@pytest.fixture(scope="module")
def image():
    path = os.path.join(INPUTS, f"{STEM}-image.tif")
    if not os.path.exists(path):
        pytest.skip("example image not present")
    return io.load_stack(path, IMAGE_SPACING, channel="pHH3")


def test_intensity_lands_in_the_image(basal, image):
    volume, spacing = image
    values = fields.intensity_on_surface(basal, volume, spacing, depth_um=15.0, n_samples=4)
    assert values.shape == (basal.npoints,)
    assert np.isfinite(values).all()
    # uint8 data: a mean over depths cannot leave the byte range.
    assert 0 <= values.min() and values.max() <= 255


def test_outward_normals_point_out_of_the_tissue(basal):
    """"In" must mean into the tissue.

    VTK's normals on a (z, y, x) mesh point inward, and the first version of the
    intensity sampling trusted them, so "just inside the surface" was measured
    just outside it. Pinned against the mask the surface was made from.
    """
    import tifffile

    mask = tifffile.imread(os.path.join(INPUTS, f"{STEM}-outer_mask.tif")) > 0
    spacing = np.array([4.5, 5.535, 5.535])
    verts = np.asarray(basal.vertices)
    out = fields.outward_normals(basal)

    def in_mask(p):
        idx = np.clip(np.round(p / spacing).astype(int), 0, np.array(mask.shape) - 1)
        return mask[tuple(idx.T)].mean()

    assert in_mask(verts - 15 * out) > 0.95
    assert in_mask(verts + 15 * out) < 0.05


def test_intensity_directions(basal, image):
    volume, spacing = image
    got = {d: fields.intensity_on_surface(basal, volume, spacing, depth_um=15.0,
                                          n_samples=4, direction=d)
           for d in ("in", "out", "both")}
    # the tissue is brighter than the background around it
    assert got["in"].mean() > got["out"].mean()
    # "both" is the average of the two sides
    np.testing.assert_allclose(got["both"], (got["in"] + got["out"]) / 2)
    with pytest.raises(ValueError, match="direction"):
        fields.intensity_on_surface(basal, volume, spacing, direction="sideways")


def test_intensity_rejects_a_mesh_outside_the_volume(basal, image):
    volume, spacing = image
    far = basal.clone()
    far.vertices = np.asarray(basal.vertices) + 1e6
    with pytest.raises(RuntimeError, match="no mesh vertex falls inside"):
        fields.intensity_on_surface(far, volume, spacing)


def test_intensity_rejects_a_multichannel_volume(basal):
    with pytest.raises(ValueError, match="single channel"):
        fields.intensity_on_surface(basal, np.zeros((2, 4, 4, 4)), IMAGE_SPACING)


def test_intensity_rejects_bad_spacing(basal, image):
    volume, _ = image
    with pytest.raises(ValueError, match="three positive numbers"):
        fields.intensity_on_surface(basal, volume, (4.5, 0.0, 22.14))
