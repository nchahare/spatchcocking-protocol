"""`mask_to_mesh` must reproduce the shipped meshes, and fail loudly otherwise.

The protocol starts at a segmentation mask, so this is the first step a reader
runs and the one they have no way of checking themselves. Two things are
therefore asserted: that the route from raw TIFF to finished surface gives back
exactly the meshes the paper used, and that the failure mode `get_mesh` has -
returning an empty mesh and printing "Saved:" - cannot happen quietly.

Marching cubes over a 462 x 768 x 1024 stack takes about 45 s, so the
reproduction test is opt-in: run with `pytest -m slow`.
"""
import numpy as np
import pytest
from vedo import Mesh

from spatchcocking import meshing

SCALING = (4.5, 5.535, 5.535)      # z, xy, xy in micron; notebook 01
# The example's reference surface has the vertices of the paper's basal mesh.
PAIRS = [("data/example_inputs/2025-10-22-12-30-outer_mask.tif",
          "data/example_outputs/reference_surface.ply")]


@pytest.mark.slow
@pytest.mark.parametrize("mask,shipped", PAIRS)
def test_reproduces_the_shipped_mesh(mask, shipped):
    """Raw TIFF in, the paper's mesh out, vertex for vertex."""
    import os
    if not os.path.exists(mask):
        pytest.skip(f"{mask} not present")
    got = meshing.mask_to_mesh(mask, SCALING[0], SCALING[1])
    want = Mesh(shipped)
    assert got.npoints == want.npoints
    np.testing.assert_array_equal(np.asarray(got.vertices),
                                  np.asarray(want.vertices))


def test_empty_mask_raises_instead_of_writing_nothing(tmp_path):
    """The defect this function exists to fix.

    `get_mesh` on a mask it cannot surface returns a zero-point mesh, writes an
    empty STL and prints "Saved:". Anything here must raise instead.
    """
    import tifffile
    p = tmp_path / "empty.tif"
    tifffile.imwrite(str(p), np.zeros((8, 16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="no surface"):
        meshing.mask_to_mesh(str(p), *SCALING[:2])


def test_never_returns_an_empty_mesh(tmp_path):
    """A mask that does surface must come back with something in it."""
    import tifffile
    vol = np.zeros((40, 48, 48), dtype=np.uint8)
    zz, yy, xx = np.ogrid[:40, :48, :48]
    vol[((zz - 20) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) < 14 ** 2] = 1
    p = tmp_path / "ball.tif"
    tifffile.imwrite(str(p), vol)

    # The defaults are tuned for a ~600k-vertex marching-cubes surface; on a
    # mesh this small they collapse it to a few per cent of its size. That is
    # documented behavior, not a bug, so scale them down here.
    mesh = meshing.mask_to_mesh(str(p), *SCALING[:2], smooth_iter=5,
                                decimate_fraction=0.9, final_fraction=0.9,
                                final_smooth_iter=2)
    assert mesh.npoints > 0
    assert mesh.ncells > 0
    # A sphere of radius 14 voxels, 4.5 um in z and 5.535 in xy.
    ext = np.ptp(np.asarray(mesh.vertices), axis=0)
    assert 100 < ext[0] < 145, ext          # z, ~14*2*4.5 = 126 um
    assert 130 < ext[1] < 180, ext          # y, ~14*2*5.535 = 155 um
    assert 130 < ext[2] < 180, ext          # x


def test_writes_only_when_asked(tmp_path):
    """get_mesh always wrote two files into the working directory."""
    import os
    import tifffile
    vol = np.zeros((40, 48, 48), dtype=np.uint8)
    zz, yy, xx = np.ogrid[:40, :48, :48]
    vol[((zz - 20) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) < 14 ** 2] = 1
    p = tmp_path / "ball.tif"
    tifffile.imwrite(str(p), vol)

    small = dict(smooth_iter=5, decimate_fraction=0.9, final_fraction=0.9,
                 final_smooth_iter=2)
    before = set(os.listdir(tmp_path))
    meshing.mask_to_mesh(str(p), *SCALING[:2], **small)
    assert set(os.listdir(tmp_path)) == before

    out = tmp_path / "out.ply"
    meshing.mask_to_mesh(str(p), *SCALING[:2], out_path=str(out), **small)
    assert out.exists()
