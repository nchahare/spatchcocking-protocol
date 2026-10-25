"""Loading, and the two things about this dataset that are easy to get wrong.

Both tests here exist because of a specific near-miss rather than for coverage:
the voxel spacing in the file is wrong by a factor of 1.5, and which channel is
which was decided by eye and is not recorded in the image at all.
"""

import os

import numpy as np
import pytest

from spatchcocking import io

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(HERE, "data", "example_inputs")
OUTPUTS = os.path.join(HERE, "data", "example_outputs")
STEM = "2025-10-22-12-30"
IMAGE = os.path.join(INPUTS, f"{STEM}-image.tif")
ANCHORS = os.path.join(OUTPUTS, "reference_surface-anchors.json")
IMAGE_SPACING = (4.5, 22.14, 22.14)

needs_image = pytest.mark.skipif(
    not os.path.exists(IMAGE),
    reason="example image not present",
)


@needs_image
def test_spacing_is_required_and_validated():
    with pytest.raises(ValueError, match="three positive numbers"):
        io.load_stack(IMAGE, (4.5, 22.14))
    with pytest.raises(ValueError, match="three positive numbers"):
        io.load_stack(IMAGE, (4.5, -1.0, 22.14))


@needs_image
def test_spacing_is_returned_exactly_as_given():
    """Never adjusted, never sniffed from the header.

    The file's own metadata would give z = 3.0 um, which is wrong for this
    pipeline by the 1.5x refractive-index correction for DBE clearing.
    """
    _, spacing = io.load_stack(IMAGE, IMAGE_SPACING, channel=0)
    assert spacing == IMAGE_SPACING


@needs_image
def test_channel_can_be_named_and_matches_its_index():
    by_name, _ = io.load_stack(IMAGE, IMAGE_SPACING, channel="pHH3")
    by_index, _ = io.load_stack(IMAGE, IMAGE_SPACING, channel=0)
    assert np.array_equal(by_name, by_index)
    assert by_name.ndim == 3


@needs_image
def test_channel_names_are_case_insensitive_and_distinct():
    phh3, _ = io.load_stack(IMAGE, IMAGE_SPACING, channel="pHH3")
    dapi, _ = io.load_stack(IMAGE, IMAGE_SPACING, channel="dapi")
    assert not np.array_equal(phh3, dapi)


@needs_image
def test_an_unknown_channel_name_lists_the_real_ones():
    with pytest.raises(ValueError, match="pHH3"):
        io.load_stack(IMAGE, IMAGE_SPACING, channel="GFP")


@needs_image
def test_the_sidecar_records_the_spacing_trap():
    meta = io.find_meta(IMAGE)
    assert meta is not None, "the image should have a *-meta.json beside it"
    assert tuple(meta["spacing_um_zyx"]) == IMAGE_SPACING
    assert meta["channels"] == ["pHH3", "DAPI"]
    assert "3.0" in meta["z_spacing_note"] and "4.5" in meta["z_spacing_note"]


def test_points_must_be_n_by_3(tmp_path):
    good, bad = os.fspath(tmp_path / "good.npy"), os.fspath(tmp_path / "bad.npy")
    np.save(good, np.zeros((5, 3), dtype=np.float32))
    np.save(bad, np.zeros((5, 2)))
    assert io.load_points(good).shape == (5, 3)
    with pytest.raises(ValueError, match="expected"):
        io.load_points(bad)


def test_point_tables_split_positions_from_values():
    folder = INPUTS
    cells, values = io.load_point_table(os.path.join(folder, f"{STEM}-points_phh3.csv"))
    assert cells.shape == (22430, 3) and values == {}
    points, values = io.load_point_table(os.path.join(folder, f"{STEM}-points_thickness.csv"))
    assert points.shape == (3749, 3) and list(values) == ["thickness_um"]


def test_a_point_table_without_z_y_x_names_what_it_has(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("X,Y,Z\n1,2,3\n")
    with pytest.raises(ValueError, match="z, y, x"):
        io.load_point_table(path)


def test_field_round_trip(tmp_path):
    coords = np.random.default_rng(0).normal(size=(10, 3))
    values = np.arange(10, dtype=float)
    path = os.fspath(tmp_path / "field.npy")
    io.save_field(path, coords, values)
    back_coords, back_values = io.load_field(path)
    assert np.allclose(back_coords, coords)
    assert np.allclose(np.ravel(back_values), values)


def test_field_refuses_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="same vertices"):
        io.save_field(os.fspath(tmp_path / "x.npy"), np.zeros((5, 3)), np.zeros(4))


def test_anchors_expose_stations_directions_and_points():
    car = io.load_anchors(ANCHORS)
    assert car["schema"].startswith("spatchcocking.dorsal_anchors/")
    assert len(car["stations"]) == len(car["directions"])
    # This sidecar was written after the clicked point started being stored, so
    # the geodesic frame route is available for it.
    assert car["points"] is not None
    assert car["points"].shape == (len(car["stations"]), 3)


def test_anchors_bring_their_axis_and_check_the_surface():
    from vedo import Mesh

    path = ANCHORS
    surface = Mesh(os.path.join(OUTPUTS, "reference_surface.ply"))
    car = io.load_anchors(path, surface)  # picked on this surface: passes
    assert car["axis"].shape == (car["axis_stations"], 3)

    moved = surface.clone()
    moved.vertices = np.asarray(surface.vertices) + 1.0
    with pytest.raises(ValueError, match="different surface"):
        io.load_anchors(path, moved)


def test_anchors_moved_away_from_their_axis_say_so(tmp_path):
    # Copied to another folder, the sidecar's axis file is not beside it. That
    # used to return axis None and crash later with an unrelated NumPy error.
    src = ANCHORS
    path = tmp_path / "anchors.json"
    path.write_text(open(src, encoding="utf-8").read(), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="medial axis"):
        io.load_anchors(os.fspath(path))


def test_anchors_reject_the_older_click_trace_schema(tmp_path):
    # The schema of the hand traces made before dorsal anchors existed.
    path = tmp_path / "legacy-dorsal.json"
    path.write_text('{"schema": "spatchcocking.dorsal_reference/1", '
                    '"points": [[0, 0, 0], [1, 1, 1]]}', encoding="utf-8")
    with pytest.raises(ValueError, match="dorsal_reference"):
        io.load_anchors(path)
