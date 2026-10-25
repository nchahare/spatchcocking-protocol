"""The command line, and the SUMMARY block it promises.

A reader checks their own run against the SUMMARY keys, so the keys are a
public interface. These tests pin them.

Everything here must run headless. `view` and `axis --pick` are excluded by
design, being the two steps that need a human looking at something.
"""

import os

import pytest

from spatchcocking.cli import main

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(HERE, "data", "example_inputs")
OUTPUTS = os.path.join(HERE, "data", "example_outputs")  # the protocol's run on INPUTS
STEM = "2025-10-22-12-30"
SURFACE = os.path.join(OUTPUTS, "reference_surface.ply")
FIELDS = os.path.join(OUTPUTS, "reference_surface_fields.vtk")
ANCHORS = os.path.join(OUTPUTS, "reference_surface-anchors.json")
AXIS = os.path.join(OUTPUTS, "reference_surface-axis.npy")
MAP = os.path.join(OUTPUTS, "spatchcocked.csv")


def summary_keys(text):
    keys, seen = [], False
    for line in text.splitlines():
        if line.strip() == "SUMMARY":
            seen = True
            continue
        if seen and line.startswith("  ") and ":" in line:
            keys.append(line.split(":", 1)[0].strip())
        elif seen and line.startswith("==="):
            break
    return keys


def test_measure_reports_every_field_it_computed(tmp_path, capsys):
    import numpy as np

    from spatchcocking import io

    # `measure --density` reads an (N, 3) .npy; the example ships its points as a table.
    points, _ = io.load_point_table(os.path.join(INPUTS, f"{STEM}-points_phh3.csv"))
    spots = os.fspath(tmp_path / "spots.npy")
    np.save(spots, points)
    out = os.fspath(tmp_path / "measured.vtk")
    code = main([
        "measure", SURFACE,
        "--curvature",
        "--density", spots,
        "--out", out,
    ])
    assert code == 0
    keys = summary_keys(capsys.readouterr().out)
    for expected in ("mesh", "Mean_Curvature", "phh3 density", "fields", "elapsed"):
        assert expected in keys, f"{expected!r} missing from the SUMMARY block"
    assert os.path.exists(out)


def test_axis_prints_the_overlap_table(capsys):
    code = main(["axis", SURFACE, "--stations", "25"])
    assert code == 0
    text = capsys.readouterr().out
    assert "r/rho" in text, "the plane-overlap diagnostic table is a protocol step"
    for expected in ("stations", "overlapping", "r_max p95"):
        assert expected in summary_keys(text)


def test_project_derives_its_threshold_and_reports_it(tmp_path, capsys):
    out = os.fspath(tmp_path / "spatchcocked.csv")
    code = main([
        "project", FIELDS, "--anchors", ANCHORS, "--axis", AXIS,
        "--out", out,
    ])
    assert code == 0
    text = capsys.readouterr().out
    keys = summary_keys(text)
    for expected in ("mesh in", "anchors", "dists_threshold", "points out",
                     "norm_height", "angle_degrees", "columns"):
        assert expected in keys
    assert "(derived)" in text, "the threshold must be derived, not a magic constant"
    assert os.path.exists(out)


def test_project_output_matches_the_example_row_count(tmp_path):
    import pandas as pd

    out = os.fspath(tmp_path / "spatchcocked.csv")
    main([
        "project", FIELDS, "--anchors", ANCHORS, "--axis", AXIS,
        "--out", out,
    ])
    df = pd.read_csv(out)
    assert len(df) == len(pd.read_csv(MAP)) == 7986, "the example map has 7986 rows"
    assert {"radius", "norm_height", "angle_degrees"} <= set(df.columns)
    assert 0.0 <= df.norm_height.min() and df.norm_height.max() <= 1.0
    assert -180.0 <= df.angle_degrees.min() and df.angle_degrees.max() <= 180.0


def test_the_published_functions_stay_quiet_unless_asked(tmp_path, capsys):
    out = os.fspath(tmp_path / "quiet.csv")
    main([
        "project", FIELDS, "--anchors", ANCHORS, "--axis", AXIS,
        "--out", out,
    ])
    text = capsys.readouterr().out
    assert "i am warpping" not in text, "published chatter should not reach protocol output"


def test_map_renders_a_figure(tmp_path):
    csv = MAP
    png = os.fspath(tmp_path / "map.png")
    assert main(["map", csv, "--field", "Mean_Curvature", "--out", png]) == 0
    assert os.path.getsize(png) > 0


def test_dorsal_lands_at_zero_degrees(tmp_path):
    """In the example, the thinnest part of the tube is dorsal (the roof plate).

    So a correctly oriented map puts the thickness minimum at theta = 0. This is
    the check that catches a wrong azimuthal origin, which no amount of staring
    at a heatmap will: the map looks entirely plausible rotated by 90 degrees.

    Azimuth is measured from cross(dorsal, tangent) while dorsal points along the
    frame's y axis, so dorsal comes out at +90 and the default shift of -90
    brings it to zero. Without the shift the minimum sits at +95.
    """
    import numpy as np
    import pandas as pd

    out = os.fspath(tmp_path / "oriented.csv")
    assert main([
        "project", FIELDS, "--anchors", ANCHORS, "--axis", AXIS,
        "--out", out,
    ]) == 0

    df = pd.read_csv(out)
    bins = pd.cut(df["angle_degrees"], np.arange(-180, 190, 10))
    means = df.groupby(bins, observed=False)["thickness_um"].mean().to_numpy()
    centers = np.arange(-175, 180, 10)
    thinnest = centers[int(np.nanargmin(means))]
    assert abs(thinnest) <= 15, (
        f"the thickness minimum is at {thinnest:+.0f} deg, not near dorsal at 0. "
        "The azimuthal origin is wrong -- check shift_deg."
    )


def test_map_puts_rostro_caudal_on_the_vertical_axis(tmp_path):
    """Orientation, pinned.

    `grid_map` returns the grid with s along the first axis and an extent of
    [s_min, s_max, theta_min, theta_max]. The paper plots rostro-caudal
    vertically, so `map` transposes. Getting this wrong produces a perfectly
    plausible figure whose axis labels describe the wrong axis, which is exactly
    the kind of error a picture does not announce.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv = MAP
    png = os.fspath(tmp_path / "oriented.png")
    assert main(["map", csv, "--field", "Mean_Curvature", "--out", png]) == 0

    ax = plt.gcf().axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x1 - x0 > 300, f"x should span the ~360 deg azimuth, spans {x1 - x0:.1f}"
    assert y1 - y0 <= 1.01, f"y should span normalized s in [0, 1], spans {y1 - y0:.3f}"
    plt.close("all")


def test_map_names_the_columns_it_has_when_the_field_is_wrong(tmp_path):
    csv = MAP
    with pytest.raises(SystemExit, match="Mean_Curvature"):
        main(["map", csv, "--field", "not_a_field", "--out", os.fspath(tmp_path / "x.png")])


def test_compare_pools_and_bins(tmp_path, capsys):
    csv = MAP
    out = os.fspath(tmp_path / "pooled.csv")
    code = main(["compare", csv, "--field", "thickness_um", "--labels", "HH20", "--out", out])
    assert code == 0
    keys = summary_keys(capsys.readouterr().out)
    assert "pooled rows" in keys and "grid" in keys and "HH20" in keys


def test_compare_counts_each_specimen_once_per_bin():
    # Two specimens in one bin: one with 9 points at value 0, one with 1 point at
    # value 10. Averaging the points would give 1; each specimen counting once
    # gives 5.
    import pandas as pd

    from spatchcocking import api

    many = pd.DataFrame({"norm_height": [0.5] * 9, "angle_degrees": [1.0] * 9, "v": [0.0] * 9})
    one = pd.DataFrame({"norm_height": [0.5], "angle_degrees": [1.0], "v": [10.0]})
    _, grids = api.compare([many, one], "v", labels=["g", "g"])
    assert float(grids["g"].stack().dropna().iloc[0]) == 5.0


def test_an_absent_axis_file_says_what_to_run(tmp_path, capsys):
    with pytest.raises(SystemExit, match="spatchcock axis"):
        main([
            "project", FIELDS, "--anchors", ANCHORS,
            "--axis", os.fspath(tmp_path / "nope.npy"),
        ])


def test_pick_points_at_the_interactive_script():
    with pytest.raises(SystemExit, match="pick_anchors.py"):
        main(["axis", SURFACE, "--pick"])
