# Example outputs: the protocol's run on `../example_inputs/`

What the protocol produces for the example embryo, HH20 `2025-10-22-12-30`,
with the default settings of every script. Compare a run of your own against
these files; the tests use them as the reference.

## The files

| File | Made by | What it is |
|---|---|---|
| `reference_surface.ply` | `protocol/build_surface.py` | The reference surface |
| `reference_surface_fields.vtk` | `protocol/measure.py` | The same surface with its fields: `thickness_um`, `Gauss_Curvature`, `Mean_Curvature`, `K1`, `K2`, `pHH3_intensity`, `DAPI_intensity`, `pHH3_density` |
| `reference_surface-axis.npy` | `protocol/pick_dorsal_points.py` | The medial axis, (z, y, x) in µm |
| `reference_surface-anchors.json` | from the dorsal clicks, below | The dorsal points |
| `spatchcocked.csv` | `protocol/unwrap.py` | The map |
| `2025-10-22-12-30-dorsal_clicks.npy` | a person, in the picker | The dorsal midline clicks, (z, y, x) in µm |

## The dorsal points

Marking the dorsal points is the one manual step. For the example, the clicks
made for the published analysis are kept in `2025-10-22-12-30-dorsal_clicks.npy`,
and `reference_surface-anchors.json` was made from them, with no new clicks:

    python scripts/anchors_from_picks.py --mesh <results>/reference_surface.ply \
        --picks data/example_outputs/2025-10-22-12-30-dorsal_clicks.npy \
        --axis <results>/reference_surface-axis.npy

To unwrap the example without clicking, copy `reference_surface-anchors.json`
and `reference_surface-axis.npy` into the sample's results folder after
`protocol/measure.py`, then run `protocol/unwrap.py`; in notebook 04,
`USE_EXAMPLE_ANCHORS = True` does the same. The anchors only fit the
surface they were made on, so the surface must be built with the default
settings.

Your own clicks will differ slightly from these, and so will your map.
