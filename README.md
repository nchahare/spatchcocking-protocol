# spatchcocking: protocol

Scripts, example data, and code for the protocol *spatchcocking*, which unwraps
the surface of a tubular tissue into a standardized two-dimensional map.

Named for the culinary technique of splitting and flattening a bird along its
midline, spatchcocking places every point of the surface by its position along
the tube, from 0 at the tail to 1 at the head, and its angle around the tube,
with the dorsal midline at 0 degrees. Any field measured on the surface travels
into the map: curvature, the intensity of each image channel next to the
surface, the density of segmented points, and values carried by points, such as
wall thickness. Because every sample lands in the same coordinates, the maps of
several samples can be averaged.

The method was developed for Chahare, Imamura & Nerurkar (2026), *Biophysical
Reports* **6**:100280, [doi:10.1016/j.bpr.2026.100280](https://doi.org/10.1016/j.bpr.2026.100280),
whose code and data are at https://github.com/nchahare/spatchcocking.

## What is here

| Folder | What it holds |
|---|---|
| `protocol/` | One script per step of the protocol, with its settings at the top |
| `scripts/` | The input checks and the dorsal picker that the protocol calls |
| `notebooks/` | The same steps as notebooks, with more plots |
| `data/example_inputs/` | The example: an HH20 chick cranial neural tube, `2025-10-22-12-30` |
| `data/example_outputs/` | The protocol's run on the example, to check a run against |
| `data/example_maps/` | The maps of six embryos (three HH17, three HH20), for pooling |
| `src/spatchcocking/` | The Python package |
| `tests/` | The test suite |

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge), then, in the
unpacked folder:

```bash
conda env create -f environment.yml
conda activate spatchcocking
```

This installs every package at the version the protocol was tested with, and
this package itself.

## Run the protocol on the example

Run each script from this folder, with the environment active. Every script
reads the sample's description from `protocol/sample.py`, which is set to the
example; for your own sample, change that file only.

| Step | Command |
|---|---|
| Check the inputs | `python scripts/check_inputs.py` |
| Build the reference surface | `python protocol/build_surface.py` |
| Measure on the surface | `python protocol/measure.py` |
| Mark the dorsal points | `python protocol/pick_dorsal_points.py` |
| Straighten and unwrap | `python protocol/unwrap.py` |
| Show the map | `python protocol/show_map.py` |
| Pool several maps | `python protocol/pool_maps.py` |

Results are written to `results/2025-10-22-12-30/`. Compare them with
`data/example_outputs/`, whose README also shows how to unwrap the example with
the dorsal points made for it, without clicking.

## Notebooks

The same steps, interactively, in `notebooks/`, stored as jupytext `.py` files.
Open them in Jupyter or Visual Studio Code, or convert one with
`jupytext --to notebook notebooks/02_mesh.py`. They call the same functions as
the scripts.

| Notebook | Step |
|---|---|
| `00_environment.py` | Is everything installed and working? |
| `01_inputs.py` | Are the inputs right? |
| `02_mesh.py` | Mask to reference surface |
| `03_fields.py` | Every measurement onto the surface |
| `04_spatchcocking.py` | Medial axis, dorsal points, straighten, unwrap |
| `05_figure_plots.py` | Every map, the dorsal check, and pooling |

## Voxel size is always given, never read

No function reads the voxel size from a file header, on purpose. A header can
be wrong, for example when the z step needs a refractive-index correction for
clearing, and a wrong voxel size gives wrong maps without any error. Give the
voxel size of each file in `protocol/sample.py`.

## Validity

The map is built from planes normal to the medial axis, and two such planes meet
on the inside of a bend, at the bend's radius of curvature. The map is therefore
one-to-one only where the tube's radius is smaller than that radius of
curvature.

Image intensity next to the surface is not a count of cells. To count cells,
use point data.

## The command line

The package also installs a `spatchcock` command for scripting: `view`,
`mesh`, `measure`, `axis`, `project`, `map`, and `compare`, each printing a
summary of counts, ranges, and units. `spatchcock <command> --help` lists the options.

## Tests

```bash
pytest
pytest -m slow
```

The tests compare every field and the map with `data/example_outputs/`;
`-m slow` adds building the reference surface from the mask.

## Where the code lives

| Module | What it holds |
|---|---|
| `api.py` | The protocol's steps: condition, measure, medial_axis, project, unwrap, compare |
| `cli.py` | The `spatchcock` command |
| `fields.py` | Curvature, thickness, point density, image intensity |
| `io.py` | Loading, with the voxel size given by the caller |
| `frame.py` | The dorsal frame along the medial axis |
| `viewer.py` | The napari view |
| `plotting.py`, `colors.py` | Drawing the maps |
| `pipeline.py` | The published projection, unchanged |
| `meshing.py` | Mask to surface mesh |
| `attic.py` | Published functions no longer used |

## License

MIT, see `LICENSE`. Vendored third-party code keeps its own notice in
`src/spatchcocking/_vendor/`.
