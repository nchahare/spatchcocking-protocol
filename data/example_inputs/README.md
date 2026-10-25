# Example inputs: HH20 chick cranial neural tube, `2025-10-22-12-30`

Everything a reader brings to the protocol, and nothing it produces. The
protocol's outputs for this embryo are in `../example_outputs/`, to check a run against.

## The files

| File | What it is |
|---|---|
| `2025-10-22-12-30-image.tif` | Raw image, two channels: **pHH3 first, DAPI second** |
| `2025-10-22-12-30-outer_mask.tif` | Outer (basal) surface mask, 0 and 1: **the reference surface** |
| `2025-10-22-12-30-points_phh3.csv` | pHH3+ nuclei, one row per cell; columns `z, y, x` |
| `2025-10-22-12-30-points_thickness.csv` | Wall thickness at points on the outer surface; columns `z, y, x, thickness_um` |
| `2025-10-22-12-30-meta.json` | The image's voxel size, channel names, and the notes behind them |

## Voxel sizes (z, y, x, in µm)

| File | Voxel size |
|---|---|
| image | **4.5, 22.14, 22.14**. The image was shrunk 4× in x and y to fit in the repository |
| outer mask | **4.5, 5.535, 5.535**: the full-resolution grid |

**z is 4.5 µm, not the 3.0 µm the microscope metadata reports.** The 1.5× factor
is the refractive-index correction for DBE clearing. Never read the voxel size
from the files.

The two grids differ, but cover the same space. The image was binned, never
cropped, so the two register by voxel size alone.

## Frame of reference

**Every file is in the image's frame: z, y, x order, in µm, with (0, 0, 0) at the
image's first voxel.**

- The mask was segmented from this image, so it is in that frame by
  construction.
- Both point files are too. **Point data from any other tool must be converted
  to this frame before use.** Imaris, for example, exports x, y, z in
  microscope stage coordinates.

## The point files

Plain CSV, one row per point.

- **Columns `z, y, x` are required**, in µm.
- **Every other column is a value carried by that point.** Its header names it,
  with units in the name (`thickness_um`). The protocol doesn't know what a value
  means, so the header has to say.

`points_phh3.csv` has positions only; the value it gives is the *density* of the
cells. `points_thickness.csv` carries one value per point.

**Where the thickness points come from:** they are not an independent
measurement. Each row is a vertex of the outer surface mesh, the example's
reference surface (`../example_outputs/reference_surface.ply`), with the wall
thickness measured there, from the outer to the lumen surface, in Chahare et al.
(2026). They stand in for the kind of per-point measurement a reader might bring from another tool, e.g. cell area or perimeter.

## Provenance

The image, the mask, and the pHH3 points come from the published example of
Chahare et al. (2026), `data/example/` in https://github.com/nchahare/spatchcocking:
the image and the mask are byte-identical copies of `2025-10-22-12-30-image.tif`
and `2025-10-22-12-30-Outer_Mask.tif`, and the pHH3 points are
`2025-10-22-12-30-spots.npy` (detected in Imaris, filtered to the tissue wall,
already converted to this frame). Acquisition details are in
`2025-10-22-12-30-meta.json`, here.
