# Example maps for pooling

One map per embryo, in the layout and columns that `protocol/unwrap.py` writes
(`<sample>/spatchcocked.csv`), for the pooling step (`protocol/pool_maps.py`).

| Folder | Stage |
|---|---|
| `2025-09-18-13-02` | HH17 |
| `2025-09-18-14-26` | HH17 |
| `2025-09-18-16-46` | HH17 |
| `2025-09-23-15-48` | HH20 |
| `2025-10-22-12-30` | HH20 |
| `2025-10-23-13-06` | HH20 |

Columns: `radius` (µm), `norm_height` (0 at the tail to 1 at the head), `angle_degrees`
(0 = dorsal midline), `thickness_um`, `Gauss_Curvature`, `Mean_Curvature`, `K1`,
`K2`, `pHH3_density` (pHH3+ cells within 100 µm).

These are the published maps of Chahare et al. (2026), split from the pooled
table `data/csv/spatchcocked_measurements.csv` of https://github.com/nchahare/spatchcocking,
one file per `timepoint`, with `thickness` renamed `thickness_um` and `phh3`
renamed `pHH3_density`. They carry no image intensity columns.
