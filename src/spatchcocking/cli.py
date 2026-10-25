"""One protocol step, one command.

    spatchcock view | mesh | measure | axis | project | map | compare

Each subcommand is a line a reader can be told to type, and each ends with a
SUMMARY block of counts, ranges and units. That block is part of the public
interface, not debug output: a reader checks their run against it, so its keys
should not drift between versions.

Every subcommand reports its own elapsed time, because a time measured on the
example data is worth more than an estimate.

Everything here is headless except `view` and `axis --pick`, which are the two
steps that genuinely need a human looking at something.
"""

import argparse
import contextlib
import io as _io
import os
import sys
import time

import numpy as np

from .colors import DEFAULT_FIELD

SPACING_HELP = (
    "voxel size in micrometers as three numbers, z y x. Never guessed from the "
    "file: this acquisition's metadata reports z=3.0 where the true step is 4.5."
)


@contextlib.contextmanager
def _quiet(verbose):
    """Swallow the published functions' own printing.

    `getDeformedmesh2` and its neighbors print progress lines and whole mesh
    descriptions. They are byte-for-byte copies of the published code and
    `tests/test_pipeline_parity.py` keeps them that way, so the noise is
    suppressed here rather than removed there. `--verbose` puts it back.
    """
    if verbose:
        yield
        return
    with contextlib.redirect_stdout(_io.StringIO()):
        yield


def _summary(title, rows, elapsed):
    print("\n" + "=" * 68)
    print("SUMMARY")
    for key, value in rows:
        print(f"  {key:<16}: {value}")
    print(f"  {'elapsed':<16}: {elapsed:.1f}s")
    print("=" * 68)


def _range(values, unit=""):
    v = np.asarray(values, dtype=float)
    suffix = f" {unit}" if unit else ""
    return f"min {v.min():.4g}{suffix}  max {v.max():.4g}{suffix}  mean {v.mean():.4g}{suffix}"


def _load_mesh(path):
    from vedo import Mesh

    if not os.path.exists(path):
        raise SystemExit(f"no mesh at {path}")
    return Mesh(path)


# ------------------------------------------------------------------ subcommands

def cmd_view(args):
    from . import io, viewer

    t0 = time.perf_counter()
    image, spacing = io.load_stack(args.image, args.spacing, channel=args.channel)
    mesh = _load_mesh(args.mesh)
    points = io.load_points(args.points) if args.points else None

    meta = io.find_meta(args.image)
    names = None
    if image.ndim == 4 and meta:
        names = meta.get("channels")

    viewer.show_in_context(
        image, spacing, mesh, points=points, channel_names=names,
        screenshot=args.screenshot,
    )
    _summary("view", [
        ("image", f"{image.shape} at {spacing} um"),
        ("mesh", f"{mesh.npoints} vertices"),
        ("points", "none" if points is None else f"{len(points)}"),
        ("screenshot", args.screenshot or "interactive"),
    ], time.perf_counter() - t0)


def cmd_mesh(args):
    from . import api

    t0 = time.perf_counter()
    mesh = api.condition(args.mask, args.spacing, out_path=args.out)
    _summary("mesh", [
        ("mask", os.path.basename(args.mask)),
        ("spacing", f"{tuple(args.spacing)} um"),
        ("vertices", mesh.npoints),
        ("faces", mesh.ncells),
        ("bounds um", " ".join(f"{b:.0f}" for b in mesh.bounds())),
        ("written", args.out or "not saved"),
    ], time.perf_counter() - t0)


def cmd_measure(args):
    from . import api, io

    t0 = time.perf_counter()
    mesh = _load_mesh(args.mesh)
    rows = [("mesh", f"{os.path.basename(args.mesh)}, {mesh.npoints} vertices")]

    if args.curvature:
        with _quiet(args.verbose):
            mesh = api.measure(mesh, "curvature")
        rows.append(("Mean_Curvature", _range(mesh.pointdata["Mean_Curvature"], "um^-1")))
        rows.append(("Gauss_Curvature", _range(mesh.pointdata["Gauss_Curvature"], "um^-2")))

    if args.thickness:
        mesh = api.measure(mesh, "thickness", inner=_load_mesh(args.thickness))
        rows.append(("thickness", _range(mesh.pointdata["thickness"], "um")))

    if args.density:
        pts = io.load_points(args.density)
        mesh = api.measure(mesh, "density", points=pts, radius=args.density_radius)
        rows.append(("phh3 density", _range(mesh.pointdata["phh3"])))
        rows.append(("spots", f"{len(pts)} at radius {args.density_radius:g} um"))

    if args.intensity:
        if args.spacing is None:
            raise SystemExit("--intensity needs --spacing")
        volume, spacing = io.load_stack(args.intensity, args.spacing, channel=args.channel)
        if volume.ndim == 4:
            raise SystemExit(
                f"{os.path.basename(args.intensity)} has several channels; name one "
                "with --channel (for the shipped example: pHH3 or DAPI)."
            )
        # Named for the channel, with the measurement appended: an intensity
        # field and a density field of the same stain are different quantities
        # and must not end up as columns distinguishable only by case.
        name = args.intensity_name or (
            f"{args.channel}_intensity" if isinstance(args.channel, str)
            else "intensity"
        )
        mesh = api.measure(
            mesh, "intensity", volume=volume, spacing=spacing,
            depth_um=args.depth, n_samples=args.samples, direction=args.direction,
            name=name,
        )
        rows.append((name, _range(mesh.pointdata[name])))
        side = {"both": "on both sides", "in": "inward", "out": "outward"}[args.direction]
        rows.append(("sampled", f"{args.samples} depths over {args.depth:g} um {side}"))

    if args.out:
        mesh.write(args.out)
    rows.append(("fields", list(mesh.pointdata.keys())))
    rows.append(("written", args.out or "not saved"))
    _summary("measure", rows, time.perf_counter() - t0)


def cmd_axis(args):
    from . import api

    t0 = time.perf_counter()
    if args.pick:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        picker = os.path.join(here, "scripts", "pick_anchors.py")
        raise SystemExit(
            "interactive picking lives in scripts/pick_anchors.py:\n"
            f"    python {os.path.relpath(picker)} {args.mesh}\n"
            "It writes a dorsal-anchor sidecar, which is what `project` reads. "
            "Keeping the clicks in a separate step is deliberate: the projection "
            "then runs from the sidecar alone, so a reviewer with no display "
            "reproduces every number."
        )

    mesh = _load_mesh(args.mesh)
    with _quiet(args.verbose):
        axis = api.medial_axis(mesh, num_points=args.stations, n_iter=args.iterations)
    diag = api.axis_diagnostics(axis, mesh.vertices)

    print(f"\n{'stn':>4} {'s':>6} {'rho um':>10} {'r_max um':>9} {'r/rho':>7}")
    for i in range(len(axis)):
        flag = "  <-- planes overlap" if i in diag["violating"] else ""
        print(f"{i:>4} {diag['s'][i]:>6.3f} {diag['rho'][i]:>10.1f} "
              f"{diag['r_max'][i]:>9.1f} {diag['ratio'][i]:>7.2f}{flag}")

    if args.out:
        np.save(args.out, axis)
    _summary("axis", [
        ("mesh", f"{mesh.npoints} vertices"),
        ("stations", len(axis)),
        ("overlapping", f"{len(diag['violating'])} {[int(i) for i in diag['violating']]}"),
        ("r_max p95", f"{np.nanpercentile(diag['r_max'], 95):.0f} um"),
        ("written", args.out or "not saved"),
    ], time.perf_counter() - t0)


def cmd_project(args):
    from . import api, io

    t0 = time.perf_counter()
    mesh = _load_mesh(args.mesh)
    car = io.load_anchors(args.anchors)
    axis_path = args.axis or os.path.join(
        os.path.dirname(os.path.abspath(args.anchors)), car["axis_file"]
    )
    if not os.path.exists(axis_path):
        raise SystemExit(
            f"the sidecar names axis file {car['axis_file']!r}, which is not at "
            f"{axis_path}. Pass --axis, or run `spatchcock axis` to make one."
        )
    axis = np.load(axis_path)
    # Read before projecting: getDeformedmesh2 decimates the mesh it is given,
    # in place, so asking afterwards reports the working resolution rather than
    # what the reader supplied.
    n_in = mesh.npoints

    with _quiet(args.verbose):
        deformed, info = api.project(
            mesh, axis, car["stations"], car["directions"],
            dists_threshold=args.dists_threshold,
        )
        shift = api.DORSAL_SHIFT_DEG if args.shift_deg is None else args.shift_deg
        df = api.unwrap(deformed, shift_deg=shift)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)

    _summary("project", [
        ("mesh in", f"{n_in} vertices"),
        ("warped at", f"{deformed.npoints} vertices"),
        ("anchors", f"{info['n_anchors']} at {sorted(int(i) for i in car['stations'])}"),
        ("axis stations", len(axis)),
        ("overlapping", f"{len(info['violating'])} {[int(i) for i in info['violating']]}"),
        ("dists_threshold", f"{info['dists_threshold']:.0f} um (derived)"),
        ("shift_deg", f"{shift:+.0f} (dorsal at 0)"),
        ("points out", len(df)),
        ("norm_height", f"{df.norm_height.min():.3f} to {df.norm_height.max():.3f}"),
        ("angle_degrees", f"{df.angle_degrees.min():.1f} to {df.angle_degrees.max():.1f}"),
        ("columns", list(df.columns)),
        ("written", args.out or "not saved"),
    ], time.perf_counter() - t0)


def cmd_map(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from .colors import field_style
    from .pipeline import getTightercmap
    from .plotting import grid_map

    t0 = time.perf_counter()
    df = pd.read_csv(args.csv)
    if args.field not in df.columns:
        raise SystemExit(
            f"{args.csv} has no column {args.field!r}. It carries: {list(df.columns)}"
        )
    values = df[args.field].to_numpy()
    label, cmap_name, symmetric = field_style(args.field)

    image, extent = grid_map(
        df["norm_height"].to_numpy(), df["angle_degrees"].to_numpy(),
        values, resolution=args.resolution,
    )

    # Median +/- 3 sigma, as the published figures scale them, rather than the
    # raw min and max -- a handful of outliers otherwise flatten the whole map.
    vmin, vmax = getTightercmap(values, sigma=3)
    if symmetric:
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    else:
        vmin = 0.0

    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    # grid_map returns the grid with s along the first axis and its extent as
    # [s_min, s_max, theta_min, theta_max]. The paper plots rostro-caudal
    # vertically and dorso-ventral horizontally, so transpose both.
    im = ax.imshow(image.T, extent=[extent[2], extent[3], extent[0], extent[1]],
                   origin="lower", aspect="auto",
                   cmap=cmap_name, vmin=vmin, vmax=vmax)
    # s = 1 is the forebrain end, so with origin="lower" rostral is already at the
    # top, as in the paper. (The published notebook plots 1 - s and then calls
    # invert_yaxis(); the two cancel, and doing only one of them flips the map.)
    ax.set_xlabel("dorso-ventral axis")
    ax.set_ylabel("rostro-caudal axis")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["V", "", "D", "", "V"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["C", "R"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    _summary("map", [
        ("csv", os.path.basename(args.csv)),
        ("field", args.field),
        ("colormap", f"{cmap_name}, {'symmetric about 0' if symmetric else 'from 0'}"),
        ("points", len(df)),
        ("value range", _range(values)),
        ("color limits", f"{vmin:.4g} to {vmax:.4g} (median +/- 3 sigma)"),
        ("grid", f"{args.resolution} x {args.resolution}"),
        ("written", args.out),
    ], time.perf_counter() - t0)


def cmd_compare(args):
    from . import api

    t0 = time.perf_counter()
    pooled, grids = api.compare(
        args.csv, args.field, rc_bins=args.rc_bins, dv_bin_width=args.dv_bins,
        labels=args.labels,
    )
    if args.out:
        pooled.to_csv(args.out, index=False)

    rows = [
        ("files", len(args.csv)),
        ("pooled rows", len(pooled)),
        ("field", args.field),
        ("grid", f"{args.rc_bins} RC x {360 // args.dv_bins} DV bins"),
    ]
    for label, grid in grids.items():
        filled = int(grid.notna().sum().sum())
        rows.append((label, f"{filled}/{grid.size} bins filled, "
                            f"mean {np.nanmean(grid.to_numpy()):.4g}"))
    rows.append(("written", args.out or "not saved"))
    _summary("compare", rows, time.perf_counter() - t0)


# ------------------------------------------------------------------------ parser

def build_parser():
    p = argparse.ArgumentParser(prog="spatchcock", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("view", help="see the mesh inside the raw image")
    v.add_argument("image")
    v.add_argument("mesh")
    v.add_argument("--spacing", nargs=3, type=float, required=True, help=SPACING_HELP)
    v.add_argument("--channel", default=None,
                   help="channel name from the *-meta.json sidecar, or an index")
    v.add_argument("--points", default=None, help="optional (N, 3) point cloud in um")
    v.add_argument("--screenshot", default=None, help="render once to this path and exit")
    v.set_defaults(func=cmd_view)

    m = sub.add_parser("mesh", help="segmentation mask -> surface mesh")
    m.add_argument("mask")
    m.add_argument("--spacing", nargs=3, type=float, required=True, help=SPACING_HELP)
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_mesh)

    s = sub.add_parser("measure", help="attach fields to a surface")
    s.add_argument("mesh")
    s.add_argument("--curvature", action="store_true")
    s.add_argument("--thickness", metavar="INNER_MESH", default=None,
                   help="inner surface, for wall thickness")
    s.add_argument("--density", metavar="SPOTS_NPY", default=None,
                   help="(N, 3) point cloud, for local density")
    s.add_argument("--density-radius", type=float, default=100.0,
                   help="kernel radius in um; should span the wall (default 100)")
    s.add_argument("--intensity", metavar="IMAGE", default=None,
                   help="image stack, for mean intensity next to the surface")
    s.add_argument("--channel", default=None, help="which channel of --intensity")
    s.add_argument("--intensity-name", default=None, help="field name to store it under")
    s.add_argument("--direction", choices=("both", "in", "out"), default="both",
                   help="which side of the surface to sample (default both)")
    s.add_argument("--depth", type=float, default=15.0,
                   help="how far from the surface to sample, on each side, um (default 15)")
    s.add_argument("--samples", type=int, default=4,
                   help="number of depths on each side (default 4)")
    s.add_argument("--spacing", nargs=3, type=float, default=None, help=SPACING_HELP)
    s.add_argument("--out", default=None, help="write the mesh with its fields, .vtk")
    s.add_argument("--verbose", action="store_true",
                   help="show the published functions' own output")
    s.set_defaults(func=cmd_measure)

    a = sub.add_parser("axis", help="medial axis and plane-overlap diagnostics")
    a.add_argument("mesh")
    a.add_argument("--stations", type=int, default=25)
    a.add_argument("--iterations", type=int, default=10)
    a.add_argument("--pick", action="store_true", help="point at the interactive picker")
    a.add_argument("--out", default=None)
    a.add_argument("--verbose", action="store_true",
                   help="show the published functions' own output")
    a.set_defaults(func=cmd_axis)

    j = sub.add_parser("project", help="straighten and unwrap into a 2D map")
    j.add_argument("mesh")
    j.add_argument("--anchors", required=True, help="dorsal-anchor sidecar json")
    j.add_argument("--axis", default=None, help="override the sidecar's axis file")
    j.add_argument("--dists-threshold", type=float, default=None,
                   help="override the value derived from this specimen's geometry")
    j.add_argument("--shift-deg", type=float, default=None,
                   help="rotate the azimuthal origin, degrees. Default -90, which "
                        "puts dorsal at 0: azimuth is measured from "
                        "cross(dorsal, tangent) while dorsal points along y, so it "
                        "lands at +90 without it")
    j.add_argument("--out", default=None, help="write the unwrapped table, .csv")
    j.add_argument("--verbose", action="store_true",
                   help="show the published functions' own output")
    j.set_defaults(func=cmd_project)

    g = sub.add_parser("map", help="render a standardized 2D map from a csv")
    g.add_argument("csv")
    g.add_argument("--field", default=DEFAULT_FIELD,
                   help=f"column to render (default {DEFAULT_FIELD})")
    g.add_argument("--resolution", type=int, default=500)
    g.add_argument("--out", default="map.png")
    g.set_defaults(func=cmd_map)

    c = sub.add_parser("compare", help="pool several maps onto a common grid")
    c.add_argument("csv", nargs="+")
    c.add_argument("--field", default=DEFAULT_FIELD,
                   help=f"column to pool (default {DEFAULT_FIELD})")
    c.add_argument("--labels", nargs="+", default=None)
    c.add_argument("--rc-bins", type=int, default=50)
    c.add_argument("--dv-bins", type=int, default=10, help="bin width in degrees")
    c.add_argument("--out", default=None, help="write the pooled table, .csv")
    c.set_defaults(func=cmd_compare)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, RuntimeError, ImportError) as exc:
        # Failures are reported in the words the Troubleshooting section uses,
        # not as a traceback the reader has to interpret.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
