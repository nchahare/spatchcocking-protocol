"""Pick dorsal ANCHORS on a mesh, with the medial axis visible and a live seam.

Three things make the dorsal points reliable:

  * the medial axis is computed FIRST and drawn, so anchors can be placed where
    they are needed rather than guessed at
  * each click pins the dorsal direction at one axis station; between anchors a
    rotation-minimizing frame carries it, so no part of the axis is ever left
    without a direction (the published scheme leaves the stations near the ends
    without a usable direction, however carefully anyone clicks)
  * the predicted seam redraws after every click, so the consequence of an
    anchor is visible immediately instead of surfacing in the final heatmap

About five anchors are enough to follow a full hand trace of the midline. Add
anchors until the seam stops moving.

What the user does:
  1. click the TAIL end of the tube. This sets which end is posterior; it used
     to be guessed from the first anchor, which silently flipped maps
  2. click about five dorsal points along the dorsal midline; press q

So that what is shown can be trusted:
  * the red line is drawn at each station's own radius, so it lies on the
    surface; at one fixed radius it floated off the thin tail and sank into the
    head, and correct anchors looked wrong
  * only the stations inside the tube are drawn and clickable; the medial axis
    runs past both ends of the tissue
  * each station is numbered, and a click far from the axis (further than
    1.4x the tube's radius there) is flagged, to undo with u

Usage:
    python scripts/pick_anchors.py <mesh> [--out <sidecar.json>]
                                   [--iterations 10] [--recompute]
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

import numpy as np
from vedo import Line, Mesh, Plotter, Points, Sphere, Text2D

from spatchcocking.api import inside_stations, station_radii
from spatchcocking.frame import dorsal_frame_from_anchors, tangents

NUM_AXIS_POINTS = 25
MLS_ITERATIONS = 10


def fingerprint(mesh):
    # the same function io.load_anchors checks against, so the two cannot disagree
    from spatchcocking.io import mesh_fingerprint

    return mesh_fingerprint(mesh)


def tube_ends(mesh):
    """The two ends of the tube: the pair of vertices farthest apart over the surface.

    Farthest along the surface, not in a straight line. The extremes of the first
    principal component are the ends only of a straight tube; on a curled tube one
    of them can lie partway along the curl, and an axis seeded from it runs out
    to the end and back. Two sweeps of Dijkstra over the mesh edges
    find the ends of a curled tube as well as a straight one.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    verts = mesh.vertices
    faces = np.asarray(mesh.cells)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    w = np.linalg.norm(verts[i] - verts[j], axis=1)
    graph = coo_matrix((w, (i, j)), shape=(len(verts), len(verts))).tocsr()
    far = lambda start: int(np.argmax(np.nan_to_num(  # noqa: E731
        dijkstra(graph, directed=False, indices=start), posinf=-1)))
    a = far(0)
    b = far(a)
    return verts[a], verts[b]


def compute_axis(mesh, cache, iterations=MLS_ITERATIONS, recompute=False):
    """Medial axis, seeded automatically from the two ends of the tube.

    The published getAxis needs a point near each end to order the skeleton.
    That does not have to be a human click: tube_ends finds them. Which end is
    posterior is decided later by the tail click.

    `iterations` is the number of smoothing rounds that pull the sampled surface
    onto its center line; more of them remove a loop left at an end. A cached
    axis is reused unless `recompute` is set, so a new `iterations` needs both.
    """
    if cache and os.path.exists(cache) and not recompute:
        print(f"axis: cached {cache} (recompute to change it)")
        return np.load(cache)

    from spatchcocking import pipeline as sp
    start, end = tube_ends(mesh)
    print(f"axis: seeding from the tube's ends {np.round(start, 1)} and {np.round(end, 1)}")
    print(f"axis: computing with {iterations} iterations "
          "(this takes about a minute on a dense mesh)")
    # The published getAxis always saves "<namefile>-axis.npy", and without a name
    # it writes "noname-axis.npy" into the working folder. Named after the cache,
    # the file it saves is the cache itself; with no cache, it saves into a
    # temporary folder that is then removed.
    with tempfile.TemporaryDirectory() as scratch:
        if cache and cache.endswith("-axis.npy"):
            namefile = cache[:-len("-axis.npy")]
        else:
            namefile = os.path.join(scratch, "axis")
        axis = np.asarray(sp.getAxis(mesh, np.array([start, end]),
                                     namefile=namefile, num_points=NUM_AXIS_POINTS,
                                     N=iterations, check=None))
    if cache:
        np.save(cache, axis)
        print(f"axis: cached to {cache}")
    return axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=MLS_ITERATIONS,
                    help="smoothing rounds for the medial axis (default 10); raise it, up to 20, if the axis loops at an end")
    ap.add_argument("--recompute", action="store_true",
                    help="compute the axis afresh instead of reusing the saved one")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.mesh)[0] + "-anchors.json"
    cache = os.path.splitext(args.mesh)[0] + "-axis.npy"

    mesh = Mesh(args.mesh)
    fp = fingerprint(mesh)
    verts = np.asarray(mesh.vertices)
    state = {"axis": compute_axis(mesh, cache, args.iterations, args.recompute),
             "tail_set": False}

    def geometry():
        axis = state["axis"]
        state["tang"] = tangents(axis)
        state["radii"] = station_radii(verts, axis)
        state["valid"] = inside_stations(mesh, axis)

    geometry()
    n = len(state["axis"])

    print(f"\nmesh        : {args.mesh}")
    print(f"vertices    : {mesh.npoints}")
    print(f"axis        : {n} stations, {len(state['valid'])} of them inside the tube "
          f"({state['valid'][0]} to {state['valid'][-1]})")
    print(f"fingerprint : {fp}")
    print(f"sidecar     : {out}\n")
    print("1. Click the tail end of the tube.")
    print("2. Click about five dorsal points along the dorsal midline. The red line shows dorsal.")
    print("   u undo   c start again   q save\n")

    idxs, dirs, picked, markers = [], [], [], []
    drawn = {"seam": None, "axis": [], "tail": None}

    plt = Plotter(title=f"Dorsal anchors - {os.path.basename(args.mesh)}", axes=9)
    banner = Text2D("", pos="top-left", c="k", bg="w", alpha=0.8, s=0.9)
    note = Text2D("", pos="bottom-left", c="red7", bg="w", alpha=0.8, s=0.8)

    def draw_axis():
        for a in drawn["axis"]:
            plt.remove(a)
        axis, valid = state["axis"], state["valid"]
        line = Line(axis[valid], c="blue4", lw=4)
        dots = Points(axis[valid], r=9)
        dots.pointdata["station"] = np.array(valid)
        dots.c("blue2")  # a plain color, not colored by the station array
        # 2D labels: drawn on top, so the tube doesn't hide them
        labels = dots.labels2d("station", scale=1.2, c="blue5")
        drawn["axis"] = [line, dots, labels]
        plt.add(*drawn["axis"])

    def redraw():
        if drawn["seam"] is not None:
            plt.remove(drawn["seam"])
            drawn["seam"] = None
        axis, valid, radii = state["axis"], state["valid"], state["radii"]
        if idxs:
            frame = dorsal_frame_from_anchors(axis, idxs, np.array(dirs))
            # each station's own radius, so the line lies on the surface
            r = np.interp(valid, [i for i in valid if np.isfinite(radii[i])],
                          [radii[i] for i in valid if np.isfinite(radii[i])])
            seam = axis[valid] + frame[valid] * (r * 1.05)[:, None]
            drawn["seam"] = Line(seam, c="red4", lw=6)
            plt.add(drawn["seam"])
        if not state["tail_set"]:
            banner.text("1. Click the tail end of the tube.")
        else:
            banner.text(
                f"2. Click about five dorsal points along the dorsal midline  |  {len(idxs)} placed"
                "\nThe red line shows dorsal."
                "\nu undo   c start again   q save")
        plt.render()

    def set_tail(p):
        valid = state["valid"]
        first, last = state["axis"][valid[0]], state["axis"][valid[-1]]
        if np.linalg.norm(p - last) < np.linalg.norm(p - first):
            state["axis"] = state["axis"][::-1].copy()  # station 0 is the tail
            geometry()
            draw_axis()
            print("  tail set: axis reversed so that it runs from the tail")
        else:
            print("  tail set")
        state["tail_set"] = True
        tail = state["axis"][state["valid"][0]]
        drawn["tail"] = Sphere(pos=tail, r=state["scale"] * 1.5, c="orange5")
        plt.add(drawn["tail"])
        note.text("")

    def on_click(evt):
        if evt.actor is None or evt.picked3d is None:
            return
        p = np.asarray(evt.picked3d, dtype=float)
        if not state["tail_set"]:
            set_tail(p)
            redraw()
            return
        axis, valid, tang, radii = state["axis"], state["valid"], state["tang"], state["radii"]
        i = valid[int(np.argmin(np.linalg.norm(axis[valid] - p, axis=1)))]
        if i in idxs:
            print(f"  station {i} already has a click; undo first to move it")
            return
        v = p - axis[i]
        v = v - (v @ tang[i]) * tang[i]
        if np.linalg.norm(v) < 1e-6:
            print("  click landed on the axis; click the surface")
            return
        away = np.linalg.norm(p - axis[i])
        if np.isfinite(radii[i]) and away > 1.4 * radii[i]:
            warning = (f"station {i}: that click is {away:.0f} um from the axis, the tube "
                       f"there is about {radii[i]:.0f} um. Is it on another part of the tube? u to undo")
            note.text(warning)
            print("  WARNING " + warning)
        else:
            note.text("")
        idxs.append(i)
        dirs.append(v / np.linalg.norm(v))
        picked.append(p)
        m = Sphere(pos=p, r=state["scale"], c="red5")
        markers.append(m)
        plt.add(m)
        print(f"  click {len(idxs)} -> station {i}")
        redraw()

    def on_key(evt):
        k = (evt.keypress or "").lower()
        if k == "c":
            for m in markers:
                plt.remove(m)
            markers.clear(); idxs.clear(); dirs.clear(); picked.clear()
            if drawn["tail"] is not None:
                plt.remove(drawn["tail"])
                drawn["tail"] = None
            state["tail_set"] = False
            note.text("")
            print("  started again: click the tail first")
            redraw()
        elif k == "u" and idxs:
            plt.remove(markers.pop())
            i = idxs.pop(); dirs.pop(); picked.pop()
            note.text("")
            print(f"  removed the click at station {i}")
            redraw()

    state["scale"] = float(np.nanmedian(state["radii"])) * 0.06
    plt.add_callback("left_click", on_click)
    plt.add_callback("key_press", on_key)
    draw_axis()
    redraw()
    plt.show(mesh.c("lightblue").alpha(0.55), banner, note, interactive=True).close()

    if not idxs:
        print("\nNo dorsal clicks. Nothing saved.")
        sys.exit(1)

    np.save(cache, state["axis"])  # in the orientation used: station 0 is the tail
    # io.load_anchors looks for the axis relative to the sidecar, so name it from
    # there: a bare file name breaks as soon as --out points at another folder.
    try:
        axis_file = os.path.relpath(cache, os.path.dirname(os.path.abspath(out)))
    except ValueError:  # another drive on Windows: no relative path exists
        axis_file = os.path.abspath(cache)
    payload = {
        "schema": "spatchcocking.dorsal_anchors/1",
        "mesh": args.mesh.replace("\\", "/"),
        "mesh_npoints": int(mesh.npoints),
        "mesh_fingerprint": fp,
        "axis_file": axis_file.replace("\\", "/"),
        "axis_stations": int(n),
        "inside_stations": [int(i) for i in state["valid"]],
        # The clicked point is kept as well as the direction: a curve along the
        # dorsal surface has to be built from points ON the surface, and a
        # direction alone cannot be turned back into one unambiguously.
        "anchors": [{"station": int(i), "direction": list(map(float, v)),
                     "point": list(map(float, p))}
                    for i, v, p in zip(idxs, dirs, picked)],
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{len(idxs)} dorsal clicks at stations {sorted(idxs)}; "
          f"tail end at station {state['valid'][0]}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
