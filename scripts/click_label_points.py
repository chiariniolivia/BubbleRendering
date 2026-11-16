"""Interactive click-to-label tool: shows a background image and records where you
click, so you can build your own ground-truth point set (e.g. "where I see a SiPM
cover slot by eye") to compare against a detector's output instead of re-eyeballing
overlays every time.

Usage:
    ~/py-envs/sbc/bin/python scripts/click_label_points.py
    ~/py-envs/sbc/bin/python scripts/click_label_points.py --image path/to/image.png --out my_points.json

With no --image, defaults to the same real photo CameraPoseCalibrationFromSiPMs.ipynb
compares against (run 20251114_37, first single-bubble event, cam2 frame16).

With --corners, each SiPM cover is marked by clicking its 4 corners (any order, any
direction -- they are sorted around the cover for you) instead of one click at its
middle. The point that gets saved for the cover is where its two diagonals cross, which
is the true image of a rectangle's center under perspective, so it no longer depends on
where you eyeballed "the middle". For a cover whose corners are too hard to make out
(thin, edge-on, partly hidden), press Enter after 1 click (its center) or after 2
clicks (two opposite corners -> their midpoint) to finish it early.

Controls:
    left click   add a point (or the next corner of the current cover)
    Enter        (--corners) finish the current cover early with 1 or 2 clicks
    right click  remove the point / cover nearest the click (undo-by-location)
    u            undo the most recently added click
    s            save to --out without closing
    close window save to --out automatically

Output is JSON: {"image": "<path>", "points": [[x, y], ...]}; with --corners "points" holds
one center per cover (diagonal crossing, midpoint, or the click) and "corners" holds each
cover's raw clicks (4 sorted corners, or the 1-2 clicks it was finished early with).
Load "points" back with
pose_calibration.load_clicked_points(path) -- an (n, 2) array, same shape
detect_sipm_covers()/detect_light_markers() return, so nearest-neighbor comparison
against a detector's output is a direct drop-in.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
import real_data_loading as rdl


def default_image_path():
    """The same real photo CameraPoseCalibrationFromSiPMs.ipynb already compares
    against: run 20251114_37's first single-bubble event, cam2 frame16."""
    run_dir = rdl.RUN_DIRS["20251114_37"]
    ev = rdl.single_bubble_events(run_dir)[0]
    return rdl.camera_image_path(run_dir, ev, camera=2, frame=16)


def default_out_path(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(__file__), "..", "outputs", "clicked_points", f"{stem}.json")


CORNERS_PER_COVER = 4


def order_corners(corners):
    """The 4 corner clicks sorted by angle around their average, i.e. in order around
    the cover whatever order they were clicked in (valid for any convex quad, which a
    rectangle seen from the camera always is)."""
    c = np.asarray(corners, dtype=float)
    mean = c.mean(axis=0)
    return c[np.argsort(np.arctan2(c[:, 1] - mean[1], c[:, 0] - mean[0]))]


def diagonal_center(corners):
    """Center of a 4-corner quad: where its two diagonals cross. For a rectangle seen in
    perspective this is the image of its true center (the plain corner average is not,
    once the quad is foreshortened). The corners may be given in any order.
    Returns ((x, y), ok); ok is False, and the plain corner average is returned instead,
    if the diagonals don't cross inside the quad (the 4 points aren't a convex quad,
    e.g. one click landed inside the other three)."""
    c = order_corners(corners)
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) > 1e-9:
        t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denom
        u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / denom
        if 0 < t < 1 and 0 < u < 1:
            return tuple(p + t * r), True
    return tuple(c.mean(axis=0)), False


class ClickSet:
    """The clicked points, kept free of any plotting so it can be tested headless.
    cornersPerPoint=1 is the original one-click-per-point behavior; 4 groups clicks into
    covers of 4 corners each, where finish_early() can close a cover with 1 or 2 clicks."""

    def __init__(self, cornersPerPoint=1):
        self.n = cornersPerPoint
        self.groups = []    # completed points: each a list of (x, y) clicks (n, or 1-2 if finished early)
        self.pending = []   # clicks of the point currently being built

    def add(self, x, y):
        self.pending.append((x, y))
        if len(self.pending) == self.n:
            self._finish()

    def _finish(self):
        if len(self.pending) == CORNERS_PER_COVER:
            self.pending = [tuple(pt) for pt in order_corners(self.pending)]
        self.groups.append(self.pending)
        self.pending = []

    def finish_early(self):
        """Closes the current cover with the 1 or 2 clicks made so far (its center, or two
        opposite corners). Returns False, doing nothing, for any other number of clicks."""
        if self.n > 1 and len(self.pending) in (1, 2):
            self._finish()
            return True
        return False

    def undo(self):
        """Removes the most recent click (so a finished cover goes back to one fewer corner)."""
        if self.pending:
            self.pending.pop()
        elif self.groups:
            self.pending = self.groups.pop()
            self.pending.pop()

    def remove_nearest(self, x, y):
        """Removes whichever is nearest the click: a pending corner, or a whole cover."""
        def dist(pt):
            return (pt[0] - x) ** 2 + (pt[1] - y) ** 2
        pendingBest = min(self.pending, key=dist, default=None)
        groupBest = min(self.groups, key=lambda g: min(dist(pt) for pt in g), default=None)
        pendingD = dist(pendingBest) if pendingBest is not None else np.inf
        groupD = min(dist(pt) for pt in groupBest) if groupBest is not None else np.inf
        if pendingD <= groupD and pendingBest is not None:
            self.pending.remove(pendingBest)
        elif groupBest is not None:
            self.groups.remove(groupBest)

    def centers(self):
        """One (x, y) per completed point, plus a list of the indices whose 4 corners
        weren't a convex quad (average used). 4 clicks -> diagonal crossing, 2 -> their
        midpoint, 1 -> the click itself."""
        centers, badIndices = [], []
        for i, group in enumerate(self.groups):
            if len(group) == CORNERS_PER_COVER:
                center, ok = diagonal_center(group)
                if not ok:
                    badIndices.append(i)
            else:
                center = tuple(np.mean(group, axis=0))
            centers.append(center)
        return centers, badIndices


def save_points(clickSet, image_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    centers, badIndices = clickSet.centers()
    data = {"image": image_path, "points": [list(map(float, c)) for c in centers]}
    if clickSet.n > 1:
        data["corners"] = [[list(map(float, pt)) for pt in group] for group in clickSet.groups]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"saved {len(centers)} point(s) to {out_path}")
    if badIndices:
        print(f"warning: cover(s) {badIndices} didn't have 4 corners forming a convex quad; "
              f"used the corner average for those -- redo them")
    if clickSet.pending:
        print(f"note: {len(clickSet.pending)} unfinished corner click(s) were not saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="path to the background image")
    parser.add_argument("--out", default=None, help="path to save clicked points as JSON")
    parser.add_argument("--corners", action="store_true",
                        help="click each cover's 4 corners in order around it instead of its middle")
    args = parser.parse_args()

    imagePath = args.image or default_image_path()
    outPath = args.out or default_out_path(imagePath)

    image = plt.imread(imagePath)
    clickSet = ClickSet(CORNERS_PER_COVER if args.corners else 1)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    cornerScatter = ax.scatter([], [], s=40, facecolors="none", edgecolors="red", linewidths=1.2)
    pendingScatter = ax.scatter([], [], s=40, facecolors="yellow", edgecolors="orange", linewidths=1.2)
    centerScatter = ax.scatter([], [], s=70, marker="x", color="lime", linewidths=1.5)
    artists = []

    def redraw():
        centers, badIndices = clickSet.centers()
        allClicks = [pt for g in clickSet.groups for pt in g] if clickSet.n > 1 else []
        cornerScatter.set_offsets(np.array(allClicks) if allClicks else np.empty((0, 2)))
        pendingScatter.set_offsets(np.array(clickSet.pending) if clickSet.pending else np.empty((0, 2)))
        centerScatter.set_offsets(np.array(centers) if centers else np.empty((0, 2)))
        if clickSet.n == 1:  # original look: red circles are the points themselves
            cornerScatter.set_offsets(np.array(centers) if centers else np.empty((0, 2)))
            centerScatter.set_offsets(np.empty((0, 2)))
        for artist in artists:
            artist.remove()
        artists.clear()
        for i, (x, y) in enumerate(centers):
            group = clickSet.groups[i]
            early = clickSet.n > 1 and len(group) < CORNERS_PER_COVER
            color = "red" if i in badIndices else ("magenta" if early else ("lime" if clickSet.n > 1 else "red"))
            artists.append(ax.annotate(str(i), (x, y), color=color, fontsize=8,
                                        xytext=(4, 4), textcoords="offset points"))
            if clickSet.n > 1 and not early:
                quad = np.array(group + [group[0]])
                artists.extend(ax.plot(quad[:, 0], quad[:, 1], "-", color=color, linewidth=0.8))
            elif early and len(group) == 2:
                line = np.array(group)
                artists.extend(ax.plot(line[:, 0], line[:, 1], "-", color=color, linewidth=0.8))
        if clickSet.n > 1:
            hint = (f"corner {len(clickSet.pending) + 1}/{clickSet.n} of cover {len(centers)} "
                    f"(Enter: finish with 1-2 clicks)")
        else:
            hint = "left click: add"
        ax.set_title(f"{len(centers)} point(s) -- {hint}, right click: remove nearest, u: undo, s: save")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            clickSet.add(event.xdata, event.ydata)
        elif event.button == 3:
            clickSet.remove_nearest(event.xdata, event.ydata)
        redraw()

    def on_key(event):
        if event.key == "enter":
            if not clickSet.finish_early():
                print("Enter only finishes a cover after 1 or 2 clicks (a cover with 3 needs a 4th)")
            redraw()
        elif event.key == "u":
            clickSet.undo()
            redraw()
        elif event.key == "s":
            save_points(clickSet, imagePath, outPath)

    def on_close(event):
        save_points(clickSet, imagePath, outPath)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    print(f"image: {imagePath}")
    print(f"will save to: {outPath}")
    what = "corner (4 per cover, any order; Enter after 1-2 clicks for a hard one)" if args.corners else "point"
    print(f"left click: add {what} | right click: remove nearest | u: undo | s: save | close window when done")
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
