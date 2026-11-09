"""Interactive click-to-label tool: shows a background image and records where you
click, so you can build your own ground-truth point set (e.g. "where I see a SiPM
cover slot by eye") to compare against a detector's output instead of re-eyeballing
overlays every time.

Usage:
    ~/py-envs/sbc/bin/python scripts/click_label_points.py
    ~/py-envs/sbc/bin/python scripts/click_label_points.py --image path/to/image.png --out my_points.json

With no --image, defaults to the same real photo CameraPoseCalibrationFromSiPMs.ipynb
compares against (run 20251114_37, first single-bubble event, cam2 frame16).

Controls:
    left click   add a point
    right click  remove the point nearest the click (undo-by-location)
    u            undo the most recently added point
    s            save to --out without closing
    close window save to --out automatically

Output is JSON: {"image": "<path>", "points": [[x, y], ...]}. Load it back with
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


def save_points(points, image_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"image": image_path, "points": [list(p) for p in points]}, f, indent=1)
    print(f"saved {len(points)} point(s) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="path to the background image")
    parser.add_argument("--out", default=None, help="path to save clicked points as JSON")
    args = parser.parse_args()

    imagePath = args.image or default_image_path()
    outPath = args.out or default_out_path(imagePath)

    image = plt.imread(imagePath)
    points = []

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    scatter = ax.scatter([], [], s=60, facecolors="none", edgecolors="red", linewidths=1.5)
    labels = []

    def redraw():
        if points:
            scatter.set_offsets(np.array(points))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        for label in labels:
            label.remove()
        labels.clear()
        for i, (x, y) in enumerate(points):
            labels.append(ax.annotate(str(i), (x, y), color="red", fontsize=8,
                                       xytext=(4, 4), textcoords="offset points"))
        ax.set_title(f"{len(points)} point(s) -- left click: add, right click: remove nearest, "
                     f"u: undo, s: save")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            points.append((event.xdata, event.ydata))
        elif event.button == 3 and points:
            dists = [((x - event.xdata) ** 2 + (y - event.ydata) ** 2) for x, y in points]
            points.pop(int(np.argmin(dists)))
        redraw()

    def on_key(event):
        if event.key == "u" and points:
            points.pop()
            redraw()
        elif event.key == "s":
            save_points(points, imagePath, outPath)

    def on_close(event):
        save_points(points, imagePath, outPath)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    print(f"image: {imagePath}")
    print(f"will save to: {outPath}")
    print("left click: add point | right click: remove nearest | u: undo | s: save | close window when done")
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
