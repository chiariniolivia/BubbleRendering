"""Locates this repo's real background-run data (camera images, handscan
labels, and precomputed analysis outputs) for camera pose calibration.

Everything under ~/coop is external, shared data — this module only reads
it, never writes/modifies it.
"""
import os

import numpy as np
from sbcbinaryformat import Streamer

COOP_DIR = os.path.expanduser("~/coop")

RUN_DIRS = {
    "20251114_37": os.path.join(COOP_DIR, "local-runs", "20251114_37"),
    "20251115_0": os.path.join(COOP_DIR, "local-runs", "20251115_0"),
}

# scan_source code -> label, matching the handscan column of the same name
SOURCE_LABELS = {0: "bulk", 1: "wall", 2: "dome", 3: "bellows", 4: "other"}


def event_dir(run_dir, event_number):
    """Path to a single event's directory, e.g. RUN_DIRS["20251114_37"]/0."""
    return os.path.join(run_dir, str(event_number))


def camera_image_path(run_dir, event_number, camera, frame):
    """Path to one camN-imgNN.png inside an event directory."""
    return os.path.join(event_dir(run_dir, event_number), f"cam{camera}-img{frame:02d}.png")


def parse_handscan_row(line):
    """One handscan.txt line -> a dict, or None for header/blank lines.

    Row format: run, ev, scanner, scan_source, scan_nbub, scan_trigger,
    scan_crosshairsgood, scan_comment (quoted, optional).
    """
    parts = line.split(None, 7)
    if len(parts) < 7:
        return None
    try:
        ev, src, nbub, trig, cross = (int(parts[i]) for i in (1, 3, 4, 5, 6))
    except ValueError:
        return None

    comment = parts[7].strip() if len(parts) == 8 else ""
    if len(comment) >= 2 and comment[0] == comment[-1] == "'":
        comment = comment[1:-1]

    return {
        "run": parts[0],
        "ev": ev,
        "scanner": parts[2],
        "scan_source": src,
        "scan_source_label": SOURCE_LABELS.get(src, f"unknown_{src}"),
        "scan_nbub": nbub,
        "scan_trigger": trig,
        "scan_crosshairsgood": cross,
        "scan_comment": comment,
    }


def load_handscan(run_dir):
    """Parse a run's handscan.txt into a list of dicts (one per scanned event)."""
    path = os.path.join(run_dir, "handscan.txt")
    rows = []
    with open(path) as f:
        for line in f:
            row = parse_handscan_row(line)
            if row is not None:
                rows.append(row)
    return rows


def load_analysis(run_dir, name):
    """Load one of a run's precomputed analysis/<name>.sbc outputs as a dict
    of arrays (e.g. name="reco" for 3D bubble positions, "exposure" for
    livetimes, "event" for pset info, "bubble"/"clustering" for per-camera
    bubble detections)."""
    path = os.path.join(run_dir, "analysis", f"{name}.sbc")
    return Streamer(path).to_dict()


def single_bubble_events(run_dir, require_good_crosshairs=True):
    """Event numbers the handscanner confirmed as single-bubble (scan_nbub == 1),
    optionally restricted to ones where they also marked the crosshairs good
    (scan_crosshairsgood == 1) -- multi-bubble events need cross-camera bubble
    correspondence to know which detection in each image is the same bubble,
    which this loader doesn't attempt; single-bubble events skip that problem
    entirely since there's only one detection per camera to begin with."""
    events = []
    for row in load_handscan(run_dir):
        if row["scan_nbub"] != 1:
            continue
        if require_good_crosshairs and not row["scan_crosshairsgood"]:
            continue
        events.append(row["ev"])
    return sorted(events)


def load_single_bubble_track(run_dir, ev):
    """Per-camera, per-frame pixel track for a single-bubble event, from
    analysis/bubble_single.sbc -- the bubble finder run under the assumption
    that there is exactly one bubble, rather than the general-purpose finder
    behind bubble.sbc/clustering.sbc. Frame numbers here line up directly with
    load_reco_single()'s, since that's the reconstruction triangulated from
    these same per-frame positions.

    Returns (tracks, t0_info):
      tracks: {camera_number: [{frame, pos: (x, y), rad, confidence}, ...]},
        sorted by frame within each camera. A camera with no entry saw
        nothing for this event.
      t0_info: {"frame": t0_frame, "cams": (cam_a, cam_b)}, the frame and
        camera pair the pipeline used to fix the bubble's nucleation time
        (same for every row in the event), or None if the event has no rows.
    """
    d = load_analysis(run_dir, "bubble_single")
    mask = d["ev"] == ev
    tracks = {}
    for cam, frame, pos, rad, confidence in zip(
        d["cam"][mask], d["frame"][mask], d["pos"][mask], d["rad"][mask], d["confidence"][mask],
    ):
        tracks.setdefault(int(cam), []).append({
            "frame": int(frame),
            "pos": (float(pos[0]), float(pos[1])),
            "rad": float(rad),
            "confidence": float(confidence),
        })
    for cam_track in tracks.values():
        cam_track.sort(key=lambda row: row["frame"])

    t0_frame, t0_cams = d["t0_frame"][mask], d["t0_cams"][mask]
    t0_info = {"frame": int(t0_frame[0]), "cams": tuple(int(c) for c in t0_cams[0])} \
        if len(t0_frame) else None
    return tracks, t0_info


# Coordinates this far out (mm) can't be a real bubble position -- the chamber's own wall
# radius is ~115mm and its cylindrical depth ~222mm, so this leaves a wide margin around the
# physical volume while still catching every failure mode seen in reco_single.sbc: the explicit
# (-999,-999,-999)/(-1000,-1000,-1000) sentinels *and* near-parallel-ray triangulation blowups
# (coordinates in the thousands of mm) that still report a tiny, misleadingly-good reprojError
PHYSICALLY_PLAUSIBLE_COORD_BOUND_MM = 500.0


def load_reco_single(run_dir, ev):
    """Per-frame triangulated 3D bubble position for a single-bubble event,
    from analysis/reco_single.sbc -- the reconstruction run under the same
    single-bubble assumption load_single_bubble_track() uses, so (unlike
    reco.sbc) there's no ambiguity about which per-camera detection maps to
    which 3D point.

    Returns a list of {frame, coord: (x, y, z) or None, reproj_error or None},
    one entry per frame, sorted by frame; coord/reproj_error are None for a
    frame where the reconstruction wasn't valid (no triangulation, one of the
    sentinel failure values, or a physically implausible result -- see
    PHYSICALLY_PLAUSIBLE_COORD_BOUND_MM).
    """
    d = load_analysis(run_dir, "reco_single")
    mask = d["ev"] == ev
    frames, coords, reproj_errors = d["frame"][mask], d["coords_3D"][mask], d["reprojError"][mask]
    rows = []
    for i in np.argsort(frames):
        coord = coords[i]
        valid = (
            not np.isnan(coord).any()
            and np.abs(coord).max() <= PHYSICALLY_PLAUSIBLE_COORD_BOUND_MM
        )
        rows.append({
            "frame": int(frames[i]),
            "coord": tuple(coord.tolist()) if valid else None,
            "reproj_error": float(reproj_errors[i]) if valid else None,
        })
    return rows
