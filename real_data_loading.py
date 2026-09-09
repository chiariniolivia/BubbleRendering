"""Loads the external LAr10Ana analysis modules and locates this repo's real
background-run data, so notebooks can use real camera images/handscan data for
camera pose calibration.

Everything under ~/coop is external, shared data — this module only reads it,
never writes/modifies it.
"""
import os
import sys

COOP_DIR = os.path.expanduser("~/coop")
LAr10ANA_DIR = os.path.join(COOP_DIR, "LAr10Ana")

RUN_DIRS = {
    "20251114_37": os.path.join(COOP_DIR, "local-runs", "20251114_37"),
    "20251115_0": os.path.join(COOP_DIR, "local-runs", "20251115_0"),
}

if LAr10ANA_DIR not in sys.path:
    sys.path.insert(0, LAr10ANA_DIR)

from ana import EventAnalysis, RunAnalysis, ExposureAnalysis
from ana.BubbleValidator import combine_handscans

try:
    from GetEvent import GetEvent, NEvent
except ImportError as e:
    # GetEvent depends on sbcbinaryformat, which isn't part of this repo's
    # environment; the rest of this module still works without it.
    GetEvent = None
    NEvent = None
    _get_event_import_error = e


def event_dir(run_dir, event_number):
    """Path to a single event's directory, e.g. RUN_DIRS["20251114_37"]/0."""
    return os.path.join(run_dir, str(event_number))


def camera_image_path(run_dir, event_number, camera, frame):
    """Path to one camN-imgNN.png inside an event directory."""
    return os.path.join(event_dir(run_dir, event_number), f"cam{camera}-img{frame:02d}.png")


def load_handscan(run_dir):
    """Parse a run's handscan.txt into a list of dicts (one per scanned event).

    Row format matches combine_handscans.parse_row: run, ev, scanner,
    scan_source, scan_nbub, scan_trigger, scan_crosshairsgood, scan_comment.
    """
    path = os.path.join(run_dir, "handscan.txt")
    rows = []
    with open(path) as f:
        for line in f:
            row = combine_handscans.parse_row(line)
            if row is not None:
                rows.append(row)
    return rows
