# CameraPoseCalibrationFromSiPMs progress log (camera 2, 20251114_37 event 1 frame 16)

Each version = a full notebook snapshot here + the cached fits in `outputs/history/<version>/`.
Per-run figures/renders live in `outputs/poseSolve/<date>/`.

| | v0 hand-placed centers | v1 corner clicks, sorted |
|---|---|---|
| date | 2026-11-09 (fits) | 2026-11-16 |
| clicks | 33 single clicks (file overwritten, not kept) | 31 covers x 4 corners, diagonal center |
| rotation-only: total rotation | 8.29 deg | 8.49 deg |
| rotation-only: matched / median residual (predicted) | 24/33, 15.9 px | 23/31, 18.6 px |
| rotation-only: median, real render | 14.4 px | 17.9 px |
| FOV solve (assumed 86) | 98.53 deg, leave-one-out 98.33-98.94 | 99.37 deg, leave-one-out 98.63-99.47 |
| FOV solve: rotation total | 7.31 deg | 7.50 deg |
| FOV solve: matched, median predicted | 30, 7.6 px | 27/31, 6.9 px |
| FOV solve: median, real render | 9.3 px | 9.3 px |
| k1 on top of FOV | -0.091, leave-one-out -0.119..-0.0005 | -0.065, leave-one-out -0.075..+0.023 |
| render median, FOV only vs FOV + k1 | 10.0 vs 10.9 px | 9.8 vs 10.0 px |

Notes
- v0 notebook was saved with the lens section (FOV/k1) not run in it; the v0 FOV/k1 numbers come from the cached JSONs
  (`outputs/history/v0_hand-placed-centers/pose_calibration/`) and the 2026-11-09 run folder.
- v1: clicks #0, #1, #2 (image-edge covers) still land >30 px from any render marker.

## v2 (2026-11-16): match on all 4 cover corners (new notebook section, `lc.fit_pose_correction_covers`)

Same sorted clicks as v1; each clicked cover's 4 corners are matched to the CAD cover plate's 4 corners
(best of 8 alignments), fit with a shared cover-size factor. FOV solve (rotation + focal):

| | v1 midpoint match | v2 corner match |
|---|---|---|
| FOV | 99.37 deg (leave-one-out 98.63-99.47) | 98.73 deg (leave-one-cover-out 97.80-99.12) |
| rotation total | 7.50 deg | 7.41 deg |
| matched | 27/31 | 29/31 |
| center error, same 26 covers (median / max) | 6.9 / 18.6 px | 8.1 / 23.8 px |
| mean corner distance (median / max) | - | 8.8 / 39.0 px |
| fitted cover size | - | 1.50x the 1.4 cm CAD plate = ~2.1 cm |

- Clicked quads are ~1.5x the CAD plates (shape/orientation agree to ~3 px after rescaling): what's clicked is a larger feature.
- Worst corner matches: SiPM 46, 55, 44, 35 (left-edge, thin/edge-on covers). The 8 SiPMs without a CAD plate use center-only matching.

## v3 (2026-11-16): robust (Huber) weighting and small light markers on the cover corners

FOV solve (rotation + focal), camera 2, leave-one-cover-out stability:

| | v1/v2 midpoint match | v2 corner, plain | v3 corner, robust 10px | v3 corner, robust + rendered corners |
|---|---|---|---|---|
| FOV | 99.37 (98.63-99.47) | 98.73 (97.80-99.12) | 98.39 (97.96-98.48) | 98.55 (98.29-98.78) |
| rotation total | 7.50 deg | 7.41 deg | 6.44 deg | 6.77 deg |
| matched | 27/31 | 29/31 | 27/31 | 28/31 |
| median error (px) | 6.9 (centre) | 8.8 (corners) | 6.3 (corners) | 7.7 (corners) |

- Corner markers: 0.1 cm lights on each plate's 4 corners, one render per SiPM (`lc.render_corner_marker_positions`,
  cached in `outputs/pose_calibration/cam2_corner_render_positions.json`); 104/144 corners found (4 SiPMs found none).
- A corner's own refraction offset differs from its SiPM centre's by a median 4.0 px (90th pct 11.9, max 24.4).
- Using the measured offsets did not lower the fit error (6.3 -> 7.7 px); some predicted quads twist (a corner blob matched
  to the wrong marker or a missing corner falling back to the centre's offset).
- Huber down-weights SiPMs 44, 35, 72, 45, 73 (thin edge-on covers at the left).

## v4 (2026-11-16): filter the measured corner blobs (`lc.filter_corner_offsets`, 6 px)

A corner is dropped if its measured refraction offset is >6 px from the median of its own SiPM's corners (good SiPMs agree to ~3 px);
a SiPM with <3 trusted corners is dropped; dropped/missing corners take their SiPM's median offset (not the centre's).
Kept 57 of the 104 corners found. FOV solve, camera 2:

| | corner, robust (centre offsets) | rendered, raw (v3) | rendered, filtered (v4) |
|---|---|---|---|
| FOV | 98.39 (97.96-98.48) | 98.60 (98.31-98.83) | 98.40 (98.26-98.49) |
| rotation total | 6.44 deg | 6.78 deg | 6.56 deg |
| z-rotation leave-one-out spread | 0.85 deg | 1.06 deg | 0.30 deg |
| matched | 27/31 | 28/31 | 27/31 |
| median corner error | 6.3 px | 7.9 px | 6.5 px |

Threshold sweep (outlier distance): 3-6 px all give FOV 98.38-98.43, rotation 6.56-6.60, z spread ~0.3 deg; 8 px and up lose the stability.

## v5 (2026-11-16): SiPMs 35, 65, 72 pinned to full weight (`fullWeightSipms`)

Those three had mean corner errors of 24, 12, 16 px (weights 0.41, 0.80, 0.62 at the 10 px cutoff) but were judged fine by eye, so they keep weight 1.
Still down-weighted: 44 (0.32), 45 (0.72), 73 (0.76). FOV solve, camera 2, final fit (robust 10 px, filtered rendered corners):

| | v4 | v5 |
|---|---|---|
| FOV | 98.40 (98.26-98.49) | 98.54 (98.28-98.65) |
| rotation total | 6.56 deg | 6.66 deg |
| matched | 27/31 | 27/31 |
| median corner error | 6.5 px | 6.9 px |

Cutoff sweep on v4 for context (FOV / rotation): none 98.83 / 7.50; 30px 98.83 / 6.97; 20px 98.44 / 6.77; 15px 98.41 / 6.69; 10px 98.40 / 6.56; 6px 98.38 / 6.45.
