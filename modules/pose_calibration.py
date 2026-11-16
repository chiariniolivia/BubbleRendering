"""Calibrates a camera's pose against known real-world reference points in the chamber:
the SiPM mounts' fiducial-marker tabs and the SiPM detectors themselves, and (eventually)
the LED ring lights.
"""
import csv
import json

import numpy as np

from utils import *
import real_data_loading as rdl

SIPM_FIDUCIAL_POSITIONS_CSV = "chamber_model/sipm_fiducial_positions.csv"


def load_sipm_fiducial_positions(csvPath=SIPM_FIDUCIAL_POSITIONS_CSV):
    """Loads chamber_model/sipm_fiducial_positions.csv -- SiPM and fiducial-marker
    positions straight off the chamber's mechanical drawings (inches; Z=0 at the top of
    the SiPM panel barrel, +Z towards the cameras, -Z towards the piston; X=Y=0 on the
    vessel's center axis) -- and converts each into this repo's render coordinate frame
    (cm).

    The drawings' frame isn't the render frame, but it turns out to be the same real-world
    frame real_data_loading.real_position_to_render_cm() already converts from (reco'd
    bubble positions, in LAr10Ana's own frame): feeding this CSV's 5 fiducial rows through
    it lands within ~0.01mm of fiducials.ply's own mesh centroids, i.e. this scene's
    existing, already-placed fiducial geometry -- checked, not assumed. So the same
    function is reused here rather than re-deriving another rotation/offset.

    Returns {"fiducial": {id: (x, y, z) cm}, "sipm": {id: (x, y, z) cm}}, id taken
    verbatim from the CSV's id column (e.g. "1".."5" for fiducials, "11".."86" for
    SiPMs -- ring number then position-within-ring)."""
    positions = {"fiducial": {}, "sipm": {}}
    with open(csvPath, newline="") as csvFile:
        for row in csv.DictReader(csvFile):
            coordIn = np.array([float(row["x_in"]), float(row["y_in"]), float(row["z_in"])])
            positions[row["type"]][row["id"]] = rdl.real_position_to_render_cm(coordIn * 25.4)
    return positions


def add_point_light_markers(components, positionsCm, radiusCm=0.3, radiance=80.0):
    """Adds one small emissive sphere per position in positionsCm (a {id: (x, y, z)}
    dict, e.g. one of load_sipm_fiducial_positions()'s or load_assumed_positions()'s
    sub-dicts) to components, named 'light_marker_<id>'. Mutates and returns components,
    so it chains with load_components() the same way RealDataRenderComparison.ipynb's own
    bubble marker does.

    Unlike project_positions_to_pixels() (a flat 2D projection of a point), rendering an
    actual light source at a real-world position shows what a camera would really see
    there -- whether it's directly visible, occluded by the copper panel, or (like the
    LED rings) only visible via reflection off the curved dome -- since the renderer
    traces real light paths through this scene's actual geometry rather than assuming a
    straight line to the camera."""
    for posId, position in positionsCm.items():
        components[f"light_marker_{posId}"] = mi.load_dict({
            "type": "sphere",
            "focused-emitter": {
                "type": "area",
                "radiance": {"type": "spectrum", "value": radiance},
            },
            "to_world": mi.ScalarTransform4f.translate(np.asarray(position)).scale(float(radiusCm)),
        })
    return components


def project_positions_to_pixels(worldPositionsCm, camera):
    """Projects each render-space cm position in worldPositionsCm (a {id: (x, y, z)}
    dict, e.g. one of load_sipm_fiducial_positions()'s two sub-dicts) onto camera N's
    *assumed* pose -- create_sensor(sensor_number=camera)'s own to_world, the same one
    every render in this repo uses. Returns {id: (px, py) pixel coords}.

    Useful for overlaying a known real-world position on a render of the assumed pose:
    wherever the overlaid marker lands relative to that position's actual rendered
    feature (if visible) is a direct visual read on how far off the assumed pose/
    geometry is from reality, at that point."""
    pose = np.array(create_sensor(sensor_number=camera)["to_world"].matrix)[:3, :]
    invPose = compute_inverse_pose_matrices(pose[None])[0]
    return {
        posId: normalized_to_pixel(reproject(invPose, np.asarray(worldPos)))
        for posId, worldPos in worldPositionsCm.items()
    }


def detect_light_markers(image, baseline=None, brightnessThreshold=None, minArea=3, maxArea=300,
                          minSize=1, maxSize=40):
    """Finds the bright blobs left by add_point_light_markers() in a render of its own
    output (e.g. CameraPoseCalibrationFromSiPMs.ipynb's sipmLightImage). Returns an
    (n, 2) array of pixel centroids, one row per candidate blob -- unlabeled (no attempt
    to say *which* SiPM each is; match against project_positions_to_pixels()'s expected
    pixels for that).

    Unlike detect_sipm_covers()/detect_viewport_circle() (built for an 8-bit real photo),
    this thresholds the render's own *raw linear* pixel values, no local background
    subtraction -- but a plain absolute threshold on image alone turned out to badly
    overcount: a single-marker test render crossed a threshold of 1.0 about 38 times,
    almost all from the chamber's own reflections/specular highlights (e.g. off the jar)
    rather than the marker, not from Monte-Carlo noise as first assumed.

    Pass baseline -- a render of the *same* scene/sensor with no light markers added (no
    add_point_light_markers() call) -- to threshold image - baseline instead: any surface
    already bright enough to cross the threshold on its own cancels out, leaving only
    what the marker(s) actually added. Checked with all 48 SiPM markers lit at once
    (radiance=80 each, the add_point_light_markers() default): the diff's own pixel
    distribution has a clean gap between its 99th percentile (~1.1, Monte-Carlo noise and
    the markers' own modest contribution to ambient/bounced light across the whole scene)
    and 99.9th percentile (~46, direct marker hits) -- so brightnessThreshold defaults to
    10.0 with a baseline (down in that gap) and 1.0 without one (raw-image ceiling, see
    above). At 10.0 this found 38 blobs for that 48-marker render, tracing the two visible
    SiPM-ring arcs cleanly -- checked by eye, not just by count. minArea/minSize still
    reject single-pixel Monte-Carlo noise spikes ("fireflies")."""
    if brightnessThreshold is None:
        brightnessThreshold = 10.0 if baseline is not None else 1.0
    signal = image if baseline is None else image - baseline
    mask = (signal >= brightnessThreshold).astype(np.uint8)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    longSide = np.maximum(widths, heights)
    shortSide = np.maximum(np.minimum(widths, heights), 1)

    keep = np.where(
        (areas >= minArea) & (areas <= maxArea)
        & (shortSide >= minSize) & (longSide <= maxSize)
    )[0]
    keep = keep[keep != 0]  # label 0 is the background
    return centroids[keep]


def load_clicked_points(path):
    """Loads a scripts/click_label_points.py output file. Returns an (n, 2) array of
    (x, y) pixel positions -- same shape detect_sipm_covers()/detect_light_markers()
    return, so e.g. nearest-neighbor distance against a detector's own output is a
    direct drop-in: np.linalg.norm(detected[:, None] - clicked[None, :], axis=-1)."""
    with open(path) as f:
        return np.array(json.load(f)["points"])


def load_clicked_corners(path):
    """Loads the raw clicks from a `click_label_points.py --corners` output file: a list
    with one (k, 2) array per cover -- k = 4 corners (sorted in order around the cover),
    or k = 1 or 2 for a cover that was finished early (its center, or two opposite
    corners). load_clicked_points() on the same file gives each cover's center, in the
    same order. Raises KeyError for a file saved without --corners."""
    with open(path) as f:
        return [np.array(group) for group in json.load(f)["corners"]]


POSE_SOLVE_OUTPUT_DIR = "outputs/poseSolve"


def new_run_dir(baseDir=POSE_SOLVE_OUTPUT_DIR):
    """A fresh outputs/poseSolve/<date ran>/ folder (YYYY_MM_DD_HH_MM_SS) for one
    notebook run's figures and renders. Only the path -- folders are made on first save."""
    return pathlib.Path(baseDir) / time.strftime("%Y_%m_%d_%H_%M_%S")


def save_figure(fig, name, runDir):
    """Saves a matplotlib figure to <runDir>/figures/<name>.png."""
    figPath = pathlib.Path(runDir) / "figures" / f"{name}.png"
    figPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figPath, dpi=150, bbox_inches="tight")
    print("saved figure:", figPath)


def save_render(image, name, runDir, isDiff=False):
    """Saves a mono render to <runDir>/renders/<name>.png as an 8-bit PNG: gamma 2.2 for
    a plain render, or scaled to its own max for a diff image (so a faint marker still
    shows). 8-bit and gamma'd, so for looking at -- not for re-running detection on."""
    renderPath = pathlib.Path(runDir) / "renders" / f"{name}.png"
    renderPath.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=float)
    if isDiff:
        image = np.clip(image / max(image.max(), 1e-9), 0, 1)
    else:
        image = np.clip(image, 0, 1) ** (1 / 2.2)
    plt.imsave(renderPath, image, cmap="gray", vmin=0, vmax=1)
    print("saved render:", renderPath)


def perturb_pose(toWorld, poseCorrection):
    """Applies a 6-vector pose correction [rx, ry, rz, tx, ty, tz] to a 4x4 camera
    to_world matrix. The rotation (Rodrigues vector, radians) and translation (cm) are
    in the camera's *own* local frame -- so a pure rotation pivots the camera in place
    about its own center -- the same convention fit_pose_correction() solves in.
    Returns the corrected 4x4 to_world as a numpy array (pass it to
    create_sensor(to_world=mi.ScalarTransform4f(...)) to render from it)."""
    rotation, _ = cv2.Rodrigues(np.asarray(poseCorrection[:3], dtype=float))
    correction = np.eye(4)
    correction[:3, :3] = rotation
    correction[:3, 3] = poseCorrection[3:6]  # anything past [5] is lens distortion, not pose
    return np.asarray(toWorld) @ correction


def flat_project(toWorld, worldPositionsCm):
    """Array version of project_positions_to_pixels() for an arbitrary 4x4 to_world:
    (n, 3) render-space cm positions -> (n, 2) pixels, straight-line (no refraction)."""
    invPose = np.linalg.inv(np.asarray(toWorld))[:3]
    return np.array([normalized_to_pixel(reproject(invPose, np.asarray(pos))) for pos in worldPositionsCm])


def match_points_one_to_one(predictedPixels, observedPixels, maxDistancePx):
    """Greedy one-to-one nearest-neighbor matching: repeatedly pairs the closest
    remaining (predicted, observed) pair until the next-closest is farther than
    maxDistancePx. Unlike a plain per-point nearest neighbor, two observed points can't
    both claim the same prediction (which plain NN did a lot in
    CameraPoseCalibrationFromSiPMs.ipynb -- e.g. three clicks -> one SiPM).
    Returns a list of (predictedIndex, observedIndex) pairs."""
    distances = np.linalg.norm(predictedPixels[:, None] - observedPixels[None], axis=-1)
    return match_from_distances(distances, maxDistancePx)


def match_from_distances(distances, maxDistancePx):
    """The greedy one-to-one step of match_points_one_to_one(), on an already-computed
    (predicted, observed) distance matrix -- so the distance can be anything, e.g. the
    best-corner-alignment cost of match_covers()."""
    pairs, usedPredicted, usedObserved = [], set(), set()
    for flatIndex in np.argsort(distances, axis=None):
        predIdx, obsIdx = np.unravel_index(flatIndex, distances.shape)
        if distances[predIdx, obsIdx] > maxDistancePx:
            break
        if predIdx in usedPredicted or obsIdx in usedObserved:
            continue
        pairs.append((int(predIdx), int(obsIdx)))
        usedPredicted.add(predIdx)
        usedObserved.add(obsIdx)
    return pairs


def apply_radial_distortion(pixels, k1, k2=0.0, focalScale=1.0, cameraMatrix=None):
    """Applies Brown-Conrady radial lens distortion to (n, 2) pinhole pixel positions:
    the pixel at normalized offset xy from the image center moves to
    xy * (1 + k1 r^2 + k2 r^4), r^2 = |xy|^2. Same model and sign convention as OpenCV's
    distCoeffs [k1, k2, 0, 0, 0] -- k1 < 0 is barrel (edges pulled inward), the usual
    case for a wide lens. cameraMatrix defaults to the ideal fov=86 pinhole every render
    in this repo uses, so pixels straight from a render or from flat_project() fit as-is.

    focalScale zooms the result about the image center -- exactly what a different fov
    (focal length) does to a pinhole image, refraction included, since the refraction
    happens in the scene, not the sensor. 1.0 = the fov=86 every render uses; >1 means
    the real lens has a longer focal length (narrower fov) than assumed."""
    if cameraMatrix is None:
        cameraMatrix = get_ideal_camera_matrix(fov=86)
    focal, cx, cy = cameraMatrix[0, 0], cameraMatrix[0, 2], cameraMatrix[1, 2]
    xy = (np.asarray(pixels, dtype=float) - [cx, cy]) / focal
    r2 = (xy ** 2).sum(axis=1, keepdims=True)
    return np.array([cx, cy]) + focalScale * focal * xy * (1 + k1 * r2 + k2 * r2 ** 2)


def distort_render(image, k1, k2=0.0, focalScale=1.0, cameraMatrix=None):
    """Warps a (pinhole, undistorted) render so it matches what a real lens with radial
    distortion k1/k2 would see -- Mitsuba 3 has no distorting camera, so this is how a
    render gets compared against a raw real photo. Uses cv2.remap: for every output
    (distorted) pixel, cv2.undistortPoints finds which render pixel lands there.
    With barrel distortion (k1 < 0) the corners of the output come from outside the
    render's own frame and stay black; fine here, since the real photos' corners are
    black outside the viewport anyway. Pixel for pixel this matches
    apply_radial_distortion() with the same k1/k2/focalScale, as long as the model
    stays monotonic out to the image edge (see distortion_is_monotonic())."""
    if cameraMatrix is None:
        cameraMatrix = get_ideal_camera_matrix(fov=86)
    height, width = image.shape[:2]
    gridX, gridY = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    distortedPixels = np.stack([gridX.ravel(), gridY.ravel()], axis=1)[:, None, :]
    distCoeffs = np.array([k1, k2, 0.0, 0.0, 0.0])
    outputMatrix = cameraMatrix.copy()
    outputMatrix[0, 0] *= focalScale
    outputMatrix[1, 1] *= focalScale
    sourcePixels = cv2.undistortPoints(distortedPixels, outputMatrix, distCoeffs, P=cameraMatrix)
    mapX = sourcePixels[:, 0, 0].reshape(height, width).astype(np.float32)
    mapY = sourcePixels[:, 0, 1].reshape(height, width).astype(np.float32)
    return cv2.remap(np.asarray(image, dtype=np.float32), mapX, mapY, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def distortion_is_monotonic(k1, k2=0.0, imageWidth=1280, imageHeight=800, fov=86):
    # fov: the lens's actual (e.g. solved) fov, which sets how far out the corner is.
    """True if the radial model keeps pushing points outward all the way to the image
    corner, i.e. d/dr [r (1 + k1 r^2 + k2 r^4)] > 0 there. A strong enough barrel k1
    makes it fold back on itself (two different scene points land on one pixel), which
    no real lens does -- a fitted k1 that fails this is soaking up some other error."""
    cameraMatrix = get_ideal_camera_matrix(fov, image_width=imageWidth, image_height=imageHeight)
    rMax = np.hypot(imageWidth / 2, imageHeight / 2) / cameraMatrix[0, 0]
    r = np.linspace(0, rMax, 200)
    return bool(np.all(1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 > 0))


# freeParams presets for fit_pose_correction(): which of
# [rx, ry, rz, tx, ty, tz, k1, k2, focalScale] get solved for.
ROTATION_ONLY = (True, True, True, False, False, False, False, False, False)
ROTATION_AND_K1 = (True, True, True, False, False, False, True, False, False)
ROTATION_AND_FOCAL = (True, True, True, False, False, False, False, False, True)
ROTATION_FOCAL_AND_K1 = (True, True, True, False, False, False, True, False, True)


def fit_pose_correction(worldPositionsCm, renderPixels, observedPixels, toWorld,
                        cutoffSchedulePx=(130, 90, 60, 40), freeParams=ROTATION_ONLY,
                        gaussNewtonIters=15, initialCorrection=None):
    """Solves for the pose correction (see perturb_pose()) that moves each known 3D point
    from where a render at toWorld put it (renderPixels) onto where it was actually seen
    in a real photo (observedPixels, unlabeled -- e.g. load_clicked_points()).

    Refraction: a flat pinhole projection is 15-70px off from where a render actually
    puts a point (jar + fluid refraction), so it can't be compared against a real photo
    directly. Instead each point's refraction offset is measured once from the render,
    delta = renderPixels - flat_project(toWorld), and assumed to stay fixed for small
    pose changes: predicted(correction) = flat_project(perturb_pose(...)) + delta.
    Checked on cam2 by re-rendering at the solved pose and re-measuring delta: the
    follow-up correction was <0.6deg and didn't change the residual, so one pass is
    enough.

    Correspondence (which observed point is which id) is solved jointly with the pose,
    ICP-style: match one-to-one within a cutoff, fit, then shrink the cutoff, per
    cutoffSchedulePx.

    Lens distortion: the real camera's lens distorts, renders don't, so the prediction
    is finally passed through apply_radial_distortion(k1, k2, focalScale) -- params
    [6], [7] and [8] (stored as focalScale - 1, so all-zero = the assumed camera).

    freeParams picks which of [rx, ry, rz, tx, ty, tz, k1, k2, focalScale] are solved
    for (a 6-long tuple means pose only). Parameters not freed stay at their
    initialCorrection value (default all zeros = the assumed camera) -- e.g. pass a
    previous fit's "correction" to hold its solved focalScale fixed while fitting k1. Defaults to
    rotation only: on cam2, also freeing translation fit the clicks better on paper
    (6px vs. 16px median), but with a 2-3cm camera shift, which a validation render
    showed was wrong -- it moved the camera into the viewport geometry and only 14 of
    the 36 SiPMs stayed visible. The flat model was absorbing what's more likely
    unmodeled lens distortion or fov error into a fake translation.

    worldPositionsCm, renderPixels: (n, 3) / (n, 2), same order. Returns a dict with
    "correction" (9,), "toWorld" (4x4 corrected), "distortion" {"k1", "k2",
    "focalScale"} (pass straight to distort_render(**...)), "pairs" [(pointIndex, observedIndex)]
    at the final cutoff, "residualsPx" (per pair), and "deltaPx" (the (n, 2) refraction
    offsets)."""
    worldPositionsCm = np.asarray(worldPositionsCm, dtype=float)
    observedPixels = np.asarray(observedPixels, dtype=float)
    freeMask = np.zeros(9, dtype=bool)
    freeMask[:len(freeParams)] = freeParams
    deltaPx = np.asarray(renderPixels, dtype=float) - flat_project(toWorld, worldPositionsCm)

    def predict(correction):
        renderPx = flat_project(perturb_pose(toWorld, correction), worldPositionsCm) + deltaPx
        return apply_radial_distortion(renderPx, correction[6], correction[7], 1 + correction[8])

    correction = np.zeros(9) if initialCorrection is None else np.array(initialCorrection, dtype=float)
    for cutoffPx in cutoffSchedulePx:
        pairs = match_points_one_to_one(predict(correction), observedPixels, cutoffPx)
        if len(pairs) < 3:
            raise RuntimeError(f"only {len(pairs)} matches within {cutoffPx}px, can't fit a pose")
        pointIdx = np.array([p for p, _ in pairs])
        obsIdx = np.array([o for _, o in pairs])
        for _ in range(gaussNewtonIters):
            residual = (predict(correction)[pointIdx] - observedPixels[obsIdx]).ravel()
            jacobian = np.zeros((len(residual), freeMask.sum()))
            for col, param in enumerate(np.flatnonzero(freeMask)):
                stepped = correction.copy()
                stepped[param] += 1e-5
                jacobian[:, col] = ((predict(stepped)[pointIdx] - observedPixels[obsIdx]).ravel() - residual) / 1e-5
            correction[freeMask] += np.linalg.lstsq(jacobian, -residual, rcond=None)[0]

    pairs = match_points_one_to_one(predict(correction), observedPixels, cutoffSchedulePx[-1])
    residualsPx = np.array([np.linalg.norm(predict(correction)[p] - observedPixels[o]) for p, o in pairs])
    return {
        "correction": correction,
        "toWorld": perturb_pose(toWorld, correction),
        "distortion": {"k1": float(correction[6]), "k2": float(correction[7]),
                       "focalScale": float(1 + correction[8])},
        "pairs": pairs,
        "residualsPx": residualsPx,
        "deltaPx": deltaPx,
    }


COVER_CORNER_PERMUTATIONS = [
    list(np.roll(order, shift))
    for order in ([0, 1, 2, 3], [0, 3, 2, 1]) for shift in range(4)
]  # 4 rotations x 2 directions: a square cover has no preferred first corner


def load_cover_corners(covers_ply="chamber_model/components/sipm_covers.ply"):
    """3D corners (render-space cm) of every SiPM cover plate the CAD models, from
    sipm_covers.ply: {id: (4, 3) array, in order around the plate}, id taken from the
    nearest load_sipm_fiducial_positions() SiPM. The plates are 1.4 x 1.4 cm squares
    (corners slightly chamfered), 40 of them -- the first SiPM of each of the 8 rings
    has no cover mesh, same gap as sipm_detectors.ply (see load_assumed_positions()),
    so those 8 ids are simply absent from the result.

    Each plate's 4 corners are the minimum-area rectangle around its vertices in the
    plate's own plane (so the chamfers don't make 8 corners)."""
    mesh = mi.load_dict({"type": "ply", "filename": covers_ply})
    params = mi.traverse(mesh)
    vertices = np.array(params["vertex_positions"]).reshape(-1, 3)
    faces = np.array(params["faces"]).reshape(-1, 3)
    parent = np.arange(len(vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(i) for i in range(len(vertices))])

    positions = load_sipm_fiducial_positions()["sipm"]
    ids = list(positions)
    centers = np.array([positions[i] for i in ids])
    corners = {}
    for root in np.unique(roots):
        pts = vertices[roots == root]
        center = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - center)
        normal = vt[2]
        up = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        if np.linalg.norm(up) < 1e-6:
            up = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        up /= np.linalg.norm(up)
        side = np.cross(normal, up)
        planar = np.stack([(pts - center) @ up, (pts - center) @ side], axis=1)
        box = cv2.boxPoints(cv2.minAreaRect(planar.astype(np.float32)))
        sid = ids[int(np.linalg.norm(centers - center, axis=1).argmin())]
        corners[sid] = center + box[:, :1] * up + box[:, 1:] * side
    return corners


def match_covers(predictedCenters, predictedCorners, observedCovers, maxDistancePx):
    """One-to-one matching of clicked covers to SiPMs by *corner alignment*.
    predictedCenters (n, 2); predictedCorners (n, 4, 2), NaN for a SiPM with no modeled
    cover; observedCovers a list of (k, 2) arrays (k = 4 clicked corners, or the 1-2
    clicks of a cover finished early).

    Cost of a (SiPM, cover) pair, in px: for 4 corners vs. 4 corners, the mean corner
    distance under the best of the 8 ways to line the corners up (a square cover has no
    preferred first corner); otherwise just the distance between the two centers.
    Returns (pairs [(sipmIndex, coverIndex)], cornerOrders {pair: 4-long index list or
    None}) -- cornerOrders[(i, j)][c] is the predicted corner that observed corner c
    was matched to."""
    n, m = len(predictedCenters), len(observedCovers)
    cost = np.full((n, m), np.inf)
    bestOrder = {}
    for j, obs in enumerate(observedCovers):
        obs = np.asarray(obs, dtype=float)
        for i in range(n):
            if len(obs) == 4 and not np.isnan(predictedCorners[i]).any():
                costs = [np.linalg.norm(predictedCorners[i][order] - obs, axis=1).mean()
                         for order in COVER_CORNER_PERMUTATIONS]
                k = int(np.argmin(costs))
                cost[i, j] = costs[k]
                bestOrder[(i, j)] = COVER_CORNER_PERMUTATIONS[k]
            else:
                cost[i, j] = np.linalg.norm(predictedCenters[i] - obs.mean(axis=0))
    pairs = match_from_distances(cost, maxDistancePx)
    return pairs, {pair: bestOrder.get(pair) for pair in pairs}


def filter_corner_offsets(offsets, outlierPx=6.0, minCorners=3):
    """Cleans up the corner refraction offsets measured by render_corner_marker_positions().
    offsets (n, 4, 2): each corner's measured offset minus its SiPM center's, NaN where no
    blob was found. Across one small plate those offsets should agree to a few px (on cam2 the
    good SiPMs' corners sit within ~3 px of each other), so a corner is dropped when it's more
    than outlierPx from the median of its own SiPM's corners (a blob matched to the wrong
    marker, or a reflection), and a SiPM left with fewer than minCorners trusted corners is
    dropped entirely. Dropped / missing corners then take the median offset of their SiPM's
    remaining corners -- not the SiPM center's own offset, which would twist the plate when its
    neighbours carry a measured one -- and a SiPM with no trusted corner gets 0 (center's offset).
    outlierPx=None skips the outlier test (missing corners are still filled in). Returns
    (filled (n, 4, 2) with no NaN, trusted (n, 4) bool -- which corners are real measurements
    that passed)."""
    offsets = np.array(offsets, dtype=float)
    out = np.zeros_like(offsets)
    trusted = np.zeros(offsets.shape[:2], dtype=bool)
    for i, row in enumerate(offsets):
        valid = ~np.isnan(row[:, 0])
        if outlierPx is not None and valid.sum() >= minCorners:
            median = np.median(row[valid], axis=0)
            valid &= np.linalg.norm(row - median, axis=1) <= outlierPx
        if valid.sum() < (minCorners if outlierPx is not None else 1):
            continue
        out[i] = np.where(valid[:, None], row, np.median(row[valid], axis=0))
        trusted[i] = valid
    return out, trusted


def render_corner_marker_positions(camera, sipmIds, centersCm, cornersCm, centerRenderPixels,
                                  coverScale=1.0, radiusCm=0.1, radiance=2000.0, sampleCount=64,
                                  baselineSamples=128, searchRadiusPx=25, minDiff=100.0, progress=print):
    """Where a real render puts small light markers at each SiPM cover's 4 corners.

    For every SiPM: puts a tiny emissive sphere (radiusCm -- smaller than the 0.3 cm markers used
    for the SiPM centers, so each corner is a tighter blob) on each of its cover plate's 4 corners
    (cornersCm[i], (4, 3); optionally scaled about the plate center by coverScale), renders
    camera N at its assumed pose, and diffs against one no-marker baseline. Rendering the corners
    (instead of reusing the center's refraction offset for them) measures the refraction across
    the cover directly. Markers placed 1.5x out from the plate (where the clicked corners are) were
    mostly hidden by the mount around the plate, so the default is the plate's own corners, where
    radiance 2000 gives blob peaks of ~500-1100 against noise under 100 (minDiff=100 keeps only
    those); fit_pose_correction_covers() extends the measured offset out to the clicked size.

    Blobs in the diff are matched to the 4 corners within searchRadiusPx of where the flat
    projection plus the center's own refraction offset (centerRenderPixels[i], the SiPM's cached
    render position) says each should be; each position is an intensity-weighted centroid
    (sub-pixel). Corners with no blob are NaN, e.g. when two corners of an edge-on cover merge
    into one blob (blobs larger than 2.5x the median area are dropped as merged).

    Returns {sipmId: {"corners": (4, 2) array, NaN where not found, "blobAreaPx": (4,)}}.
    centersCm/cornersCm/centerRenderPixels are indexed like sipmIds."""
    assumedToWorld = np.array(create_sensor(sensor_number=camera)["to_world"].matrix)
    baselineComponents = load_components(materials=create_materials(with_fluids=True), use_distorted_jar=True)
    baseline = to_mono(np.array(render(scene=load_scene(
        components=baselineComponents, sensor=create_sensor(sensor_number=camera, sample_count=baselineSamples)),
        denoise=False)))

    results = {}
    for n, sid in enumerate(sipmIds):
        center = np.asarray(centersCm[n], dtype=float)
        scaled = center + coverScale * (np.asarray(cornersCm[n], dtype=float) - center)
        if np.isnan(scaled).any():
            continue
        components = load_components(materials=create_materials(with_fluids=True), use_distorted_jar=True)
        add_point_light_markers(components, {f"{sid}_corner{k}": scaled[k] for k in range(4)},
                                radiusCm=radiusCm, radiance=radiance)
        image = to_mono(np.array(render(scene=load_scene(
            components=components, sensor=create_sensor(sensor_number=camera, sample_count=sampleCount)),
            denoise=False)))
        diff = image - baseline

        centerDelta = np.asarray(centerRenderPixels[n], dtype=float) - flat_project(assumedToWorld, center[None])[0]
        predicted = flat_project(assumedToWorld, scaled) + centerDelta
        count, labels, stats, _ = cv2.connectedComponentsWithStats((diff >= minDiff).astype(np.uint8), connectivity=8)
        blobs, areas = [], []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            ys, xs = np.nonzero(labels == label)
            weight = diff[ys, xs]
            blobs.append([(xs * weight).sum() / weight.sum(), (ys * weight).sum() / weight.sum()])
            areas.append(int(stats[label, cv2.CC_STAT_AREA]))
        found = np.full((4, 2), np.nan)
        blobAreas = np.zeros(4)
        if blobs:
            blobs = np.array(blobs)
            for cornerIdx, blobIdx in match_points_one_to_one(predicted, blobs, searchRadiusPx):
                found[cornerIdx] = blobs[blobIdx]
                blobAreas[cornerIdx] = areas[blobIdx]
        results[sid] = {"corners": found, "blobAreaPx": blobAreas}
        progress(f"[{n + 1}/{len(sipmIds)}] sipm {sid}: {int((~np.isnan(found[:, 0])).sum())}/4 corner blobs")

    allAreas = np.concatenate([r["blobAreaPx"][r["blobAreaPx"] > 0] for r in results.values()])
    if len(allAreas):
        tooBig = 2.5 * np.median(allAreas)
        for r in results.values():
            merged = r["blobAreaPx"] > tooBig
            r["corners"][merged] = np.nan
    return results


def fit_pose_correction_covers(centersCm, cornersCm, renderPixels, observedCovers, toWorld,
                               cutoffSchedulePx=(130, 90, 60, 40), freeParams=ROTATION_ONLY,
                               gaussNewtonIters=15, initialCorrection=None,
                               fitCoverScale=True, initialCoverScale=1.0,
                               cornerRenderPixels=None, cornerRenderScale=1.5, robustPx=None,
                               cornerOutlierPx=6.0, fullWeightSipms=None):
    """fit_pose_correction(), but matching and fitting on each cover's *corners* instead
    of one midpoint per cover.

    centersCm (n, 3) / renderPixels (n, 2): each SiPM's CAD center and where a render
    puts it (as fit_pose_correction()). cornersCm (n, 4, 3): the cover plate's 4 corners
    (load_cover_corners(), arranged per SiPM, NaN rows for the 8 SiPMs with no cover
    mesh). observedCovers: a list of (k, 2) arrays, one per clicked cover (the "corners"
    of a click_label_points.py --corners file).

    Cover size: the corners clicked on cam2 enclose ~1.5x the CAD plate's 1.4 cm (shape
    and orientation agree to ~3 px once that is scaled out), i.e. what's being clicked is
    a larger feature than the plate. Left alone, that size gap would be fit as pose
    error, so with fitCoverScale each plate's corners are scaled about its center in
    the plate's own plane by one shared factor "coverScale", solved with the pose
    (clicked size = coverScale x 1.4 cm). initialCoverScale is the starting guess, or
    the fixed value when fitCoverScale is False.

    Refraction: as in fit_pose_correction(), measured at each SiPM's center from its
    render and assumed unchanged for a small pose change -- here also assumed the same
    across the cover (the offset changes slowly across the image; this is the one
    approximation the corners add).

    cornerRenderPixels (n, 4, 2), optional: where render_corner_marker_positions() put small
    markers at the cover corners (scaled by cornerRenderScale about the plate center), NaN where
    a corner wasn't found. With it, each corner's own refraction offset is measured instead of
    borrowing the SiPM center's: the offset at plate scale s is taken as the center's plus
    s / cornerRenderScale times the extra offset the corner render measured (linear across the
    cover). Outlier corners are rejected and missing ones filled from their SiPM's other
    corners by filter_corner_offsets() (cornerOutlierPx; None = no outlier test).

    robustPx: if set, iteratively reweighted Huber fit -- a matched cover whose RMS point error
    (px) exceeds robustPx has its residuals weighted by robustPx / error, so a few bad matches
    (edge-on covers, wrong neighbor) can't drag the pose; None = plain least squares.
    fullWeightSipms: SiPM indices (rows of centersCm) that keep weight 1 whatever their error --
    for covers you've checked by eye and trust despite a large residual. "weights" (per pair) is
    returned either way.

    Correspondence is redone at every cutoff from the corner-alignment cost of
    match_covers(). Each matched 4-corner cover then adds 8 residuals (its 4 corners,
    lined up the way that matches best), each center-only match adds 2, so a well-seen
    cover also constrains the in-plane twist the midpoint alone can't.

    Returns the same keys as fit_pose_correction(): "correction", "toWorld",
    "distortion", "pairs" [(sipmIndex, coverIndex)], "deltaPx"; plus "coverScale", "weights",
    "cornerRendersUsed" (measured corners that passed filter_corner_offsets()),
    "predictedCenterPx" (n, 2) / "predictedCornerPx" (n, 4, 2) at the solved pose (NaN corners for
    SiPMs with no plate),
    "residualsPx" (per pair: mean corner distance, or center distance for a center-only
    match) and "cornerOrders" (see match_covers())."""
    centersCm = np.asarray(centersCm, dtype=float)
    cornersCm = np.asarray(cornersCm, dtype=float)
    observedCovers = [np.asarray(o, dtype=float) for o in observedCovers]
    freeMask = np.zeros(9, dtype=bool)
    freeMask[:len(freeParams)] = freeParams
    deltaPx = np.asarray(renderPixels, dtype=float) - flat_project(toWorld, centersCm)
    hasCorners = ~np.isnan(cornersCm).any(axis=(1, 2))
    # extra refraction offset (beyond the center's) measured at the corners, per unit of plate scale
    cornerDeltaSlope = np.zeros((len(centersCm), 4, 2))
    keptCorners = 0
    if cornerRenderPixels is not None:
        cornerRenderPixels = np.asarray(cornerRenderPixels, dtype=float)
        atRender = centersCm[hasCorners][:, None, :] + cornerRenderScale * (
            cornersCm[hasCorners] - centersCm[hasCorners][:, None, :])
        flatAtRender = flat_project(toWorld, atRender.reshape(-1, 3)).reshape(-1, 4, 2)
        measured = cornerRenderPixels[hasCorners] - flatAtRender - deltaPx[hasCorners][:, None, :]
        measured, trusted = filter_corner_offsets(measured, cornerOutlierPx)
        keptCorners = int(trusted.sum())
        cornerDeltaSlope[hasCorners] = measured / cornerRenderScale

    def predict(correction, coverScale):
        pose = perturb_pose(toWorld, correction)
        centerPx = flat_project(pose, centersCm) + deltaPx
        cornerPx = np.full((len(centersCm), 4, 2), np.nan)
        if hasCorners.any():
            scaled = centersCm[hasCorners][:, None, :] + coverScale * (
                cornersCm[hasCorners] - centersCm[hasCorners][:, None, :])
            flat = flat_project(pose, scaled.reshape(-1, 3)).reshape(-1, 4, 2)
            cornerPx[hasCorners] = (flat + deltaPx[hasCorners][:, None, :]
                                    + coverScale * cornerDeltaSlope[hasCorners])
        distort = lambda px: apply_radial_distortion(px, correction[6], correction[7], 1 + correction[8])
        centerPx = distort(centerPx)
        cornerPx[hasCorners] = distort(cornerPx[hasCorners].reshape(-1, 2)).reshape(-1, 4, 2)
        return centerPx, cornerPx

    def residuals(correction, coverScale, pairs, orders):
        centerPx, cornerPx = predict(correction, coverScale)
        out = []
        for (i, j) in pairs:
            order = orders[(i, j)]
            if order is not None:
                out.append((cornerPx[i][order] - observedCovers[j]).ravel())
            else:
                out.append(centerPx[i] - observedCovers[j].mean(axis=0))
        return np.concatenate(out)

    correction = np.zeros(9) if initialCorrection is None else np.array(initialCorrection, dtype=float)
    coverScale = float(initialCoverScale)
    nPose = int(freeMask.sum())
    pinned = set() if fullWeightSipms is None else {int(i) for i in fullWeightSipms}

    def unpack(theta):
        c = correction.copy()
        c[freeMask] = theta[:nPose]
        return c, (float(theta[nPose]) if fitCoverScale else coverScale)

    for cutoffPx in cutoffSchedulePx:
        centerPx, cornerPx = predict(correction, coverScale)
        pairs, orders = match_covers(centerPx, cornerPx, observedCovers, cutoffPx)
        if len(pairs) < 3:
            raise RuntimeError(f"only {len(pairs)} matches within {cutoffPx}px, can't fit a pose")
        theta = np.concatenate([correction[freeMask], [coverScale] if fitCoverScale else []])
        rowsPerPair = np.array([8 if orders[pair] is not None else 2 for pair in pairs])
        rowPair = np.repeat(np.arange(len(pairs)), rowsPerPair)
        for _ in range(gaussNewtonIters):
            residual = residuals(*unpack(theta), pairs, orders)
            weights = np.ones(len(pairs))
            if robustPx is not None:
                pairError = np.sqrt(np.bincount(rowPair, residual ** 2) / (rowsPerPair / 2))
                weights = np.minimum(1.0, robustPx / np.maximum(pairError, 1e-9))
                weights[[i in pinned for i, _ in pairs]] = 1.0
            rootW = np.sqrt(weights[rowPair])
            jacobian = np.zeros((len(residual), len(theta)))
            for col in range(len(theta)):
                stepped = theta.copy()
                stepped[col] += 1e-5
                jacobian[:, col] = (residuals(*unpack(stepped), pairs, orders) - residual) / 1e-5
            theta = theta + np.linalg.lstsq(jacobian * rootW[:, None], -residual * rootW, rcond=None)[0]
        correction, coverScale = unpack(theta)

    centerPx, cornerPx = predict(correction, coverScale)
    pairs, orders = match_covers(centerPx, cornerPx, observedCovers, cutoffSchedulePx[-1])
    residualsPx = []
    rmsPx = []
    for (i, j) in pairs:
        if orders[(i, j)] is not None:
            diffs = np.linalg.norm(cornerPx[i][orders[(i, j)]] - observedCovers[j], axis=1)
        else:
            diffs = np.array([np.linalg.norm(centerPx[i] - observedCovers[j].mean(axis=0))])
        residualsPx.append(diffs.mean())
        rmsPx.append(np.sqrt((diffs ** 2).mean()))
    finalWeights = np.ones(len(pairs)) if robustPx is None else np.minimum(1.0, robustPx / np.maximum(rmsPx, 1e-9))
    finalWeights[[i in pinned for i, _ in pairs]] = 1.0
    return {
        "correction": correction,
        "toWorld": perturb_pose(toWorld, correction),
        "distortion": {"k1": float(correction[6]), "k2": float(correction[7]),
                       "focalScale": float(1 + correction[8])},
        "weights": finalWeights,
        "cornerRendersUsed": keptCorners,
        "predictedCenterPx": centerPx,
        "predictedCornerPx": cornerPx,
        "coverScale": coverScale,
        "pairs": pairs,
        "cornerOrders": orders,
        "residualsPx": np.array(residualsPx),
        "deltaPx": deltaPx,
    }


def render_light_marker_pair(camera, positionsCm, toWorld=None, sampleCount=128, radiance=80.0,
                             fov=86):
    """Renders camera N's scene twice -- with a light marker at each of positionsCm, and
    without -- from toWorld (4x4, defaults to create_sensor()'s assumed pose). Returns
    (litImage, baselineImage) mono arrays, ready for detect_light_markers(lit, baseline=...).
    fov (degrees, x axis) defaults to the 86 every render uses; pass a solved fov (e.g.
    from fit_pose_correction(..., freeParams=ROTATION_AND_FOCAL)) to render the real lens's
    view natively instead of warping a fov=86 render."""
    images = []
    for withMarkers in (True, False):
        components = load_components(materials=create_materials(with_fluids=True), use_distorted_jar=True)
        if withMarkers:
            add_point_light_markers(components, positionsCm, radiance=radiance)
        sensorArgs = {} if toWorld is None else {"to_world": mi.ScalarTransform4f(np.asarray(toWorld))}
        sensor = create_sensor(sensor_number=camera, sample_count=sampleCount, fov=fov, **sensorArgs)
        images.append(to_mono(np.array(render(scene=load_scene(components=components, sensor=sensor), denoise=False))))
    return tuple(images)


def detect_viewport_circle(image, backgroundThreshold=30):
    """Fits the outer edge of the whole bright viewport visible in a real frame
    (chamber interior + outer rim together, vs. the black background around it).
    Returns ((cx, cy), radius). Used by both detectors below to stay off the rim's
    own bolt-hole highlights and seams, which are otherwise similar in size/contrast
    to a real SiPM cover or fiducial marker."""
    mask = (image > backgroundThreshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(biggest)
    return np.array(center), radius


def detect_sipm_covers(image, localBackgroundKernel=31, contrastThreshold=35, minArea=20, maxArea=400,
                        minSize=4, maxSize=30, minAspectRatio=1.3, maxAspectRatio=5.0,
                        maxRadiusFraction=0.70):
    """Finds small dark rectangular blobs (SiPM cover openings) in a real camN-imgNN.png
    frame. Returns an (n, 2) array of pixel positions, one row per candidate -- unlabeled
    (no attempt to say *which* SiPM each is), since there's no 1:1 correspondence to the
    real ids without also solving the matching problem this detector is meant to feed
    into.

    Mirrors detect_viewport_circle()'s neighbor (the LED detector this module used to
    have, see git history) in structure: each pixel is compared to its own local
    median-blurred surroundings (localBackgroundKernel), but here looking for a
    significantly *darker* local patch (`cv2.subtract(background, image)`) rather than a
    brighter one, since a SiPM cover reads as a small dark rectangle against the lighter
    copper panel around it. Connected components are then filtered by size (minArea..
    maxArea) and elongation (minAspectRatio..maxAspectRatio) to keep roughly
    rectangular/slot-shaped blobs -- the panel seams and crosshair calibration lines that
    also show up dark are much more elongated (aspectRatio excluded by maxAspectRatio),
    while sensor noise speckle is excluded by minArea/minSize.

    Tuned by eye against run 20251114_37 cam1 frame16: lands cleanly on the real dark
    slots around the chamber wall, but the bright LED dot pattern's own local brightness
    gradient (median-blurred background sitting high near a cluster of lit LEDs, with gaps
    between them reading as locally "dark") is a known, not-yet-fixed source of false
    positives inside that region -- not checked against other cameras/frames."""
    background = cv2.medianBlur(image, localBackgroundKernel)
    contrast = cv2.subtract(background, image)
    mask = (contrast >= contrastThreshold).astype(np.uint8)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    longSide = np.maximum(widths, heights)
    shortSide = np.maximum(np.minimum(widths, heights), 1)
    aspectRatio = longSide / shortSide

    keep = np.where(
        (areas >= minArea) & (areas <= maxArea)
        & (shortSide >= minSize) & (longSide <= maxSize)
        & (aspectRatio >= minAspectRatio) & (aspectRatio <= maxAspectRatio)
    )[0]
    keep = keep[keep != 0]  # label 0 is the background

    points = centroids[keep]
    if len(points) == 0:
        return points
    viewportCenter, viewportRadius = detect_viewport_circle(image)
    distanceFromCenter = np.linalg.norm(points - viewportCenter, axis=1)
    return points[distanceFromCenter <= maxRadiusFraction * viewportRadius]


def detect_fiducial_markers(image, assumedPixels, searchRadiusPx=60, cannyLow=10, cannyHigh=40,
                             houghThreshold=20, minLineLength=30, maxLineGap=10,
                             minAngleDeg=20, clusterDistPx=8, minClusterVotes=5):
    """Searches each fiducial marker's window (around where the assumed pose projects it
    -- assumedPixels is a {id: (px, py)} dict, e.g. project_positions_to_pixels()'s output
    for load_assumed_positions()["fiducial"] on the same camera) for where its thin
    reference lines cross, rather than for the marker's own faint photometric signature.
    Returns ({id: (px, py)}, {id: votes}) -- one entry per id with at least
    minClusterVotes of line-pair agreement in its window (an id can be missing --
    occluded, out of frame, or too few lines detected). votes is how many independent
    line-pair intersections agreed on that id's point -- a rough per-id confidence signal,
    since this function has no other way to flag a shaky result.

    This module's first attempt at this function scored circle candidates by photometric
    contrast and was unreliable (see git history): a fiducial marker is a tiny real-world
    cylindrical tab (~0.2in) whose only visible signature is a faint lit/shadowed shading
    gradient, easily lost in or confused with ordinary wall texture. But every real marker
    in this chamber also sits where several thin reference lines cross -- visible in every
    real photo checked, both filled and (per chamber_model/sipm_fiducial_positions.csv's
    source material) dry -- which turns out to be a far stronger, more specific cue than
    the marker's own appearance. Within each window: cv2.HoughLinesP proposes line
    segments, near-parallel pairs are skipped (minAngleDeg) since they don't meaningfully
    intersect, every remaining pair's intersection point (within the window) is kept, and
    intersections within clusterDistPx of each other are merged into one cluster -- the
    marker is the cluster with the most line-pair agreement (its "votes"), provided that's
    at least minClusterVotes (guards against a window with only 1-2 incidental line
    crossings, e.g. two panel seams, being mistaken for a real marker).

    Checked, not just unverified, against two markers whose real pixel position was
    confirmed by eye (at a visible line crossing) in chamber_images/led1-cam2.bmp: this
    found them 13.1px and 4.3px away, vs. 40-80px and inconsistent-in-direction for the
    old contrast-based scoring on a different frame. Still: not yet checked against enough
    independently-known marker positions to trust blindly -- a low vote count (tens, not
    hundreds) is itself a signal to look at that window before trusting its result."""
    results, voteCounts = {}, {}
    height, width = image.shape
    minAngleRad = np.deg2rad(minAngleDeg)
    for posId, (expectedX, expectedY) in assumedPixels.items():
        expectedX, expectedY = int(round(expectedX)), int(round(expectedY))
        x0, y0 = max(expectedX - searchRadiusPx, 0), max(expectedY - searchRadiusPx, 0)
        x1, y1 = min(expectedX + searchRadiusPx, width), min(expectedY + searchRadiusPx, height)
        windowImage = image[y0:y1, x0:x1]
        if windowImage.size == 0:
            continue

        edges = cv2.Canny(windowImage, cannyLow, cannyHigh)
        segments = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 360, threshold=houghThreshold,
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
        if segments is None or len(segments) < 2:
            continue
        segments = segments.reshape(-1, 4)

        intersections = []
        for i in range(len(segments)):
            angleI = np.arctan2(segments[i][3] - segments[i][1], segments[i][2] - segments[i][0])
            for j in range(i + 1, len(segments)):
                angleJ = np.arctan2(segments[j][3] - segments[j][1], segments[j][2] - segments[j][0])
                angleDiff = abs((angleI - angleJ + np.pi / 2) % np.pi - np.pi / 2)
                if angleDiff < minAngleRad:
                    continue
                point = _line_intersection(segments[i], segments[j])
                if point is None:
                    continue
                px, py = point
                if 0 <= px < windowImage.shape[1] and 0 <= py < windowImage.shape[0]:
                    intersections.append((px, py))
        if not intersections:
            continue
        intersections = np.array(intersections)

        used = np.zeros(len(intersections), dtype=bool)
        bestCenter, bestVotes = None, minClusterVotes
        for i in range(len(intersections)):
            if used[i]:
                continue
            group = np.linalg.norm(intersections - intersections[i], axis=1) < clusterDistPx
            used |= group
            if group.sum() > bestVotes:
                bestVotes = group.sum()
                bestCenter = intersections[group].mean(axis=0)
        if bestCenter is not None:
            results[posId] = (bestCenter[0] + x0, bestCenter[1] + y0)
            voteCounts[posId] = int(bestVotes)
    return results, voteCounts


def _line_intersection(lineA, lineB):
    """Intersection point of two infinite lines through segments lineA, lineB (each
    (x1, y1, x2, y2)), or None if they're (numerically) parallel."""
    x1, y1, x2, y2 = lineA
    x3, y3, x4, y4 = lineB
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    pointX = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    pointY = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return pointX, pointY


SIPM_CLUSTER_DISTANCE_CM = 1.5  # merges one physical SiPM's own disconnected mesh
# pieces (~0.1cm apart) without also merging in the next SiPM over in the same ring
# (~2.4-3cm apart) -- see load_assumed_positions().
ASSUMED_MATCH_DISTANCE_CM = 1.0  # see load_assumed_positions()


def load_assumed_positions(matchDistanceCm=ASSUMED_MATCH_DISTANCE_CM):
    """Loads this scene's own already-placed geometry for the fiducial tabs
    (fiducials.ply) and SiPMs (sipm_detectors.ply) and returns their *assumed*
    render-space positions (cm) -- i.e. what's actually modeled in the scene, as
    opposed to load_sipm_fiducial_positions()'s *actual* measured positions.

    The raw mesh has no id of its own, so each mesh centroid is labeled with the id of
    its nearest load_sipm_fiducial_positions() match, and dropped if that's farther than
    matchDistanceCm away. That cutoff matters for the SiPMs: sipm_detectors.ply doesn't
    model all 48 -- the first (piston-side) SiPM of each of the 8 rings is simply
    missing from the mesh, so without a cutoff those 8 ids would silently get labeled
    with their ring's *next* SiPM's geometry (~1.6cm away, same z) instead of being left
    out -- checked by comparing match distances with and without one, not assumed; the
    other 40 SiPMs and all 5 fiducials match to within ~0.2cm.

    Returns {"fiducial": {id: (x, y, z) cm}, "sipm": {id: (x, y, z) cm}}."""
    actualPositions = load_sipm_fiducial_positions()

    fiducialMesh = mi.load_dict({"type": "ply", "filename": scene_dir + "fiducials" + object_format})
    fiducialParams = mi.traverse(fiducialMesh)
    fiducialCentroids = _connected_component_centroids(
        np.array(fiducialParams["vertex_positions"]).reshape(-1, 3),
        np.array(fiducialParams["faces"]).reshape(-1, 3),
    )

    sipmMesh = mi.load_dict({"type": "ply", "filename": scene_dir + "sipm_detectors" + object_format})
    sipmParams = mi.traverse(sipmMesh)
    sipmPieceCentroids = _connected_component_centroids(
        np.array(sipmParams["vertex_positions"]).reshape(-1, 3),
        np.array(sipmParams["faces"]).reshape(-1, 3),
    )
    sipmCentroids = _distance_clustered_centroids(sipmPieceCentroids, SIPM_CLUSTER_DISTANCE_CM)

    assumedPositions = {"fiducial": {}, "sipm": {}}
    for posType, meshCentroids in (("fiducial", fiducialCentroids), ("sipm", sipmCentroids)):
        for posId, actualPos in actualPositions[posType].items():
            distances = np.linalg.norm(meshCentroids - actualPos, axis=1)
            nearestIndex = distances.argmin()
            if distances[nearestIndex] <= matchDistanceCm:
                assumedPositions[posType][posId] = meshCentroids[nearestIndex]
    return assumedPositions


def _distance_clustered_centroids(points, thresholdCm):
    """Union-find over pairwise distance (not mesh topology) -- merges a physical
    part's own few nearby disconnected mesh pieces into one centroid, for a mesh where
    face-connectivity alone leaves each piece separate."""
    numPoints = points.shape[0]
    parent = np.arange(numPoints)

    def _find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a, b):
        rootA, rootB = _find(a), _find(b)
        if rootA != rootB:
            parent[rootA] = rootB

    for i in range(numPoints):
        for j in range(i + 1, numPoints):
            if np.linalg.norm(points[i] - points[j]) < thresholdCm:
                _union(i, j)

    roots = np.array([_find(i) for i in range(numPoints)])
    _, inverse = np.unique(roots, return_inverse=True)

    numClusters = inverse.max() + 1
    centroids = np.zeros((numClusters, 3))
    counts = np.zeros(numClusters)
    np.add.at(centroids, inverse, points)
    np.add.at(counts, inverse, 1)
    return centroids / counts[:, None]


def _connected_component_centroids(vertexPositions, faceIndices):
    """Union-find over face edges, then mean vertex position per connected component --
    e.g. for a mesh where each physical part (one LED, one fiducial tab) is its own
    disconnected piece, this isolates one centroid per part with nothing to tune."""
    numVerts = vertexPositions.shape[0]
    parent = np.arange(numVerts)

    def _find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a, b):
        rootA, rootB = _find(a), _find(b)
        if rootA != rootB:
            parent[rootA] = rootB

    for a, b, c in faceIndices:
        _union(a, b)
        _union(b, c)

    roots = np.array([_find(i) for i in range(numVerts)])
    _, inverse = np.unique(roots, return_inverse=True)

    numComponents = inverse.max() + 1
    centroids = np.zeros((numComponents, 3))
    counts = np.zeros(numComponents)
    np.add.at(centroids, inverse, vertexPositions)
    np.add.at(counts, inverse, 1)
    return centroids / counts[:, None]
