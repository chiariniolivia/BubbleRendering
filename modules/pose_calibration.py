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
    correction[:3, 3] = poseCorrection[3:]
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


ROTATION_ONLY = (True, True, True, False, False, False)


def fit_pose_correction(worldPositionsCm, renderPixels, observedPixels, toWorld,
                        cutoffSchedulePx=(130, 90, 60, 40), freeParams=ROTATION_ONLY,
                        gaussNewtonIters=15):
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

    freeParams picks which of [rx, ry, rz, tx, ty, tz] are solved for. Defaults to
    rotation only: on cam2, also freeing translation fit the clicks better on paper
    (6px vs. 16px median), but with a 2-3cm camera shift, which a validation render
    showed was wrong -- it moved the camera into the viewport geometry and only 14 of
    the 36 SiPMs stayed visible. The flat model was absorbing what's more likely
    unmodeled lens distortion or fov error into a fake translation.

    worldPositionsCm, renderPixels: (n, 3) / (n, 2), same order. Returns a dict with
    "correction" (6,), "toWorld" (4x4 corrected), "pairs" [(pointIndex, observedIndex)]
    at the final cutoff, "residualsPx" (per pair), and "deltaPx" (the (n, 2) refraction
    offsets)."""
    worldPositionsCm = np.asarray(worldPositionsCm, dtype=float)
    observedPixels = np.asarray(observedPixels, dtype=float)
    freeMask = np.array(freeParams, dtype=bool)
    deltaPx = np.asarray(renderPixels, dtype=float) - flat_project(toWorld, worldPositionsCm)

    def predict(correction):
        return flat_project(perturb_pose(toWorld, correction), worldPositionsCm) + deltaPx

    correction = np.zeros(6)
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
        "pairs": pairs,
        "residualsPx": residualsPx,
        "deltaPx": deltaPx,
    }


def render_light_marker_pair(camera, positionsCm, toWorld=None, sampleCount=128, radiance=80.0):
    """Renders camera N's scene twice -- with a light marker at each of positionsCm, and
    without -- from toWorld (4x4, defaults to create_sensor()'s assumed pose). Returns
    (litImage, baselineImage) mono arrays, ready for detect_light_markers(lit, baseline=...)."""
    images = []
    for withMarkers in (True, False):
        components = load_components(materials=create_materials(with_fluids=True), use_distorted_jar=True)
        if withMarkers:
            add_point_light_markers(components, positionsCm, radiance=radiance)
        sensorArgs = {} if toWorld is None else {"to_world": mi.ScalarTransform4f(np.asarray(toWorld))}
        sensor = create_sensor(sensor_number=camera, sample_count=sampleCount, **sensorArgs)
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
