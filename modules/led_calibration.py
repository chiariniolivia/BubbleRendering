"""Calibrates a camera's pose against the real chamber's LED ring lights.

Each camera has its own LED ring (`led_ring_1/2/3` in the CAD model) mounted at a
known, fixed 3D position -- unlike bubbles, these don't move frame-to-frame, so
matching their real pixel positions against their known CAD positions gives a
PnP calibration input independent of LAr10Ana's triangulation or the fiducial
pentagon `PnP_Test_Chamber.ipynb` uses.
"""
import cv2
import numpy as np

from utils import *

LED_RING_NAMES = ("led_ring_1", "led_ring_2", "led_ring_3")


def load_led_centroids(ringName):
    """Loads one led_ring_N.ply and returns its per-LED 3D centroids (cm), one
    row per LED. Each LED is modeled as its own disconnected mesh piece, so
    clustering by face connectivity (not distance) cleanly isolates one
    centroid per LED with nothing to tune."""
    mesh = mi.load_dict({
        "type": "ply",
        "filename": scene_dir + ringName + object_format,
    })
    params = mi.traverse(mesh)
    vertexPositions = np.array(params["vertex_positions"]).reshape(-1, 3)
    faceIndices = np.array(params["faces"]).reshape(-1, 3)
    return _connected_component_centroids(vertexPositions, faceIndices)


def _connected_component_centroids(vertexPositions, faceIndices):
    """Union-find over face edges, then mean vertex position per component."""
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


def predict_led_pixels(camera, sampleCount=1024, contrastThreshold=150, **detectKwargs):
    """Renders this repo's *assumed* pose for the given camera number (no bubble --
    create_sensor()'s own to_world, the same pose RealDataRenderComparison.ipynb
    renders against) and runs detect_led_glints() on the render itself. Returns
    an (n, 2) array of predicted glint pixel positions, same format as
    detect_led_glints()'s own output -- there's no per-LED correspondence here,
    just "where the pattern lands if the assumed pose is right."

    A straight-line pinhole projection of the known 3D LED positions doesn't work
    for this (checked): what's actually visible of the LED rings in a real photo
    is entirely a *reflection* off the curved dome/jar, not a direct view, so
    there's no straight line from an LED to the camera to project along in the
    first place. Rendering the real scene geometry/materials at the assumed pose
    reproduces that same reflected pattern -- the same way this repo's own
    render() was used to originally identify these LEDs as the pattern's source.

    A render's Monte-Carlo grain is a different kind of noise than a real photo's
    sensor noise -- it's not simple per-pixel speckle (a light median pre-blur
    barely touched it; a heavy enough one to actually clean it also smeared the
    LED glints into the background), so pre-denoising the render isn't the fix.
    What does work: the grain rarely gets anywhere near as bright as an LED glint
    (checked at sampleCount=1024), so contrastThreshold is raised well past
    detect_led_glints()'s real-photo default of 60 -- high enough that the noise
    floor doesn't cross it, while the glints (near-saturated, same as in a real
    photo) still do. This is passed as a keyword default rather than baked into
    detect_led_glints() itself, since it's specific to render grain, not photos.
    The disc's glossier surface converges slower than the rest of the interior,
    so its reflected LEDs are underrepresented in the prediction at this sample
    count -- same region that's hardest to detect in the real photo, just for a
    different reason. Any other keyword arguments are passed through to
    detect_led_glints() too."""
    materials = create_materials(with_fluids=True)
    components = load_components(materials=materials, use_distorted_jar=True)
    sensor = create_sensor(sensor_number=camera, sample_count=sampleCount)
    scene = load_scene(components=components, sensor=sensor)
    image = np.array(render(scene=scene, denoise=False))

    grayscale = np.clip(to_mono(image) ** (1 / 2.2), 0, 1)
    renderImage = (grayscale * 255).astype(np.uint8)
    return detect_led_glints(renderImage, contrastThreshold=contrastThreshold, **detectKwargs)


def detect_led_glints(image, localBackgroundKernel=31, contrastThreshold=60, minArea=2, maxArea=220,
                       minSize=2, maxSize=35, maxAspectRatio=3.0, peakWindow=5, maxRadiusFraction=0.70):
    """Finds small bright blobs (LED glints) in a real camN-imgNN.png frame. Returns
    an (n, 2) array of pixel positions, one row per detected glint.

    LEDs reflected off the far side of the chamber can land on top of the brightly
    illuminated central disc, where a glint that would be obviously bright against
    the dark wall barely stands out at all against a background that's already at
    ~180/255 -- an absolute brightness cutoff misses these almost entirely. Instead,
    each pixel is compared to its own local surroundings: `localBackgroundKernel`
    sets a median-blur so large that individual glints get smoothed away but broad
    features (the disc, the wall) survive, and a pixel only counts as a glint
    candidate if it's at least contrastThreshold brighter than that local backdrop
    -- so a glint on the dark wall and an equally-real one sitting on the bright
    disc both clear the same relative bar.

    A real frame's other bright features (the illuminated disc's own edge, the metal
    jar rim, thin crosshair calibration lines) are much larger or more elongated
    than a single LED glint, so local contrast alone isn't enough either -- connected
    components are first filtered by size/elongation (short side >= minSize, long side
    <= maxSize, long/short side ratio <= maxAspectRatio) to keep only glint-shaped
    blobs -- minSize specifically throws out single-pixel-wide slivers that show up
    along crosshair calibration lines and other sharp bright edges, which are much
    thinner than any real glint but would otherwise pass the other checks. Adjacent
    LEDs on the same ring are only ~9-11px apart in a real frame (see the ring radius
    from load_led_centroids and the camera's fov/distance), close enough that a whole
    run of several in a row can merge into one blob -- maxSize/maxArea are sized to
    admit a run of up to ~4 merged LEDs rather than rejecting the whole run outright
    (a first pass capped at a single LED's own size silently dropped every LED in
    a merged run, not just the extras). Each kept blob is then split into one point
    per local peak of its distance transform (a run of several merged LEDs has that
    many peaks; a lone LED just has one), rather than collapsing it to a single
    centroid.

    The chamber's SiPM-bearing interior wall (where the LEDs live) is a fair bit
    smaller than the whole bright viewport visible in frame -- outside it is a
    reflective outer rim/mounting flange whose bolt-hole highlights and seams are
    glint-shaped too. detect_viewport_circle() fits the outer edge of the whole
    bright viewport (interior + rim, vs. the black background around it); any blob
    farther than maxRadiusFraction of that fitted radius from its center is dropped.
    Checked against cam1 frame16 (run 20251114_37 ev1): every detection past the
    interior/rim boundary lands either on the rim seam itself or on a shiny tool
    inserted into the interior -- none are real LED glints -- while the fraction is
    generous enough to keep the entire real dot pattern, which tops out well short
    of it.

    Known limitation: that tool, and a couple of interior wall panel-seam corners
    with their own specular highlight, sit *inside* the interior boundary and
    happen to be a similar size/roundness to a run of merged LEDs -- no combination
    of the shape checks above can separate them from real glints without also
    losing real ones (checked: a stricter size/shape pass does cut some of these,
    but it costs real detections too, so it's left permissive here on purpose).
    They're expected to fall out at the matching step instead (nearest-neighbor
    against the pose-predicted LED positions, with a max-distance cutoff), which
    has a signal -- 3D geometric consistency with the known LED layout -- that a
    2D shape/contrast heuristic can't approximate. Tuned against cam1 frame16 of
    run 20251114_37 event 1; not yet verified against other cameras/frames."""
    background = cv2.medianBlur(image, localBackgroundKernel)
    contrast = cv2.subtract(image, background)
    mask = (contrast >= contrastThreshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    lefts = stats[:, cv2.CC_STAT_LEFT]
    tops = stats[:, cv2.CC_STAT_TOP]

    longSide = np.maximum(widths, heights)
    shortSide = np.maximum(np.minimum(widths, heights), 1)
    aspectRatio = longSide / shortSide

    keepLabels = np.where(
        (areas >= minArea) & (areas <= maxArea)
        & (shortSide >= minSize) & (widths <= maxSize) & (heights <= maxSize)
        & (aspectRatio <= maxAspectRatio)
    )[0]
    keepLabels = keepLabels[keepLabels != 0]  # label 0 is the background

    points = []
    for label in keepLabels:
        points.extend(_blob_peaks(
            labels, label, lefts[label], tops[label], widths[label], heights[label], peakWindow,
        ))
    points = np.array(points)
    if len(points) == 0:
        return points

    viewportCenter, viewportRadius = detect_viewport_circle(image)
    distanceFromCenter = np.linalg.norm(points - viewportCenter, axis=1)
    return points[distanceFromCenter <= maxRadiusFraction * viewportRadius]


def detect_viewport_circle(image, backgroundThreshold=30):
    """Fits the outer edge of the whole bright viewport visible in a real frame
    (interior wall + outer rim together, vs. the black background around it).
    Returns ((cx, cy), radius). Used to tell the interior (where the LEDs live)
    apart from the rim -- see detect_led_glints()."""
    mask = (image > backgroundThreshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(biggest)
    return np.array(center), radius


def _blob_peaks(labels, label, left, top, width, height, peakWindow):
    """Local maxima of one blob's distance transform, merging duplicate plateau
    pixels -- one point per LED even when several merged into a single blob."""
    pad = 2
    x0, y0 = max(left - pad, 0), max(top - pad, 0)
    blobMask = (labels[y0:top + height + pad, x0:left + width + pad] == label).astype(np.uint8)

    distance = cv2.distanceTransform(blobMask, cv2.DIST_L2, 5)
    dilated = cv2.dilate(distance, np.ones((peakWindow, peakWindow), np.uint8))
    isPeak = (distance == dilated) & (distance > 0.5)
    peakYs, peakXs = np.nonzero(isPeak)
    peakPoints = np.stack([peakXs, peakYs], axis=1).astype(float)

    merged = []
    used = np.zeros(len(peakPoints), dtype=bool)
    for i in range(len(peakPoints)):
        if used[i]:
            continue
        group = np.linalg.norm(peakPoints - peakPoints[i], axis=1) < 4
        used |= group
        center = peakPoints[group].mean(axis=0)
        merged.append([center[0] + x0, center[1] + y0])
    return merged
