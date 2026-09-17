"""Calibrates a camera's pose against the real chamber's LED ring lights.

Each camera has its own LED ring (`led_ring_1/2/3` in the CAD model) mounted at a
known, fixed 3D position -- unlike bubbles, these don't move frame-to-frame, so
matching their real pixel positions against their known CAD positions gives a
PnP calibration input independent of LAr10Ana's triangulation or the fiducial
pentagon `PnP_Test_Chamber.ipynb` uses.
"""
import numpy as np

from setup import *

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
