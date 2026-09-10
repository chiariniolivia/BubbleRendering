# Bubble Localization and Rendering for the SBC

Forked from [MA-Khatri/BubbleLocalizationAndRendering](https://github.com/MA-Khatri/BubbleLocalizationAndRendering). See that repo for his thesis and some more general documentation.

The primary files are [modules/setup.py](modules/setup.py) which contains functions for setting up and rendering images of the chambers, and [modules/utils.py](modules/utils.py) which contains several functions that are used for triangulation, remapping, creating plots, etc. Examples of how the functions from those files are used to create renders and triangulate bubbles can be found within the Jupyter notebooks, under [notebooks/](notebooks/).

[modules/real_data_loading.py](modules/real_data_loading.py) loads real background-run data (camera images, handscanned bubble multiplicity, and precomputed `.sbc` analysis outputs) for calibrating the rendered cameras' poses against the physical detector. It expects that data locally under `~/coop/local-runs/<run>/<ev>/`,  `~/coop/local-runs/<run>/analysis/`, and `~/coop/local-runs/<run>/handscan.txt`. These can be fetched from gpv, in the unpacked format. The expectation is that the events have been handscanned for multiplicity before analysis here. For further information on the analysis modules, see [LAr10Ana](https://github.com/SBC-Collaboration/LAr10Ana).

## Setting Up the Environment

The code now runs from a plain Python virtual environment (`venv`) rather than Anaconda. Any Python 3.12+ interpreter should work; this was last verified against Python 3.14.

```console
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

If you use a different shell (like me who uses fish!!) use the appropriate activate script. Should work for power shell (ps1) and csh, there is probably a way to make it work in nu or elvish but your mileage may vary. That installs everything needed for both rendering (Mitsuba/Dr.Jit/OpenCV) and working with real background-run data (`sbcbinaryformat`, pulled straight from its GitHub repo since it isn't on PyPI). GPU-accelerated rendering requires an NVIDIA card; without one it falls back to CPU rendering automatically.

### Registering a Jupyter kernel

So the notebooks can find this environment, register it with Jupyter once (with the venv activated):

```console
python -m ipykernel install --user --name sbc --display-name "Python (sbc)"
```

Then open a notebook and select the **Python (sbc)** kernel. Every notebook in this repo already has its kernelspec set to `sbc`.

### Running Notebooks

To launch Jupyter Lab directly from this environment (with the venv activated):

```console
jupyter lab
```

which opens a lab environment where you can edit and run the notebooks in your default browser. If you're using a code editor like VS Code instead, point it at the same venv's interpreter and select the **Python (sbc)** kernel when opening a notebook.

## Notebooks

All notebooks live in [`notebooks/`](notebooks/); their first cell adds [`modules/`](modules/) to `sys.path` before importing `setup`/`utils`/`real_data_loading`, so they work whether you launch Jupyter from the repo root or open them directly (Jupyter's usual default working directory is the notebook's own folder).

- [`BasicRender.ipynb`](notebooks/BasicRender.ipynb) Walks through the basics of setting up a Mitsuba scene of the chamber and creating a render.
- [`PixelRemappingAndTriangulation.ipynb`](notebooks/PixelRemappingAndTriangulation.ipynb) Creates renders of the testing grid and uses those renders to determine the remapping functions and triangulations. This was the primary notebook used to test different remapping functions, triangulation methods, and optimization techniques.
- [`PixelRemappingAndTriangulation_Wrong_IoRs.ipynb`](notebooks/PixelRemappingAndTriangulation_Wrong_IoRs.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that creates renders of a smaller test grid to determine the effect that errors in the indices of refraction of chamber elements have on triangulation errors.
- [`PixelRemappingAndTriangulation_Multiple.ipynb`](notebooks/PixelRemappingAndTriangulation_Multiple.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that tests all of the available optimization methods simultaneously and plots their resulting loss curves and triangulation error histograms.
- [`PnP_Test_Chamber.ipynb`](notebooks/PnP_Test_Chamber.ipynb) Determines the pose of camera 2 of the chamber using the pixel positions of the fiducial markers in a real image.
- [`PoseMatrixEstimation.ipynb`](notebooks/PoseMatrixEstimation.ipynb) Estimates the pose of the cameras used for rendering by using estimated pixel positions of the fiducial markers within each camera's image. It also saves the resulting estimated pose matrices.
- [`TriangulationWithPixelLocalizationError.ipynb`](notebooks/TriangulationWithPixelLocalizationError.ipynb) Plots the relationship between the average error in the pixel localization of bubbles vs the resulting error in triangulation.
- [`TriangulationWithPixelOffsetPnPPoses.ipynb`](notebooks/TriangulationWithPixelOffsetPnPPoses.ipynb) Plots the relationship between the error in pixel localization of the fiducial markers vs the resulting error in the estimated pose and average triangulation error.
- [`TriangulationWithPnPEstimatedPoseMatrix.ipynb`](notebooks/TriangulationWithPnPEstimatedPoseMatrix.ipynb) Runs n-view triangulation on the test grid using the estimated poses calculated in `PoseMatrixEstimation.ipynb`.
- [`CorrespondingBubbleDetectionTest.ipynb`](notebooks/CorrespondingBubbleDetectionTest.ipynb) Features a test for a possible solution to determining corresponding bubble locations between images.
- [`BulkRendering.ipynb`](notebooks/BulkRendering.ipynb) Shows how multiple renders of a grid of bubbles can be created using the `bubble_grid_renders` function.
- [`CameraPoseOptimizationFromRealData.ipynb`](notebooks/CameraPoseOptimizationFromRealData.ipynb) In-progress work using real background-run camera images (via `real_data_loading.py`) to check/refine the rendering pipeline's camera pose estimates against the physical detector.
