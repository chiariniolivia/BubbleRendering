# Bubble Localization and Rendering for the SBC

![Flow chart of steps in bubble localization for the SBC chambers.](thesis/figures/localization_flow_chart_2.png)

This is the repository for the code associated with my thesis that can be found [here](thesis/Bubble_Localization_and_Rendering_for_the_SBC.pdf).

The primary files are [setup.py](setup.py) which contains functions for setting up and rendering images of the chambers, and [utils.py](utils.py) which contains several functions that are used for triangulation, remapping, creating plots, etc. Examples of how the functions from those files are used to create renders and triangulate bubbles can be found within the Jupyter notebooks.

[real_data_loading.py](real_data_loading.py) loads real background-run data (camera images, handscanned bubble multiplicity, and precomputed `.sbc` analysis outputs) for calibrating the rendered cameras' poses against the physical detector. It expects that data locally under `~/coop/local-runs/<run>/` and `~/coop/local-runs/<run>/analysis/`; that data isn't part of this repo and is treated as read-only.

The associated Blender files used to clean up and export the chamber model can be found [here](https://drive.google.com/drive/folders/1IG_XLitbM6101vEFMDmiwda3bSZK6OpN?usp=drive_link). If you wish to make edits to any of the chamber elements, you can do so in Blender and export the selected components to the `chamber_model/components/` folder. Make sure to make any corresponding necessary changes to the `setup.py` file!

## Setting Up the Environment

The code now runs from a plain Python virtual environment (`venv`) rather than Anaconda. Any Python 3.12+ interpreter should work; this was last verified against Python 3.14.

```console
python3 -m venv ~/py-envs/sbc
~/py-envs/sbc/bin/pip install -r requirements.txt
```

That installs everything needed for both rendering (Mitsuba/Dr.Jit/OpenCV) and working with real background-run data (`sbcbinaryformat`, pulled straight from its GitHub repo since it isn't on PyPI).

Rendering picks the best available Mitsuba variant automatically (`setup.py` tries `cuda_ad_rgb`, then `llvm_ad_rgb`, then `scalar_rgb`). GPU acceleration needs an **NVIDIA** card — Mitsuba/Dr.Jit currently only ship CUDA (NVIDIA) and LLVM (CPU) backends, so an AMD GPU won't be used and rendering will run on CPU via `llvm_ad_rgb` regardless.

### Registering a Jupyter kernel

So the notebooks can find this environment, register it with Jupyter once:

```console
~/py-envs/sbc/bin/python -m ipykernel install --user --name sbc --display-name "Python (sbc)"
```

Then open a notebook and select the **Python (sbc)** kernel. Every notebook in this repo (root and `archive/`) already has its kernelspec set to `sbc`.

### Running Notebooks

To launch Jupyter Lab directly from this environment:

```console
~/py-envs/sbc/bin/jupyter lab
```

which opens a lab environment where you can edit and run the notebooks in your default browser. If you're using a code editor like VS Code instead, point it at the same venv's interpreter and select the **Python (sbc)** kernel when opening a notebook.

### Legacy: Anaconda on Windows

The original code was developed in an Anaconda environment on Windows; `environment.yml` documents that setup and can still be used with `conda env create -f environment.yml` if you'd rather use conda. It hasn't been kept in sync with `requirements.txt` and may be out of date — the venv + `requirements.txt` path above is what's actually in use now.

## Notebooks

- [`BasicRender.ipynb`](BasicRender.ipynb) Walks through the basics of setting up a Mitsuba scene of the chamber and creating a render.
- [`PixelRemappingAndTriangulation.ipynb`](PixelRemappingAndTriangulation.ipynb) Creates renders of the testing grid and uses those renders to determine the remapping functions and triangulations. This was the primary notebook used to test different remapping functions, triangulation methods, and optimization techniques.
- [`PixelRemappingAndTriangulation_Wrong_IoRs.ipynb`](PixelRemappingAndTriangulation_Wrong_IoRs.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that creates renders of a smaller test grid to determine the effect that errors in the indices of refraction of chamber elements have on triangulation errors.
- [`PixelRemappingAndTriangulation_Multiple.ipynb`](PixelRemappingAndTriangulation_Multiple.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that tests all of the available optimization methods simultaneously and plots their resulting loss curves and triangulation error histograms.
- [`PnP_Test_Chamber.ipynb`](PnP_Test_Chamber.ipynb) Determines the pose of camera 2 of the chamber using the pixel positions of the fiducial markers in a real image.
- [`PoseMatrixEstimation.ipynb`](PoseMatrixEstimation.ipynb) Estimates the pose of the cameras used for rendering by using estimated pixel positions of the fiducial markers within each camera's image. It also saves the resulting estimated pose matrices.
- [`TriangulationWithPixelLocalizationError.ipynb`](TriangulationWithPixelLocalizationError.ipynb) Plots the relationship between the average error in the pixel localization of bubbles vs the resulting error in triangulation.
- [`TriangulationWithPixelOffsetPnPPoses.ipynb`](TriangulationWithPixelOffsetPnPPoses.ipynb) Plots the relationship between the error in pixel localization of the fiducial markers vs the resulting error in the estimated pose and average triangulation error.
- [`TriangulationWithPnPEstimatedPoseMatrix.ipynb`](TriangulationWithPnPEstimatedPoseMatrix.ipynb) Runs n-view triangulation on the test grid using the estimated poses calculated in `PoseMatrixEstimation.ipynb`.
- [`CorrespondingBubbleDetectionTest.ipynb`](CorrespondingBubbleDetectionTest.ipynb) Features a test for a possible solution to determining corresponding bubble locations between images.
- [`BulkRendering.ipynb`](BulkRendering.ipynb) Shows how multiple renders of a grid of bubbles can be created using the `bubble_grid_renders` function.
- [`CameraPoseOptimizationFromRealData.ipynb`](CameraPoseOptimizationFromRealData.ipynb) In-progress work using real background-run camera images (via `real_data_loading.py`) to check/refine the rendering pipeline's camera pose estimates against the physical detector.

---

Note: notebooks located in the archive folder are not well documented and feature code that may be deprecated. They were mainly used in initial testing.

- [`archive/PnPEstimationWithCheckerboard.ipynb`](archive/PnPEstimationWithCheckerboard.ipynb) Tries to estimate the pose matrices of the cameras in the chamber by placing a checkerboard in the scene where the fiducial markers would normally be.
- [`archive/PnP_Test_Render.ipynb`](archive/PnP_Test_Render.ipynb) An earlier version of `archive/PnPEstimationWithCheckerboard.ipynb` that computed the pose for a single camera by placing the checkerboard in the scene where the fiducial markers would be.
- [`archive/CameraCalibrationWithRenderedCheckerboard.ipynb`](archive/CameraCalibrationWithRenderedCheckerboard.ipynb) Calibrated the camera used to render images by rendering a series of images of a checkerboard. Also features code that could be adapted to calibrate real cameras with real images.
- [`archive/LocationRemappingTest.ipynb`](archive/LocationRemappingTest.ipynb) Features some of the initial tests for determining the remapping coefficients of distortion functions.
- [`archive/LocationMappingComparedWithAndWithoutDistortedSurface.ipynb`](archive/LocationMappingComparedWithAndWithoutDistortedSurface.ipynb) Similar to `archive/LocationRemappingTest.ipynb`, performs initial tests for pixel remapping, now comparing the results with and without the distorted jar surfaces.
- [`archive/TwoCameraTriangulationError.ipynb`](archive/TwoCameraTriangulationError.ipynb) Initial triangulation tests for two-camera midpoint triangulation.
- [`archive/PoseOptimizationWithRenders.ipynb`](archive/PoseOptimizationWithRenders.ipynb) Attempts to use Mitsuba's inverse rendering to optimize for the pose of the cameras. Based on [this Mitsuba tutorial](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/object_pose_estimation.html).
- [`archive/BasicChamberSurfaceOptimizationRender.ipynb`](archive/BasicChamberSurfaceOptimizationRender.ipynb) Attempts at optimizing a heightmap to determine jar surface distortion. Optimizations were attempted by comparing to a rendered image with a distorted jar surface. Based on [this Mitsuba tutorial](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/caustics_optimization.html).
