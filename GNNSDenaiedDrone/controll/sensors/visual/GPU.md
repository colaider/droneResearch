# OpenCL vision operations

`gpuVision.py` supplies one shared `VisionBackend` to temporal LK, stereo LK,
and feature replenishment (GFTT, Canny, dilation). OpenCL is requested by default.
The two most recent frames' left/right pyramid layers are cached as `cv2.UMat`.
An image is uploaded on first use and reused across operations. Cached NumPy
images must remain immutable; use the backend from one processing thread.

GFTT returns NumPy corners. Canny passes its UMat directly to dilation, which
returns the edge mask to NumPy. LK returns NumPy points/status/error for the
existing validity checks. Stereo patch search, filters and display remain as
before. OpenCV still builds internal LK pyramids per call.

Select CPU for comparison with `RB3_VISION_BACKEND=cpu` in the controller's
environment, or pass `vision_backend="cpu"` to `Sensors` or `VisulaEst`.
An explicit constructor selection takes precedence over the environment.
Use `opencl` to request the GPU path.

OpenCV can silently fall back to CPU. UMat and OpenCL activation alone do not
prove GPU execution. Validate device execution, tracking/depth quality and
end-to-end latency on the RB3 before relying on a speed improvement.

The LK wrapper constructs `SparsePyrLKOpticalFlow` and restores its requested
flags before `calc`. This preserves stereo's initial predictions around an
OpenCV constructor bug: its OpenCL `useInitialFlow` member is initialized from
the eigenvalue flag rather than the initial-flow flag. The workaround sets both
at construction only; the actual operation receives the original flags.
Source: https://github.com/opencv/opencv/blob/4.12.0/modules/video/src/lkpyramid.cpp

Run the camera-free tests from the repository root:

```sh
python -m unittest discover -s GNNSDenaiedDrone/controll/tests -v
```

Tests exercise both CPU and UMat inputs. On a host without an active OpenCL GPU,
UMat tests validate fallback/integration, not accelerator performance.
