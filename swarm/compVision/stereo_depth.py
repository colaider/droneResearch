"""Binocular depth for an ideal rectified pinhole pair, independent of drone pose.

OpenCV optical frame: X right, Y down, Z forward. Depth is optical Z in metres,
not Euclidean range. Images must be captured at the same simulation step, with
identical intrinsics and parallel cameras separated along optical X.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    """Known simulated camera geometry; real cameras require measured calibration.

    width/height: image dimensions in pixels. vertical_fov_deg: Genesis vertical
    field of view in degrees. baseline_m: positive left-to-right spacing in m.
    Both cameras have square pixels, centred principal points and no distortion.
    """
    width: int = 640
    height: int = 480
    vertical_fov_deg: float = 65.0
    baseline_m: float = 0.2

    def __post_init__(self):
        if self.width < 64 or self.height < 32:
            raise ValueError('Image resolution must be at least 64x32')
        if not np.isfinite(self.baseline_m) or self.baseline_m <= 0:
            raise ValueError('baseline_m must be positive and finite')
        if not 0 < self.vertical_fov_deg < 180:
            raise ValueError('vertical_fov_deg must lie between 0 and 180 degrees')

    @property
    def focal_px(self):
        """Focal length in pixels, calculated from VERTICAL field of view."""
        return self.height / (2 * np.tan(np.deg2rad(self.vertical_fov_deg) / 2))

    @property
    def K(self):
        """(3,3) pixel intrinsic matrix shared by both cameras."""
        f = self.focal_px
        return np.array([[f, 0., self.width/2], [0., f, self.height/2], [0., 0., 1.]])

    @property
    def projection_matrices(self):
        """Return P_left/P_right (3,4), projecting LEFT-frame 3D coordinates.

        The right camera centre is [B,0,0] in left coordinates; consequently
        its world-to-camera translation is [-B,0,0], not [+B,0,0].
        """
        left = np.column_stack((np.eye(3), np.zeros(3)))
        right = np.column_stack((np.eye(3), [-self.baseline_m, 0., 0.]))
        return self.K @ left, self.K @ right


def triangulate_matches(left_uv, right_uv, calibration, max_vertical_error=1.5,
                        min_depth=0.5, max_depth=80.):
    """Triangulate matched pixels using two projection matrices.

    Args:
        left_uv/right_uv: matching (N,2) arrays of [u,v] pixel coordinates.
            Row i must refer to the same scene point in both images.
        calibration: StereoCalibration for this rectified, distortion-free pair.
        max_vertical_error: allowed |v_left-v_right| in pixels (epipolar check).
        min_depth/max_depth: accepted optical Z range in metres.

    Returns:
        points: (N,3) XYZ in the LEFT optical frame, metres; invalid rows are NaN.
        valid: (N,) boolean mask. No drone altitude or simulator depth is used.
    """
    left = np.asarray(left_uv, dtype=float)
    right = np.asarray(right_uv, dtype=float)
    if left.ndim != 2 or left.shape[1] != 2 or right.shape != left.shape:
        raise ValueError('Matching coordinates must both have shape (N, 2)')
    points = np.full((len(left), 3), np.nan)
    valid = (np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
             & (np.abs(left[:, 1]-right[:, 1]) <= max_vertical_error)
             & (left[:, 0]-right[:, 0] > 0.5))
    for pixels in (left, right):
        valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))
    idx = np.flatnonzero(valid)
    if len(idx):
        P_left, P_right = calibration.projection_matrices
        homogeneous = cv2.triangulatePoints(P_left, P_right, left[idx].T, right[idx].T)
        # OpenCV returns [X*w,Y*w,Z*w,w]; dividing by w recovers metric XYZ.
        good_w = np.abs(homogeneous[3]) > 1e-12
        xyz = np.full((len(idx), 3), np.nan)
        xyz[good_w] = (homogeneous[:3, good_w]/homogeneous[3, good_w]).T
        good = (np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= min_depth)
                & (xyz[:, 2] <= max_depth))
        # Reprojection checks catch incompatible row pairs, not every false match
        # on repetitive facades: texture ambiguity remains a stereo limitation.
        for P, observed in ((P_left, left[idx]), (P_right, right[idx])):
            projected = np.column_stack((xyz, np.ones(len(xyz)))) @ P.T
            uv = np.full((len(xyz), 2), np.nan)
            np.divide(projected[:, :2], projected[:, 2, None], out=uv,
                      where=np.abs(projected[:, 2, None]) > 1e-12)
            good &= np.linalg.norm(uv-observed, axis=1) <= max_vertical_error
        points[idx[good]] = xyz[good]
        valid[idx] = good
    return points, valid


class StereoDepth:
    """Dense disparity plus sparse feature triangulation for a calibrated rig."""
    def __init__(self, calibration, num_disparities=128, block_size=5,
                 min_depth=0.5, max_depth=80.):
        """Args: calibration geometry; search width in pixels; odd patch size.

        num_disparities must be a positive multiple of 16 smaller than image
        width. Larger values allow nearer objects but cost time and image border.
        Depth limits are metres. Unknown depth always stays NaN, never zero.
        """
        if num_disparities <= 0 or num_disparities % 16 or num_disparities >= calibration.width:
            raise ValueError('num_disparities must be a multiple of 16 below image width')
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError('block_size must be odd and at least 3')
        if not 0 < min_depth < max_depth:
            raise ValueError('Require 0 < min_depth < max_depth')
        self.calibration = calibration
        self.min_depth, self.max_depth = min_depth, max_depth
        self.num_disparities = num_disparities
        parameters = dict(numDisparities=num_disparities, blockSize=block_size,
                          P1=8*block_size**2, P2=32*block_size**2,
                          disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=80,
                          speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, **parameters)
        self.right_matcher = cv2.StereoSGBM_create(minDisparity=-num_disparities, **parameters)
        self.detector = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _gray(self, image):
        """Validate uint8 HxWx3 BGR (or HxW grayscale) and return grayscale."""
        c = self.calibration
        if image.dtype != np.uint8 or image.shape[:2] != (c.height, c.width):
            raise ValueError('Images must be uint8 with the calibrated height and width')
        if image.ndim == 2:
            return image
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected grayscale or 3-channel BGR images')
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def depth_from_disparity(self, disparity, valid=None):
        """Convert HxW disparity d=u_left-u_right (PIXELS) to optical depth (m).

        Z=f*B/d follows from two pinhole projections of the same 3D point.
        fx is in pixels, B in metres. Tiny, negative and out-of-range disparities
        are invalid; valid optionally masks occlusion/matching failures.
        """
        disparity = np.asarray(disparity, dtype=float)
        mask = np.isfinite(disparity) & (disparity > 0.5)
        if valid is not None:
            mask &= valid
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        np.divide(self.calibration.focal_px*self.calibration.baseline_m,
                  disparity, out=depth, where=mask)
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = np.nan
        return depth

    def compute(self, left_bgr, right_bgr):
        """Return disparity (px), depth (m), valid mask and sparse matched XYZ.

        Inputs are synchronized rectified BGR images. The rig is rectified by
        construction; real camera images must first be undistorted/rectified.
        """
        left, right = self._gray(left_bgr), self._gray(right_bgr)
        # SGBM encodes disparity as int16 with FOUR fractional bits, hence /16.
        dl = self.left_matcher.compute(left, right).astype(np.float32)/16.
        dr = self.right_matcher.compute(right, left).astype(np.float32)/16.
        rows, cols = np.indices(dl.shape)
        right_u = np.rint(cols-dl).astype(int)
        inside = (right_u >= 0) & (right_u < dl.shape[1])
        reverse = dr[rows, np.clip(right_u, 0, dl.shape[1]-1)]
        valid = (inside & (dl > 0.5) & (dl < self.num_disparities-1)
                 & (reverse < -0.5) & (reverse > -self.num_disparities)
                 & (np.abs(dl+reverse) <= 1.0))
        # Textureless walls have ambiguous correspondence even if the matcher
        # fills them in. Remove locally flat regions rather than inventing depth.
        intensity = left.astype(np.float32)
        variance = cv2.blur(intensity**2, (7, 7))-cv2.blur(intensity, (7, 7))**2
        valid &= variance > 4.0
        depth = self.depth_from_disparity(dl, valid)
        valid = np.isfinite(depth)
        disparity = np.where(valid, dl, np.nan).astype(np.float32)
        sparse = self.sparse_matches(left, right)
        return dict(disparity_px=disparity, depth_m=depth, valid=valid, **sparse)

    def sparse_matches(self, left_gray, right_gray):
        """Match ORB features mutually, then triangulate with P_left/P_right.

        Returns matching left_uv/right_uv (N,2) in pixels and points_m (N,3)
        in the left optical frame. All returned rows passed geometry checks.
        """
        kl, dl = self.detector.detectAndCompute(left_gray, None)
        kr, dr = self.detector.detectAndCompute(right_gray, None)
        empty = dict(left_uv=np.empty((0,2)), right_uv=np.empty((0,2)), points_m=np.empty((0,3)))
        if dl is None or dr is None or min(len(dl), len(dr)) < 2:
            return empty
        def accepted(a, b):
            return {pair[0].queryIdx:pair[0].trainIdx for pair in self.matcher.knnMatch(a,b,k=2)
                    if len(pair)==2 and pair[0].distance < 0.75*pair[1].distance}
        forward, reverse = accepted(dl,dr), accepted(dr,dl)
        matches = [(i,j) for i,j in forward.items() if reverse.get(j)==i]
        if not matches:
            return empty
        left = np.array([kl[i].pt for i,j in matches])
        right = np.array([kr[j].pt for i,j in matches])
        points, valid = triangulate_matches(left,right,self.calibration,
                                            min_depth=self.min_depth,max_depth=self.max_depth)
        return dict(left_uv=left[valid],right_uv=right[valid],points_m=points[valid])

    def point_cloud(self, depth, stride=4):
        """Back-project depth into left-frame XYZ and return image sample indices.

        depth: HxW metres with NaN for unknown; stride: positive pixel sampling
        interval. Returns points (N,3) metres and rows/cols for looking up colors.
        """
        if stride < 1:
            raise ValueError('stride must be positive')
        rows, cols = np.indices(depth.shape)
        keep = np.isfinite(depth) & (rows % stride == 0) & (cols % stride == 0)
        z = depth[keep]; K = self.calibration.K
        xyz = np.column_stack(((cols[keep]-K[0,2])*z/K[0,0],
                               (rows[keep]-K[1,2])*z/K[1,1], z))
        return xyz, rows[keep], cols[keep]

    def preview(self, left, right, result):
        """Return a 2x2 BGR panel: left/right images, depth, epipolar matches."""
        depth = result['depth_m']; mask = result['valid']
        scale = np.nan_to_num(1.-depth/self.max_depth, nan=0.)
        heat = cv2.applyColorMap(np.uint8(np.clip(scale,0,1)*255),cv2.COLORMAP_TURBO)
        heat[~mask] = 0
        matches = left.copy()
        for l,r,p in zip(result['left_uv'],result['right_uv'],result['points_m']):
            u,v = np.rint(l).astype(int)
            cv2.circle(matches,(u,v),3,(50,220,50),-1)
        panels = [left.copy(),right.copy(),heat,matches]
        labels = ['LEFT','RIGHT',f'Depth Z: 0-{self.max_depth:g} m; black=unknown',
                  f'{len(result["points_m"])} triangulated features; valid {mask.mean():.0%}']
        for panel,label in zip(panels,labels):
            cv2.rectangle(panel,(0,0),(panel.shape[1],30),(20,20,20),-1)
            cv2.putText(panel,label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        return np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
