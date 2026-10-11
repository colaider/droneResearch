"""scripts/calibrate_stereo.py on checkerboards rendered through a known stereo rig."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import calibrate_stereo as calib_script   # noqa: E402

from swarm.compVision.denseStereo import StereoRectifier   # noqa: E402

W, H = 960, 600
K1 = np.array([[700.0, 0, 470], [0, 700, 310], [0, 0, 1]])
K2 = np.array([[600.0, 0, 490], [0, 600, 290], [0, 0, 1]])   # different focal, like the USB pair
R = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
T = np.array([-0.125, 0.002, 0.001])
COLS, ROWS, SQUARE, PX = 9, 6, 0.03, 40


def render_pairs(n=15, seed=0):
    tex = np.full(((ROWS + 3) * PX, (COLS + 3) * PX), 255, np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                tex[(r + 1) * PX:(r + 2) * PX, (c + 1) * PX:(c + 2) * PX] = 0
    # Texture pixels -> board metres, origin at the first inner corner (2 squares in).
    to_board = np.array([[SQUARE / PX, 0, -2 * SQUARE], [0, SQUARE / PX, -2 * SQUARE], [0, 0, 1]])
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        rb = cv2.Rodrigues(rng.uniform(-0.35, 0.35, 3))[0]
        tb = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.02), rng.uniform(0.6, 0.9)])
        images = []
        for K, Rc, tc in ((K1, rb, tb), (K2, R @ rb, R @ tb + T)):
            homography = K @ np.column_stack([Rc[:, 0], Rc[:, 1], tc]) @ to_board
            image = cv2.warpPerspective(tex, homography, (W, H), borderValue=128)
            images.append(cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0.7), cv2.COLOR_GRAY2BGR))
        pairs.append(tuple(images))
    return pairs


class CalibrateStereoTests(unittest.TestCase):
    def test_recovers_rig_and_rectifies_rows(self):
        calib = calib_script.calibrate(render_pairs(), COLS, ROWS, SQUARE)
        np.testing.assert_allclose(calib['T'].ravel(), T, atol=0.005)
        np.testing.assert_allclose(calib['K1'][0, 0], 700, rtol=0.01)
        np.testing.assert_allclose(calib['K2'][0, 0], 600, rtol=0.01)
        rect, err = calib_script.epipolar_error(calib)
        self.assertLess(err, 0.3)
        self.assertAlmostEqual(rect.baseline, 0.125, delta=0.002)
        np.testing.assert_allclose(rect.disparity_direction, [-1, 0], atol=0.01)

    def test_cli_from_image_folder_writes_loadable_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, (left, right) in enumerate(render_pairs(10, seed=1)):
                cv2.imwrite(f'{tmp}/left_{i:02d}.png', left)
                cv2.imwrite(f'{tmp}/right_{i:02d}.png', right)
            out = f'{tmp}/calib.npz'
            args = calib_script.parse_args(['--images', tmp, '--board', f'{COLS}x{ROWS}',
                                            '--square', str(SQUARE), '--out', out])
            self.assertEqual(calib_script.main(args), 0)
            rect = StereoRectifier.from_npz(out)
        self.assertEqual(rect.size, (W, H))
        self.assertAlmostEqual(rect.baseline, 0.125, delta=0.003)

    def test_too_few_boards_rejected(self):
        blank = np.full((H, W, 3), 128, np.uint8)
        with self.assertRaises(ValueError):
            calib_script.calibrate([(blank, blank)] * 3, COLS, ROWS, SQUARE)


if __name__ == '__main__':
    unittest.main()
