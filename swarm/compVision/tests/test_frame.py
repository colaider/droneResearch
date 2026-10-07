"""Check preprocessing equivalence and isolation of display annotations."""

import unittest

import cv2
import numpy as np

from swarm.compVision.frame import Frame, FrameProcessor


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(13).integers(
            0, 256, size=(49, 65, 3), dtype=np.uint8
        )

    def test_default_preprocessing_preserves_existing_clahe_behavior(self):
        original = self.image.copy()
        processor = FrameProcessor()
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = processor.clahe.apply(lab[:, :, 0])
        expected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        np.testing.assert_array_equal(processor.preprocess(self.image), expected)
        np.testing.assert_array_equal(self.image, original)

    def test_filter_order_and_runtime_configuration(self):
        processor = FrameProcessor(image_filters=())
        processor.image_filters = ("gaussian", "median")
        processor.filter_kernel_size = 5
        expected = cv2.medianBlur(cv2.GaussianBlur(self.image, (5, 5), 0), 5)
        np.testing.assert_array_equal(processor.preprocess(self.image), expected)
        processor.image_filters = ("unknown",)
        with self.assertRaises(ValueError):
            processor.preprocess(self.image)
        processor.image_filters = ()
        processor.filter_kernel_size = 2
        with self.assertRaises(ValueError):
            processor.preprocess(self.image)

    def test_pyramids_and_display_images_do_not_corrupt_inputs(self):
        original = self.image.copy()
        prepared = FrameProcessor(image_filters=()).process([self.image, self.image], 7)
        self.assertEqual(prepared.idx, 7)
        self.assertEqual([p.shape for p in prepared.pyramids[0]], [(49, 65), (25, 33), (13, 17)])
        self.assertIs(prepared.pyramids[0][0], prepared.gray_frames[0])
        self.assertTrue(all(p.dtype == np.uint8 for p in prepared.pyramids[0]))
        display = prepared.display_copy()
        display.frames[0][:] = 0
        np.testing.assert_array_equal(prepared.frames[0], original)
        self.assertIs(display.gray_frames, prepared.gray_frames)
        self.assertIs(display.pyramids, prepared.pyramids)
        prepared.frames[1][:] = 0
        np.testing.assert_array_equal(self.image, original)

    def test_invalid_stereo_inputs(self):
        processor = FrameProcessor()
        for frames in (
            [],
            [self.image],
            [self.image, self.image, self.image],
            [self.image, self.image[:-1]],
            [self.image, self.image[:, :, 0]],
            [self.image, self.image.astype(np.float32)],
            [self.image, self.image[:0]],
        ):
            with self.subTest(shapes=[getattr(f, "shape", None) for f in frames]):
                with self.assertRaises(ValueError):
                    processor.process(frames, 0)

    def test_empty_frames_have_independent_defaults(self):
        first, second = Frame(), Frame()
        first.frames.append(self.image)
        self.assertEqual(second.frames, [])
        self.assertEqual(second.idx, 0)


if __name__ == "__main__":
    unittest.main()
