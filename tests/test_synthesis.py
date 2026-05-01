"""
Basic sanity tests for synthesizer_app.core.synthesis.synthesize_one.

Run with the glass_env Python from the GLASS/ directory:

    "C:/Users/seong/anaconda3/envs/glass_env/python.exe" -m unittest \
        synthesizer_app.tests.test_synthesis -v
"""
import os
import sys
import unittest

import imgaug
import numpy as np
import PIL.Image
import torch


def _seed_all(seed):
    """Seed every RNG that LAS touches: np, torch, and imgaug (used by perlin.py)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    imgaug.seed(seed)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLASS = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _GLASS not in sys.path:
    sys.path.insert(0, _GLASS)

from synthesizer_app.core.synthesis import SynthParams, synthesize_one  # noqa: E402


def _solid_image(color, size=320):
    return PIL.Image.fromarray(np.full((size, size, 3), color, dtype=np.uint8))


class SynthesisShapeTest(unittest.TestCase):

    def test_output_shape_preserves_original(self):
        image = _solid_image((128, 128, 128), size=320)
        texture = _solid_image((200, 50, 50), size=180)
        params = SynthParams(working_size=288, output_size=None,
                             beta_mean=0.5, beta_std=0.0, rand_aug=False)

        _seed_all(0)
        result = synthesize_one(image, texture, params)

        self.assertEqual(result.ng_image_bgr.shape, (320, 320, 3))
        self.assertEqual(result.mask_uint8.shape, (320, 320))
        self.assertEqual(result.mask_uint8.dtype, np.uint8)
        self.assertTrue(set(np.unique(result.mask_uint8).tolist()).issubset({0, 255}))

    def test_output_shape_forced(self):
        image = _solid_image((128, 128, 128), size=320)
        texture = _solid_image((200, 50, 50), size=180)
        params = SynthParams(working_size=288, output_size=288, rand_aug=False)

        _seed_all(0)
        result = synthesize_one(image, texture, params)

        self.assertEqual(result.ng_image_bgr.shape, (288, 288, 3))
        self.assertEqual(result.mask_uint8.shape, (288, 288))


class SynthesisDeterminismTest(unittest.TestCase):

    def test_same_seed_same_output(self):
        image = _solid_image((128, 128, 128), size=320)
        texture = _solid_image((200, 50, 50), size=320)
        params = SynthParams(working_size=288, output_size=288, rand_aug=True)

        _seed_all(42)
        a = synthesize_one(image, texture, params)

        _seed_all(42)
        b = synthesize_one(image, texture, params)

        np.testing.assert_array_equal(a.ng_image_bgr, b.ng_image_bgr)
        np.testing.assert_array_equal(a.mask_uint8, b.mask_uint8)
        self.assertAlmostEqual(a.beta_used, b.beta_used)

    def test_different_seed_different_output(self):
        image = _solid_image((128, 128, 128), size=320)
        texture = _solid_image((200, 50, 50), size=320)
        params = SynthParams(working_size=288, output_size=288, rand_aug=True)

        _seed_all(1)
        a = synthesize_one(image, texture, params)

        _seed_all(2)
        b = synthesize_one(image, texture, params)

        # Either the mask differs or the texture aug differs; at least one of
        # the two outputs must change for unequal seeds.
        same_image = np.array_equal(a.ng_image_bgr, b.ng_image_bgr)
        same_mask = np.array_equal(a.mask_uint8, b.mask_uint8)
        self.assertFalse(same_image and same_mask)


class SynthesisFgValidationTest(unittest.TestCase):

    def test_use_foreground_without_mask_raises(self):
        image = _solid_image((128, 128, 128))
        texture = _solid_image((200, 50, 50))
        params = SynthParams(use_foreground=True)

        _seed_all(0)
        with self.assertRaises(ValueError):
            synthesize_one(image, texture, params, fg_mask=None)


if __name__ == "__main__":
    unittest.main()
