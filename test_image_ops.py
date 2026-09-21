"""Regression tests for deterministic handle compositing.

Run with: python -m unittest test_image_ops.py
"""

import io
import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

from app import app, auto_remove_background, erase_old_handle_area


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class ImageOperationTests(unittest.TestCase):
    def test_white_background_does_not_remove_internal_highlight(self):
        image = Image.new("RGB", (120, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 99, 99), fill=(180, 180, 180))
        draw.rectangle((40, 40, 79, 79), fill=(245, 245, 245))

        result = auto_remove_background(image, tolerance=32, feather=1)

        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertGreater(result.getpixel((60, 60))[3], 200)

    def test_erasing_stays_inside_selected_box(self):
        pixels = np.full((160, 200, 4), 200, dtype=np.uint8)
        pixels[:, 98:102, :3] = 40  # a dark door seam
        pixels[55:105, 80:120, :3] = (40, 120, 220)  # the old handle
        pixels[:, :, 3] = 255
        original = pixels.copy()

        result = np.asarray(erase_old_handle_area(Image.fromarray(pixels), (60, 30, 80, 100)))

        self.assertTrue(np.array_equal(original[:30], result[:30]))
        self.assertTrue(np.array_equal(original[130:], result[130:]))
        self.assertTrue(np.array_equal(original[:, :60], result[:, :60]))
        self.assertTrue(np.array_equal(original[:, 140:], result[:, 140:]))
        self.assertLess(result[80, 100, 2], 210)

    def test_rotated_product_stays_inside_box(self):
        base = Image.new("RGBA", (240, 240), (230, 230, 230, 255))
        handle = Image.new("RGBA", (30, 110), (0, 0, 0, 0))
        ImageDraw.Draw(handle).rectangle((5, 5, 24, 104), fill=(255, 0, 0, 255))
        box = {"x": 100, "y": 60, "w": 40, "h": 120}

        with app.test_client() as client:
            response = client.post(
                "/composite",
                data={
                    "original_image": (io.BytesIO(png_bytes(base)), "base.png"),
                    "handle_image": (io.BytesIO(png_bytes(handle)), "handle.png"),
                    "boxes": json.dumps([box]),
                    "scale": "100",
                    "rotation": "45",
                    "shadow_strength": "none",
                    "edge_darkening": "false",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        result = np.asarray(Image.open(io.BytesIO(response.data)).convert("RGB"))
        red = (result[:, :, 0] > 200) & (result[:, :, 1] < 60) & (result[:, :, 2] < 60)
        ys, xs = np.where(red)
        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(xs.min(), box["x"])
        self.assertLessEqual(xs.max() + 1, box["x"] + box["w"])
        self.assertGreaterEqual(ys.min(), box["y"])
        self.assertLessEqual(ys.max() + 1, box["y"] + box["h"])

    def test_wide_rotated_product_stays_inside_box_after_pixel_rounding(self):
        base = Image.new("RGBA", (260, 220), (230, 230, 230, 255))
        handle = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
        ImageDraw.Draw(handle).rounded_rectangle((2, 2, 197, 57), radius=10, fill=(255, 0, 0, 255))
        box = {"x": 70, "y": 50, "w": 101, "h": 101}

        with app.test_client() as client:
            response = client.post(
                "/composite",
                data={
                    "original_image": (io.BytesIO(png_bytes(base)), "base.png"),
                    "handle_image": (io.BytesIO(png_bytes(handle)), "handle.png"),
                    "boxes": json.dumps([box]),
                    "scale": "100",
                    "rotation": "45",
                    "shadow_strength": "none",
                    "edge_darkening": "false",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        result = np.asarray(Image.open(io.BytesIO(response.data)).convert("RGB"))
        red = (result[:, :, 0] > 200) & (result[:, :, 1] < 60) & (result[:, :, 2] < 60)
        ys, xs = np.where(red)
        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(xs.min(), box["x"])
        self.assertLessEqual(xs.max() + 1, box["x"] + box["w"])
        self.assertGreaterEqual(ys.min(), box["y"])
        self.assertLessEqual(ys.max() + 1, box["y"] + box["h"])


if __name__ == "__main__":
    unittest.main()
