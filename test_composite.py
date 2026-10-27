"""Portable HTTP integration checks with synthetic images; no running server required."""
import io
import json
import unittest

from PIL import Image, ImageDraw
from app import app


def upload(image):
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)
    return stream, "test.png"


class CompositeApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.cabinet = Image.new("RGBA", (800, 600), (245, 242, 238, 255))
        draw = ImageDraw.Draw(self.cabinet)
        draw.line([(400, 0), (400, 600)], fill=(200, 197, 192, 255), width=3)
        draw.rounded_rectangle((120, 250, 200, 350), 15, (120, 180, 100, 255))
        self.handle = Image.new("RGBA", (300, 200), "white")
        ImageDraw.Draw(self.handle).rounded_rectangle((50, 70, 250, 130), 12, (180, 185, 190, 255))

    def test_remove_background_returns_transparent_png(self):
        response = self.client.post("/remove-bg", data={"image": upload(self.handle)})
        self.assertEqual(response.status_code, 200)
        result = Image.open(io.BytesIO(response.data)).convert("RGBA")
        # The endpoint deliberately crops transparent margins around the handle.
        self.assertLess(result.width, self.handle.width)
        self.assertLess(result.height, self.handle.height)
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertGreater(result.getpixel((result.width // 2, result.height // 2))[3], 200)

    def test_composite_preserves_canvas_and_door_seam(self):
        response = self.client.post("/composite", data={
            "original_image": upload(self.cabinet), "handle_image": upload(self.handle),
            "boxes": json.dumps([{"x": 120, "y": 250, "w": 80, "h": 100}]),
            "scale": "92", "rotation": "0", "shadow_strength": "none",
        })
        self.assertEqual(response.status_code, 200)
        result = Image.open(io.BytesIO(response.data)).convert("RGBA")
        self.assertEqual(result.size, self.cabinet.size)
        self.assertEqual(result.getpixel((400, 300)), self.cabinet.getpixel((400, 300)))


if __name__ == "__main__":
    unittest.main()
