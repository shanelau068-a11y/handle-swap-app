"""
拉手替换工具 - 后端服务
提供两个接口：
  /swap-handle-openai  -> 使用 OpenAI gpt-image-1 的 images.edit + mask 精确替换圈选区域
  /swap-handle-gemini  -> 使用 Gemini 2.5 Flash Image 多图对话式编辑（无需蒙版）

环境变量：
  OPENAI_API_KEY  - 使用 OpenAI 方案时必需
  GEMINI_API_KEY  - 使用 Gemini 方案时必需

启动：
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-xxxx
  export GEMINI_API_KEY=xxxx
  python app.py
  然后浏览器打开 index.html（或用 Flask 静态托管）
"""

import os
import io
import base64
from flask import Flask, request, send_file, abort
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

DEFAULT_PROMPT_TEMPLATE = (
    "This is a photo of a wardrobe/cabinet door with a handle. "
    "Replace ONLY the handle inside the masked (transparent) region with a new handle. "
    "The new handle should match this description/reference: {extra}. "
    "Keep the cabinet door, wood texture, lighting, shadows, and camera angle exactly the same. "
    "Do not change anything outside the masked area. Make the new handle look realistically "
    "mounted, with correct perspective, shadow, and screw holes if needed."
)

GEMINI_PROMPT_TEMPLATE = (
    "You are given two images. The FIRST image is a wardrobe/cabinet door with an old handle. "
    "The SECOND image is a new replacement handle. "
    "Replace the old handle in the first image with the new handle shown in the second image, "
    "matching its shape, color and material as closely as possible. "
    "Keep everything else in the first image (door, wood texture, lighting, shadows, camera angle) "
    "exactly unchanged. Make the new handle look realistically mounted with correct perspective and shadow. "
    "Extra instructions: {extra}"
)


@app.route("/swap-handle-openai", methods=["POST"])
def swap_handle_openai():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        abort(500, "OPENAI_API_KEY 未设置")

    original_file = request.files.get("original_image")
    mask_file = request.files.get("mask")
    new_handle_file = request.files.get("new_handle_image")
    extra_prompt = request.form.get("extra_prompt", "")

    if not original_file or not mask_file:
        abort(400, "缺少原图或蒙版")

    orig_img = Image.open(original_file.stream).convert("RGBA")
    orig_buf = io.BytesIO()
    orig_img.save(orig_buf, format="PNG")
    orig_buf.seek(0)

    mask_img = Image.open(mask_file.stream).convert("RGBA")
    if mask_img.size != orig_img.size:
        mask_img = mask_img.resize(orig_img.size)
    mask_buf = io.BytesIO()
    mask_img.save(mask_buf, format="PNG")
    mask_buf.seek(0)

    extra_desc = extra_prompt.strip() or "a new handle with a modern, clean style"
    prompt = DEFAULT_PROMPT_TEMPLATE.format(extra=extra_desc)

    client = OpenAI(api_key=api_key)
    result = client.images.edit(
        model="gpt-image-1",
        image=orig_buf,
        mask=mask_buf,
        prompt=prompt,
        size="1024x1024",
        quality="high",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    return send_file(io.BytesIO(image_bytes), mimetype="image/png")


@app.route("/swap-handle-gemini", methods=["POST"])
def swap_handle_gemini():
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        abort(500, "GEMINI_API_KEY 未设置")

    original_file = request.files.get("original_image")
    new_handle_file = request.files.get("new_handle_image")
    extra_prompt = request.form.get("extra_prompt", "")

    if not original_file or not new_handle_file:
        abort(400, "缺少原图或新拉手图片")

    orig_bytes = original_file.read()
    new_bytes = new_handle_file.read()

    extra_desc = extra_prompt.strip() or "none"
    prompt = GEMINI_PROMPT_TEMPLATE.format(extra=extra_desc)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Part.from_bytes(data=orig_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=new_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return send_file(io.BytesIO(part.inline_data.data), mimetype="image/png")

    abort(500, "Gemini 未返回图片，请检查 prompt 或重试")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
