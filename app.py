"""
拉手智能替换工具 - 后端服务
核心原则：新拉手真实像素只做等比例缩放、旋转和透视，绝不变形。

接口：
  /remove-bg      -> 自动去除新拉手图片的白底/纯色背景，返回透明PNG
  /composite      -> 将新拉手合成到柜门原图的指定位置（支持多个位置）
  /swap-handle-openai  -> (可选高级) 使用 OpenAI gpt-image-1 生成式替换
  /swap-handle-gemini  -> (可选高级) 使用 Gemini 图像编辑

启动：
  pip install -r requirements.txt
  python app.py
"""

import os
import io
import json
import math
import base64
from flask import Flask, request, send_file, abort, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"))


# ---------------------------------------------------------------------------
# 背景去除：自动检测并移除纯色/白色背景
# ---------------------------------------------------------------------------

def auto_remove_background(img, tolerance=32, feather=2):
    """
    自动去除图片的纯色背景（numpy 加速）。
    策略：采样四角颜色，将与背景色接近的像素设为透明。
    """
    if img.mode == "RGBA":
        alpha = np.array(img.split()[3])
        corners = [alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]]
        if all(c < 10 for c in corners):
            return img
        rgb = img.convert("RGB")
    else:
        rgb = img.convert("RGB")
        img = rgb.convert("RGBA")

    arr = np.array(rgb)  # (h, w, 3)
    h, w = arr.shape[:2]

    # 采样四角各 5x5 区域
    def sample_corner(cx, cy):
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        region = arr[y0:y1, x0:x1].reshape(-1, 3)
        return region.mean(axis=0)

    bg_colors = [
        sample_corner(0, 0),
        sample_corner(w - 1, 0),
        sample_corner(0, h - 1),
        sample_corner(w - 1, h - 1),
    ]

    # 取最亮的角作为背景色参考
    bg_color = max(bg_colors, key=lambda c: sum(c))

    # 计算每个像素与背景色的差异
    diff = np.abs(arr - bg_color).max(axis=2)  # (h, w)

    # 创建 alpha mask
    alpha_mask = np.zeros((h, w), dtype=np.uint8)
    # 完全背景 -> 0
    alpha_mask[diff < tolerance] = 0
    # 羽化过渡区
    feather_zone = (diff >= tolerance) & (diff < tolerance + feather * 2)
    alpha_mask[feather_zone] = ((diff[feather_zone] - tolerance) / (feather * 2) * 255).astype(np.uint8)
    # 前景 -> 255
    alpha_mask[diff >= tolerance + feather * 2] = 255

    # 轻微模糊边缘
    alpha_img = Image.fromarray(alpha_mask, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather))

    result = img.convert("RGBA")
    result.putalpha(alpha_img)
    return result


# ---------------------------------------------------------------------------
# 裁剪透明边距，获取拉手实际内容区域
# ---------------------------------------------------------------------------

def trim_to_content(img):
    """裁剪图片的透明边距，只保留有内容的区域。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


# ---------------------------------------------------------------------------
# 接触阴影生成
# ---------------------------------------------------------------------------

def create_shadow(handle_img, blur_radius, opacity, offset_x, offset_y):
    """
    从拉手的 alpha 通道生成接触阴影（仅阴影，不含拉手本身）。
    阴影使用拉手的真实轮廓，而不是矩形阴影。
    """
    if handle_img.mode != "RGBA":
        handle_img = handle_img.convert("RGBA")

    alpha = np.array(handle_img.split()[3], dtype=np.float32)
    # 阴影 alpha = 拉手 alpha * opacity
    shadow_alpha = np.clip(alpha * opacity / 255.0, 0, 255).astype(np.uint8)

    shadow_img = Image.new("RGBA", handle_img.size, (0, 0, 0, 0))
    shadow_img.putalpha(Image.fromarray(shadow_alpha, mode="L"))

    # 模糊阴影
    if blur_radius > 0:
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return shadow_img


def erase_old_handle_area(base_img, box, feather=3):
    """
    擦除旧拉手区域：采样选区周围的柜门颜色，填充选区内部。
    这样旧拉手不会在新拉手下方透出来。
    """
    bx, by, bw, bh = box
    img_w, img_h = base_img.size

    # 扩展选区范围用于采样周围颜色
    margin = max(5, min(bw, bh) // 4)
    sample_x0 = max(0, bx - margin)
    sample_y0 = max(0, by - margin)
    sample_x1 = min(img_w, bx + bw + margin)
    sample_y1 = min(img_h, by + bh + margin)

    arr = np.array(base_img)  # (h, w, 4)

    # 采样选区外围环带的颜色（排除选区内部）
    ring_pixels = []
    # 上边
    if by - margin >= 0:
        ring_pixels.append(arr[max(0, by - margin):by, sample_x0:sample_x1].reshape(-1, 4))
    # 下边
    if by + bh + margin <= img_h:
        ring_pixels.append(arr[by + bh:min(img_h, by + bh + margin), sample_x0:sample_x1].reshape(-1, 4))
    # 左边
    if bx - margin >= 0:
        ring_pixels.append(arr[sample_y0:sample_y1, max(0, bx - margin):bx].reshape(-1, 4))
    # 右边
    if bx + bw + margin <= img_w:
        ring_pixels.append(arr[sample_y0:sample_y1, bx + bw:min(img_w, bx + bw + margin)].reshape(-1, 4))

    if not ring_pixels:
        return base_img

    ring = np.concatenate(ring_pixels, axis=0)
    # 取中位数颜色（比平均值更抗干扰）
    fill_color = np.median(ring, axis=0).astype(np.uint8)

    # 填充选区内部
    fill_x0 = max(0, bx)
    fill_y0 = max(0, by)
    fill_x1 = min(img_w, bx + bw)
    fill_y1 = min(img_h, by + bh)

    arr[fill_y0:fill_y1, fill_x0:fill_x1, :3] = fill_color[:3]
    arr[fill_y0:fill_y1, fill_x0:fill_x1, 3] = 255

    result = Image.fromarray(arr, mode="RGBA")

    # 对填充区域边缘做轻微模糊，使其与周围柜门融合
    if feather > 0:
        # 只模糊选区周围一圈
        blur_margin = feather + 2
        blur_x0 = max(0, fill_x0 - blur_margin)
        blur_y0 = max(0, fill_y0 - blur_margin)
        blur_x1 = min(img_w, fill_x1 + blur_margin)
        blur_y1 = min(img_h, fill_y1 + blur_margin)
        region = result.crop((blur_x0, blur_y0, blur_x1, blur_y1))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=feather))
        result.paste(blurred, (blur_x0, blur_y0))

    return result


# ---------------------------------------------------------------------------
# 边缘暗化：让拉手与柜门接触处有自然暗边
# ---------------------------------------------------------------------------

def add_contact_edge_darkening(base_img, handle_img, pos_x, pos_y, radius=2, darkness=30):
    """在拉手边缘下方添加一圈轻微暗边，增强安装真实感（numpy 加速）。"""
    if handle_img.mode != "RGBA":
        return base_img

    alpha = handle_img.split()[3]
    # 膨胀 alpha（向外扩展几像素）
    dilated = alpha.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))

    # 暗边 = 膨胀区域 - 原始区域
    dilated_arr = np.array(dilated, dtype=np.int16)
    alpha_arr = np.array(alpha, dtype=np.int16)
    edge_arr = np.clip(dilated_arr - alpha_arr, 0, 255).astype(np.uint8)

    # 模糊暗边
    edge = Image.fromarray(edge_arr, mode="L")
    edge = edge.filter(ImageFilter.GaussianBlur(radius=radius))

    # 创建暗色叠加并合成到 base_img
    img_w, img_h = base_img.size
    paste_x = max(0, pos_x)
    paste_y = max(0, pos_y)
    crop_x = max(0, -pos_x)
    crop_y = max(0, -pos_y)
    crop_w = min(edge.width - crop_x, img_w - paste_x)
    crop_h = min(edge.height - crop_y, img_h - paste_y)

    if crop_w <= 0 or crop_h <= 0:
        return base_img

    edge_crop = edge.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    dark_patch = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
    dark_patch.putalpha(edge_crop)

    # 应用暗度
    r, g, b, a = dark_patch.split()
    r = r.point(lambda v: max(0, v - darkness))
    g = g.point(lambda v: max(0, v - darkness))
    b = b.point(lambda v: max(0, v - darkness))
    dark_patch = Image.merge("RGBA", (r, g, b, a))
    base_img.paste(dark_patch, (paste_x, paste_y), dark_patch)

    return base_img


# ---------------------------------------------------------------------------
# 合成接口
# ---------------------------------------------------------------------------

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    """去除新拉手图片的背景，返回透明PNG。"""
    file = request.files.get("image")
    if not file:
        abort(400, "缺少图片文件")

    tolerance = int(request.form.get("tolerance", 32))
    feather = int(request.form.get("feather", 2))

    img = Image.open(file.stream)
    result = auto_remove_background(img, tolerance=tolerance, feather=feather)
    result = trim_to_content(result)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/composite", methods=["POST"])
def composite():
    """
    将新拉手合成到柜门原图的指定位置。

    输入：
      - original_image: 柜门原图
      - handle_image: 新拉手图片（会自动去背景）
      - boxes: JSON 格式的位置列表 [{"x":100,"y":200,"w":80,"h":120}, ...]
      - scale: 缩放因子（0-200，默认92，表示92%）
      - rotation: 旋转角度（默认0）
      - shadow_strength: 阴影强度（none/light/standard/strong）
      - light_direction: 光源方向（auto/left/right/top）
      - tolerance: 背景去除容差（默认32）
      - feather: 背景去除羽化（默认2）
      - edge_darkening: 是否启用边缘暗化（true/false）

    输出：合成后的 PNG 图片
    """
    orig_file = request.files.get("original_image")
    handle_file = request.files.get("handle_image")
    boxes_json = request.form.get("boxes", "[]")

    if not orig_file or not handle_file:
        abort(400, "缺少原图或新拉手图片")

    try:
        boxes = json.loads(boxes_json)
    except json.JSONDecodeError:
        abort(400, "boxes 格式错误")

    if not boxes or not isinstance(boxes, list):
        abort(400, "至少需要一个选区位置")

    # 参数
    scale_percent = float(request.form.get("scale", 92))
    rotation = float(request.form.get("rotation", 0))
    shadow_strength = request.form.get("shadow_strength", "standard")
    light_direction = request.form.get("light_direction", "auto")
    tolerance = int(request.form.get("tolerance", 32))
    feather = int(request.form.get("feather", 2))
    edge_darkening = request.form.get("edge_darkening", "true").lower() == "true"

    # 加载原图
    base_img = Image.open(orig_file.stream).convert("RGBA")

    # 加载并处理新拉手
    handle_raw = Image.open(handle_file.stream)
    handle_img = auto_remove_background(handle_raw, tolerance=tolerance, feather=feather)
    handle_img = trim_to_content(handle_img)

    handle_w, handle_h = handle_img.size

    if handle_w == 0 or handle_h == 0:
        abort(400, "新拉手图片处理后为空，请检查背景去除设置")

    # 阴影参数
    shadow_params = {
        "none": {"blur": 0, "opacity": 0, "offset_x": 0, "offset_y": 0},
        "light": {"blur": 3, "opacity": 60, "offset_x": 2, "offset_y": 3},
        "standard": {"blur": 5, "opacity": 90, "offset_x": 3, "offset_y": 5},
        "strong": {"blur": 8, "opacity": 120, "offset_x": 4, "offset_y": 8},
    }
    sp = shadow_params.get(shadow_strength, shadow_params["standard"])

    # 光源方向影响阴影偏移
    light_offsets = {
        "left": {"offset_x": 4, "offset_y": 2},
        "right": {"offset_x": -4, "offset_y": 2},
        "top": {"offset_x": 0, "offset_y": 5},
        "auto": {"offset_x": sp["offset_x"], "offset_y": sp["offset_y"]},
    }
    lo = light_offsets.get(light_direction, light_offsets["auto"])
    shadow_offset_x = lo["offset_x"]
    shadow_offset_y = lo["offset_y"]

    user_scale = scale_percent / 100.0

    # 逐个位置合成
    for box in boxes:
        bx = int(box.get("x", 0))
        by = int(box.get("y", 0))
        bw = int(box.get("w", 0))
        bh = int(box.get("h", 0))

        if bw <= 0 or bh <= 0:
            continue

        # 1. 擦除旧拉手区域（用周围柜门颜色填充）
        base_img = erase_old_handle_area(base_img, (bx, by, bw, bh), feather=3)

        # 2. 计算等比例缩放：取宽和高方向的最小比例，确保不变形
        scale_w = bw / handle_w
        scale_h = bh / handle_h
        uniform_scale = min(scale_w, scale_h) * user_scale

        new_w = max(1, int(handle_w * uniform_scale))
        new_h = max(1, int(handle_h * uniform_scale))

        # 等比例缩放（高质量重采样）
        resized = handle_img.resize((new_w, new_h), Image.LANCZOS)

        # 旋转
        if abs(rotation) > 0.5:
            resized = resized.rotate(rotation, expand=True, resample=Image.BICUBIC)

        final_w, final_h = resized.size

        # 计算安装位置：以选区中心为安装中心
        box_cx = bx + bw / 2
        box_cy = by + bh / 2
        paste_x = int(box_cx - final_w / 2)
        paste_y = int(box_cy - final_h / 2)

        # 3. 边缘暗化（在拉手下方）
        if edge_darkening and sp["opacity"] > 0:
            base_img = add_contact_edge_darkening(
                base_img, resized, paste_x, paste_y,
                radius=2, darkness=25
            )

        # 4. 生成并合成阴影（仅阴影，不含拉手）
        if sp["opacity"] > 0:
            shadow_img = create_shadow(
                resized,
                blur_radius=sp["blur"],
                opacity=sp["opacity"],
                offset_x=shadow_offset_x,
                offset_y=shadow_offset_y,
            )
            shadow_paste_x = paste_x + shadow_offset_x
            shadow_paste_y = paste_y + shadow_offset_y
            if shadow_paste_x < base_img.width and shadow_paste_y < base_img.height:
                temp = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                temp.paste(shadow_img, (shadow_paste_x, shadow_paste_y), shadow_img)
                base_img = Image.alpha_composite(base_img, temp)

        # 5. 合成新拉手
        temp2 = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        temp2.paste(resized, (paste_x, paste_y), resized)
        base_img = Image.alpha_composite(base_img, temp2)

    # 转为 RGB 输出
    output = base_img.convert("RGB")
    buf = io.BytesIO()
    output.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---------------------------------------------------------------------------
# 可选高级接口：AI 生成式替换（保留但非默认）
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = (
    "This is a photo of a wardrobe/cabinet door with a handle. "
    "Replace ONLY the handle inside the masked (transparent) region with a new handle. "
    "The new handle should match this description/reference: {extra}. "
    "Keep the cabinet door, wood texture, lighting, shadows, and camera angle exactly the same. "
    "Do not change anything outside the masked area."
)

GEMINI_PROMPT_TEMPLATE = (
    "You are given two images. The FIRST image is a wardrobe/cabinet door with an old handle. "
    "The SECOND image is a new replacement handle. "
    "Replace the old handle in the first image with the new handle shown in the second image, "
    "matching its shape, color and material as closely as possible. "
    "Keep everything else in the first image exactly unchanged."
)


@app.route("/swap-handle-openai", methods=["POST"])
def swap_handle_openai():
    """可选：使用 OpenAI gpt-image-1 的生成式替换。"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        abort(500, "OPENAI_API_KEY 未设置")

    original_file = request.files.get("original_image")
    mask_file = request.files.get("mask")
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
    """可选：使用 Gemini 图像编辑。"""
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
