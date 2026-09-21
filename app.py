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
    自动去除从图片边缘连通的纯色背景。

    只从四角做颜色容差洪水填充，因此产品内部的白色珍珠、高光和
    浅色细节不会因为“接近白色”而被整片抠除。
    """
    result = img.convert("RGBA")
    rgba = np.array(result)
    rgb = result.convert("RGB")
    h, w = rgba.shape[:2]
    original_alpha = rgba[:, :, 3].copy()
    if h == 0 or w == 0:
        return result

    # Pillow 在 C 层执行洪水填充，比 Python 逐像素 BFS 更适合大图。
    # 使用一个不太可能出现在产品中的标记色来读回每个连通区域。
    marker = (1, 2, 3)
    remove_mask = np.zeros((h, w), dtype=bool)
    seeds = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
    for sx, sy in seeds:
        if original_alpha[sy, sx] < 10:
            continue
        work = rgb.copy()
        ImageDraw.floodfill(work, (sx, sy), marker, thresh=max(0, int(tolerance)))
        remove_mask |= np.all(np.asarray(work) == marker, axis=2)

    # 原本透明的像素始终透明；产品内部没有从边缘连通的白色区域不会被删除。
    mask = np.where(remove_mask, 0, 255).astype(np.uint8)
    if feather > 0:
        # 只在前景边界做很窄的过渡，避免产生明显白边。
        mask = np.asarray(
            Image.fromarray(mask, mode="L").filter(
                ImageFilter.GaussianBlur(radius=max(0.5, float(feather)))
            )
        )
    alpha = (original_alpha.astype(np.float32) * mask.astype(np.float32) / 255.0).astype(np.uint8)
    result.putalpha(Image.fromarray(alpha, mode="L"))
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


def alpha_composite_clipped(base_img, overlay, x, y):
    """将 RGBA 图层放到 base 上，自动裁剪越过画布边界的部分。"""
    x, y = int(x), int(y)
    src_x0, src_y0 = max(0, -x), max(0, -y)
    src_x1 = min(overlay.width, base_img.width - x)
    src_y1 = min(overlay.height, base_img.height - y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return base_img
    crop = overlay.crop((src_x0, src_y0, src_x1, src_y1))
    base_img.alpha_composite(crop, (max(0, x), max(0, y)))
    return base_img


def erase_old_handle_area(base_img, box, feather=3):
    """
    擦除旧拉手区域：采样选区周围的柜门颜色，填充选区内部。
    这样旧拉手不会在新拉手下方透出来。
    """
    bx, by, bw, bh = [int(v) for v in box]
    img_w, img_h = base_img.size
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(img_w, bx + max(0, bw)), min(img_h, by + max(0, bh))
    if x1 <= x0 or y1 <= y0:
        return base_img

    # 仅在选区内部生成填充，选区外的像素完全不改。左右、上下边界的
    # 插值能保留柜门的渐变，比整块使用一个中位数颜色更不容易留下色块。
    arr = np.array(base_img).copy()
    original = arr.copy()
    margin = max(2, min(8, int(min(x1 - x0, y1 - y0) * 0.08)))
    left_x, right_x = max(0, x0 - margin), min(img_w - 1, x1 + margin - 1)
    top_y, bottom_y = max(0, y0 - margin), min(img_h - 1, y1 + margin - 1)
    width, height = x1 - x0, y1 - y0
    tx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    ty = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    left = original[y0:y1, left_x:left_x + 1, :3].astype(np.float32)
    right = original[y0:y1, right_x:right_x + 1, :3].astype(np.float32)
    top = original[top_y:top_y + 1, x0:x1, :3].astype(np.float32)
    bottom = original[bottom_y:bottom_y + 1, x0:x1, :3].astype(np.float32)
    horizontal = left * (1.0 - tx) + right * tx
    vertical = top * (1.0 - ty) + bottom * ty
    fill = np.clip((horizontal + vertical) * 0.5, 0, 255).astype(np.uint8)

    arr[y0:y1, x0:x1, :3] = fill
    arr[y0:y1, x0:x1, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


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
    # 暗度作用在 alpha，而不是把透明区域的 RGB 变黑。使用 alpha_composite
    # 避免 paste(..., mask=同一张 RGBA 图) 造成 alpha 被重复相乘。
    edge_alpha = edge_crop.point(lambda v: int(v * max(0, min(255, darkness)) / 255))
    dark_patch = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
    dark_patch.putalpha(edge_alpha)
    return alpha_composite_clipped(base_img, dark_patch, paste_x, paste_y)


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
    # 100% 是选区内允许的最大尺寸；超过 100 的旧参数仍被安全截断。
    scale_percent = max(1.0, min(100.0, float(request.form.get("scale", 92))))
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

        # 2. 先计算旋转后的包围盒，再按最小比例缩放。这样 45°、90°
        # 等角度也不会因为 expand=True 把产品推出原选区。
        angle = math.radians(rotation % 180.0)
        cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
        rotated_w = handle_w * cos_a + handle_h * sin_a
        rotated_h = handle_w * sin_a + handle_h * cos_a
        uniform_scale = min(bw / max(1.0, rotated_w), bh / max(1.0, rotated_h)) * user_scale

        new_w = max(1, int(handle_w * uniform_scale))
        new_h = max(1, int(handle_h * uniform_scale))

        # 等比例缩放（高质量重采样）
        resized = handle_img.resize((new_w, new_h), Image.LANCZOS)

        # 旋转
        if abs(rotation) > 0.5:
            resized = resized.rotate(rotation, expand=True, resample=Image.BICUBIC)

        # Pillow 的 expand=True 会按像素取整，实际包围盒可能比上面的
        # 三角函数估算多 1–2px。用旋转后的真实尺寸再做一次等比例收缩，
        # 避免横长/窄长拉手在 30°/45° 等角度越过选区或被边缘裁掉。
        actual_fit = min(
            1.0,
            bw / max(1, resized.width),
            bh / max(1, resized.height),
        )
        if actual_fit < 1.0:
            resized = resized.resize(
                (
                    max(1, int(resized.width * actual_fit)),
                    max(1, int(resized.height * actual_fit)),
                ),
                Image.LANCZOS,
            )

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
                # alpha_composite 只使用一次 alpha，保留正确的阴影强度。
                base_img = alpha_composite_clipped(base_img, shadow_img, shadow_paste_x, shadow_paste_y)

        # 5. 合成新拉手
        base_img = alpha_composite_clipped(base_img, resized, paste_x, paste_y)

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
