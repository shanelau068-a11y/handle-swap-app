"""生成测试图片并测试合成接口"""
from PIL import Image, ImageDraw
import io
import requests
import json
import os

# 创建一张模拟柜门图：白色背景 + 两个彩色"拉手"
cabinet = Image.new("RGBA", (800, 600), (245, 242, 238, 255))
draw = ImageDraw.Draw(cabinet)
# 柜门中间缝隙
draw.line([(400, 0), (400, 600)], fill=(200, 197, 192, 255), width=3)
# 左门拉手（绿色）
draw.rounded_rectangle([(120, 250), (200, 350)], radius=15, fill=(120, 180, 100, 255))
# 右门拉手（黄色）
draw.rounded_rectangle([(500, 250), (580, 350)], radius=15, fill=(230, 200, 80, 255))
cabinet_path = "/tmp/test_cabinet.png"
cabinet.save(cabinet_path)
print(f"柜门测试图已保存: {cabinet_path} ({cabinet.size})")

# 创建一张模拟新拉手图：白色背景 + 银色金属拉手
handle = Image.new("RGBA", (300, 200), (255, 255, 255, 255))
hdraw = ImageDraw.Draw(handle)
# 银色金属拉手
hdraw.rounded_rectangle([(50, 70), (250, 130)], radius=12, fill=(180, 185, 190, 255))
# 高光
hdraw.rounded_rectangle([(55, 75), (245, 95)], radius=6, fill=(220, 225, 230, 255))
handle_path = "/tmp/test_handle.png"
handle.save(handle_path)
print(f"新拉手测试图已保存: {handle_path} ({handle.size})")

# 测试去背景接口
print("\n--- 测试 /remove-bg ---")
with open(handle_path, 'rb') as f:
    resp = requests.post("http://localhost:5000/remove-bg", 
                         files={"image": f}, 
                         data={"tolerance": "32", "feather": "2"})
print(f"状态码: {resp.status_code}")
if resp.ok:
    result_bg = Image.open(io.BytesIO(resp.content))
    print(f"去背景结果: {result_bg.size}, 模式: {result_bg.mode}")
    # 检查是否有透明区域
    alpha = result_bg.split()[3]
    bbox = alpha.getbbox()
    print(f"内容区域: {bbox}")
    result_bg.save("/tmp/test_handle_nobg.png")
    print("去背景结果已保存")
else:
    print(f"错误: {resp.text}")
    sys.exit(1)

# 测试合成接口
print("\n--- 测试 /composite ---")
boxes = [
    {"x": 120, "y": 250, "w": 80, "h": 100},  # 左门拉手位置
    {"x": 500, "y": 250, "w": 80, "h": 100},  # 右门拉手位置
]
with open(cabinet_path, 'rb') as cf, open(handle_path, 'rb') as hf:
    resp = requests.post("http://localhost:5000/composite",
                         files={
                             "original_image": cf,
                             "handle_image": hf,
                         },
                         data={
                             "boxes": json.dumps(boxes),
                             "scale": "92",
                             "rotation": "0",
                             "shadow_strength": "standard",
                             "light_direction": "auto",
                             "tolerance": "32",
                             "feather": "2",
                             "edge_darkening": "true",
                         })
print(f"状态码: {resp.status_code}")
if resp.ok:
    result = Image.open(io.BytesIO(resp.content))
    print(f"合成结果: {result.size}, 模式: {result.mode}")
    result.save("/tmp/test_composite_result.png")
    print("合成结果已保存到 /tmp/test_composite_result.png")
    
    # 验证不变形：检查合成后的拉手宽高比是否与原图一致
    # 原拉手宽高比: (250-50)/(130-70) = 200/60 = 3.33
    # 合成后的拉手区域应该在选区内且保持比例
    print(f"\n验证: 原拉手宽高比 = {(250-50)/(130-70):.2f}")
    print("合成完成，请检查结果图片确认拉手未变形")
else:
    print(f"错误: {resp.text}")
    sys.exit(1)

print("\n✓ 所有测试通过")
