from PIL import Image, ImageDraw, ImageFont

# 获取用户输入的图片宽度和高度
width = 3000
height = 1622

# 背景颜色和字体设置
background_color = (255, 255, 255)  # 白色背景
text_color = (0, 0, 0)              # 黑色文字
font_size = 800                     # 字体大小
margin_bottom = 200                 # 文字离下边缘的距离

# 创建图片对象
image = Image.new('RGB', (width, height), background_color)
draw = ImageDraw.Draw(image)

# 选择字体和大小
try:
    font = ImageFont.truetype("Arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

# 设置文本内容
text = "Unit 2"

# 计算文字的尺寸和位置
text_bbox = draw.textbbox((0, 0), text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]

# 计算文字的位置，使其水平居中并离下边缘有一定的距离
text_x = (width - text_width) / 2
text_y = height - text_height - margin_bottom

# 确保文字不超出图片边界
if text_y < 0:
    text_y = 0

# 绘制文字到图片上
draw.text((text_x, text_y), text, fill=text_color, font=font)

# 保存图片
image.save("fire.jpg")

print(f"图片已生成并保存为 'unit1_image_custom_size.png'，尺寸为 {width}x{height}。")

