"""多色建模示例：平滑融合渐变、挖孔洞壁上色、扭转条纹、图片贴色。

运行后生成 colored.ply（顶点色，用于预览）和 colored.3mf（材质，用于切片）。
"""

import numpy as np

from sdf import *

IMAGE = 'examples/butterfly.png'

# 1. 平滑融合渐变：红球与蓝盒，k 过渡带的颜色自然混合
f = sphere(1).color('#e63946') | box(1.4).color((40, 120, 200)).translate((0.9, 0, 0)).k(0.35)

# 2. 挖孔：洞壁带上刀具的黄色（刀具不上色时会沿用基体颜色）
f -= cylinder(0.3).orient(X).color('yellow')

# 3. 扭转条纹：颜色函数按局部坐标出条纹，跟随 twist 一起扭
def stripes(p):
    w = (np.sin(p[:, 2] * 6) > 0).reshape((-1, 1))
    return np.where(w, (1.0, 1.0, 1.0), (0.1, 0.4, 0.9))

bar = box((0.7, 0.7, 2.5)).color(stripes).twist(0.8).translate((0, 2, 0))
f |= bar

# 4. 图片贴色铭牌：image() 自动携带图片本身的颜色
w, h = measure_image(IMAGE)
plate = rounded_box((w * 1.1, h * 1.1, 0.1), 0.05).color((60, 60, 60))
logo = image(IMAGE).extrude(1) & slab(z0=0, z1=0.075)
plate = (plate | logo).translate((0, -2, 0))
f |= plate

# 未上色的部分会使用默认颜色，可用 default_color 指定
f.save('colored.ply')
f.save('colored.3mf')
