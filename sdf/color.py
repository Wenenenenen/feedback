"""颜色支持：颜色解析与颜色函数组合。

颜色函数约定：接收 (N, 2) 或 (N, 3) 的点数组，返回 (N, 3) 的 float RGB（0-1）。
返回值为 NaN 表示该点未上色，导出时会替换为默认颜色。
"""

import numpy as np

# 未上色部分的默认颜色（浅灰）
DEFAULT_COLOR = (0.8, 0.8, 0.8)


def to_rgb(c):
    """把颜色规格解析为 (3,) 的 0-1 float RGB。

    支持：'#e63946' 等十六进制字符串、'yellow' 等颜色名称、
    0-255 的 RGB 三元组、0-1 的 RGB 三元组。
    """
    if isinstance(c, str):
        from matplotlib.colors import to_rgb as _to_rgb
        return np.array(_to_rgb(c), dtype=float)
    a = np.asarray(c, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError('颜色必须是 3 个分量的 RGB')
    if np.nanmax(np.abs(a)) > 1:
        a = a / 255.0
    return a


def as_color_func(c):
    """把颜色规格或自定义函数转成统一的颜色函数。

    可调用对象接收局部坐标点数组 p，需返回 (N, 3) 的 RGB，
    可用于编写程序化花纹（条纹、渐变等）。
    """
    if callable(c):
        def f(p):
            a = np.asarray(c(p), dtype=float)
            return a.reshape((-1, 3))
        return f
    rgb = to_rgb(c)

    def f(p):
        return np.tile(rgb, (len(p), 1))
    return f


def child_colors(sdf, p):
    """采样子对象的颜色；未上色的子对象返回全 NaN。"""
    fn = getattr(sdf, '_color', None)
    if fn is None:
        return np.full((len(p), 3), np.nan)
    return np.asarray(fn(p), dtype=float).reshape((-1, 3))


def has_colors(sdfs):
    """任一子对象已上色时返回 True。"""
    return any(getattr(s, '_color', None) is not None for s in sdfs)


def sample_colors(sdf, p, default_color=None):
    """采样颜色，并把未上色（NaN）的点替换为默认颜色。"""
    fn = getattr(sdf, '_color', None)
    if fn is None:
        colors = np.full((len(p), 3), np.nan)
    else:
        colors = np.asarray(fn(p), dtype=float).reshape((-1, 3))
    dc = to_rgb(DEFAULT_COLOR if default_color is None else default_color)
    bad = ~np.isfinite(colors).all(axis=1)
    colors[bad] = dc
    return np.clip(colors, 0, 1)


def _fill_nan(a, b):
    """a 中的 NaN 用 b 对应位置填充。"""
    return np.where(np.isnan(a), b, a)


def mix(c1, c2, t):
    """按 t 混合两组颜色（t=0 取 c1，t=1 取 c2），NaN 用另一侧填充。"""
    c1f = _fill_nan(c1, c2)
    c2f = _fill_nan(c2, c1)
    return c2f * t + c1f * (1 - t)


def smooth_mix(c1, c2, h):
    """平滑布尔运算的颜色混合（h=1 取 c1，h=0 取 c2）。

    过渡区内一侧未上色时用另一侧颜色填充；过渡区外保持原值（含 NaN），
    保证未上色区域不会被远处的颜色"污染"。
    """
    c = mix(c1, c2, 1 - h)
    c = np.where(h <= 0, c2, c)
    c = np.where(h >= 1, c1, c)
    return c


def boolean_color(kind, a, bs, k=None):
    """构造布尔运算的颜色函数，距离折叠规则与 dn.py 完全一致。"""
    def f(p):
        d1 = a(p)
        c1 = child_colors(a, p)
        for b in bs:
            d2 = b(p)
            c2 = child_colors(b, p)
            K = k or getattr(b, '_k', None)
            if kind == 'union':
                if K is None:
                    # 硬并集：逐点取距离较小者的颜色
                    w = (d1 <= d2).reshape((-1, 1))
                    c1 = np.where(w, c1, c2)
                    d1 = np.minimum(d1, d2)
                else:
                    h = np.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                    c1 = smooth_mix(c1, c2, h)
                    m = d2 + (d1 - d2) * h
                    d1 = m - K * h * (1 - h)
            elif kind == 'intersection':
                # 交集：胜出者无颜色时回落到另一方的颜色
                c2e = _fill_nan(c2, c1)
                c1e = _fill_nan(c1, c2)
                if K is None:
                    w = (d1 >= d2).reshape((-1, 1))
                    c1 = np.where(w, c1e, c2e)
                    d1 = np.maximum(d1, d2)
                else:
                    h = np.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                    c1 = smooth_mix(c1e, c2e, h)
                    m = d2 + (d1 - d2) * h
                    d1 = m + K * h * (1 - h)
            elif kind == 'difference':
                # 差集：洞壁取刀具颜色；刀具未上色时沿用被切物体的颜色
                c2e = _fill_nan(c2, c1)
                if K is None:
                    w = (d1 >= -d2).reshape((-1, 1))
                    c1 = np.where(w, c1, c2e)
                    d1 = np.maximum(d1, -d2)
                else:
                    h = np.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                    c1 = smooth_mix(c2e, c1, h)
                    m = d1 + (-d2 - d1) * h
                    d1 = m + K * h * (1 - h)
        return c1
    return f


def blend_color(a, bs, k):
    """blend 操作的颜色函数：使用与距离相同的权重混合。"""
    def f(p):
        c1 = child_colors(a, p)
        for b in bs:
            c2 = child_colors(b, p)
            K = k or getattr(b, '_k', None)
            c1 = mix(c1, c2, K)
        return c1
    return f


def transformed(other, transform):
    """变换类操作的颜色函数：在变换后的局部坐标处采样子对象颜色。

    子对象未上色时返回 None。
    """
    fn = getattr(other, '_color', None)
    if fn is None:
        return None

    def f(p):
        return fn(transform(p))
    return f
