"""颜色功能测试：颜色语义、变换跟随、导出、兼容性与多线程一致性。"""

import hashlib
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sdf import *
from sdf import d3, d2
from sdf.color import to_rgb, DEFAULT_COLOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

RED = (1, 0, 0)
BLUE = (0, 0, 1)


# ---------------------------------------------------------------------------
# 颜色解析
# ---------------------------------------------------------------------------

def test_parse_hex():
    assert np.allclose(to_rgb('#e63946'), (0.90196078, 0.22352941, 0.2745098))


def test_parse_name():
    assert np.allclose(to_rgb('yellow'), (1, 1, 0))


def test_parse_rgb_255():
    assert np.allclose(to_rgb((40, 120, 200)), (40 / 255, 120 / 255, 200 / 255))


def test_parse_rgb_float():
    assert np.allclose(to_rgb((0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))


def test_parse_callable():
    # 可调用对象作为程序化花纹
    f = sphere(1).color(lambda p: np.tile((0.1, 0.2, 0.3), (len(p), 1)))
    c = f.colors_at(np.array([[0.0, 0, 0], [1, 1, 1]]))
    assert np.allclose(c, (0.1, 0.2, 0.3))


def test_default_color():
    # 未上色的形状使用默认颜色，且可以自定义
    f = sphere(1)
    c = f.colors_at(np.array([[0.0, 0, 0]]))
    assert np.allclose(c, DEFAULT_COLOR)
    c = f.colors_at(np.array([[0.0, 0, 0]]), default_color=(255, 0, 0))
    assert np.allclose(c, (1, 0, 0))


def test_color_does_not_mutate():
    # .color() 返回新对象，原对象保持未上色
    a = sphere(1)
    b = a.color('red')
    assert a._color is None
    assert b._color is not None


# ---------------------------------------------------------------------------
# 布尔运算颜色语义
# ---------------------------------------------------------------------------

def test_union_colors():
    f = sphere(1).color('red') | sphere(1).color('blue').translate((3, 0, 0))
    c = f.colors_at(np.array([[0.0, 0, 0], [3.0, 0, 0]]))
    assert np.allclose(c[0], RED)
    assert np.allclose(c[1], BLUE)


def test_intersection_colors():
    f = sphere(1).color('red') & box(1.5).color('blue')
    # 交集 = 盒子被球切掉角：盒面中心 -> 蓝色；球面（靠近角部）-> 红色
    s = 1 / np.sqrt(3)
    c = f.colors_at(np.array([[0.75, 0.0, 0], [s, s, s]]))
    assert np.allclose(c[0], BLUE)
    assert np.allclose(c[1], RED)


def test_difference_tool_color():
    # 洞壁使用刀具（被减去的形状）的颜色
    base = box(2).color('red')
    tool = cylinder(0.5).color('blue')
    f = base - tool
    c = f.colors_at(np.array([[0.5, 0.0, 0], [1.0, 0.2, 0.3]]))
    assert np.allclose(c[0], BLUE)  # 洞壁上的点
    assert np.allclose(c[1], RED)   # 外表面上的点


def test_difference_uncolored_tool():
    # 刀具未上色时，洞壁沿用被切物体的颜色
    base = box(2).color('red')
    f = base - cylinder(0.5)
    c = f.colors_at(np.array([[0.5, 0.0, 0]]))
    assert np.allclose(c[0], RED)


def test_smooth_union_gradient():
    # 平滑并集的交界处颜色应自然过渡（介于两色之间）
    a = sphere(1).color(RED)
    b = sphere(1).color(BLUE).translate((1.5, 0, 0))
    f = a | b.k(0.5)
    # 两球心连线的中点处 h = 0.5，颜色应为两者各半
    c = f.colors_at(np.array([[0.75, 0.0, 0]]))[0]
    assert np.allclose(c, (0.5, 0, 0.5), atol=1e-6)


def test_smooth_difference_gradient():
    base = box(2).color(RED)
    tool = cylinder(0.5).color(BLUE).k(0.2)
    f = base - tool
    # 过渡区内的点（d1 + d2 ≈ 0）颜色应介于红蓝之间
    c = f.colors_at(np.array([[0.75, 0.0, 0]]))[0]
    assert 0 < c[0] < 1
    assert 0 < c[2] < 1


def test_blend_colors():
    f = blend(sphere(1).color(RED), sphere(1).color(BLUE).translate((1, 0, 0)), k=0.25)
    c = f.colors_at(np.array([[0.0, 0, 0]]))[0]
    assert np.allclose(c, (0.75, 0, 0.25))


def test_transition_colors():
    f = transition_linear(
        sphere(1).color(RED), sphere(1).color(BLUE),
        p0=(0, 0, -1), p1=(0, 0, 1))
    c = f.colors_at(np.array([[0.0, 0, 0]]))[0]
    assert np.allclose(c, (0.5, 0, 0.5), atol=1e-6)


def test_repeat_colors():
    f = sphere(0.4).color('lime').repeat((2, 0, 0))
    c = f.colors_at(np.array([[2.0, 0, 0], [4.0, 0, 0]]))
    assert np.allclose(c, to_rgb('lime'))


def test_circular_array_colors():
    f = box(0.5).color((0.1, 0.2, 0.3)).circular_array(4, 1)
    c = f.colors_at(np.array([[1.0, 0, 0], [0.0, 1, 0]]))
    assert np.allclose(c, (0.1, 0.2, 0.3))


# ---------------------------------------------------------------------------
# 变换后花纹跟随
# ---------------------------------------------------------------------------

def _stripes(p):
    # 沿局部 x 的红蓝条纹
    w = (np.sin(p[:, 0] * 4) > 0).reshape((-1, 1))
    return np.where(w, (1.0, 0, 0), (0, 0, 1.0))


def test_pattern_follows_twist():
    s = box(2).color(_stripes)
    t = s.twist(1.0)
    pt = np.array([[0.9, 0.0, 1.0]])
    # 手工计算 twist 的局部坐标变换
    c, sn = np.cos(1.0), np.sin(1.0)
    q = np.array([[c * 0.9 - sn * 0.0, sn * 0.9 + c * 0.0, 1.0]])
    assert np.allclose(t.colors_at(pt), s.colors_at(q))
    # 花纹随物体扭转，而不是停在世界坐标
    assert not np.allclose(t.colors_at(pt), s.colors_at(pt))


def test_pattern_follows_translate_rotate():
    s = box(2).color(_stripes)
    t = s.translate((10, 0, 0))
    pt = np.array([[10.3, 0.0, 0]])
    assert np.allclose(t.colors_at(pt), s.colors_at(np.array([[0.3, 0, 0]])))
    r = s.rotate(np.pi / 2, Z)
    pt = np.array([[0.0, 0.4, 0]])
    assert np.allclose(r.colors_at(pt), s.colors_at(np.array([[0.4, 0, 0]])))


def test_pattern_follows_bend():
    s = box(2).color(_stripes)
    b = s.bend(0.5)
    pt = np.array([[0.9, 0.0, 0.5]])
    c, sn = np.cos(0.5 * 0.9), np.sin(0.5 * 0.9)
    q = np.array([[c * 0.9, sn * 0.9, 0.5]])
    assert np.allclose(b.colors_at(pt), s.colors_at(q))


def test_pattern_follows_wrap_around():
    s = box(0.2).color(_stripes).translate((1, 0, 0))
    w = s.wrap_around(0, 2 * np.pi, r=1)
    # 世界点 (0, -1, 0) 对应原物体局部坐标 (π/2, 0, 0)
    c = w.colors_at(np.array([[0.0, -1, 0]]))
    expected = s.colors_at(np.array([[np.pi / 2, 0.0, 0]]))
    assert np.allclose(c, expected)


# ---------------------------------------------------------------------------
# 2D -> 3D 与 image()
# ---------------------------------------------------------------------------

def test_extrude_keeps_color():
    f = circle(1).color('green').extrude(1)
    c = f.colors_at(np.array([[0.5, 0.0, 0.3]]))
    assert np.allclose(c, to_rgb('green'))


def test_extrude_to_blends_colors():
    f = extrude_to(circle(1).color(RED), circle(0.5).color(BLUE), 2)
    c = f.colors_at(np.array([[0.5, 0.0, 0.0]]))[0]
    assert np.allclose(c, (0.5, 0, 0.5), atol=1e-6)


def test_revolve_keeps_color():
    f = circle(0.5).color('orange').translate((1, 0)).revolve()
    c = f.colors_at(np.array([[1.0, 0, 0]]))
    assert np.allclose(c, to_rgb('orange'))


def test_image_colors(tmp_path):
    from PIL import Image
    # 左半红、右半蓝的测试图片
    im = Image.new('RGB', (4, 4))
    for y in range(4):
        for x in range(4):
            im.putpixel((x, y), (255, 0, 0) if x < 2 else (0, 0, 255))
    path = str(tmp_path / 'test.png')
    im.save(path)
    f = image(path, width=2, height=2).extrude(0.5)
    c = f.colors_at(np.array([[-0.5, 0.0, 0], [0.5, 0.0, 0]]))
    assert np.allclose(c[0], RED, atol=0.05)
    assert np.allclose(c[1], BLUE, atol=0.05)


# ---------------------------------------------------------------------------
# 自定义形状
# ---------------------------------------------------------------------------

def test_custom_sdf3_color():
    @sdf3
    def blob(r=1):
        def f(p):
            return np.linalg.norm(p, axis=1) - r
        return f
    f = blob(1).color('purple')
    c = f.colors_at(np.array([[0.0, 0, 0]]))
    assert np.allclose(c, to_rgb('purple'))
    # 自定义形状未上色时也能正常生成
    points = blob(1).generate(step=0.5, verbose=False)
    assert len(points) > 0


# ---------------------------------------------------------------------------
# 网格质量（小 batch 下封闭、顶点合并）
# ---------------------------------------------------------------------------

def _manifold_check(points):
    points = np.asarray(points).reshape((-1, 3))
    verts, inverse = np.unique(points, axis=0, return_inverse=True)
    tris = inverse.reshape((-1, 3))
    # 顶点应被合并：唯一顶点数明显少于三角形 soup 点数
    assert len(verts) < len(points)
    # 每条边应恰好被两个三角形共享（封闭流形）
    edges = np.concatenate([
        tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    return verts, tris


def test_mesh_closed_small_batches():
    # 小 batch_size 强制多批次拼接，验证跨批次封闭
    f = sphere(1)
    points = f.generate(step=0.15, batch_size=8, verbose=False)
    _manifold_check(points)


def test_mesh_closed_boolean():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    points = f.generate(step=0.2, batch_size=8, verbose=False)
    _manifold_check(points)


# ---------------------------------------------------------------------------
# 导出文件可以读回
# ---------------------------------------------------------------------------

def _read_ply(path):
    """手工解析二进制 PLY，返回 (顶点, 颜色, 面)。"""
    with open(path, 'rb') as fp:
        data = fp.read()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    header = data[:end].decode('ascii')
    nv = int(re.search(r'element vertex (\d+)', header).group(1))
    nf = int(re.search(r'element face (\d+)', header).group(1))
    vdtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vdata = np.frombuffer(data, dtype=vdtype, count=nv, offset=end)
    foffset = end + nv * vdtype.itemsize
    fdtype = np.dtype([('n', 'u1'), ('i', '<i4', 3)])
    fdata = np.frombuffer(data, dtype=fdtype, count=nf, offset=foffset)
    verts = np.stack([vdata['x'], vdata['y'], vdata['z']], axis=1)
    colors = np.stack([vdata['red'], vdata['green'], vdata['blue']], axis=1)
    return verts, colors, fdata['i']


def test_ply_roundtrip(tmp_path):
    f = sphere(1).color('red') | box(1.2).color('blue').translate((1, 0, 0))
    path = str(tmp_path / 'out.ply')
    f.save(path, step=0.2, verbose=False)
    verts, colors, faces = _read_ply(path)
    assert len(verts) > 0 and len(faces) > 0
    assert faces.max() < len(verts)
    # 左侧应偏红、右侧应偏蓝
    left = colors[verts[:, 0] < -0.5]
    right = colors[verts[:, 0] > 1.0]
    assert left[:, 0].mean() > 200
    assert right[:, 2].mean() > 200
    # 导出的网格封闭
    _manifold_check(verts[faces.reshape(-1)])


def test_3mf_roundtrip(tmp_path):
    f = sphere(1).color('red') | box(1.2).color('blue').translate((1, 0, 0))
    path = str(tmp_path / 'out.3mf')
    f.save(path, step=0.2, verbose=False)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert '[Content_Types].xml' in names
        assert '_rels/.rels' in names
        assert '3D/3dmodel.model' in names
        xml = z.read('3D/3dmodel.model')
    root = ET.fromstring(xml)
    ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    bases = root.findall('.//m:basematerials/m:base', ns)
    assert len(bases) >= 2
    displaycolors = {b.get('displaycolor') for b in bases}
    assert '#FF0000FF' in displaycolors
    assert '#0000FFFF' in displaycolors
    verts = root.findall('.//m:vertices/m:vertex', ns)
    tris = root.findall('.//m:triangles/m:triangle', ns)
    assert len(verts) > 0 and len(tris) > 0
    # 三角形的材质引用必须合法
    for t in tris:
        assert t.get('pid') == '1'
        assert 0 <= int(t.get('p1')) < len(bases)
        assert int(t.get('v1')) < len(verts)


def test_3mf_max_materials(tmp_path):
    # 渐变色谱应被量化到材质上限以内
    f = sphere(1).color(RED) | sphere(1).color(BLUE).translate((1.5, 0, 0)).k(0.5)
    path = str(tmp_path / 'out.3mf')
    f.save(path, step=0.15, verbose=False, max_materials=8)
    with zipfile.ZipFile(path) as z:
        xml = z.read('3D/3dmodel.model')
    root = ET.fromstring(xml)
    ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    bases = root.findall('.//m:basematerials/m:base', ns)
    assert len(bases) <= 8


def test_save_default_color(tmp_path):
    # 完全不上色时也能导出，全部使用指定的默认色
    f = sphere(1)
    path = str(tmp_path / 'out.ply')
    f.save(path, step=0.3, verbose=False, default_color=(10, 20, 30))
    _, colors, _ = _read_ply(path)
    assert np.all(colors == (10, 20, 30))


# ---------------------------------------------------------------------------
# 兼容性：STL 不变、generate 返回值不变
# ---------------------------------------------------------------------------

def _baseline_model():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    g = box(1).twist(0.5).bend(0.2)
    g = g | sphere(0.8).translate((1.5, 0, 0)).wrap_around(-1, 1, r=2)
    g = g | circle(0.5).revolve(1.5).translate((0, 2, 0))
    g = g | box(0.5).bend_linear(-Z, Z, (0.5, 0.5, 0)).bend_radial(0, 1, 0.2).translate((-2, 0, 0))
    g = g | sphere(0.3).repeat((1, 1, 0), count=1).translate((0, 0, 2))
    return f | g


def test_stl_unchanged(tmp_path):
    # 与原始版本生成的基准文件逐字节一致
    path = str(tmp_path / 'out.stl')
    _baseline_model().save(path, step=0.2, verbose=False, workers=4)
    with open(path, 'rb') as fp:
        digest = hashlib.sha256(fp.read()).hexdigest()
    with open(os.path.join(DATA_DIR, 'baseline.stl'), 'rb') as fp:
        expected = hashlib.sha256(fp.read()).hexdigest()
    assert digest == expected


def test_generate_return_value():
    # generate() 的返回值结构保持不变：顶点行组成的列表
    points = _baseline_model().generate(step=0.3, verbose=False)
    assert isinstance(points, list)
    arr = np.asarray(points)
    assert arr.ndim == 2 and arr.shape[1] == 3


# ---------------------------------------------------------------------------
# 多线程一致性
# ---------------------------------------------------------------------------

def test_multithread_consistency():
    f = _baseline_model().color('red') if False else _baseline_model()
    p1 = np.asarray(f.generate(step=0.2, verbose=False, workers=1))
    p4 = np.asarray(f.generate(step=0.2, verbose=False, workers=4))
    assert p1.shape == p4.shape
    assert np.array_equal(p1, p4)


def test_multithread_color_consistency():
    f = sphere(1).color(RED) | box(1.2).color(BLUE).translate((1, 0, 0)).k(0.3)
    p1 = np.asarray(f.generate(step=0.2, verbose=False, workers=1))
    p4 = np.asarray(f.generate(step=0.2, verbose=False, workers=4))
    assert np.array_equal(p1, p4)
    # 同一组点的颜色采样结果一致
    assert np.array_equal(f.colors_at(p1), f.colors_at(p4))
