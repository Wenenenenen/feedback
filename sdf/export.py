"""带颜色的网格导出：PLY（顶点色）与 3MF（材质）。

两个格式都会先用 np.unique 合并重复顶点（包括批次边界的缝合点），
保证导出的网格封闭、无重复顶点。
"""

import numpy as np
import zipfile

from .color import sample_colors

# 3MF 中材质数量的默认上限，超出时自动降低颜色量化精度
MAX_MATERIALS = 256


def _colored_mesh(sdf, points, default_color):
    """合并重复顶点，并采样每个顶点的颜色。

    返回 (顶点, 三角形索引, 顶点颜色)，颜色为 0-1 的 (N, 3) RGB。
    相邻批次共享边界上的顶点存在 1 ulp 级浮点差异（各批次分别做
    缩放平移运算所致），先按微小容差取整再合并，保证网格封闭。
    """
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    tol = 1e-6 * max(1.0, float(np.abs(points).max()))
    keys = np.round(points / tol).astype(np.int64)
    _, index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    verts = points[index]
    colors = sample_colors(sdf, verts, default_color)
    triangles = inverse.reshape((-1, 3))
    return verts, triangles, colors


def write_ply(path, sdf, points, default_color=None):
    """写出带顶点色的二进制 PLY（MeshLab / Blender / 3D Builder 可读）。"""
    verts, triangles, colors = _colored_mesh(sdf, points, default_color)
    rgb = np.round(colors * 255).astype(np.uint8)

    header = '\n'.join([
        'ply',
        'format binary_little_endian 1.0',
        'element vertex %d' % len(verts),
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'element face %d' % len(triangles),
        'property list uchar int vertex_indices',
        'end_header',
        '',
    ]).encode('ascii')

    vdtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vdata = np.empty(len(verts), dtype=vdtype)
    vdata['x'] = verts[:, 0]
    vdata['y'] = verts[:, 1]
    vdata['z'] = verts[:, 2]
    vdata['red'] = rgb[:, 0]
    vdata['green'] = rgb[:, 1]
    vdata['blue'] = rgb[:, 2]

    fdtype = np.dtype([('n', 'u1'), ('i', '<i4', 3)])
    fdata = np.empty(len(triangles), dtype=fdtype)
    fdata['n'] = 3
    fdata['i'] = triangles

    with open(path, 'wb') as fp:
        fp.write(header)
        fp.write(vdata.tobytes())
        fp.write(fdata.tobytes())


def _quantize(colors, max_materials):
    """把 0-1 颜色量化为不超过 max_materials 种的 8-bit RGB。"""
    q0 = np.round(colors * 255).astype(np.uint16)
    q = q0
    shift = 0
    while len(np.unique(q, axis=0)) > max_materials and shift < 7:
        shift += 1
        # 始终从原始值重新量化，量化到桶中心避免颜色整体偏移
        q = ((q0 >> shift) << shift) + (1 << (shift - 1))
        q = np.clip(q, 0, 255)
    return q.astype(np.uint8)


_CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
'''

_RELS = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
'''


def _model_xml(verts, triangles, palette, tindex):
    """生成 3D/3dmodel.model 的 XML 内容。"""
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<model unit="millimeter" xml:lang="en-US" '
               'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">')
    out.append('<resources>')
    out.append('<basematerials id="1">')
    for i, c in enumerate(palette):
        hexcolor = '#%02X%02X%02XFF' % (c[0], c[1], c[2])
        out.append('<base name="color%d" displaycolor="%s"/>' % (i, hexcolor))
    out.append('</basematerials>')
    out.append('<object id="2" type="model" pid="1" pindex="0">')
    out.append('<mesh>')
    out.append('<vertices>')
    for v in verts:
        out.append('<vertex x="%.6f" y="%.6f" z="%.6f"/>' % (v[0], v[1], v[2]))
    out.append('</vertices>')
    out.append('<triangles>')
    for t, m in zip(triangles, tindex):
        out.append('<triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d"/>' %
                   (t[0], t[1], t[2], m))
    out.append('</triangles>')
    out.append('</mesh>')
    out.append('</object>')
    out.append('</resources>')
    out.append('<build><item objectid="2"/></build>')
    out.append('</model>')
    return '\n'.join(out)


def write_3mf(path, sdf, points, default_color=None, max_materials=MAX_MATERIALS):
    """写出 3MF：每种颜色对应一个 basematerial。

    三角形按质心颜色归到最接近的材质，Bambu Studio / PrusaSlicer /
    Windows 3D Builder 均可直接识别。
    """
    verts, triangles, colors = _colored_mesh(sdf, points, default_color)

    # 按三角形质心颜色量化并建立材质调色板
    centroids = colors[triangles].mean(axis=1)
    q = _quantize(centroids, max_materials)
    palette, tindex = np.unique(q, axis=0, return_inverse=True)

    model = _model_xml(verts, triangles, palette, tindex)

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _CONTENT_TYPES)
        z.writestr('_rels/.rels', _RELS)
        z.writestr('3D/3dmodel.model', model)
