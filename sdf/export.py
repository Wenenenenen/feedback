"""
Color-aware mesh exporters: binary PLY with vertex colors and 3MF with one
basematerial per distinct color. Only the Python standard library plus numpy
is used, so no new dependencies are introduced.
"""

import zipfile
import numpy as np

# Colors are quantized to this many levels per channel before being assigned
# to 3MF materials, keeping smooth-blend gradients from exploding into
# thousands of materials while staying visually smooth in the slicer.
# 32 levels (5 bits) reproduces the standard CSS/Bambu palette within 8/255.
COLOR_LEVELS = 32


def _quantize_rgb(colors):
    rgb = np.clip(colors[:, :3], 0, 1)
    q = np.floor(rgb * (COLOR_LEVELS - 1) + 0.5).astype(np.uint8)
    return q


def _quantized_to_hex(q):
    """Expand 5-bit channel values back to 8-bit for the 3MF hex color."""
    rgb8 = np.round(q.astype(float) / (COLOR_LEVELS - 1) * 255).astype(int)
    return rgb8


def _merge_vertices(points, colors, key_colors):
    """
    Merge coincident vertices that share the same (quantized) color.

    Returns (unique points, face indices into them, per-unique-vertex key
    colors). Marching cubes produces identical float coordinates for shared
    vertices, so exact equality is sufficient and keeps the mesh watertight
    without tolerance-based snapping.
    """
    dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
             ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    key = np.empty(len(points), dtype=dtype)
    key['x'] = points[:, 0]
    key['y'] = points[:, 1]
    key['z'] = points[:, 2]
    key['r'] = key_colors[:, 0]
    key['g'] = key_colors[:, 1]
    key['b'] = key_colors[:, 2]
    unique_key, inverse = np.unique(key, return_inverse=True)
    verts = np.column_stack([
        unique_key['x'], unique_key['y'], unique_key['z']])
    vert_colors = np.column_stack([
        unique_key['r'], unique_key['g'], unique_key['b']])
    return verts, inverse, vert_colors


def _merge_positions_only(points):
    """Merge coincident vertices regardless of color."""
    dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    key = np.empty(len(points), dtype=dtype)
    key['x'] = points[:, 0]
    key['y'] = points[:, 1]
    key['z'] = points[:, 2]
    unique_key, inverse = np.unique(key, return_inverse=True)
    verts = np.column_stack([
        unique_key['x'], unique_key['y'], unique_key['z']])
    return verts, inverse


# ---------------------------------------------------------------------------
# PLY (binary little-endian, vertex colors)
# ---------------------------------------------------------------------------

def write_ply(path, points, colors):
    q = _quantize_rgb(colors)
    verts, faces_idx, vert_q = _merge_vertices(points, colors, q)
    vert_rgb8 = _quantized_to_hex(vert_q)
    faces = faces_idx.reshape((-1, 3))

    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        'element vertex %d\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
        'element face %d\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    ) % (len(verts), len(faces))

    with open(path, 'wb') as fp:
        fp.write(header.encode('ascii'))
        vertex_data = np.empty(len(verts), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        vertex_data['x'] = verts[:, 0].astype('<f4')
        vertex_data['y'] = verts[:, 1].astype('<f4')
        vertex_data['z'] = verts[:, 2].astype('<f4')
        vertex_data['r'] = vert_rgb8[:, 0]
        vertex_data['g'] = vert_rgb8[:, 1]
        vertex_data['b'] = vert_rgb8[:, 2]
        fp.write(vertex_data.tobytes())
        face_data = np.empty(len(faces), dtype=[
            ('n', 'u1'), ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4')])
        face_data['n'] = 3
        face_data['i0'] = faces[:, 0]
        face_data['i1'] = faces[:, 1]
        face_data['i2'] = faces[:, 2]
        fp.write(face_data.tobytes())


# ---------------------------------------------------------------------------
# 3MF (core spec + basematerials extension, stdlib zipfile + xml)
# ---------------------------------------------------------------------------

_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels"
    ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model"
    ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0"
    Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""

_MODEL_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US"
  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <metadata name="Application">sdf</metadata>
"""


def _hex_color(rgb):
    return '#%02X%02X%02X' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _fmt(v):
    s = ('%.6f' % v).rstrip('0').rstrip('.')
    return s if s else '0'


def write_3mf(path, points, colors):
    q = _quantize_rgb(colors)

    # one material per distinct quantized color
    unique_colors, inv = np.unique(
        q.reshape(-1, 3), axis=0, return_inverse=True)
    unique_rgb8 = _quantized_to_hex(unique_colors)
    per_vertex_mat = inv

    # merge coincident positions (independent of color); each triangle keeps
    # a single material id derived from its three vertices' colors.
    verts, face_idx = _merge_positions_only(points)
    faces = face_idx.reshape((-1, 3))
    tri_mats = per_vertex_mat.reshape((-1, 3))
    # triangle material = majority vote of its vertices' materials. Most
    # triangles are unanimous; only blend-zone triangles mix, so the Python
    # fallback loop runs over a small subset rather than every triangle.
    tri_mat = tri_mats[:, 0].copy()
    mixed_idx = np.where(np.any(tri_mats != tri_mat[:, None], axis=1))[0]
    for idx in mixed_idx:
        counts = np.bincount(tri_mats[idx], minlength=len(unique_colors))
        tri_mat[idx] = counts.argmax()

    lines = [_MODEL_HEADER]
    lines.append('  <resources>\n')
    lines.append(
        '    <m:basematerials id="1">\n')
    for rgb in unique_rgb8:
        lines.append(
            '      <m:base name="%s" displaycolor="%s"/>\n'
            % (_hex_color(rgb), _hex_color(rgb)))
    lines.append('    </m:basematerials>\n')
    lines.append('    <object id="2" type="model">\n')
    lines.append('      <mesh>\n')
    lines.append('        <vertices>\n')
    for v in verts:
        lines.append(
            '          <vertex x="%s" y="%s" z="%s"/>\n'
            % (_fmt(v[0]), _fmt(v[1]), _fmt(v[2])))
    lines.append('        </vertices>\n')
    lines.append('        <triangles>\n')
    for (a, b, c), m in zip(faces, tri_mat):
        lines.append(
            '          <triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d"/>\n'
            % (a, b, c, m))
    lines.append('        </triangles>\n')
    lines.append('      </mesh>\n')
    lines.append('    </object>\n')
    lines.append('  </resources>\n')
    lines.append('  <build>\n')
    lines.append('    <item objectid="2"/>\n')
    lines.append('  </build>\n')
    lines.append('</model>\n')
    model_xml = ''.join(lines)

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', _CONTENT_TYPES)
        zf.writestr('_rels/.rels', _RELS)
        zf.writestr('3D/3dmodel.model', model_xml)


def read_3mf(path):
    """
    Minimal 3MF reader used by the test-suite: parses vertices, triangles and
    the basematerial colors with the standard library only.
    """
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as zf:
        xml = zf.read('3D/3dmodel.model')

    root = ET.fromstring(xml)
    core = '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}'
    mat = '{http://schemas.microsoft.com/3dmanufacturing/material/2015/02}'

    materials = []
    for bm in root.iter(mat + 'basematerials'):
        for base in bm.iter(mat + 'base'):
            materials.append(base.attrib['displaycolor'])

    verts = []
    for vertex in root.iter(core + 'vertex'):
        verts.append((
            float(vertex.attrib['x']),
            float(vertex.attrib['y']),
            float(vertex.attrib['z'])))

    faces = []
    face_mats = []
    for tri in root.iter(core + 'triangle'):
        faces.append((
            int(tri.attrib['v1']),
            int(tri.attrib['v2']),
            int(tri.attrib['v3'])))
        face_mats.append(int(tri.attrib.get('p1', 0)))

    return (np.array(verts, dtype=float),
            np.array(faces, dtype=np.int64),
            materials,
            np.array(face_mats, dtype=np.int64))
