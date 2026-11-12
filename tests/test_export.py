"""Export tests: PLY vertex colors and 3MF materials round-trip."""

import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from sdf import *
from sdf.export import read_3mf, write_3mf, write_ply


def _model():
    f = sphere(1).color('#e63946') | box(1.5).color((40, 120, 200)).k(0.2)
    f -= cylinder(0.4).orient(X).color('yellow')
    return f


def test_ply_roundtrip_with_vertex_colors(tmp_path):
    path = tmp_path / 'out.ply'
    f = _model()
    f.save(str(path), samples=2 ** 16, verbose=False)

    import meshio
    mesh = meshio.read(str(path))
    assert {'red', 'green', 'blue'} <= set(mesh.point_data)
    # meshio exposes uchar as signed int8; reinterpret as unsigned bytes
    rgb = np.stack([
        mesh.point_data['red'],
        mesh.point_data['green'],
        mesh.point_data['blue']], axis=1).astype(np.uint8).view(np.uint8)
    rgb = rgb.astype(int)
    assert rgb.min() >= 0 and rgb.max() <= 255
    # the three input colors must all survive quantization in the output
    for target in ((230, 57, 70), (40, 120, 200), (255, 255, 0)):
        assert np.min(np.linalg.norm(
            rgb - np.array(target), axis=1)) <= 16


def test_ply_header_is_binary_little_endian(tmp_path):
    path = tmp_path / 'out.ply'
    write_ply(str(path),
              np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
              np.array([[1., 0., 0., 1.]] * 3))
    with open(str(path), 'rb') as fp:
        header = fp.read(200).split(b'end_header')[0].decode()
    assert 'format binary_little_endian 1.0' in header
    assert 'property uchar red' in header


def test_3mf_is_valid_zip_with_required_parts(tmp_path):
    path = tmp_path / 'out.3mf'
    _model().save(str(path), samples=2 ** 16, verbose=False)
    with zipfile.ZipFile(str(path)) as zf:
        names = set(zf.namelist())
        assert '[Content_Types].xml' in names
        assert '_rels/.rels' in names
        assert '3D/3dmodel.model' in names
        # every zip entry must pass its CRC check (done by testzip)
        assert zf.testzip() is None


def test_3mf_materials_and_triangle_assignment(tmp_path):
    path = tmp_path / 'out.3mf'
    f = _model()
    f.save(str(path), samples=2 ** 16, verbose=False)

    verts, faces, materials, face_mats = read_3mf(str(path))
    assert len(verts) > 0 and len(faces) > 0
    assert len(materials) >= 3  # red, blue, yellow (+ blend intermediates)
    # every triangle references a valid material id and a pid
    assert set(face_mats.tolist()) <= set(range(len(materials)))
    assert face_mats.min() >= 0

    with zipfile.ZipFile(str(path)) as zf:
        root = ET.fromstring(zf.read('3D/3dmodel.model'))
    core = '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}'
    tris = list(root.iter(core + 'triangle'))
    assert all('pid' in t.attrib and 'p1' in t.attrib for t in tris)
    # every requested color appears as a base material (within 5-bit quant)
    rgb_mats = np.array([
        [int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in materials])
    for target in ((230, 57, 70), (40, 120, 200), (255, 255, 0)):
        assert np.min(np.linalg.norm(
            rgb_mats - np.array(target), axis=1)) <= 16


def test_3mf_quantization_caps_material_count(tmp_path):
    # a fully procedural rainbow could otherwise create thousands of mats
    def rainbow(p):
        lo, hi = p[:, 0].min(), p[:, 0].max()
        t = (p[:, 0] - lo) / (np.ptp(p[:, 0]) or 1)
        out = np.zeros((len(p), 3))
        out[:, 0] = t
        out[:, 2] = 1 - t
        return out

    f = box(2).color(rainbow)
    path = tmp_path / 'rainbow.3mf'
    f.save(str(path), samples=2 ** 16, verbose=False,
           bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)))
    _, _, materials, _ = read_3mf(str(path))
    # quantization bounds material count far below the vertex count
    assert len(materials) <= 32 ** 2
    assert len(materials) > 1


def test_3mf_and_ply_share_geometry(tmp_path):
    f = _model()
    f.save(str(tmp_path / 'a.ply'), samples=2 ** 16, verbose=False)
    f.save(str(tmp_path / 'a.3mf'), samples=2 ** 16, verbose=False)
    import meshio
    ply = meshio.read(str(tmp_path / 'a.ply'))
    v3, f3, _, _ = read_3mf(str(tmp_path / 'a.3mf'))
    # same merged vertex set (order may differ)
    assert np.allclose(
        np.sort(ply.points, axis=0), np.sort(v3, axis=0), atol=1e-6)
    assert len(ply.cells[0].data) == len(f3)


def test_default_color_used_for_uncolored_export(tmp_path):
    f = sphere(1)  # no .color() at all
    path = tmp_path / 'plain.3mf'
    f.save(str(path), samples=2 ** 15, verbose=False,
           default_color='#112233')
    _, _, materials, face_mats = read_3mf(str(path))
    assert len(materials) == 1
    # exact input on the unquantized path; tolerate 5-bit quantization
    rgb = [int(materials[0][i:i + 2], 16) for i in (1, 3, 5)]
    assert np.max(np.abs(np.array(rgb) - np.array((0x11, 0x22, 0x33)))) <= 8
    assert np.all(face_mats == 0)


def test_3mf_xml_uses_declared_namespaces(tmp_path):
    path = tmp_path / 'x.3mf'
    write_3mf(str(path),
              np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
              np.array([[1., 0., 0., 1.]] * 3))
    with zipfile.ZipFile(str(path)) as zf:
        xml = zf.read('3D/3dmodel.model').decode()
    assert 'xmlns:m=' in xml
    assert 'basematerials' in xml
    assert 'unit="millimeter"' in xml


def test_explicit_colorless_export_skips_vertex_colors(tmp_path):
    import meshio
    f = sphere(1).color('red')
    path = tmp_path / 'plain.ply'
    f.save(str(path), samples=2 ** 14, verbose=False, colored=False)
    assert not meshio.read(str(path)).point_data


def test_uncolored_sdf_still_exports_other_formats(tmp_path):
    # meshio-backed formats keep using the original colorless path
    path = tmp_path / 'plain.vtk'
    sphere(1).save(str(path), samples=2 ** 14, verbose=False)
    assert path.exists()
