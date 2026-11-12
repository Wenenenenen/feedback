"""image() carries the source picture's colors through extrude()."""

import numpy as np

from PIL import Image

from sdf import image


def _make_image(tmp_path):
    # 8x8 image with four solid 4x4 quadrants: red | green / blue | white
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[:4, :4] = (255, 0, 0, 255)      # top-left
    arr[:4, 4:] = (0, 255, 0, 255)      # top-right
    arr[4:, :4] = (0, 0, 255, 255)      # bottom-left
    arr[4:, 4:] = (255, 255, 255, 255)  # bottom-right
    path = tmp_path / 'logo.png'
    Image.fromarray(arr).save(str(path))
    return path, arr


def test_image_color_function_samples_texture(tmp_path):
    path, _ = _make_image(tmp_path)
    s = image(str(path), width=2, height=2)
    # quadrant centers (well away from bilinear seam pixels)
    pts = np.array([
        [-0.5, 0.5], [0.5, 0.5],
        [-0.5, -0.5], [0.5, -0.5]])
    got = s.color_fn(pts)[:, :3]
    expected = np.array([
        [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 1, 1]], dtype=float)
    assert np.allclose(got, expected, atol=0.15)


def test_image_color_nan_outside_bounds(tmp_path):
    path, _ = _make_image(tmp_path)
    s = image(str(path), width=2, height=2)
    out = s.color_fn(np.array([[5.0, 5.0]]))
    assert np.all(np.isnan(out))


def test_image_extrude_keeps_colors(tmp_path):
    path, _ = _make_image(tmp_path)
    solid = image(str(path), width=2, height=2).extrude(0.4)
    pts, cols = solid.generate_colored(
        samples=2 ** 15, verbose=False,
        bounds=((-1.2, -1.2, -0.4), (1.2, 1.2, 0.4)))
    # all four quadrant colors must appear in the quantized output
    q = np.unique((cols[:, :3] * 31 + 0.5).astype(int), axis=0)
    targets = np.array([
        [31, 0, 0], [0, 31, 0], [0, 0, 31], [31, 31, 31]])
    for t in targets:
        assert np.min(np.linalg.norm(q - t, axis=1)) <= 2.0


def test_image_distance_path_unchanged_under_color(tmp_path):
    path, _ = _make_image(tmp_path)
    s = image(str(path), width=2, height=2)
    p = np.array([[0.0, 0.0], [0.3, 0.3], [1.5, 1.5]])
    d = s(p)
    assert d.shape == (3, 1)
    assert np.isfinite(d).all()


def test_real_png_asset_loads_with_color():
    # the bundled butterfly asset used by examples/image.py
    s = image('examples/butterfly.png')
    center = s.color_fn(np.array([[0.0, 0.0]]))
    assert center.shape == (1, 4)
    assert 0.0 <= center[0, 0] <= 1.0
