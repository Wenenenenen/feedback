"""Color parsing and procedural color-function construction."""

import numpy as np
import pytest

from sdf.color import parse_color, make_color_function, NAMED_COLORS


def test_hex_6():
    assert parse_color('#e63946') == pytest.approx(
        (0xe6 / 255, 0x39 / 255, 0x46 / 255, 1.0))


def test_hex_without_hash_and_short_form():
    assert parse_color('fff')[:3] == pytest.approx((1, 1, 1))
    assert parse_color('#000')[:3] == pytest.approx((0, 0, 0))


def test_hex_8_alpha():
    r, g, b, a = parse_color('#ff000080')
    assert (r, g, b) == pytest.approx((1, 0, 0))
    assert a == pytest.approx(0x80 / 255)


def test_named_colors():
    assert parse_color('yellow')[:3] == pytest.approx((1, 1, 0))
    assert parse_color('YELLOW')[:3] == pytest.approx((1, 1, 0))
    assert parse_color('steelblue') == parse_color(NAMED_COLORS['steelblue'])


def test_tuple_01_and_0_255():
    assert parse_color((1.0, 0.5, 0.0))[:3] == pytest.approx((1, 0.5, 0))
    rgb = parse_color((40, 120, 200))[:3]
    assert rgb == pytest.approx((40 / 255, 120 / 255, 200 / 255))
    # mixed / fractional 0-255 still detected by the >1 rule
    assert parse_color((255, 128, 0))[1] == pytest.approx(128 / 255)


def test_rgba_tuple():
    assert parse_color((0, 0, 0, 0.5))[3] == pytest.approx(0.5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        parse_color('not-a-color')
    with pytest.raises(ValueError):
        parse_color('#12')
    with pytest.raises(ValueError):
        parse_color((1, 2))


def test_constant_color_function():
    fn = make_color_function('red')
    out = fn(np.zeros((5, 3)))
    assert out.shape == (5, 4)
    assert np.allclose(out[:, :3], (1, 0, 0))
    assert np.all(out[:, 3] == 1)


def test_procedural_color_function():
    def stripes(p):
        red = np.array([1, 0, 0])
        blue = np.array([0, 0, 1])
        return np.where((p[:, 0] > 0)[:, None], red, blue)

    fn = make_color_function(stripes)
    p = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    out = fn(p)
    assert np.allclose(out[0, :3], (1, 0, 0))
    assert np.allclose(out[1, :3], (0, 0, 1))


def test_procedural_color_function_0_255():
    fn = make_color_function(lambda p: np.full((len(p), 3), 255))
    out = fn(np.zeros((3, 3)))
    assert np.allclose(out[:, :3], 1.0)
