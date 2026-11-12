"""
Binary STL output must remain byte-for-byte identical to the pre-color
library, and generate()'s return value must be unchanged.
"""

import hashlib
import os

import numpy as np

from sdf import (
    X, Y, Z, sphere, box, cylinder, generate, generate_colored,
    write_binary_stl)
from stl_baselines import BASELINE_HASHES, SAMPLES, build_model


def _sha(path):
    with open(path, 'rb') as fp:
        return hashlib.sha256(fp.read()).hexdigest()


def test_stl_hashes_match_baselines(tmp_path):
    for name, expected in BASELINE_HASHES.items():
        model, extra = build_model(name)
        path = tmp_path / name
        model.save(
            str(path), samples=SAMPLES, verbose=False, **extra)
        assert _sha(str(path)) == expected, name


def test_stl_from_colored_object_is_identical(tmp_path):
    # attaching colors must not change the STL geometry path
    plain, extra = build_model('csg.stl')
    colored = (sphere(1).color('red') & box(1.5).color('blue'))
    c = cylinder(0.5).color('green')
    colored -= c.orient(X) | c.orient(Y) | c.orient(Z)

    p1 = tmp_path / 'a.stl'
    p2 = tmp_path / 'b.stl'
    plain.save(str(p1), samples=SAMPLES, verbose=False)
    colored.save(str(p2), samples=SAMPLES, verbose=False)
    assert _sha(str(p1)) == _sha(str(p2))


def test_generate_return_value_unchanged():
    f = sphere(1)
    points = generate(f, samples=2 ** 15, verbose=False)
    # original contract: a list of triangle vertices, shape (n*3, 3)
    assert isinstance(points, list)
    arr = np.asarray(points)
    assert arr.ndim == 2 and arr.shape[1] == 3
    assert arr.shape[0] % 3 == 0


def test_generate_colored_is_separate_interface():
    f = sphere(1).color('red')
    points, colors = f.generate_colored(samples=2 ** 15, verbose=False)
    assert isinstance(points, np.ndarray)
    assert colors.shape == (points.shape[0], 4)
    # and it agrees with what generate() produces for geometry
    plain = np.asarray(f.generate(samples=2 ** 15, verbose=False))
    assert np.array_equal(plain, points)


def test_write_binary_stl_directly(tmp_path):
    points = sphere(1).generate(samples=2 ** 14, verbose=False)
    path = tmp_path / 'direct.stl'
    write_binary_stl(str(path), points)
    assert os.path.getsize(str(path)) == 84 + 50 * (len(points) // 3)
