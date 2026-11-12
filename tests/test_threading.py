"""Determinism: single-threaded and multi-threaded results must match."""

import numpy as np

from sdf import *  # noqa: F401,F403  (X/Y/Z, sphere, box, ...)


def _model():
    f = sphere(1).color('#e63946') | box(1.5).color('blue').k(0.25)
    f -= cylinder(0.4).orient(X).color('yellow')
    f |= sphere(0.3).translate((0, 1.1, 0)).color(
        lambda p: np.hstack([
            np.abs(p), np.ones((len(p), 1))]))
    return f


def test_thread_counts_give_identical_points_and_colors():
    f = _model()
    ref_pts, ref_col = f.generate_colored(
        samples=2 ** 16, workers=1, verbose=False)
    for workers in (2, 4, 8):
        pts, col = f.generate_colored(
            samples=2 ** 16, workers=workers, verbose=False)
        assert np.array_equal(pts, ref_pts), workers
        assert np.array_equal(col, ref_col), workers


def test_thread_counts_identical_without_color():
    f = sphere(1) & box(1.5)
    ref = np.asarray(f.generate(
        samples=2 ** 16, workers=1, verbose=False))
    for workers in (2, 8):
        out = np.asarray(f.generate(
            samples=2 ** 16, workers=workers, verbose=False))
        assert np.array_equal(out, ref), workers


def test_batch_size_does_not_change_manifold_geometry():
    # different batch sizes may reorder triangles and introduce only
    # floating-point-scale vertex jitter (also true of the original library);
    # the merged vertex sets must agree within a tight tolerance. Geometry is
    # independent of color, so a plain (uncolored) SDF is used here.
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    merged = {}
    for bs in (2, 8, 32):
        pts = np.asarray(f.generate(
            samples=2 ** 16, batch_size=bs, verbose=False))
        verts = np.unique(pts, axis=0)
        merged[bs] = np.sort(verts, axis=0)
    assert merged[2].shape == merged[32].shape
    assert np.allclose(merged[2], merged[32], atol=1e-6)
    assert np.allclose(merged[8], merged[32], atol=1e-6)


def test_colored_batch_size_manifold():
    # colored output must stay edge-manifold at adversarial batch sizes.
    # Uses a single connected solid (the multi-body _model produces the same
    # batch-related boundary edges whether colored or not).
    from sdf.export import _merge_positions_only
    f = sphere(1).color('red') | box(1.5).color('blue').k(0.25)
    f -= cylinder(0.4).orient(X).color('yellow')
    for bs in (2, 8):
        pts, _ = f.generate_colored(
            samples=2 ** 16, batch_size=bs, verbose=False)
        verts, inv = _merge_positions_only(pts)
        faces = inv.reshape(-1, 3)
        edges = np.vstack([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_counts=True)
        assert np.all(counts == 2), bs
