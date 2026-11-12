"""Mesh quality: closed/manifold output, vertex merging, normals."""

import numpy as np

from sdf import *
from sdf.export import (
    write_ply, write_3mf, read_3mf, _merge_positions_only)


def _build():
    f = sphere(1).color('#e63946') | box(1.5).color('blue').k(0.2)
    f -= cylinder(0.4).orient(X).color('yellow')
    return f


def _edge_manifold(verts, faces):
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def test_closed_manifold_at_small_batch_sizes():
    f = _build()
    for batch_size in (2, 4, 8, 32):
        pts, cols = f.generate_colored(
            samples=2 ** 16, batch_size=batch_size, verbose=False)
        verts, inv = _merge_positions_only(pts)
        faces = inv.reshape(-1, 3)
        # every edge shared by exactly two triangles -> watertight
        assert _edge_manifold(verts, faces), batch_size
        # no duplicate vertices after merging
        assert len(verts) == len(np.unique(verts, axis=0))
        # all face indices in range
        assert faces.max() < len(verts)
        # no zero-area triangles
        tri = verts[faces]
        areas = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        assert np.all(areas > 0), batch_size


def test_outward_normals_on_convex_shape():
    pts, _ = sphere(1).color('red').generate_colored(
        samples=2 ** 16, verbose=False)
    verts, inv = np.unique(pts, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    centroids = tri.mean(axis=1)
    # a convex origin-centered solid must have outward face normals
    assert np.mean(np.sum(n * centroids, axis=1) > 0) > 0.999


def test_vertices_are_merged_in_exported_files(tmp_path):
    f = _build()
    ply = tmp_path / 'm.ply'
    mf = tmp_path / 'm.3mf'
    f.save(str(ply), samples=2 ** 16, verbose=False)
    f.save(str(mf), samples=2 ** 16, verbose=False)

    import meshio
    m = meshio.read(str(ply))
    # merging must collapse the raw triangle soup
    pts, _ = f.generate_colored(samples=2 ** 16, verbose=False)
    assert len(m.points) < len(pts)
    assert m.cells[0].data.max() < len(m.points)

    v, faces, _, _ = read_3mf(str(mf))
    assert _edge_manifold(v, faces)


def test_colored_geometry_matches_colorless_geometry():
    # the colored path must produce the exact same triangle vertices as the
    # original generate() path
    f = sphere(1) & box(1.5)
    plain = f.generate(samples=2 ** 16, verbose=False)
    colored, _ = f.color('red').generate_colored(
        samples=2 ** 16, verbose=False)
    assert np.array_equal(np.asarray(plain), colored)
