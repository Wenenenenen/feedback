"""Color semantics of boolean operations, transforms and 2D->3D ops."""

import numpy as np

from sdf import *


RED = (1, 0, 0)
BLUE = (0, 0, 1)
GREEN = (0, 1, 0)


def _rgb(sdf, points):
    return sdf.color_fn(np.asarray(points, dtype=float))[:, :3]


def test_union_picks_closest_surface_color():
    a = sphere(1).color(RED)
    b = sphere(0.7).translate((0.5, 0, 0)).color(BLUE)
    u = a | b
    c = _rgb(u, [[-0.9, 0, 0], [0.9, 0, 0], [0.5, 0, 0]])
    assert np.allclose(c[0], RED)      # only inside a
    assert np.allclose(c[1], BLUE)     # only inside b
    assert np.allclose(c[2], BLUE)     # b's boundary is closer


def test_intersection_color():
    a = sphere(1).color(RED)
    b = sphere(0.7).translate((0.5, 0, 0)).color(BLUE)
    i = a & b
    c = _rgb(i, [[0.0, 0, 0], [0.5, 0, 0]])
    assert np.allclose(c[0], BLUE)     # d_b > d_a -> b dominates
    assert np.allclose(c[1], RED)


def test_difference_hole_wall_takes_cutter_color():
    a = sphere(1).color(RED)
    b = sphere(0.7).translate((0.5, 0, 0)).color(BLUE)
    d = a - b
    c = _rgb(d, [[-0.9, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
    assert np.allclose(c[0], RED)      # untouched base region
    assert np.allclose(c[1], BLUE)     # inside cutter -> cutter color
    assert np.allclose(c[2], BLUE)


def test_difference_uncolored_cutter_keeps_base_color():
    a = sphere(1).color(RED)
    b = sphere(0.7).translate((0.5, 0, 0))  # no color
    d = a - b
    c = d.color_fn(np.array([[0.5, 0, 0], [-0.9, 0, 0]]))
    # every defined row falls back to red; nan only where neither has color
    defined = ~np.isnan(c[:, 0])
    assert np.allclose(c[defined, 0], 1.0)
    assert np.allclose(c[defined, 1:3], 0.0)


def test_smooth_union_is_gradient_not_hard_cut():
    a = sphere(1).color(RED)
    b = sphere(1).translate((0.6, 0, 0)).color(BLUE)
    u = a | b.k(0.5)
    p = np.linspace(-0.9, 1.5, 40).reshape(-1, 1)
    p = np.hstack([p, np.zeros((40, 2))])
    inside = (u(p).ravel() < 0)
    c = u.color_fn(p)[:, :3][inside]
    # a range of distinct intermediate colors must appear in the blend zone
    intermediates = c[(c[:, 0] > 0.05) & (c[:, 2] > 0.05)]
    assert len(intermediates) >= 5
    # red -> blue progression along +x inside the solid
    assert c[0, 2] > c[0, 0]          # blue side
    assert c[-1, 0] > c[-1, 2]        # red side


def test_smooth_difference_gradient():
    a = sphere(1).color(RED)
    b = sphere(1).translate((0.5, 0, 0)).color(BLUE)
    d = a - b.k(0.4)
    p = np.linspace(-0.9, 0.9, 19).reshape(-1, 1)
    p = np.hstack([p, np.zeros((19, 2))])
    inside = d(p).ravel() < 0
    c = d.color_fn(p)[:, :3][inside]
    # surviving shell spans red (far from cutter) to blue (near hole wall),
    # with genuine mixtures in between -- no hard color block boundary.
    assert c[-1, 2] > 0.8            # hole wall is cutter blue
    assert c[0, 0] > 0.5             # far side keeps base red
    mixes = c[(c[:, 0] > 0.1) & (c[:, 2] > 0.1)]
    assert len(mixes) >= 3


def test_blend_convex_color_mix():
    a = sphere(1).color(RED)
    b = sphere(1).color(BLUE)
    c = _rgb(a.blend(b, k=0.3), [[0.0, 0, 0]])[0]
    assert np.allclose(c, (0.7, 0.0, 0.3))


def test_transition_linear_color_easing():
    f0 = sphere(1).color(RED)
    f1 = sphere(1).color(GREEN)
    t = transition_linear(f0, f1, p0=(0, 0, -1), p1=(0, 0, 1))
    c = _rgb(t, [[0, 0, -1], [0, 0, 0], [0, 0, 1]])
    assert np.allclose(c[0], RED)
    assert np.allclose(c[1], (0.5, 0.5, 0))
    assert np.allclose(c[2], GREEN)


def test_procedural_pattern_follows_twist():
    def xstripes(p):
        t = np.sin(p[:, 0] * 12)
        return np.where((t > 0)[:, None],
                        np.array(RED), np.array(BLUE))

    plain = cylinder(0.5).color(xstripes)
    twisted = cylinder(0.5).color(xstripes).twist(2.0)
    p = np.array([[0.3, 0.1, 0.2], [-0.2, 0.4, -0.3], [0.0, 0.45, 0.1]])
    # colors differ between twisted and untwisted at the same world points
    assert not np.allclose(plain.color_fn(p), twisted.color_fn(p))
    # exact inverse-map check: twist evaluates other at R(kz) @ world,
    # so the twisted color at world p == plain color at R(kz) @ p.
    for world in p:
        x, y, z = world
        ang = 2.0 * z
        local = np.array([[
            np.cos(ang) * x - np.sin(ang) * y,
            np.sin(ang) * x + np.cos(ang) * y, z]])
        assert np.allclose(
            twisted.color_fn(world.reshape(1, 3))[:, :3],
            plain.color_fn(local)[:, :3])


def test_translate_rotate_scale_carry_color():
    g = sphere(1).color(RED)
    p = np.array([[0.1, 0.0, 0.0]])
    assert np.allclose(
        g.translate((5, 0, 0)).color_fn(p + [5, 0, 0])[:, :3], RED)
    r = g.rotate(np.pi / 2, X)
    x, angle = (1, 0, 0), np.pi / 2
    s, c, m = np.sin(angle), np.cos(angle), 1 - np.cos(angle)
    M = np.array([
        [c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    assert np.allclose(
        r.color_fn(p)[:, :3], g.color_fn(p @ M)[:, :3])
    assert np.allclose(
        g.scale(2).color_fn(np.array([[0.2, 0, 0]]))[:, :3], RED)


def test_wrap_around_carries_color():
    w = box((2, 1, 1)).color(GREEN).wrap_around(-1, 1)
    p = np.random.RandomState(0).uniform(-1.2, 1.2, (30, 3))
    c = w.color_fn(p)
    assert np.any(~np.isnan(c[:, 0]))


def test_extrude_and_revolve_preserve_2d_color():
    g = circle(0.5).color(RED)
    g |= rectangle((0.6, 0.3)).translate((0.5, 0)).color(BLUE)
    e = g.extrude(0.4)
    c = _rgb(e, [[0, 0, 0.1], [0.5, 0, -0.1]])
    assert np.allclose(c[0], RED)
    assert np.allclose(c[1], BLUE)
    rv = g.revolve(0.2)
    assert np.allclose(
        _rgb(rv, [[0.5, 0, 0]])[0], RED)


def test_repeat_and_circular_array_colors():
    r = sphere(0.3).color(RED).repeat((1, 0, 0), count=2)
    c = r.color_fn(np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    assert np.allclose(c[:, 0], 1.0)

    ca = (sphere(0.2).translate((2, 0, 0)).color(GREEN)
          .circular_array(6))
    a = 2 * np.pi / 6
    c = ca.color_fn(np.array([
        [2 * np.cos(a), 2 * np.sin(a), 0], [2, 0, 0]]))
    assert np.allclose(c[:, 1], 1.0)


def test_unary_ops_preserve_color():
    for shape in (
            sphere(1).color('white').negate(),
            sphere(1).color('white').dilate(0.1),
            sphere(1).color('white').erode(0.1),
            sphere(1).color('white').shell(0.1)):
        c = shape.color_fn(np.array([[0.0, 0, 0]]))
        assert np.allclose(c[0, :3], 1.0)


def test_uncolored_sdf_has_no_color_function():
    assert sphere(1).color_fn is None
    assert (sphere(1).color(RED) | sphere(1)).color_fn is not None
    assert (sphere(1) | sphere(1)).color_fn is None


def test_nested_booleans_keep_distance_and_color():
    a = sphere(1).color(RED)
    b = sphere(1).translate((0.5, 0, 0)).color(BLUE)
    c = sphere(1).translate((1.0, 0, 0)).color(GREEN)
    ra, rb, rc = sphere(1), sphere(1).translate((0.5, 0, 0)), \
        sphere(1).translate((1.0, 0, 0))
    p = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])
    for colored, plain in (
            ((a | b) - c, (ra | rb) - rc),
            ((a - b) | c, (ra - rb) | rc),
            ((a | b) & c, (ra | rb) & rc),
            (a & b & c, ra & rb & rc)):
        # geometry must match the uncolored reference exactly
        assert np.allclose(colored(p), plain(p))
        # and colors evaluate to a valid RGBA array
        assert colored.color_fn(p).shape == (3, 4)


def test_color_returns_new_object_without_mutating():
    a = sphere(1)
    b = a.color(RED)
    assert a is not b
    assert a.color_fn is None
    assert b.color_fn is not None


def test_custom_sdf3_distance_only_can_be_colored():
    @sdf3
    def blob():
        def f(p):
            return np.linalg.norm(p, axis=1) - 0.5
        return f

    m = blob().color('red')
    p, c = m.generate_colored(samples=2 ** 14, verbose=False)
    assert np.allclose(c[:, 0], 1.0)
    assert np.allclose(c[:, 1:3], 0.0)


def test_operations_returning_sdf_instances_keep_color():
    # regression: op2/op3 wrappers must not strip color_fn when the raw
    # operation returns an already-wrapped SDF instance.
    two_d = circle(0.2).translate((1, 0)).color('orange').circular_array(5)
    assert two_d.color_fn is not None
    angle = 2 * np.pi / 5
    col = two_d.color_fn(np.array([[np.cos(angle), np.sin(angle)]]))
    # CSS4 'orange' is #FFA500 -> (1, 165/255, 0)
    assert np.allclose(col[0, :3], (1, 165 / 255, 0), atol=0.02)

    assert sphere(1).color('red').rotate_to(X, Y).color_fn is not None
    assert sphere(1).color('red').orient(X).color_fn is not None


def test_revolve_preserves_procedural_pattern():
    def rings(p):
        r = np.hypot(p[:, 0], p[:, 1])
        return np.where(
            (np.sin(r * 8) > 0)[:, None],
            np.array(RED), np.array(GREEN))

    solid = circle(0.8).color(rings).revolve(0.1)
    wp = np.array([[0.1 + r, 0.0, 0.0]
                   for r in np.linspace(0, 0.79, 40)])
    c = solid.color_fn(wp)[:, :3]
    # both pattern colors survive the revolution
    assert np.any(np.all(np.abs(c - np.array(RED)) < 0.05, axis=1))
    assert np.any(np.all(np.abs(c - np.array(GREEN)) < 0.05, axis=1))
