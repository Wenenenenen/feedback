import functools
import itertools
import numpy as np
# import operator

from . import color as _color
from . import core, dn, d2, ease

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}

class SDF3:
    def __init__(self, f, color_fn=None):
        self.f = f
        self.color_fn = color_fn
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        return getattr(self.f, name)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self
    def color(self, spec):
        """
        Attach a color to this shape.

        ``spec`` may be a hex string, a CSS4 color name, an RGB(A) tuple
        (0-1 floats or 0-255 ints), or a callable taking the local sample
        points and returning per-point colors (procedural patterns).
        Returns a new SDF3, so the original is left unmodified.
        """
        return SDF3(self.f, _color.make_color_function(spec))
    def generate(self, *args, **kwargs):
        return core.generate(self, *args, **kwargs)
    def generate_colored(self, *args, **kwargs):
        return core.generate_colored(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return core.save(path, self, *args, **kwargs)
    def show_slice(self, *args, **kwargs):
        return core.show_slice(self, *args, **kwargs)

def sdf3(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, SDF3):
            # operation returned an already-wrapped SDF (e.g. rotate_to);
            # pass its color function through unchanged
            return result
        if isinstance(result, tuple):
            return SDF3(result[0], result[1])
        return SDF3(result)
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, SDF3):
            return result
        if isinstance(result, tuple):
            return SDF3(result[0], result[1])
        return SDF3(result)
    _ops[f.__name__] = wrapper
    return wrapper

def op32(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, d2.SDF2):
            return result
        if isinstance(result, tuple):
            return d2.SDF2(result[0], result[1])
        return d2.SDF2(result)
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length(a):
    return np.linalg.norm(a, axis=1)

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

_min = np.minimum
_max = np.maximum

# Primitives

@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - center) - radius
    return f

@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return np.dot(point - p, normal)
    return f

@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)

@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)
    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)
    return f

@sdf3
def rounded_box(size, radius):
    size = np.array(size)
    def f(p):
        q = np.abs(p) - size / 2 + radius
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - radius
    return f

@sdf3
def wireframe_box(size, thickness):
    size = np.array(size)
    def g(a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)
    def f(p):
        p = np.abs(p) - size / 2 - thickness / 2
        q = np.abs(p + thickness / 2) - thickness / 2
        px, py, pz = p[:,0], p[:,1], p[:,2]
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))
    return f

@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[:,[0,1]]
        z = p[:,2]
        a = _length(xy) - r1
        b = _length(_vec(a, z)) - r2
        return b
    return f

@sdf3
def capsule(a, b, radius):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        pa = p - a
        ba = b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - radius
    return f

@sdf3
def cylinder(radius):
    def f(p):
        return _length(p[:,[0,1]]) - radius;
    return f

@sdf3
def capped_cylinder(a, b, radius):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        ba = b - a
        pa = p - a
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba
    return f

@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d = _vec(
            _length(p[:,[0,1]]) - ra + rb,
            np.abs(p[:,2]) - h / 2 + rb)
        return (
            _min(_max(d[:,0], d[:,1]), 0) +
            _length(_max(d, 0)) - rb)
    return f

@sdf3
def capped_cone(a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = _dot(p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))
    return f

@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[:,[0,1]]), p[:,2])
        b = (r1 - r2) / h
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - r1
        c2 = _length(q - _vec(0, h)) - r2
        c3 = np.dot(q, _vec(a, b)) - r1
        return np.where(k < 0, c1, np.where(k > a * h, c2, c3))
    return f

@sdf3
def ellipsoid(size):
    size = np.array(size)
    def f(p):
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        return k0 * (k0 - 1) / k1
    return f

@sdf3
def pyramid(h):
    def f(p):
        a = np.abs(p[:,[0,1]]) - 0.5
        w = a[:,1] > a[:,0]
        a[w] = a[:,[1,0]][w]
        px = a[:,0]
        py = p[:,2]
        pz = a[:,1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))
    return f

# Platonic Solids

@sdf3
def tetrahedron(r):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - r) / np.sqrt(3)
    return f

@sdf3
def octahedron(r):
    def f(p):
        return (np.sum(np.abs(p), axis=1) - r) * np.tan(np.radians(30))
    return f

@sdf3
def dodecahedron(r):
    x, y, z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))
    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        q = (_max(_max(a, b), c) - x) * r
        return q
    return f

@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
    w = np.sqrt(3) / 3
    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        d = np.dot(p, (w, w, w)) - x
        return _max(_max(_max(a, b), c) - x, d) * r
    return f

# Positioning

@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)
    def cf(p):
        return _color.color_of(other, p - offset)
    return f, cf

@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))
    def f(p):
        return other(p / s) * m
    def cf(p):
        return _color.color_of(other, p / s)
    return f, cf

@op3
def rotate(other, angle, vector=Z):
    x, y, z = _normalize(vector)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T
    def f(p):
        return other(np.dot(p, matrix))
    def cf(p):
        return _color.color_of(other, np.dot(p, matrix))
    return f, cf

@op3
def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)

@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)

@op3
def circular_array(other, count, offset=0):
    other = other.translate(X * offset)
    da = 2 * np.pi / count
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        d1 = other(_vec(np.cos(a - da) * d, np.sin(a - da) * d, z))
        d2 = other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)
    def cf(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        q1 = _vec(np.cos(a - da) * d, np.sin(a - da) * d, z)
        q2 = _vec(np.cos(a) * d, np.sin(a) * d, z)
        return _color.nan_select(
            other(q2) < other(q1),
            _color.color_of(other, q2),
            _color.color_of(other, q1))
    return f, cf

# Alterations

@op3
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        z = q[:,2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w
    def cf(p):
        q = np.abs(p) - size
        return _color.color_of(other, _max(q, 0))
    return f, cf

@op3
def twist(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    def cf(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        return _color.color_of(other, _vec(x2, y2, z))
    return f, cf

@op3
def bend(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    def cf(p):
        x = p[:,0]
        y = p[:,1]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        return _color.color_of(other, _vec(x2, y2, p[:,2]))
    return f, cf

@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = -np.array(v)
    ab = p1 - p0
    def f(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)
    def cf(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return _color.color_of(other, p + t * v)
    return f, cf

@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))
    def cf(p):
        x = p[:,0]
        y = p[:,1]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = p[:,2] - dz * e(t)
        return _color.color_of(other, _vec(x, y, z))
    return f, cf

@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    ab = p1 - p0
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    def cf(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return _color.nan_blend(
            _color.color_of(f0, p), _color.color_of(f1, p), t)
    return f, cf

@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = np.hypot(p[:,0], p[:,1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    def cf(p):
        r = np.hypot(p[:,0], p[:,1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return _color.nan_blend(
            _color.color_of(f0, p), _color.color_of(f1, p), t)
    return f, cf

@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0 = X * x0
    p1 = X * x1
    v = -Y
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * np.pi)
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:,2] = z
        return other(q)
    def cf(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = (np.hypot(x, y) - r).reshape((-1, 1))
        a = np.arctan2(y, x)
        t = e((a + np.pi) / (2 * np.pi)).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:,2] = z
        return _color.color_of(other, q)
    return f, cf

# 3D => 2D Operations

@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s
    def f(p):
        p = _vec(p[:,0], p[:,1], np.zeros(len(p)))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A
    def cf(p):
        p = _vec(p[:,0], p[:,1], np.zeros(len(p)))
        return _color.color_of(other, p)
    return f, cf

# Common

def _boolean(op_name, color_fn):
    """Wrap a dn boolean so color functions are tracked alongside distances."""
    dn_fn = getattr(dn, op_name)
    def op(a, *bs, k=None):
        def f(p):
            return dn_fn(a, *bs, k=k)(p)
        def cf(p):
            c = _color.color_of(a, p)
            left = a
            for b in bs:
                c = color_fn(left, b, k)
                if c is None:
                    c = np.full((len(p), 4), np.nan)
                else:
                    c = c(p)
                # build an SDF-like stand-in carrying the accumulated color so
                # the next combinator can blend against it.
                left = SDF3(left.f, lambda q, _c=c: _c)
            return c
        return SDF3(f, cf if _color.has_color(a) or
                    any(_color.has_color(b) for b in bs) else None)
    return op

union = _boolean('union', _color.union_color)
difference = _boolean('difference', _color.difference_color)
intersection = _boolean('intersection', _color.intersection_color)

def blend(a, *bs, k=0.5):
    def f(p):
        return dn.blend(a, *bs, k=k)(p)
    def cf(p):
        c = _color.color_of(a, p)
        left = a
        for b in bs:
            K = k or getattr(b, '_k', None)
            bc = _color.blend_color(left, b, K if K is not None else 0.5)
            if bc is None:
                c = _color.color_of(left, p)
            else:
                c = bc(p)
            left = SDF3(left.f, lambda q, _c=c: _c)
        return c
    return SDF3(f, cf if _color.has_color(a) or
                any(_color.has_color(b) for b in bs) else None)

def negate(other):
    return SDF3(dn.negate(other), _color.unary_color(other))

def dilate(other, r):
    return SDF3(dn.dilate(other, r), _color.unary_color(other))

def erode(other, r):
    return SDF3(dn.erode(other, r), _color.unary_color(other))

def shell(other, thickness):
    return SDF3(dn.shell(other, thickness), _color.unary_color(other))

def repeat(other, spacing, count=None, padding=0):
    count = np.array(count) if count is not None else None
    spacing = np.array(spacing)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def indexes_for(p):
        q = np.divide(p, spacing, out=np.zeros_like(p), where=spacing != 0)
        if count is None:
            index = np.round(q)
        else:
            index = np.clip(np.round(q), -count, count)
        return [index + n for n in
                neighbors(p.shape[-1], padding, spacing)]

    # distance behavior is exactly the original dn.repeat implementation
    f = dn.repeat(other, spacing, count, padding)

    color_fn = None
    if _color.has_color(other):
        def color_fn(p):
            indexes = indexes_for(p)
            ds = [other(p - spacing * i) for i in indexes]
            cs = [_color.color_of(other, p - spacing * i)
                  for i in indexes]
            d = ds[0]
            c = cs[0]
            for d2, c2 in zip(ds[1:], cs[1:]):
                take = (d2 < d).reshape(-1)
                c = _color.nan_select(take, c2, c)
                d = _min(d, d2)
            return c

    return SDF3(f, color_fn)

# attribute access (f.shell(...), f.repeat(...), ...) -- mirrors the
# registration that op3() used to perform for the dn-backed operations.
for _name in (
        'union', 'difference', 'intersection', 'blend',
        'negate', 'dilate', 'erode', 'shell', 'repeat'):
    _ops[_name] = globals()[_name]
