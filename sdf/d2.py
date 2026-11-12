import functools
import itertools
import numpy as np
import operator

from . import color as _color
from . import dn, d3, ease

# Constants

ORIGIN = np.array((0, 0))

X = np.array((1, 0))
Y = np.array((0, 1))

UP = Y

# SDF Class

_ops = {}

class SDF2:
    def __init__(self, f, color_fn=None):
        self.f = f
        self.color_fn = color_fn
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
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
        """Attach a constant or procedural color (see sdf.color for details)."""
        return SDF2(self.f, _color.make_color_function(spec))

def sdf2(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, SDF2):
            return result
        if isinstance(result, tuple):
            return SDF2(result[0], result[1])
        return SDF2(result)
    return wrapper

def op2(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, SDF2):
            return result
        if isinstance(result, tuple):
            return SDF2(result[0], result[1])
        return SDF2(result)
    _ops[f.__name__] = wrapper
    return wrapper

def op23(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, d3.SDF3):
            return result
        if isinstance(result, tuple):
            return d3.SDF3(result[0], result[1])
        return d3.SDF3(result)
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

_min = np.minimum
_max = np.maximum

# Primitives

@sdf2
def circle(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - center) - radius
    return f

@sdf2
def line(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return np.dot(point - p, normal)
    return f

@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)

@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return rectangle(size, center)
    size = np.array(size)
    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)
    return f

@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    try:
        r0, r1, r2, r3 = radius
    except TypeError:
        r0 = r1 = r2 = r3 = radius
    def f(p):
        x = p[:,0]
        y = p[:,1]
        r = np.zeros(len(p)).reshape((-1, 1))
        r[np.logical_and(x > 0, y > 0)] = r0
        r[np.logical_and(x > 0, y <= 0)] = r1
        r[np.logical_and(x <= 0, y <= 0)] = r2
        r[np.logical_and(x <= 0, y > 0)] = r3
        q = np.abs(p) - size / 2 + r
        return (
            _min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) +
            _length(_max(q, 0)).reshape((-1, 1)) - r)
    return f

@sdf2
def equilateral_triangle():
    def f(p):
        k = 3 ** 0.5
        p = _vec(
            np.abs(p[:,0]) - 1,
            p[:,1] + 1 / k)
        w = p[:,0] + k * p[:,1] > 0
        q = _vec(
            p[:,0] - k * p[:,1],
            -k * p[:,0] - p[:,1]) / 2
        p = np.where(w.reshape((-1, 1)), q, p)
        p = _vec(
            p[:,0] - np.clip(p[:,0], -2, 0),
            p[:,1])
        return -_length(p) * np.sign(p[:,1])
    return f

@sdf2
def hexagon(r):
    r *= 3 ** 0.5 / 2
    def f(p):
        k = np.array((3 ** 0.5 / -2, 0.5, np.tan(np.pi / 6)))
        p = np.abs(p)
        p -= 2 * k[:2] * _min(_dot(k[:2], p), 0).reshape((-1, 1))
        p -= _vec(
            np.clip(p[:,0], -k[2] * r, k[2] * r),
            np.zeros(len(p)) + r)
        return _length(p) * np.sign(p[:,1])
    return f

@sdf2
def rounded_x(w, r):
    def f(p):
        p = np.abs(p)
        q = (_min(p[:,0] + p[:,1], w) * 0.5).reshape((-1, 1))
        return _length(p - q) - r
    return f

@sdf2
def polygon(points):
    points = [np.array(p) for p in points]
    def f(p):
        n = len(points)
        d = _dot(p - points[0], p - points[0])
        s = np.ones(len(p))
        for i in range(n):
            j = (i + n - 1) % n
            vi = points[i]
            vj = points[j]
            e = vj - vi
            w = p - vi
            b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:,1] >= vi[1]
            c2 = p[:,1] < vj[1]
            c3 = e[0] * w[:,1] > e[1] * w[:,0]
            c = _vec(c1, c2, c3)
            s = np.where(np.all(c, axis=1) | np.all(~c, axis=1), -s, s)
        return s * np.sqrt(d)
    return f

@sdf2
def vesica(r, d):
    def f(p):
        p = np.abs(p)
        b = np.sqrt(r * r - d * d)
        return np.where(
            ((p[:,1] - b) * d > p[:,0] * b),
            _length(p - np.array([0, b])),
            _length(p - np.array([-d, 0])) - r)
    return f

# Positioning

@op2
def translate(other, offset):
    def f(p):
        return other(p - offset)
    def cf(p):
        return _color.color_of(other, p - offset)
    return f, cf

@op2
def scale(other, factor):
    try:
        x, y = factor
    except TypeError:
        x = y = factor
    s = (x, y)
    m = min(x, y)
    def f(p):
        return other(p / s) * m
    def cf(p):
        return _color.color_of(other, p / s)
    return f, cf

@op2
def rotate(other, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [c, -s],
        [s, c],
    ]).T
    def f(p):
        return other(np.dot(p, matrix))
    def cf(p):
        return _color.color_of(other, np.dot(p, matrix))
    return f, cf

@op2
def circular_array(other, count):
    angles = [i / count * 2 * np.pi for i in range(count)]
    return union(*[other.rotate(a) for a in angles])

# Alterations

@op2
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        w = _min(_max(x, y), 0)
        return other(_max(q, 0)) + w
    def cf(p):
        q = np.abs(p) - size
        return _color.color_of(other, _max(q, 0))
    return f, cf

# 2D => 3D Operations

@op23
def extrude(other, h):
    def f(p):
        d = other(p[:,[0,1]])
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    def cf(p):
        return _color.color_of(other, p[:,[0,1]])
    return f, cf

@op23
def extrude_to(a, b, h, e=ease.linear):
    def f(p):
        d1 = a(p[:,[0,1]])
        d2 = b(p[:,[0,1]])
        t = e(np.clip(p[:,2] / h, -0.5, 0.5) + 0.5)
        d = d1 + (d2 - d1) * t.reshape((-1, 1))
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    def cf(p):
        t = e(np.clip(p[:,2] / h, -0.5, 0.5) + 0.5).reshape((-1, 1))
        return _color.nan_blend(
            _color.color_of(a, p[:,[0,1]]),
            _color.color_of(b, p[:,[0,1]]), t)
    return f, cf

@op23
def revolve(other, offset=0):
    def f(p):
        xy = p[:,[0,1]]
        q = _vec(_length(xy) - offset, p[:,2])
        return other(q)
    def cf(p):
        xy = p[:,[0,1]]
        q = _vec(_length(xy) - offset, p[:,2])
        return _color.color_of(other, q)
    return f, cf

# Common

def _boolean(op_name, color_fn):
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
                left = SDF2(left.f, lambda q, _c=c: _c)
            return c
        return SDF2(f, cf if _color.has_color(a) or
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
            left = SDF2(left.f, lambda q, _c=c: _c)
        return c
    return SDF2(f, cf if _color.has_color(a) or
                any(_color.has_color(b) for b in bs) else None)

def negate(other):
    return SDF2(dn.negate(other), _color.unary_color(other))

def dilate(other, r):
    return SDF2(dn.dilate(other, r), _color.unary_color(other))

def erode(other, r):
    return SDF2(dn.erode(other, r), _color.unary_color(other))

def shell(other, thickness):
    return SDF2(dn.shell(other, thickness), _color.unary_color(other))

def repeat(other, spacing, count=None, padding=0):
    f = dn.repeat(other, spacing, count, padding)
    color_fn = None
    if _color.has_color(other):
        count_arr = np.array(count) if count is not None else None
        spacing_arr = np.array(spacing)

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
            q = np.divide(
                p, spacing_arr, out=np.zeros_like(p),
                where=spacing_arr != 0)
            if count_arr is None:
                index = np.round(q)
            else:
                index = np.clip(np.round(q), -count_arr, count_arr)
            return [index + n for n in
                    neighbors(p.shape[-1], padding, spacing_arr)]

        def color_fn(p):
            indexes = indexes_for(p)
            ds = [other(p - spacing_arr * i) for i in indexes]
            cs = [_color.color_of(other, p - spacing_arr * i)
                  for i in indexes]
            d = ds[0]
            c = cs[0]
            for d2, c2 in zip(ds[1:], cs[1:]):
                take = (d2 < d).reshape(-1)
                c = _color.nan_select(take, c2, c)
                d = _min(d, d2)
            return c

    return SDF2(f, color_fn)

# attribute access (f.shell(...), f.repeat(...), ...) -- mirrors the
# registration that op2() used to perform for the dn-backed operations.
for _name in (
        'union', 'difference', 'intersection', 'blend',
        'negate', 'dilate', 'erode', 'shell', 'repeat'):
    _ops[_name] = globals()[_name]
