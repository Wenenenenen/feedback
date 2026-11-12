"""
Color support for SDFs.

A *color function* maps an array of sample points (shape (n, 2) or (n, 3))
to an RGBA array of shape (n, 4) with channels in [0, 1].

``nan`` values are used as the "no color here" sentinel. A color function may
return all-nan rows (e.g. an uncolored operand of a boolean operation); the
helpers in this module propagate and combine them safely. When a finished
mesh still contains nan colors at export time, a default color is substituted.
"""

import numpy as np

# ---------------------------------------------------------------------------
# CSS4 named colors (name -> #rrggbb). No new dependencies are added, so the
# table is embedded here instead of pulling in matplotlib.colors at runtime.
# ---------------------------------------------------------------------------

NAMED_COLORS = {
    'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7', 'aqua': '#00ffff',
    'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc',
    'bisque': '#ffe4c4', 'black': '#000000', 'blanchedalmond': '#ffebcd',
    'blue': '#0000ff', 'blueviolet': '#8a2be2', 'brown': '#a52a2a',
    'burlywood': '#deb887', 'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00',
    'chocolate': '#d2691e', 'coral': '#ff7f50', 'cornflowerblue': '#6495ed',
    'cornsilk': '#fff8dc', 'crimson': '#dc143c', 'cyan': '#00ffff',
    'darkblue': '#00008b', 'darkcyan': '#008b8b', 'darkgoldenrod': '#b8860b',
    'darkgray': '#a9a9a9', 'darkgreen': '#006400', 'darkgrey': '#a9a9a9',
    'darkkhaki': '#bdb76b', 'darkmagenta': '#8b008b',
    'darkolivegreen': '#556b2f', 'darkorange': '#ff8c00',
    'darkorchid': '#9932cc', 'darkred': '#8b0000', 'darksalmon': '#e9967a',
    'darkseagreen': '#8fbc8f', 'darkslateblue': '#483d8b',
    'darkslategray': '#2f4f4f', 'darkslategrey': '#2f4f4f',
    'darkturquoise': '#00ced1', 'darkviolet': '#9400d3', 'deeppink': '#ff1493',
    'deepskyblue': '#00bfff', 'dimgray': '#696969', 'dimgrey': '#696969',
    'dodgerblue': '#1e90ff', 'firebrick': '#b22222', 'floralwhite': '#fffaf0',
    'forestgreen': '#228b22', 'fuchsia': '#ff00ff', 'gainsboro': '#dcdcdc',
    'ghostwhite': '#f8f8ff', 'gold': '#ffd700', 'goldenrod': '#daa520',
    'gray': '#808080', 'green': '#008000', 'greenyellow': '#adff2f',
    'grey': '#808080', 'honeydew': '#f0fff0', 'hotpink': '#ff69b4',
    'indianred': '#cd5c5c', 'indigo': '#4b0082', 'ivory': '#fffff0',
    'khaki': '#f0e68c', 'lavender': '#e6e6fa', 'lavenderblush': '#fff0f5',
    'lawngreen': '#7cfc00', 'lemonchiffon': '#fffacd', 'lightblue': '#add8e6',
    'lightcoral': '#f08080', 'lightcyan': '#e0ffff',
    'lightgoldenrodyellow': '#fafad2', 'lightgray': '#d3d3d3',
    'lightgreen': '#90ee90', 'lightgrey': '#d3d3d3', 'lightpink': '#ffb6c1',
    'lightsalmon': '#ffa07a', 'lightseagreen': '#20b2aa',
    'lightskyblue': '#87cefa', 'lightslategray': '#778899',
    'lightslategrey': '#778899', 'lightsteelblue': '#b0c4de',
    'lightyellow': '#ffffe0', 'lime': '#00ff00', 'limegreen': '#32cd32',
    'linen': '#faf0e6', 'magenta': '#ff00ff', 'maroon': '#800000',
    'mediumaquamarine': '#66cdaa', 'mediumblue': '#0000cd',
    'mediumorchid': '#ba55d3', 'mediumpurple': '#9370db',
    'mediumseagreen': '#3cb371', 'mediumslateblue': '#7b68ee',
    'mediumspringgreen': '#00fa9a', 'mediumturquoise': '#48d1cc',
    'mediumvioletred': '#c71585', 'midnightblue': '#191970',
    'mintcream': '#f5fffa', 'mistyrose': '#ffe4e1', 'moccasin': '#ffe4b5',
    'navajowhite': '#ffdead', 'navy': '#000080', 'oldlace': '#fdf5e6',
    'olive': '#808000', 'olivedrab': '#6b8e23', 'orange': '#ffa500',
    'orangered': '#ff4500', 'orchid': '#da70d6', 'palegoldenrod': '#eee8aa',
    'palegreen': '#98fb98', 'paleturquoise': '#afeeee',
    'palevioletred': '#db7093', 'papayawhip': '#ffefd5',
    'peachpuff': '#ffdab9', 'peru': '#cd853f', 'pink': '#ffc0cb',
    'plum': '#dda0dd', 'powderblue': '#b0e0e6', 'purple': '#800080',
    'rebeccapurple': '#663399', 'red': '#ff0000', 'rosybrown': '#bc8f8f',
    'royalblue': '#4169e1', 'saddlebrown': '#8b4513', 'salmon': '#fa8072',
    'sandybrown': '#f4a460', 'seagreen': '#2e8b57', 'seashell': '#fff5ee',
    'sienna': '#a0522d', 'silver': '#c0c0c0', 'skyblue': '#87ceeb',
    'slateblue': '#6a5acd', 'slategray': '#708090', 'slategrey': '#708090',
    'snow': '#fffafa', 'springgreen': '#00ff7f', 'steelblue': '#4682b4',
    'tan': '#d2b48c', 'teal': '#008080', 'thistle': '#d8bfd8',
    'tomato': '#ff6347', 'turquoise': '#40e0d0', 'violet': '#ee82ee',
    'wheat': '#f5deb3', 'white': '#ffffff', 'whitesmoke': '#f5f5f5',
    'yellow': '#ffff00', 'yellowgreen': '#9acd32',
}

DEFAULT_COLOR = (0.8, 0.8, 0.8)


def parse_color(value):
    """
    Convert a color specification to an RGBA tuple of floats in [0, 1].

    Accepted inputs:
      * hex strings: '#rgb', '#rgba', '#rrggbb', '#rrggbbaa' (with or
        without the leading '#')
      * CSS4 color names, e.g. 'yellow' (case-insensitive)
      * sequences of 3 or 4 numbers. Channels are interpreted as 0-255 when
        any channel exceeds 1.0, otherwise as 0-1 floats.
    """
    if isinstance(value, str):
        return _parse_string(value)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] not in (3, 4):
        raise ValueError('color must have 3 or 4 channels, got %d' %
                         arr.shape[0])
    if np.any(arr > 1.0):
        arr = arr / 255.0
    rgb = np.clip(arr[:3], 0, 1)
    a = arr[3] if arr.shape[0] == 4 else 1.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]),
            float(np.clip(a, 0, 1)))


def _parse_string(value):
    s = value.strip().lower()
    if s in NAMED_COLORS:
        s = NAMED_COLORS[s]
    if not s.startswith('#'):
        # accept bare hex strings such as 'e63946' or 'fff'
        if all(ch in '0123456789abcdef' for ch in s) and len(s) in (3, 4, 6, 8):
            s = '#' + s
        else:
            raise ValueError('unknown color name: %r' % value)
    h = s[1:]
    if len(h) == 3:
        h = ''.join(c * 2 for c in h) + 'ff'
    elif len(h) == 4:
        h = ''.join(c * 2 for c in h)
    elif len(h) == 6:
        h = h + 'ff'
    if len(h) != 8:
        raise ValueError('invalid color: %r' % value)
    try:
        vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6)]
    except ValueError:
        raise ValueError('invalid color: %r' % value)
    return tuple(vals)


def make_color_function(spec):
    """
    Build a color function from a constant color or a callable.

    A callable receives the raw sample points (the same local-coordinate
    array the distance function sees) and may return anything
    :func:`parse_color` understands per row, or an (n, 3|4) array.
    """
    if callable(spec) and not isinstance(spec, (tuple, list, np.ndarray)):
        user_fn = spec

        def f(p):
            out = user_fn(p)
            return _coerce_rows(out, len(p))

        return f
    rgba = parse_color(spec)
    constant = np.array(rgba, dtype=float)

    def f(p):
        return np.repeat(constant.reshape(1, 4), len(p), axis=0)

    return f


def _coerce_rows(out, n):
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr, (n, arr.shape[0]))
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] == 3:
        alpha = np.ones((n, 1))
        arr = np.hstack([arr, alpha])
    # 0-255 inputs from procedural color functions are supported too.
    if np.nanmax(arr[:, :3]) > 1.0:
        arr = arr.copy()
        arr[:, :3] = arr[:, :3] / 255.0
        if np.nanmax(arr[:, 3]) > 1.0:
            arr[:, 3] = arr[:, 3] / 255.0
    return arr


def no_color(p):
    """Color function that marks every point as uncolored."""
    return np.full((len(p), 4), np.nan)


def has_color(sdf):
    return getattr(sdf, 'color_fn', None) is not None


def color_of(sdf, p):
    """Evaluate an SDF's color function, returning all-nan when uncolored."""
    fn = getattr(sdf, 'color_fn', None)
    if fn is None:
        return np.full((len(p), 4), np.nan)
    return np.asarray(fn(p), dtype=float)


# ---------------------------------------------------------------------------
# NaN-safe color combinators.
# ---------------------------------------------------------------------------

def _masked(c):
    return ~np.isnan(c[:, 0])


def nan_blend(c1, c2, w):
    """
    Interpolate between two color arrays with weight ``w`` (0 -> c1, 1 -> c2).
    Rows that are nan on one side take the other side; all-nan stays nan.
    ``w`` may be a scalar or an (n, 1) array.
    """
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    m1 = _masked(c1)
    m2 = _masked(c2)
    out = np.full_like(c1, np.nan)
    both = m1 & m2
    one = m1 ^ m2
    if np.any(both):
        out[both] = c1[both] * (1 - w[both]) + c2[both] * w[both]
    if np.any(one):
        pick2 = one & m2
        pick1 = one & m1
        if np.any(pick2):
            out[pick2] = c2[pick2]
        if np.any(pick1):
            out[pick1] = c1[pick1]
    return out


def nan_select(cond, c_if, c_else):
    """Element-wise ``where`` that is safe against all-nan rows."""
    out = np.full_like(c_if, np.nan)
    cond = cond.reshape(-1)
    out[cond] = c_if[cond]
    out[~cond] = c_else[~cond]
    return out


def transform_color(color_fn, point_fn):
    """Wrap a color function so patterns follow a geometric transform."""
    def f(p):
        return color_fn(point_fn(p))
    return f


# ---------------------------------------------------------------------------
# Color functions mirroring the boolean operations in dn.py.
# Each returns a color function (or None when nothing carries color).
# ---------------------------------------------------------------------------

def union_color(a, b, k=None):
    ca, cb = getattr(a, 'color_fn', None), getattr(b, 'color_fn', None)
    if ca is None and cb is None:
        return None

    def f(p):
        c1 = color_of(a, p)
        c2 = color_of(b, p)
        d1 = a(p)
        d2 = b(p)
        K = k or getattr(b, '_k', None)
        if K is None:
            return nan_select(d2 < d1, c2, c1)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
        return nan_blend(c1, c2, h)

    return f


def difference_color(a, b, k=None):
    ca, cb = getattr(a, 'color_fn', None), getattr(b, 'color_fn', None)
    if ca is None and cb is None:
        return None

    def f(p):
        c1 = color_of(a, p)
        c2 = color_of(b, p)
        d1 = a(p)
        d2 = b(p)
        K = k or getattr(b, '_k', None)
        if K is None:
            # Hole walls take the cutter's color; everywhere else keeps the
            # base object's color. An uncolored cutter yields the base color.
            return nan_select(-d2 > d1, c2, c1)
        h = np.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
        return nan_blend(c1, c2, h)

    return f


def intersection_color(a, b, k=None):
    ca, cb = getattr(a, 'color_fn', None), getattr(b, 'color_fn', None)
    if ca is None and cb is None:
        return None

    def f(p):
        c1 = color_of(a, p)
        c2 = color_of(b, p)
        d1 = a(p)
        d2 = b(p)
        K = k or getattr(b, '_k', None)
        if K is None:
            return nan_select(d2 > d1, c2, c1)
        h = np.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
        return nan_blend(c1, c2, h)

    return f


def blend_color(a, b, k=0.5):
    ca, cb = getattr(a, 'color_fn', None), getattr(b, 'color_fn', None)
    if ca is None and cb is None:
        return None
    K = k or getattr(b, '_k', None)
    if K is None:
        K = 0.5

    def f(p):
        c1 = color_of(a, p)
        c2 = color_of(b, p)
        return nan_blend(c1, c2, K)

    return f


def unary_color(other):
    """negate / dilate / erode / shell keep the operand's color."""
    return getattr(other, 'color_fn', None)
