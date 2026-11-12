"""
Color support showcase.

Run with:  python examples/colored.py
Writes colored_demo.ply (vertex colors, for preview) and
colored_demo.3mf (one material per color, for Bambu Studio / PrusaSlicer).

Demonstrated features:
  * constant colors as hex strings, CSS names and RGB tuples (0-1 or 0-255)
  * smooth-union gradient (the .k() blend zone becomes a color gradient)
  * difference: the drilled hole wall takes the cutter's color
  * a procedural stripe pattern that follows a twist deformation
  * an image's own colors carried through an extrusion
  * configurable default color for any uncolored surface
"""

import os

import numpy as np

from sdf import *

HERE = os.path.dirname(os.path.abspath(__file__))


def twisted_stripes():
    """Procedural RGB stripes in local coordinates; twist carries them."""
    def stripes(p):
        # sine bands along local x, in 0-255 RGB to show that convention too
        band = (np.sin(p[:, 0] * 14) > 0).astype(float)
        red = np.stack([255 * band, 40 * band, 60 * band], axis=1)
        white = np.full((len(p), 3), 235.0)
        return np.where((band > 0)[:, None], red, white)

    # rounded_cylinder stays watertight after twist (a sharp-edged cylinder
    # exposes a pre-existing marching-cubes seam in the underlying library)
    return rounded_cylinder(0.55, 0.08, 1.2).color(stripes).twist(2.0)


def smooth_blend_part():
    """Two solids fused with .k() -> their colors blend across the seam."""
    a = sphere(0.8).color('#e63946')                      # hex
    b = box(1.2).color((40, 120, 200)).k(0.35)           # 0-255 tuple
    return (a | b).translate((-2.2, 0, 0))


def drilled_part():
    """A block drilled by a colored cutter: the hole wall is the cutter color."""
    block = box((1.4, 1.4, 1.4)).color('steelblue')      # CSS color name
    drill = cylinder(0.45).orient(X).color('yellow')
    return (block - drill).translate((0.0, 0, 0))


def logo_plate():
    """A colored image extruded into a nameplate, picture colors intact."""
    image_path = os.path.join(HERE, 'butterfly.png')
    w, h = measure_image(image_path)
    plate = rounded_box((w * 1.15, h * 1.15, 0.12), 0.05).color((35, 35, 40))
    logo = image(image_path).extrude(0.1) & slab(z0=0, z1=0.08)
    return (plate | logo).translate((2.4, 0, 0))


def main():
    model = smooth_blend_part() | drilled_part()
    model |= twisted_stripes().translate((0, 1.8, 0))
    model |= logo_plate()

    # explicit bounds: image-based SDFs can confuse the automatic estimator
    bounds = ((-3.6, -2.2, -1.4), (3.6, 2.8, 1.4))
    common = dict(samples=2 ** 18, verbose=True,
                  default_color=(180, 180, 180), bounds=bounds)

    model.save('colored_demo.ply', **common)
    model.save('colored_demo.3mf', **common)
    print('wrote colored_demo.ply and colored_demo.3mf')


if __name__ == '__main__':
    main()
