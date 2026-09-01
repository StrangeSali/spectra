def lerp(a, b, t):
    """
    Linearly interpolate between a and b.

    t = 0   -> returns a
    t = 1   -> returns b
    t = 0.5 -> halfway between a and b
    """
    return a + (b - a) * t
# Future scope:
# quiet sound → small shape
# loud sound  → large shape

def lerp_color(color_a, color_b, t):
    """
    Linearly interpolate between two RGB colors.

    color_a and color_b are tuples like:
    (255, 0, 0)
    """

    r = lerp(color_a[0], color_b[0], t)
    g = lerp(color_a[1], color_b[1], t)
    b = lerp(color_a[2], color_b[2], t)

    return (int(r), int(g), int(b))
