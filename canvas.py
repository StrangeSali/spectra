import pygame

def draw_circle(surface, color, center, radius):
    """Draw a circle with a static color."""
    pygame.draw.circle(surface, color, center, radius)


def draw_polygon(surface, color, points):
    """Draw a polygon with a static color."""
    pygame.draw.polygon(surface, color, points)
