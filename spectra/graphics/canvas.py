import pygame


def draw_circle(surface, color, center, radius):
    """Draw a circle using the given RGB or RGBA color."""
    pygame.draw.circle(surface, color, center, radius)


def draw_polygon(surface, color, points):
    """Draw a polygon using the given RGB or RGBA color."""
    pygame.draw.polygon(surface, color, points)
