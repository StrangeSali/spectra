import pygame


def draw_circle(surface, color, center, radius):
    """Draw a circle using the given RGB or RGBA color."""
    pygame.draw.circle(surface, color, center, radius)


def draw_polygon(surface, color, points):
    """Draw a polygon using the given RGB or RGBA color."""
    pygame.draw.polygon(surface, color, points)

def draw_car(surface, color, center, size):
    """Draw a simple stylized car."""

    x, y = center

    # Main body
    body = pygame.Rect(
        x - size,
        y - size // 3,
        size * 2,
        size // 2
    )

    pygame.draw.rect(
        surface,
        color,
        body,
        border_radius=max(4, size // 6)
    )

    # Roof
    roof = [
        (x - size // 2, y - size // 3),
        (x - size // 4, y - size),
        (x + size // 3, y - size),
        (x + size // 2, y - size // 3),
    ]

    pygame.draw.polygon(
        surface,
        color,
        roof
    )

    # Wheels
    wheel_radius = max(
        5,
        size // 5
    )

    pygame.draw.circle(
        surface,
        color,
        (
            x - size // 2,
            y + size // 4
        ),
        wheel_radius
    )

    pygame.draw.circle(
        surface,
        color,
        (
            x + size // 2,
            y + size // 4
        ),
        wheel_radius
    )


def draw_hands(surface, color, center, size):
    """Draw two stylized hands for human/clapping sounds."""

    x, y = center

    palm_width = max(
        12,
        size // 2
    )

    palm_height = max(
        20,
        size
    )

    # Left palm
    left_palm = pygame.Rect(
        x - size,
        y - palm_height // 2,
        palm_width,
        palm_height
    )

    # Right palm
    right_palm = pygame.Rect(
        x + size // 2,
        y - palm_height // 2,
        palm_width,
        palm_height
    )

    pygame.draw.ellipse(
        surface,
        color,
        left_palm
    )

    pygame.draw.ellipse(
        surface,
        color,
        right_palm
    )

    # Fingers
    finger_length = max(
        10,
        size // 2
    )

    spacing = max(
        3,
        size // 10
    )

    for i in range(4):

        left_x = (
            x
            - size
            + 5
            + i * spacing
        )

        pygame.draw.line(
            surface,
            color,
            (
                left_x,
                y - palm_height // 3
            ),
            (
                left_x,
                y - palm_height // 3
                - finger_length
            ),
            width=max(2, size // 15)
        )

        right_x = (
            x
            + size // 2
            + 5
            + i * spacing
        )

        pygame.draw.line(
            surface,
            color,
            (
                right_x,
                y - palm_height // 3
            ),
            (
                right_x,
                y - palm_height // 3
                - finger_length
            ),
            width=max(2, size // 15)
        )
