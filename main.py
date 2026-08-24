import pygame

from canvas import draw_circle, draw_polygon
from graphics.visualizer import lerp, lerp_color
import classifier


SOUND_VISUALS = {
    "Clapping": {
        "shape": "circle",
        "color": (255, 180, 50)
    },
    "Siren": {
        "shape": "triangle",
        "color": (255, 50, 80)
    },
    "Car": {
        "shape": "polygon",
        "color": (50, 150, 255)
    }
}


pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Spectra AI")

clock = pygame.time.Clock()

running = True

current_size = 40
current_color = (255, 255, 255)


while running:

    # 1. Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. Get latest sound information
    state = classifier.get_state()

    sound_class = state["class_name"]
    rms = state["rms"]

    # 3. Find the visual style for that sound
    visual = SOUND_VISUALS.get(
        sound_class,
        {
            "shape": "circle",
            "color": (255, 255, 255)
        }
    )

    target_color = visual["color"]
    shape = visual["shape"]

    # 4. RMS controls target size
    target_size = 30 + rms * 150

    # 5. Smooth transitions
    current_size = lerp(
        current_size,
        target_size,
        0.1
    )

    current_color = lerp_color(
        current_color,
        target_color,
        0.1
    )

    # 6. Clear screen
    screen.fill((0, 0, 0))

    # 7. Draw based on sound class
    if shape == "circle":

        draw_circle(
            screen,
            current_color,
            (400, 300),
            int(current_size)
        )

    elif shape == "triangle":

        size = int(current_size)

        points = [
            (400, 300 - size),
            (400 - size, 300 + size),
            (400 + size, 300 + size)
        ]

        draw_polygon(
            screen,
            current_color,
            points
        )

    elif shape == "polygon":

        size = int(current_size)

        points = [
            (400 - size, 300 - size),
            (400 + size, 300 - size),
            (400 + size, 300 + size),
            (400 - size, 300 + size)
        ]

        draw_polygon(
            screen,
            current_color,
            points
        )

    # 8. Update display
    pygame.display.flip()

    # 9. Limit frame rate
    clock.tick(60)


pygame.quit()
