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

font = pygame.font.Font(None, 32)

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

    print(state.keys())

    sound_class = state["class_name"]
    confidence = state["confidence"]
    rms = state["rms"]

# Caution: confirm the actual keys returned by classifier.get_state():
# "class_name", "confidence", "rms"


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

    fps = clock.get_fps()

    category_text = font.render(
        f"Detected: {sound_class}",
        True,
        current_color
        )

    confidence_text = font.render(
        f"Confidence: {confidence * 100:.1f}%",
        True,
        (255, 255, 255)
        )

  #confidence * 100 assumes confidence is between 0 and 1.

    fps_text = font.render(
        f"FPS: {fps:.0f}",
        True,
        (255, 255, 255)
        )

    screen.blit(category_text, (20, 20))
    screen.blit(confidence_text, (20, 55))
    screen.blit(fps_text, (20, 90))



    # 8. Update display
    pygame.display.flip()

    # 9. Limit frame rate
    clock.tick(60)


pygame.quit()
