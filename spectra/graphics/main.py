import math
import random
import pygame

from spectra.graphics.canvas import draw_circle, draw_polygon
from spectra.graphics.visualizer import lerp, lerp_color
from spectra.models.inference_worker import YAMNetInferenceWorker
from spectra.models.category_mapping import map_to_category


SOUND_VISUALS = {
    "Alert": {
        "shape": "triangle",
        "color": (255, 50, 80)
    },
    "Human": {
        "shape": "circle",
        "color": (255, 180, 50)
    },
    "Vehicle": {
        "shape": "polygon",
        "color": (50, 150, 255)
    },
    "Animal": {
        "shape": "circle",
        "color": (120, 220, 130)
    },
    "Music": {
        "shape": "polygon",
        "color": (180, 100, 255)
    },
    "Background": {
        "shape": "circle",
        "color": (120, 130, 150)
    }
}


pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Spectra AI")

font = pygame.font.Font(None, 32)
clock = pygame.time.Clock()

worker = YAMNetInferenceWorker()
worker.start()

running = True

current_size = 40
current_color = (255, 255, 255)

previous_rms = 0.0
rings = []
particles = []


while running:

    # 1. Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. Get latest model prediction
    predictions = worker.get_latest_predictions()

    if predictions:
        class_name, confidence = predictions[0]
    else:
        class_name = "Background"
        confidence = 0.0

    # Convert raw YAMNet class to accessibility category
    category = map_to_category(
        class_name,
        confidence
    )

    # Get current audio energy
    rms = worker.get_rms()

    # 3. Find visual identity
    visual = SOUND_VISUALS.get(
        category,
        SOUND_VISUALS["Background"]
    )

    target_color = visual["color"]
    shape = visual["shape"]

    # 4. RMS controls visual energy / size
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

    # 6. Detect audio spike -> expanding ring
    spike_threshold = 0.02

    if rms - previous_rms > spike_threshold:
        rings.append({
            "radius": int(current_size),
            "alpha": 255
        })

    previous_rms = rms

    # 7. Add subtle particles when sound is active
    if rms > 0.05:
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.0)

        particles.append({
            "x": 400,
            "y": 300,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": 60
        })

    # 8. Clear screen
    screen.fill((8, 10, 25))

    # 9. Persistent listening waveform
    wave_points = []

    time_offset = pygame.time.get_ticks() * 0.003
    wave_amplitude = 10 + rms * 40

    for x in range(0, 800, 8):
        y = 500 + math.sin(
            x * 0.03 + time_offset
        ) * wave_amplitude

        wave_points.append((x, int(y)))

    pygame.draw.lines(
        screen,
        (80, 100, 140),
        False,
        wave_points,
        2
    )

    # 10. Confidence controls visual clarity
    alpha = int(80 + confidence * 175)

    shape_surface = pygame.Surface(
        (800, 600),
        pygame.SRCALPHA
    )

    rgba_color = (
        *current_color,
        alpha
    )

    # 11. Draw main shape
    if shape == "circle":

        draw_circle(
            shape_surface,
            rgba_color,
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
            shape_surface,
            rgba_color,
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
            shape_surface,
            rgba_color,
            points
        )

    screen.blit(
        shape_surface,
        (0, 0)
    )

    # 12. Draw expanding rings
    for ring in rings:

        ring["radius"] += 4
        ring["alpha"] -= 8

        if ring["alpha"] > 0:

            ring_surface = pygame.Surface(
                (800, 600),
                pygame.SRCALPHA
            )

            pygame.draw.circle(
                ring_surface,
                (
                    *current_color,
                    ring["alpha"]
                ),
                (400, 300),
                ring["radius"],
                width=3
            )

            screen.blit(
                ring_surface,
                (0, 0)
            )

    rings = [
        ring
        for ring in rings
        if ring["alpha"] > 0
    ]

    # 13. Draw particles
    for particle in particles:

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["life"] -= 1

        pygame.draw.circle(
            screen,
            current_color,
            (
                int(particle["x"]),
                int(particle["y"])
            ),
            3
        )

    particles = [
        particle
        for particle in particles
        if particle["life"] > 0
    ]

    # 14. HUD
    fps = clock.get_fps()

    category_text = font.render(
        f"{category}: {class_name}",
        True,
        current_color
    )

    confidence_text = font.render(
        f"Confidence: {confidence * 100:.1f}%",
        True,
        (255, 255, 255)
    )

    fps_text = font.render(
        f"FPS: {fps:.0f}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        category_text,
        (20, 20)
    )

    screen.blit(
        confidence_text,
        (20, 55)
    )

    screen.blit(
        fps_text,
        (20, 90)
    )

    # 15. Update display
    pygame.display.flip()

    # 16. Limit frame rate
    clock.tick(60)


worker.stop()
pygame.quit()
