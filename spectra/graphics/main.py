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

SHAPE_POSITIONS = [
    (200, 300),
    (400, 300),
    (600, 300)
]

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

    # 2. Get latest YAMNet predictions
    predictions = worker.get_latest_predictions()

    active_sounds = []
    seen_categories = set()

    for class_name, confidence in predictions:

        category = map_to_category(
            class_name,
            confidence
        )

        if (
            category != "Background"
            and category not in seen_categories
        ):
            active_sounds.append({
                "class_name": class_name,
                "category": category,
                "confidence": confidence
            })

            seen_categories.add(category)

        if len(active_sounds) == 3:
            break

    # 3. Get global audio energy
    rms = worker.get_rms()

    # RMS currently applies to the whole sound scene
    target_size = 30 + rms * 150

    current_size = lerp(
        current_size,
        target_size,
        0.1
    )

    # 4. Detect audio spike -> expanding ring
    spike_threshold = 0.02

    if rms - previous_rms > spike_threshold:
        rings.append({
            "radius": int(current_size),
            "alpha": 255
        })

    previous_rms = rms

    # 5. Add subtle particles
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

    # 6. Clear screen
    screen.fill((8, 10, 25))

    # 7. Persistent listening waveform
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

    # 8. Draw up to 3 active sounds
    for index, sound in enumerate(active_sounds):

        class_name = sound["class_name"]
        category = sound["category"]
        confidence = sound["confidence"]

        visual = SOUND_VISUALS.get(
            category,
            SOUND_VISUALS["Background"]
        )

        shape = visual["shape"]
        color = visual["color"]

        x, y = SHAPE_POSITIONS[index]

        # Confidence -> opacity
        alpha = int(80 + confidence * 175)

        rgba_color = (
            *color,
            alpha
        )

        shape_surface = pygame.Surface(
            (800, 600),
            pygame.SRCALPHA
        )

        size = int(current_size)

        if shape == "circle":

            draw_circle(
                shape_surface,
                rgba_color,
                (x, y),
                size
            )

        elif shape == "triangle":

            points = [
                (x, y - size),
                (x - size, y + size),
                (x + size, y + size)
            ]

            draw_polygon(
                shape_surface,
                rgba_color,
                points
            )

        elif shape == "polygon":

            points = [
                (x - size, y - size),
                (x + size, y - size),
                (x + size, y + size),
                (x - size, y + size)
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

    # 9. Draw expanding rings
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
                (150, 170, 220, ring["alpha"]),
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

    # 10. Draw particles
    for particle in particles:

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["life"] -= 1

        pygame.draw.circle(
            screen,
            (150, 170, 220),
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

    # 11. HUD
    fps = clock.get_fps()

    for index, sound in enumerate(active_sounds):

        category = sound["category"]
        class_name = sound["class_name"]
        confidence = sound["confidence"]

        color = SOUND_VISUALS[category]["color"]

        sound_text = font.render(
            f"{category}: {class_name} ({confidence * 100:.1f}%)",
            True,
            color
        )

        screen.blit(
            sound_text,
            (20, 20 + index * 35)
        )

    fps_text = font.render(
        f"FPS: {fps:.0f}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        fps_text,
        (20, 130)
    )

    # 12. Update display
    pygame.display.flip()

    # 13. Limit frame rate
    clock.tick(60)


worker.stop()
pygame.quit()
