import math
import random
import pygame

from spectra.graphics.canvas import draw_circle, draw_polygon
from spectra.graphics.visualizer import lerp
from spectra.models.inference_worker import YAMNetInferenceWorker
from spectra.processing.category_mapping import (
    aggregate_category_scores,
    map_audioset_class,
)


SOUND_VISUALS = {
    "Alarms": {
        "shape": "triangle",
        "color": (255, 50, 80)
    },

    "Human": {
        "shape": "circle",
        "color": (255, 180, 50)
    },

    "Traffic": {
        "shape": "polygon",
        "color": (50, 150, 255)
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


DISPLAY_CATEGORIES = {
    "Alarms",
    "Human",
    "Traffic",
    "Music"
}


SHAPE_POSITIONS = [
    (200, 300),
    (400, 300),
    (600, 300)
]


MIN_CATEGORY_SCORE = 0.20
ACTIVE_TIMEOUT_MS = 500
SILENCE_THRESHOLD = 0.60

DEBUG_INTERVAL_MS = 2000


pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Spectra AI")

font = pygame.font.Font(None, 32)
clock = pygame.time.Clock()


worker = YAMNetInferenceWorker()
worker.start()


running = True

previous_rms = 0.0

rings = []
particles = []

active_categories = {}

# Each category keeps its own animation state
shape_states = {}

last_debug_print = 0


while running:

    now = pygame.time.get_ticks()

    # 1. Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. Get latest raw YAMNet predictions
    predictions = worker.get_latest_predictions()

    # 3. Aggregate raw predictions into Spectra categories
    category_scores = aggregate_category_scores(
        predictions
    )

    # Keep only MVP categories and remove weak guesses
    category_scores = {
        category: score
        for category, score in category_scores.items()
        if (
            category in DISPLAY_CATEGORIES
            and score >= MIN_CATEGORY_SCORE
        )
    }

    # 4. Detect silence directly
    silence_score = 0.0

    for class_name, confidence in predictions:
        if class_name == "Silence":
            silence_score = confidence
            break

    is_silent = (
        silence_score >= SILENCE_THRESHOLD
        and not category_scores
    )

    # 5. Update persistent active categories
    for category, score in category_scores.items():

        matching_predictions = [
            (class_name, confidence)
            for class_name, confidence in predictions
            if map_audioset_class(class_name) == category
        ]

        if matching_predictions:
            best_class_name, best_class_confidence = max(
                matching_predictions,
                key=lambda item: item[1]
            )
        else:
            best_class_name = category

        existing = active_categories.get(category)

        if (
            existing is None
            or score >= existing["confidence"]
        ):
            active_categories[category] = {
                "class_name": best_class_name,
                "confidence": score,
                "last_seen": now
            }

        else:
            existing["last_seen"] = now

    # 6. Remove stale categories
    active_categories = {
        category: data
        for category, data in active_categories.items()
        if now - data["last_seen"] < ACTIVE_TIMEOUT_MS
    }

    # 7. Build drawable sounds
    active_sounds = [
        {
            "category": category,
            "class_name": data["class_name"],
            "confidence": data["confidence"]
        }
        for category, data in active_categories.items()
    ]

    active_sounds.sort(
        key=lambda sound: sound["confidence"],
        reverse=True
    )

    active_sounds = active_sounds[:3]

    # 8. Background fallback
    if not active_sounds:
        active_sounds = [{
            "category": "Background",
            "class_name": "Silence" if is_silent else "Background",
            "confidence": silence_score if is_silent else 0.0
        }]

    # 9. Debug every 2 seconds
    if now - last_debug_print >= DEBUG_INTERVAL_MS:

        print("\n--- DEBUG ---")
        print("RAW:", predictions)
        print("CATEGORY SCORES:", category_scores)
        print("ACTIVE SOUNDS:", active_sounds)
        print("SILENCE SCORE:", silence_score)

        last_debug_print = now

    # 10. Global audio energy
    rms = worker.get_rms()

    # 11. Detect audio spike -> expanding ring
    spike_threshold = 0.02

    if rms - previous_rms > spike_threshold:

        for index, sound in enumerate(active_sounds):

            if sound["category"] == "Background":
                continue

            x, y = SHAPE_POSITIONS[index]

            rings.append({
                "x": x,
                "y": y,
                "radius": 40,
                "alpha": 255,
                "color": SOUND_VISUALS[
                    sound["category"]
                ]["color"]
            })

    previous_rms = rms

    # 12. Create particles around active shapes
    if rms > 0.05:

        for index, sound in enumerate(active_sounds):

            category = sound["category"]

            if category == "Background":
                continue

            x, y = SHAPE_POSITIONS[index]

            angle = random.uniform(
                0,
                2 * math.pi
            )

            speed = random.uniform(
                0.5,
                2.0
            )

            particles.append({
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": 60,
                "color": SOUND_VISUALS[
                    category
                ]["color"]
            })

    # 13. Clear screen
    screen.fill(
        (8, 10, 25)
    )

    # 14. Persistent listening waveform
    wave_points = []

    time_offset = (
        pygame.time.get_ticks()
        * 0.003
    )

    wave_amplitude = (
        10
        + rms * 40
    )

    for x in range(
        0,
        800,
        8
    ):

        y = (
            500
            + math.sin(
                x * 0.03
                + time_offset
            )
            * wave_amplitude
        )

        wave_points.append(
            (x, int(y))
        )

    pygame.draw.lines(
        screen,
        (80, 100, 140),
        False,
        wave_points,
        2
    )

    # 15. Draw active sounds
    for index, sound in enumerate(
        active_sounds
    ):

        category = sound["category"]
        class_name = sound["class_name"]

        confidence = max(
            0.0,
            min(
                sound["confidence"],
                1.0
            )
        )

        visual = SOUND_VISUALS.get(
            category,
            SOUND_VISUALS["Background"]
        )

        shape = visual["shape"]
        color = visual["color"]

        if category == "Background":
            x, y = (400, 300)
        else:
            x, y = SHAPE_POSITIONS[index]

        # Create per-category animation state
        if category not in shape_states:
            shape_states[category] = {
                "size": 30.0,
                "alpha": 60.0
            }

        state = shape_states[category]

        # Stronger confidence = bigger + more opaque
        if category == "Background":
            target_size = 45
            target_alpha = 130
        else:
            target_size = 35 + confidence * 90
            target_alpha = 50 + confidence * 205

        # Smoothly animate toward target
        state["size"] = lerp(
            state["size"],
            target_size,
            0.15
        )

        state["alpha"] = lerp(
            state["alpha"],
            target_alpha,
            0.15
        )

        size = int(
            state["size"]
        )

        alpha = int(
            max(
                0,
                min(
                    state["alpha"],
                    255
                )
            )
        )

        rgba_color = (
            *color,
            alpha
        )

        shape_surface = pygame.Surface(
            (800, 600),
            pygame.SRCALPHA
        )

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

    # 16. Draw expanding rings
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
                    *ring["color"],
                    ring["alpha"]
                ),
                (
                    ring["x"],
                    ring["y"]
                ),
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

    # 17. Draw particles
    for particle in particles:

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["life"] -= 1

        pygame.draw.circle(
            screen,
            particle["color"],
            (
                int(particle["x"]),
                int(particle["y"])
            ),
            5
        )

    particles = [
        particle
        for particle in particles
        if particle["life"] > 0
    ]

    # 18. HUD
    fps = clock.get_fps()

    for index, sound in enumerate(
        active_sounds
    ):

        category = sound["category"]
        class_name = sound["class_name"]

        confidence = max(
            0.0,
            min(
                sound["confidence"],
                1.0
            )
        )

        color = SOUND_VISUALS[category]["color"]

        if category == "Background":

            if class_name == "Silence":
                label = (
                    f"Background: Silence "
                    f"({confidence * 100:.1f}%)"
                )
            else:
                label = "Background: Listening"

        else:
            label = (
                f"{category}: {class_name} "
                f"({confidence * 100:.1f}%)"
            )

        sound_text = font.render(
            label,
            True,
            color
        )

        screen.blit(
            sound_text,
            (
                20,
                20 + index * 35
            )
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

    # 19. Update display
    pygame.display.flip()

    # 20. Limit frame rate
    clock.tick(60)


worker.stop()
pygame.quit()
