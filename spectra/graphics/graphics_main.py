import math
import random

import pygame
import pyaudio

from spectra.graphics.canvas import (
    draw_circle,
    draw_polygon,
)

from spectra.graphics.visualizer import lerp
#from spectra.main import process_microphone


# ==================================================
# SCREEN CONFIGURATION
# ==================================================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


# ==================================================
# VISUAL CONFIGURATION
# ==================================================

SOUND_VISUALS = {
    "Alert": {
        "shape": "alarm",
        "color": (255, 50, 80),
    },

    "Clapping": {
        "shape": "clapping_hands",
        "color": (255, 180, 50),
    },

    "Human": {
        "shape": "talking",
        "color": (255, 180, 50),
    },

    "Vehicle": {
        "shape": "car",
        "color": (50, 150, 255),
    },

    "Animal": {
        "shape": "animal",
        "color": (120, 220, 130),
    },

    "Nature": {
        "shape": "nature",
        "color": (80, 210, 180),
    },

    "Background": {
        "shape": "circle",
        "color": (120, 130, 150),
    },
}


# ==================================================
# ICON POSITIONS
# ==================================================

SHAPE_POSITIONS = [
    (180, 345),
    (400, 345),
    (620, 345),
]


# ==================================================
# PYGAME SETUP
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption(
    "Spectra AI"
)

clock = pygame.time.Clock()


# ==================================================
# FONTS
# ==================================================

title_font = pygame.font.Font(None, 40)
subtitle_font = pygame.font.Font(None, 23)

confidence_font = pygame.font.Font(None, 40)
confidence_label_font = pygame.font.Font(None, 18)

sound_name_font = pygame.font.Font(None, 30)
category_font = pygame.font.Font(None, 21)

small_font = pygame.font.Font(None, 22)
rms_font = pygame.font.Font(None, 34)


# ==================================================
# ASSETS
# ==================================================

# --------------------------------------------------
# Background
# --------------------------------------------------

BACKGROUND_IMAGE = pygame.image.load(
    "spectra/graphics/assets/spectra-background.png"
).convert()

BACKGROUND_IMAGE = pygame.transform.smoothscale(
    BACKGROUND_IMAGE,
    (
        SCREEN_WIDTH,
        240
    )
)


# --------------------------------------------------
# Transparent PNG icons
# --------------------------------------------------

CLAPPING_HANDS_IMAGE = pygame.image.load(
    "spectra/graphics/assets/clapping-hands.png"
).convert_alpha()

CAR_IMAGE = pygame.image.load(
    "spectra/graphics/assets/car.png"
).convert_alpha()

ALARM_IMAGE = pygame.image.load(
    "spectra/graphics/assets/alarm.png"
).convert_alpha()

ANIMAL_IMAGE = pygame.image.load(
    "spectra/graphics/assets/animal.png"
).convert_alpha()

NATURE_IMAGE = pygame.image.load(
    "spectra/graphics/assets/nature.png"
).convert_alpha()

TALKING_IMAGE = pygame.image.load(
    "spectra/graphics/assets/talking.png"
).convert_alpha()

# ==================================================
# BACKGROUND OVERLAY
# ==================================================

BACKGROUND_OVERLAY = pygame.Surface(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.SRCALPHA
)

BACKGROUND_OVERLAY.fill(
    (0, 0, 0, 70)
)


# ==================================================
# MICROPHONE SETUP
# ==================================================

audio = pyaudio.PyAudio()

mic = audio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024,
)


# ==================================================
# ANIMATION STATE
# ==================================================

running = True

particles = []

shape_states = {}


# ==================================================
# FLOATING ICON
# ==================================================

def draw_floating_icon(
    surface,
    image,
    center,
    size,
    color,
    alpha,
    confidence,
):
    """
    Draw a clean PNG icon with a subtle illuminated
    socle underneath.

    No layered neon silhouette.

    Confidence controls:
        - icon size elsewhere in the animation
        - icon opacity
        - only a SMALL change in socle brightness
    """

    x, y = center

    icon_size = max(
        1,
        size * 2
    )


    # ==================================================
    # 1. SOFT SOCLE / SHADOW
    # ==================================================

    socle_width = int(
        icon_size * 0.90
    )

    socle_height = max(
        16,
        int(icon_size * 0.14)
    )


    # Surface is intentionally larger than the ellipse
    # so the edges do not feel cramped.
    socle_surface = pygame.Surface(
        (
            socle_width + 40,
            socle_height + 30
        ),
        pygame.SRCALPHA
    )


    # Keep confidence effect subtle.
    socle_alpha = int(
        30 + confidence * 35
    )


    # --------------------------------------------------
    # Soft outer ellipse
    # --------------------------------------------------

    pygame.draw.ellipse(
        socle_surface,
        (
            *color,
            socle_alpha // 2
        ),
        (
            10,
            10,
            socle_width + 20,
            socle_height + 8
        )
    )


    # --------------------------------------------------
    # Slightly brighter center
    # --------------------------------------------------

    pygame.draw.ellipse(
        socle_surface,
        (
            *color,
            socle_alpha
        ),
        (
            30,
            14,
            max(
                10,
                socle_width - 20
            ),
            max(
                4,
                socle_height
            )
        )
    )


    socle_rect = socle_surface.get_rect(
        center=(
            x,
            y + size + 18
        )
    )


    surface.blit(
        socle_surface,
        socle_rect,
        special_flags=pygame.BLEND_RGBA_ADD
    )


    # ==================================================
    # 2. CRISP ICON
    # ==================================================

    scaled_image = pygame.transform.smoothscale(
        image,
        (
            icon_size,
            icon_size
        )
    )


    scaled_image.set_alpha(
        alpha
    )


    image_rect = scaled_image.get_rect(
        center=(
            x,
            y
        )
    )


    surface.blit(
        scaled_image,
        image_rect
    )


# ==================================================
# MAIN LOOP
# ==================================================

while running:

    # ==================================================
    # 1. EVENTS
    # ==================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # ==================================================
    # 2. TEMPORARY TEST DATA
    #
    # Keep this while we finish the UX.
    #
    # Later:
    #
    # predictions, rms = process_microphone(mic)
    # rms = float(rms)
    # ==================================================

    rms = 0.05


    predictions = [
        {
            "category": "Alert",
            "display_label": "Siren",
            "confidence": 0.60,
        },

        {
            "category": "Nature",
            "display_label": "forest",
            "confidence": 0.90,
        },

        {
            "category": "Animal",
            "display_label": "Cat",
            "confidence": 0.75,
        },
    ]


    # ==================================================
    # 3. MODEL -> GRAPHICS ADAPTER
    # ==================================================

    active_sounds = []


    for prediction in predictions[:3]:

        category = prediction.get(
            "category",
            prediction.get("class_name")
        )


        display_label = prediction.get(
            "display_label",
            prediction.get(
                "class_name",
                category
            )
        )


        confidence = float(
            prediction.get(
                "confidence",
                0.0
            )
        )


        confidence = max(
            0.0,
            min(
                confidence,
                1.0
            )
        )


        if category in SOUND_VISUALS:

            active_sounds.append({
                "category": category,
                "display_label": display_label,
                "confidence": confidence,
            })


    # ==================================================
    # 4. BACKGROUND FALLBACK
    # ==================================================

    if not active_sounds:

        active_sounds = [{
            "category": "Background",
            "display_label": "Listening",
            "confidence": 0.0,
        }]


    # ==================================================
    # 5. RMS -> PARTICLES
    # ==================================================

    if rms >= 0.05:

        if rms < 0.065:
            particle_probability = 0.15

        elif rms < 0.08:
            particle_probability = 0.40

        else:
            particle_probability = 0.80


        if random.random() < particle_probability:

            for index, sound in enumerate(
                active_sounds
            ):

                category = sound["category"]


                if category == "Background":
                    continue


                x, y = SHAPE_POSITIONS[index]


                angle = random.uniform(
                    0,
                    2 * math.pi
                )


                # RMS slightly affects particle speed.
                speed = random.uniform(
                    0.8,
                    1.5 + rms * 15
                )


                particles.append({
                    "x": x,
                    "y": y,

                    "vx": (
                        math.cos(angle)
                        * speed
                    ),

                    "vy": (
                        math.sin(angle)
                        * speed
                    ),

                    "life": 55,

                    "color": SOUND_VISUALS[
                        category
                    ]["color"],
                })


    # ==================================================
    # 6. BACKGROUND
    # ==================================================

    screen.fill(
        (5, 7, 12)
    )


    # Preserve the background placement you liked.
    screen.blit(
        BACKGROUND_IMAGE,
        (0, 105)
    )


    screen.blit(
        BACKGROUND_OVERLAY,
        (0, 0)
    )


    # ==================================================
    # 7. HEADER
    # ==================================================

    title_text = title_font.render(
        "SPECTRA AI",
        True,
        (245, 245, 250),
    )


    subtitle_text = subtitle_font.render(
        "Real-time Sound Analysis",
        True,
        (175, 180, 195),
    )


    screen.blit(
        title_text,
        (35, 25)
    )


    screen.blit(
        subtitle_text,
        (37, 65)
    )


    # ==================================================
    # 8. RMS DISPLAY
    # ==================================================

    rms_label = small_font.render(
        "RMS",
        True,
        (190, 195, 205),
    )


    rms_value = rms_font.render(
        f"{rms:.2f}",
        True,
        (245, 245, 250),
    )


    screen.blit(
        rms_label,
        (690, 25)
    )


    screen.blit(
        rms_value,
        (680, 48)
    )


    # --------------------------------------------------
    # RMS meter
    # --------------------------------------------------

    meter_x = 755
    meter_y = 25

    meter_width = 10
    meter_height = 75


    pygame.draw.rect(
        screen,
        (40, 45, 55),
        (
            meter_x,
            meter_y,
            meter_width,
            meter_height
        ),
        border_radius=4
    )


    rms_normalized = max(
        0.0,
        min(
            rms / 0.10,
            1.0
        )
    )


    fill_height = int(
        meter_height
        * rms_normalized
    )


    if fill_height > 0:

        pygame.draw.rect(
            screen,
            (30, 220, 130),
            (
                meter_x,

                meter_y
                + meter_height
                - fill_height,

                meter_width,
                fill_height
            ),
            border_radius=4
        )


    # ==================================================
    # 9. CONFIDENCE LABELS
    # ==================================================

    for index, sound in enumerate(
        active_sounds
    ):

        category = sound["category"]


        if category == "Background":
            continue


        confidence = sound["confidence"]

        color = SOUND_VISUALS[
            category
        ]["color"]

        x, y = SHAPE_POSITIONS[index]


        percentage = (
            f"{confidence * 100:.0f}%"
        )


        percentage_text = confidence_font.render(
            percentage,
            True,
            color,
        )


        percentage_rect = percentage_text.get_rect(
            center=(
                x,
                215
            )
        )


        screen.blit(
            percentage_text,
            percentage_rect
        )


        confidence_text = confidence_label_font.render(
            "CONFIDENCE",
            True,
            (155, 160, 175),
        )


        confidence_rect = confidence_text.get_rect(
            center=(
                x,
                242
            )
        )


        screen.blit(
            confidence_text,
            confidence_rect
        )


    # ==================================================
    # 10. ICONS
    # ==================================================

    for index, sound in enumerate(
        active_sounds
    ):

        category = sound["category"]
        confidence = sound["confidence"]


        visual = SOUND_VISUALS[
            category
        ]


        shape = visual["shape"]
        color = visual["color"]


        if category == "Background":

            x, y = (
                400,
                345
            )

        else:

            x, y = SHAPE_POSITIONS[
                index
            ]


        # --------------------------------------------------
        # Animation memory
        # --------------------------------------------------

        if category not in shape_states:

            shape_states[category] = {
                "size": 30.0,
                "alpha": 60.0,
            }


        state = shape_states[
            category
        ]


        # ==================================================
        # CONFIDENCE -> ICON PROMINENCE
        # ==================================================

        if category == "Background":

            target_size = 45
            target_alpha = 120

        else:

            # Slightly less dramatic than before.
            target_size = (
                40
                + confidence * 40
            )

            target_alpha = (
                130
                + confidence * 125
            )


        state["size"] = lerp(
            state["size"],
            target_size,
            0.15,
        )


        state["alpha"] = lerp(
            state["alpha"],
            target_alpha,
            0.15,
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
            (
                SCREEN_WIDTH,
                SCREEN_HEIGHT
            ),
            pygame.SRCALPHA,
        )


        # ==================================================
        # BACKGROUND
        # ==================================================

        if shape == "circle":

            draw_circle(
                shape_surface,
                rgba_color,
                (x, y),
                size,
            )


        # ==================================================
        # ALERT
        #
        # Still temporary until we have a PNG.
        # ==================================================

        elif shape == "triangle":

            points = [
                (
                    x,
                    y - size
                ),

                (
                    x - size,
                    y + size
                ),

                (
                    x + size,
                    y + size
                ),
            ]


            draw_polygon(
                shape_surface,
                rgba_color,
                points,
            )


        # ==================================================
        # MUSIC
        # ==================================================

        elif shape == "polygon":

            points = [
                (
                    x - size,
                    y - size
                ),

                (
                    x + size,
                    y - size
                ),

                (
                    x + size,
                    y + size
                ),

                (
                    x - size,
                    y + size
                ),
            ]


            draw_polygon(
                shape_surface,
                rgba_color,
                points,
            )


        # ==================================================
        # CAR
        # ==================================================

        elif shape == "car":

            draw_floating_icon(
                shape_surface,
                CAR_IMAGE,
                (x, y),
                size,
                color,
                alpha,
                confidence,
            )


        # ==================================================
        # CLAPPING HANDS
        # ==================================================

        elif shape == "clapping_hands":

            draw_floating_icon(
                shape_surface,
                CLAPPING_HANDS_IMAGE,
                (x, y),
                size,
                color,
                alpha,
                confidence,
            )

        elif shape == "alarm":
            draw_floating_icon(
            shape_surface,
            ALARM_IMAGE,
            (x, y),
            size,
            color,
            alpha,
            confidence,
            )

        elif shape == "animal":
            draw_floating_icon(
            shape_surface,
            ANIMAL_IMAGE,
            (x, y),
            size,
            color,
            alpha,
            confidence,
            )

        elif shape == "nature":
            draw_floating_icon(
            shape_surface,
            NATURE_IMAGE,
            (x, y),
            size,
            color,
            alpha,
            confidence,
            )

        elif shape == "talking":
            draw_floating_icon(
            shape_surface,
            TALKING_IMAGE,
            (x, y),
            size,
            color,
            alpha,
            confidence,
            )

        screen.blit(
            shape_surface,
            (0, 0)
        )


    # ==================================================
    # 11. PARTICLES
    # ==================================================

    particle_size = max(
        3,
        int(
            3 + rms * 10
        )
    )


    for particle in particles:

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]

        particle["life"] -= 1


        # Fade particles near the end of their life.
        particle_alpha = int(
            255
            * max(
                0,
                particle["life"] / 55
            )
        )


        particle_surface = pygame.Surface(
            (12, 12),
            pygame.SRCALPHA
        )


        pygame.draw.circle(
            particle_surface,
            (
                *particle["color"],
                particle_alpha
            ),
            (6, 6),
            particle_size,
        )


        screen.blit(
            particle_surface,
            (
                int(particle["x"] - 6),
                int(particle["y"] - 6)
            )
        )


    particles = [
        particle
        for particle in particles
        if particle["life"] > 0
    ]


    # ==================================================
    # 12. SOUND LABELS
    # ==================================================

    for index, sound in enumerate(
        active_sounds
    ):

        category = sound["category"]


        if category == "Background":
            continue


        display_label = sound[
            "display_label"
        ]


        color = SOUND_VISUALS[
            category
        ]["color"]


        x, y = SHAPE_POSITIONS[
            index
        ]


        # --------------------------------------------------
        # Specific sound
        # --------------------------------------------------

        sound_name = sound_name_font.render(
            display_label,
            True,
            (245, 245, 250),
        )


        sound_name_rect = sound_name.get_rect(
            center=(
                x,
                445
            )
        )


        screen.blit(
            sound_name,
            sound_name_rect
        )


        # --------------------------------------------------
        # Broad category
        # --------------------------------------------------

        category_text = category_font.render(
            category,
            True,
            color,
        )


        category_rect = category_text.get_rect(
            center=(
                x,
                470
            )
        )


        screen.blit(
            category_text,
            category_rect
        )


    # ==================================================
    # 13. LISTENING STATUS
    # ==================================================

    pygame.draw.circle(
        screen,
        (30, 220, 130),
        (35, 570),
        5
    )


    listening_text = small_font.render(
        "Listening...",
        True,
        (185, 190, 205),
    )


    screen.blit(
        listening_text,
        (50, 560)
    )


    # ==================================================
    # 14. FPS
    # ==================================================

    fps = clock.get_fps()


    fps_text = small_font.render(
        f"FPS: {fps:.0f}",
        True,
        (140, 145, 160),
    )


    fps_rect = fps_text.get_rect(
        right=765,
        centery=570
    )


    screen.blit(
        fps_text,
        fps_rect
    )


    # ==================================================
    # 15. UPDATE DISPLAY
    # ==================================================

    pygame.display.flip()

    clock.tick(60)


# ==================================================
# CLEANUP
# ==================================================

mic.stop_stream()
mic.close()

audio.terminate()

pygame.quit()
