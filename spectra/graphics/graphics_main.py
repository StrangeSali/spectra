import math
import random
import statistics

import pygame
import pyaudio

from spectra.graphics.canvas import draw_circle
from spectra.graphics.visualizer import lerp
from spectra.main import process_microphone


# ==================================================
# SCREEN CONFIGURATION
# ==================================================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

FPS = 60


# ==================================================
# PREDICTION CONFIGURATION
# ==================================================

# Model refresh rate.
PREDICTION_INTERVAL_MS = 1500


# --------------------------------------------------
# AUTOMATIC BACKGROUND CALIBRATION
# --------------------------------------------------

# Spectra listens to a few initial samples to learn
# the normal RMS level of the microphone / room.
CALIBRATION_SAMPLES = 5


# Minimum possible quiet threshold.
ABSOLUTE_MIN_RMS = 0.0025


# A sound must rise sufficiently above the learned
# background noise floor before we classify it.
NOISE_MULTIPLIER = 1.35


# Small extra margin above the noise floor.
NOISE_MARGIN = 0.0005


# --------------------------------------------------
# DISPLAY CONFIDENCE
# --------------------------------------------------

MIN_DISPLAY_CONFIDENCE = 0.20

SECONDARY_CONFIDENCE_THRESHOLD = 0.40

MAX_DISPLAYED_SOUNDS = 3


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
    (
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )
)

pygame.display.set_caption(
    "Spectra AI"
)

clock = pygame.time.Clock()


# ==================================================
# FONTS
# ==================================================

title_font = pygame.font.Font(
    None,
    40,
)

subtitle_font = pygame.font.Font(
    None,
    23,
)

confidence_font = pygame.font.Font(
    None,
    40,
)

confidence_label_font = pygame.font.Font(
    None,
    18,
)

sound_name_font = pygame.font.Font(
    None,
    30,
)

category_font = pygame.font.Font(
    None,
    21,
)

small_font = pygame.font.Font(
    None,
    22,
)

rms_font = pygame.font.Font(
    None,
    34,
)


# ==================================================
# ASSETS
# ==================================================

BACKGROUND_IMAGE = pygame.image.load(
    "spectra/graphics/assets/spectra-background.png"
).convert()

BACKGROUND_IMAGE = pygame.transform.smoothscale(
    BACKGROUND_IMAGE,
    (
        SCREEN_WIDTH,
        240,
    ),
)


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


ICON_IMAGES = {

    "clapping_hands":
        CLAPPING_HANDS_IMAGE,

    "car":
        CAR_IMAGE,

    "alarm":
        ALARM_IMAGE,

    "animal":
        ANIMAL_IMAGE,

    "nature":
        NATURE_IMAGE,

    "talking":
        TALKING_IMAGE,
}


# ==================================================
# BACKGROUND OVERLAY
# ==================================================

BACKGROUND_OVERLAY = pygame.Surface(
    (
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    ),
    pygame.SRCALPHA,
)

BACKGROUND_OVERLAY.fill(
    (
        0,
        0,
        0,
        70,
    )
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
# STATE
# ==================================================

running = True

particles = []

shape_states = {}


predictions = []

rms = 0.0


last_prediction_time = (
    -PREDICTION_INTERVAL_MS
)


# --------------------------------------------------
# Noise-floor calibration
# --------------------------------------------------

calibration_values = []

noise_floor = None

quiet_threshold = (
    ABSOLUTE_MIN_RMS
)

is_calibrating = True


# ==================================================
# BACKGROUND RESULT
# ==================================================

def background_result(
    label="Listening",
):

    return [
        {
            "category": "Background",
            "display_label": label,
            "confidence": 0.0,
        }
    ]


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

    x, y = center

    icon_size = max(
        1,
        size * 2,
    )


    # --------------------------------------------------
    # SOCLE
    # --------------------------------------------------

    socle_width = int(
        icon_size * 0.90
    )

    socle_height = max(
        16,
        int(
            icon_size * 0.14
        ),
    )


    socle_surface = pygame.Surface(
        (
            socle_width + 40,
            socle_height + 30,
        ),
        pygame.SRCALPHA,
    )


    socle_alpha = int(
        30
        + confidence * 35
    )


    pygame.draw.ellipse(
        socle_surface,
        (
            *color,
            socle_alpha // 2,
        ),
        (
            10,
            10,
            socle_width + 20,
            socle_height + 8,
        ),
    )


    pygame.draw.ellipse(
        socle_surface,
        (
            *color,
            socle_alpha,
        ),
        (
            30,
            14,

            max(
                10,
                socle_width - 20,
            ),

            max(
                4,
                socle_height,
            ),
        ),
    )


    socle_rect = (
        socle_surface.get_rect(
            center=(
                x,
                y + size + 18,
            )
        )
    )


    surface.blit(
        socle_surface,
        socle_rect,
        special_flags=pygame.BLEND_RGBA_ADD,
    )


    # --------------------------------------------------
    # ICON
    # --------------------------------------------------

    scaled_image = (
        pygame.transform.smoothscale(
            image,
            (
                icon_size,
                icon_size,
            ),
        )
    )


    scaled_image.set_alpha(
        alpha
    )


    image_rect = (
        scaled_image.get_rect(
            center=(
                x,
                y,
            )
        )
    )


    surface.blit(
        scaled_image,
        image_rect,
    )


# ==================================================
# PREDICTION ADAPTER
# ==================================================

def adapt_predictions(
    raw_predictions,
):

    if not raw_predictions:

        return background_result()


    valid_predictions = []


    for prediction in raw_predictions:

        category = prediction.get(
            "category",
            prediction.get(
                "class_name"
            ),
        )


        display_label = prediction.get(
            "display_label",
            prediction.get(
                "class_name",
                category,
            ),
        )


        confidence = float(
            prediction.get(
                "confidence",
                0.0,
            )
        )


        if category not in SOUND_VISUALS:
            continue


        if confidence < MIN_DISPLAY_CONFIDENCE:
            continue


        valid_predictions.append(
            {
                "category": category,
                "display_label": display_label,
                "confidence": confidence,
            }
        )


    if not valid_predictions:

        return background_result()


    # --------------------------------------------------
    # Sort strongest first
    # --------------------------------------------------

    valid_predictions.sort(
        key=lambda item: item[
            "confidence"
        ],
        reverse=True,
    )


    # --------------------------------------------------
    # Always show strongest meaningful prediction
    # --------------------------------------------------

    active_sounds = [
        valid_predictions[0]
    ]


    # --------------------------------------------------
    # Secondary predictions must be much stronger
    # --------------------------------------------------

    for prediction in valid_predictions[1:]:

        if (
            prediction["confidence"]
            >= SECONDARY_CONFIDENCE_THRESHOLD
        ):

            active_sounds.append(
                prediction
            )


        if (
            len(active_sounds)
            >= MAX_DISPLAYED_SOUNDS
        ):

            break


    return active_sounds


# ==================================================
# PARTICLES
# ==================================================

def update_particles(
    active_sounds,
    current_rms,
):

    if not active_sounds:
        return


    if (
        active_sounds[0]["category"]
        == "Background"
    ):
        return


    if current_rms < 0.01:

        particle_probability = 0.10

    elif current_rms < 0.03:

        particle_probability = 0.25

    elif current_rms < 0.06:

        particle_probability = 0.50

    else:

        particle_probability = 0.80


    if (
        random.random()
        >= particle_probability
    ):
        return


    for index, sound in enumerate(
        active_sounds[:3]
    ):

        category = sound[
            "category"
        ]


        if category == "Background":
            continue


        x, y = SHAPE_POSITIONS[
            index
        ]


        angle = random.uniform(
            0,
            2 * math.pi,
        )


        speed = random.uniform(
            0.8,
            1.5
            + current_rms * 15,
        )


        particles.append(
            {
                "x": x,
                "y": y,

                "vx":
                    math.cos(angle)
                    * speed,

                "vy":
                    math.sin(angle)
                    * speed,

                "life": 55,

                "color":
                    SOUND_VISUALS[
                        category
                    ]["color"],
            }
        )


# ==================================================
# HEADER
# ==================================================

def draw_header(
    current_rms,
):

    title_text = title_font.render(
        "SPECTRA AI",
        True,
        (
            245,
            245,
            250,
        ),
    )


    subtitle_text = subtitle_font.render(
        "Real-time Sound Analysis",
        True,
        (
            175,
            180,
            195,
        ),
    )


    screen.blit(
        title_text,
        (
            35,
            25,
        ),
    )


    screen.blit(
        subtitle_text,
        (
            37,
            65,
        ),
    )


    rms_label = small_font.render(
        "RMS",
        True,
        (
            190,
            195,
            205,
        ),
    )


    rms_value = rms_font.render(
        f"{current_rms:.4f}",
        True,
        (
            245,
            245,
            250,
        ),
    )


    screen.blit(
        rms_label,
        (
            670,
            25,
        ),
    )


    screen.blit(
        rms_value,
        (
            650,
            48,
        ),
    )


    meter_x = 755
    meter_y = 25

    meter_width = 10
    meter_height = 75


    pygame.draw.rect(
        screen,
        (
            40,
            45,
            55,
        ),
        (
            meter_x,
            meter_y,
            meter_width,
            meter_height,
        ),
        border_radius=4,
    )


    rms_normalized = max(
        0.0,
        min(
            current_rms / 0.10,
            1.0,
        ),
    )


    fill_height = int(
        meter_height
        * rms_normalized
    )


    if fill_height > 0:

        pygame.draw.rect(
            screen,
            (
                30,
                220,
                130,
            ),
            (
                meter_x,

                meter_y
                + meter_height
                - fill_height,

                meter_width,
                fill_height,
            ),
            border_radius=4,
        )


# ==================================================
# CONFIDENCE
# ==================================================

def draw_confidences(
    active_sounds,
):

    for index, sound in enumerate(
        active_sounds[:3]
    ):

        category = sound[
            "category"
        ]


        if category == "Background":
            continue


        confidence = sound[
            "confidence"
        ]


        color = SOUND_VISUALS[
            category
        ]["color"]


        x, _ = SHAPE_POSITIONS[
            index
        ]


        percentage_text = (
            confidence_font.render(
                f"{confidence * 100:.0f}%",
                True,
                color,
            )
        )


        percentage_rect = (
            percentage_text.get_rect(
                center=(
                    x,
                    215,
                )
            )
        )


        screen.blit(
            percentage_text,
            percentage_rect,
        )


        confidence_text = (
            confidence_label_font.render(
                "CONFIDENCE",
                True,
                (
                    155,
                    160,
                    175,
                ),
            )
        )


        confidence_rect = (
            confidence_text.get_rect(
                center=(
                    x,
                    242,
                )
            )
        )


        screen.blit(
            confidence_text,
            confidence_rect,
        )


# ==================================================
# ICONS
# ==================================================

def draw_icons(
    active_sounds,
):

    for index, sound in enumerate(
        active_sounds[:3]
    ):

        category = sound[
            "category"
        ]

        confidence = sound[
            "confidence"
        ]


        visual = SOUND_VISUALS[
            category
        ]


        shape = visual[
            "shape"
        ]

        color = visual[
            "color"
        ]


        if category == "Background":

            x, y = (
                400,
                345,
            )

        else:

            x, y = SHAPE_POSITIONS[
                index
            ]


        if category not in shape_states:

            shape_states[
                category
            ] = {
                "size": 30.0,
                "alpha": 60.0,
            }


        state = shape_states[
            category
        ]


        if category == "Background":

            target_size = 45
            target_alpha = 120

        else:

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
                    255,
                ),
            )
        )


        shape_surface = pygame.Surface(
            (
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
            ),
            pygame.SRCALPHA,
        )


        if shape == "circle":

            draw_circle(
                shape_surface,
                (
                    *color,
                    alpha,
                ),
                (
                    x,
                    y,
                ),
                size,
            )


        elif shape in ICON_IMAGES:

            draw_floating_icon(
                shape_surface,

                ICON_IMAGES[
                    shape
                ],

                (
                    x,
                    y,
                ),

                size,
                color,
                alpha,
                confidence,
            )


        screen.blit(
            shape_surface,
            (
                0,
                0,
            ),
        )


# ==================================================
# DRAW PARTICLES
# ==================================================

def draw_particles(
    current_rms,
):

    particle_size = max(
        3,
        int(
            3
            + current_rms * 10
        ),
    )


    for particle in particles:

        particle["x"] += particle[
            "vx"
        ]

        particle["y"] += particle[
            "vy"
        ]

        particle["life"] -= 1


        particle_alpha = int(
            255
            * max(
                0,
                particle["life"] / 55,
            )
        )


        particle_surface = (
            pygame.Surface(
                (
                    12,
                    12,
                ),
                pygame.SRCALPHA,
            )
        )


        pygame.draw.circle(
            particle_surface,
            (
                *particle["color"],
                particle_alpha,
            ),
            (
                6,
                6,
            ),
            particle_size,
        )


        screen.blit(
            particle_surface,
            (
                int(
                    particle["x"]
                    - 6
                ),

                int(
                    particle["y"]
                    - 6
                ),
            ),
        )


    particles[:] = [
        particle
        for particle in particles
        if particle["life"] > 0
    ]


# ==================================================
# LABELS
# ==================================================

def draw_sound_labels(
    active_sounds,
):

    for index, sound in enumerate(
        active_sounds[:3]
    ):

        category = sound[
            "category"
        ]


        if category == "Background":

            display_label = sound[
                "display_label"
            ]


            listening_label = (
                sound_name_font.render(
                    display_label,
                    True,
                    (
                        190,
                        195,
                        205,
                    ),
                )
            )


            listening_rect = (
                listening_label.get_rect(
                    center=(
                        400,
                        445,
                    )
                )
            )


            screen.blit(
                listening_label,
                listening_rect,
            )

            continue


        display_label = sound[
            "display_label"
        ]


        color = SOUND_VISUALS[
            category
        ]["color"]


        x, _ = SHAPE_POSITIONS[
            index
        ]


        sound_name = (
            sound_name_font.render(
                display_label,
                True,
                (
                    245,
                    245,
                    250,
                ),
            )
        )


        sound_name_rect = (
            sound_name.get_rect(
                center=(
                    x,
                    445,
                )
            )
        )


        screen.blit(
            sound_name,
            sound_name_rect,
        )


        category_text = (
            category_font.render(
                category,
                True,
                color,
            )
        )


        category_rect = (
            category_text.get_rect(
                center=(
                    x,
                    470,
                )
            )
        )


        screen.blit(
            category_text,
            category_rect,
        )


# ==================================================
# FOOTER
# ==================================================

def draw_footer():

    pygame.draw.circle(
        screen,
        (
            30,
            220,
            130,
        ),
        (
            35,
            570,
        ),
        5,
    )


    listening_text = (
        small_font.render(
            "Listening...",
            True,
            (
                185,
                190,
                205,
            ),
        )
    )


    screen.blit(
        listening_text,
        (
            50,
            560,
        ),
    )


    fps = clock.get_fps()


    fps_text = small_font.render(
        f"FPS: {fps:.0f}",
        True,
        (
            140,
            145,
            160,
        ),
    )


    fps_rect = fps_text.get_rect(
        right=765,
        centery=570,
    )


    screen.blit(
        fps_text,
        fps_rect,
    )


# ==================================================
# MAIN LOOP
# ==================================================

try:

    while running:

        # --------------------------------------------------
        # EVENTS
        # --------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False


        # --------------------------------------------------
        # MODEL REFRESH
        # --------------------------------------------------

        current_time = (
            pygame.time.get_ticks()
        )


        if (
            current_time
            - last_prediction_time
            >= PREDICTION_INTERVAL_MS
        ):

            new_predictions, new_rms = (
                process_microphone(
                    mic
                )
            )


            rms = float(
                new_rms
            )


            # ==================================================
            # CALIBRATION
            # ==================================================

            if is_calibrating:

                calibration_values.append(
                    rms
                )


                predictions = []


                print(
                    f"CALIBRATING "
                    f"{len(calibration_values)}/"
                    f"{CALIBRATION_SAMPLES} "
                    f"| RMS: {rms:.5f}"
                )


                if (
                    len(calibration_values)
                    >= CALIBRATION_SAMPLES
                ):

                    noise_floor = (
                        statistics.median(
                            calibration_values
                        )
                    )


                    quiet_threshold = max(
                        ABSOLUTE_MIN_RMS,

                        noise_floor
                        * NOISE_MULTIPLIER
                        + NOISE_MARGIN,
                    )


                    is_calibrating = False


                    print(
                        "\n"
                        "Calibration complete."
                    )

                    print(
                        f"Noise floor: "
                        f"{noise_floor:.5f}"
                    )

                    print(
                        f"Quiet threshold: "
                        f"{quiet_threshold:.5f}"
                    )

                    print()


            # ==================================================
            # NORMAL OPERATION
            # ==================================================

            else:

                # ----------------------------------------------
                # BACKGROUND / QUIET
                # ----------------------------------------------

                if rms <= quiet_threshold:

                    predictions = []


                    # Slowly adapt to changing room noise.
                    noise_floor = (
                        0.95
                        * noise_floor
                        + 0.05
                        * rms
                    )


                    quiet_threshold = max(
                        ABSOLUTE_MIN_RMS,

                        noise_floor
                        * NOISE_MULTIPLIER
                        + NOISE_MARGIN,
                    )


                    print(
                        f"QUIET "
                        f"| RMS: {rms:.5f} "
                        f"| threshold: "
                        f"{quiet_threshold:.5f}"
                    )


                # ----------------------------------------------
                # MEANINGFUL SOUND
                # ----------------------------------------------

                else:

                    predictions = (
                        new_predictions
                    )


                    print(
                        f"SOUND "
                        f"| RMS: {rms:.5f} "
                        f"| threshold: "
                        f"{quiet_threshold:.5f} "
                        f"| {new_predictions}"
                    )


            last_prediction_time = (
                current_time
            )


        # --------------------------------------------------
        # GRAPHICS
        # --------------------------------------------------

        if is_calibrating:

            active_sounds = (
                background_result(
                    "Calibrating..."
                )
            )

        else:

            active_sounds = (
                adapt_predictions(
                    predictions
                )
            )


        update_particles(
            active_sounds,
            rms,
        )


        # --------------------------------------------------
        # BACKGROUND
        # --------------------------------------------------

        screen.fill(
            (
                5,
                7,
                12,
            )
        )


        screen.blit(
            BACKGROUND_IMAGE,
            (
                0,
                105,
            ),
        )


        screen.blit(
            BACKGROUND_OVERLAY,
            (
                0,
                0,
            ),
        )


        # --------------------------------------------------
        # UI
        # --------------------------------------------------

        draw_header(
            rms
        )


        draw_confidences(
            active_sounds
        )


        draw_icons(
            active_sounds
        )


        draw_particles(
            rms
        )


        draw_sound_labels(
            active_sounds
        )


        draw_footer()


        pygame.display.flip()


        clock.tick(
            FPS
        )


# ==================================================
# CLEANUP
# ==================================================

finally:

    mic.stop_stream()

    mic.close()

    audio.terminate()

    pygame.quit()
