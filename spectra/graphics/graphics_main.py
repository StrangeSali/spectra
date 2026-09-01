# ==================================================
# IMPORTS
# ==================================================

import time

import numpy as np
import pygame
import requests


# ==================================================
# IMPORTANT
#
# Initialize the desktop Pygame display BEFORE
# importing renderer.py.
#
# renderer.py is also reused by Streamlit/headless
# rendering, so graphics_main owns the desktop window.
# ==================================================

pygame.init()


# ==================================================
# API CONFIGURATION
# ==================================================

API_URL = (
    "https://spectra-1087886990522."
    "europe-west1.run.app"
)

LATEST_PREDICTION_URL = (
    f"{API_URL}/predict/latest"
)


# How often we ask the API for a newer prediction.
#
# The Pygame graphics still run at 60 FPS.
# We do NOT need 60 HTTP calls per second.
API_REFRESH_MS = 300


# ==================================================
# IMPORT RENDERER
#
# Renderer only receives data and produces pixels.
# ==================================================

from spectra.graphics.renderer import (
    render_frame,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)


# ==================================================
# PYGAME WINDOW
# ==================================================

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

FPS = 60


# ==================================================
# HTTP SESSION
#
# requests.Session reuses the HTTP connection instead
# of creating a completely new connection every time.
# ==================================================

http = requests.Session()


# ==================================================
# STATE
# ==================================================

running = True


# Latest prediction data successfully received
# from the API.
predictions = []


# Our current API /predict/latest response does not
# necessarily contain RMS.
#
# If RMS is added to the API later, the code below
# will automatically use it.
rms = 0.0


# Timestamp returned by the API.
latest_api_timestamp = None


# Used to control API polling independently from
# the 60 FPS graphics loop.
last_api_refresh = (
    -API_REFRESH_MS
)


# ==================================================
# API FETCH
# ==================================================

def fetch_latest_prediction():
    """
    Ask Spectra API for its most recent prediction.

    This function knows about HTTP/API structure.

    renderer.py does NOT.
    """

    try:

        response = http.get(
            LATEST_PREDICTION_URL,
            timeout=2,
        )


        # --------------------------------------------------
        # NO PREDICTIONS YET
        # --------------------------------------------------

        if response.status_code == 404:

            return {
                "predictions": [],
                "rms": 0.0,
                "timestamp": None,
            }


        # --------------------------------------------------
        # OTHER API ERROR
        # --------------------------------------------------

        response.raise_for_status()


        # --------------------------------------------------
        # JSON RESPONSE
        # --------------------------------------------------

        data = response.json()


        return {
            "predictions":
                data.get(
                    "predictions",
                    [],
                ),

            # Future-proof:
            # use RMS if API starts providing it.
            "rms":
                data.get(
                    "rms",
                    0.0,
                ),

            "timestamp":
                data.get(
                    "timestamp"
                ),
        }


    except requests.RequestException as error:

        print(
            "Spectra API unavailable:",
            error,
        )

        return None


    except ValueError as error:

        print(
            "Invalid API JSON:",
            error,
        )

        return None


# ==================================================
# NUMPY FRAME -> PYGAME SURFACE
# ==================================================

def numpy_frame_to_surface(
    frame,
):
    """
    renderer.py returns:

        height x width x RGB

    pygame.surfarray.make_surface expects:

        width x height x RGB

    so swap the first two axes back.
    """

    pygame_array = np.swapaxes(
        frame,
        0,
        1,
    )


    return pygame.surfarray.make_surface(
        pygame_array
    )


# ==================================================
# MAIN LOOP
# ==================================================

try:

    while running:

        # ==================================================
        # 1. PYGAME EVENTS
        # ==================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False


        # ==================================================
        # 2. CURRENT TIME
        # ==================================================

        current_time = (
            pygame.time.get_ticks()
        )


        # ==================================================
        # 3. ASK API FOR LATEST PREDICTION
        # ==================================================

        if (
            current_time
            - last_api_refresh
            >= API_REFRESH_MS
        ):

            api_result = (
                fetch_latest_prediction()
            )


            if api_result is not None:

                new_timestamp = (
                    api_result[
                        "timestamp"
                    ]
                )


                # ------------------------------------------
                # NEW API RESULT
                # ------------------------------------------

                if (
                    new_timestamp
                    != latest_api_timestamp
                ):

                    predictions = (
                        api_result[
                            "predictions"
                        ]
                    )


                    rms = float(
                        api_result.get(
                            "rms",
                            0.0,
                        )
                    )


                    latest_api_timestamp = (
                        new_timestamp
                    )


                    print(
                        "LATEST SPECTRA PREDICTION:"
                    )

                    print(
                        predictions
                    )


            last_api_refresh = (
                current_time
            )


        # ==================================================
        # 4. RENDER VISUALIZATION
        #
        # graphics_main doesn't draw the interface.
        #
        # It simply gives renderer.py:
        #
        #   predictions
        #   rms
        # ==================================================

        frame = render_frame(
            predictions,
            rms,
        )


        # ==================================================
        # 5. NUMPY -> PYGAME
        # ==================================================

        frame_surface = (
            numpy_frame_to_surface(
                frame
            )
        )


        # ==================================================
        # 6. SHOW FRAME
        # ==================================================

        screen.blit(
            frame_surface,
            (
                0,
                0,
            )
        )


        pygame.display.flip()


        # ==================================================
        # 7. KEEP DISPLAY AT 60 FPS
        # ==================================================

        clock.tick(
            FPS
        )


# ==================================================
# CLEANUP
# ==================================================

finally:

    http.close()

    pygame.quit()
