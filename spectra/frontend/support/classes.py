ESC50_CATEGORIES = {

    "clapping": [
        "clapping",
        "door_wood_knock",
        "hand_saw",
        "brushing_teeth",
        "chainsaw",
        "can_opening",
        "mouse_click"
    ],

    "human": [
        "crying_baby",
        "sneezing",
        "breathing",
        "coughing",
        "snoring",
        "drinking_sipping",
        "keyboard_typing",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "church_bells",
        "laughing",
        "footsteps",
    ],

    "animal": [
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
        "crickets",
        "chirping_birds",
    ],

    "nature": [
        "rain",
        "sea_waves",
        "crackling_fire",
        "water_drops",
        "wind",
        "pouring_water",
        "thunderstorm",
        "door_wood_creaks",
    ],

    "alert": [
        "siren",
        "glass_breaking",
        "fireworks",
    ],

    "vehicle": [
        "helicopter",
        "car_horn",
        "engine",
        "train",
        "airplane",
    ],
}

SOUNDS_DICT = {}

for category, sounds in ESC50_CATEGORIES.items():
    for sound in sounds:
        SOUNDS_DICT[sound] = category

DEFAULT_CATEGORY = "Background"
