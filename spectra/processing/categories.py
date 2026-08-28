ESC50_CLASSES = [
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

    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",

    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "snoring",
    "drinking_sipping",
    "door_knock",
    "mouse_click",
    "keyboard_typing",

    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "helicopter",
    "chainsaw",
    "siren",

    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw",
    "footsteps",
    "laughing",
    "brushing_teeth"
]


ESC50_CATEGORIES = {
    "Clapping": ["clapping", "door_knock","hand_saw", "brushing_teeth", "chainsaw"],

    "Human": [
        "crying_baby",
        "sneezing",
        "breathing",
        "coughing",
        "snoring",
        "drinking_sipping",
        "mouse_click",
        "keyboard_typing",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "church_bells",
        "laughing"
        "footsteps"
    ],

    "Animal": [
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
        "chirping_birds"
    ],

    "Nature": [
        "rain",
        "sea_waves",
        "crackling_fire",
        "water_drops",
        "wind",
        "pouring_water",
        "thunderstorm",
        "door_wood_creaks"
    ],

    "Alert": [
        "siren",
        "glass_breaking",
        "fireworks"
    ],

    "Vehicle": [
        "helicopter",
        "car_horn",
        "engine",
        "train",
        "airplane"
    ]
}


SOUNDS_DICT = {}

for category, sounds in ESC50_CATEGORIES.items():
    for sound in sounds:
        SOUNDS_DICT[sound] = category


DEFAULT_CATEGORY = "Background"
