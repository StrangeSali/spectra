# Maps YAMNet's 521 AudioSet classes into 6 simplified accessibility categories.
# Instead of listing all 521 classes manually, we match by keyword: if a class
# name contains one of these substrings, it belongs to that category.
# Order matters: categories are checked top to bottom, first match wins.
import re

CATEGORY_KEYWORDS = {
    "Alert": [
        "siren", "alarm", "emergency", "smoke detector", "fire alarm",
        "civil defense", "buzzer", "gunshot", "explosion"
    ],
    "Human": [
        "speech", "conversation", "shout", "yell", "cry", "sob",
        "laughter", "laugh", "clapping", "applause", "cough",
        "sneeze", "footsteps", "baby", "child"
    ],
    "Animal": [
        "dog", "cat", "bird", "animal", "bark", "meow", "roar",
        "growl", "insect", "livestock", "horse", "cattle"
    ],
    "Vehicle": [
        "car", "vehicle", "engine", "traffic", "motor", "truck",
        "bus", "train", "motorcycle", "horn", "honk", "aircraft"
    ],
    "Music": [
        "music", "musical instrument", "singing", "guitar", "piano",
        "drum", "violin"
    ],
}

DEFAULT_CATEGORY = "Background"


def map_to_category(class_name, confidence, threshold=0.3):
    """
    Map a raw YAMNet class prediction into one of the 6 core accessibility
    categories, applying a confidence gate.
    """
    if confidence < threshold:
        return DEFAULT_CATEGORY

    class_name_lower = class_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            # \b enforces word boundaries, so "bus" won't match inside "busy"
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, class_name_lower):
                return category

    return DEFAULT_CATEGORY

    class_name_lower = class_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in class_name_lower:
                return category

    return DEFAULT_CATEGORY

# test!
if __name__ == "__main__":
    # Quick manual test using predictions we already validated in 2.2/2.3
    test_predictions = [
        ("Telephone", 0.9955),
        ("Busy signal", 0.9789),
        ("Alarm", 0.9683),
        ("Dial tone", 0.8861),
        ("Clapping", 0.85),
        ("Siren", 0.92),
        ("Car", 0.78),
        ("Something random", 0.15),  # should fall to Background (low confidence)
    ]

    print("Testing category mapping:\n")
    for class_name, confidence in test_predictions:
        category = map_to_category(class_name, confidence)
        print(f"  {class_name} ({confidence:.2f}) -> {category}")
