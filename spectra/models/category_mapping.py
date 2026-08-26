# Maps YAMNet's 521 AudioSet classes into 6 simplified accessibility categories.
# Instead of listing all 521 classes manually, we match by keyword: if a class
# name contains one of these substrings, it belongs to that category.
# Order matters: categories are checked top to bottom, first match wins.
import re

CATEGORY_KEYWORDS = {
    "Alert": [
        "siren", "alarm", "emergency", "smoke detector", "fire alarm",
        "civil defense", "buzzer", "gunshot", "explosion",
        # clapping / hand-noise related YAMNet labels
        "hands", "slap", "smack", "finger snapping"
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


def _match_category(class_name: str) -> str:
    """
    Match a raw YAMNet class name against CATEGORY_KEYWORDS using
    word-boundary regex. Returns DEFAULT_CATEGORY if nothing matches.
    Shared by both map_to_category() and map_to_category_smooth().
    """
    class_name_lower = class_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            # \b enforces word boundaries, so "bus" won't match inside "busy"
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, class_name_lower):
                return category

    return DEFAULT_CATEGORY


def map_to_category(class_name: str, confidence: float, threshold: float = 0.3) -> str:
    """
    [Legacy / compatibility] Hard-cutoff category mapping used by classify_wav
    and the offline ESC-50 evaluation (2.2/2.3). Behavior is unchanged from
    the original implementation — kept as-is so existing calls/tests don't break.
    """
    if confidence < threshold:
        return DEFAULT_CATEGORY
    return _match_category(class_name)


def map_to_category_smooth(
    class_name: str,
    confidence: float,
    current_category: str = DEFAULT_CATEGORY,
    threshold_enter: float = 0.35,
    threshold_exit: float = 0.20,
) -> str:
    """
    Hysteresis-based category mapping for real-time/streaming use (2.6).

    Instead of a single hard cutoff, uses two thresholds to avoid flicker
    when confidence hovers near the boundary:
      - confidence >= threshold_enter -> resolve the real category by keywords.
      - confidence <= threshold_exit  -> fall back to Background.
      - in between (the "hysteresis band") -> keep whatever category was
        active before (current_category), i.e. no change.

    This function is stateless itself; the caller (YAMNetInferenceWorker)
    is responsible for tracking and passing in current_category.
    """
    if threshold_exit >= threshold_enter:
        raise ValueError("threshold_exit must be lower than threshold_enter")

    if confidence >= threshold_enter:
        return _match_category(class_name)

    if confidence <= threshold_exit:
        return DEFAULT_CATEGORY

    return current_category


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

    print("Testing legacy hard-cutoff mapping (map_to_category):\n")
    for class_name, confidence in test_predictions:
        category = map_to_category(class_name, confidence)
        print(f"  {class_name} ({confidence:.2f}) -> {category}")

    print("\nTesting hysteresis mapping (map_to_category_smooth) over a sequence:\n")
    # Simulates confidence wobbling around the threshold - should NOT flicker
    sequence = [
        ("Siren", 0.10),  # -> Background (below exit)
        ("Siren", 0.28),  # -> stays Background (in band, previous was Background)
        ("Siren", 0.40),  # -> Alert (above enter)
        ("Siren", 0.30),  # -> stays Alert (in band, previous was Alert)
        ("Siren", 0.15),  # -> Background (below exit again)
    ]
    current = DEFAULT_CATEGORY
    for class_name, confidence in sequence:
        current = map_to_category_smooth(class_name, confidence, current_category=current)
        print(f"  {class_name} ({confidence:.2f}) -> {current}")
