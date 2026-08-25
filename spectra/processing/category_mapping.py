# Maps YAMNet's 521 AudioSet classes into 6 simplified accessibility categories.
# Instead of listing all 521 classes manually, we match by keyword: if a class
# name contains one of these substrings, it belongs to that category.
# Order matters: categories are checked top to bottom, first match wins.
import re

CATEGORY_KEYWORDS = {
    "Alert": [
        "Siren", "Alarm", "Emergency", "Smoke detector", "Fire alarm",
        "Civil defense", "Buzzer", "Gunshot", "Explosion"
    ],
    "Human": [
        "Speech", "Conversation", "Shout", "Yell", "Cry", "Sob",
        "Laughter", "Laugh", "Clapping", "Applause", "Cough",
        "Sneeze", "Footsteps", "Baby", "Child"
    ],
    "Animal": [
        "Dog", "Cat", "Bird", "Animal", "Bark", "Meow", "Roar",
        "Growl", "Insect", "Livestock", "Horse", "Cattle"
    ],
    "Vehicle": [
        "Car", "Vehicle", "Engine", "Traffic", "Motor", "Truck",
        "Bus", "Train", "Motorcycle", "Horn", "Honk", "Aircraft"
    ],
    "Music": [
        "Music", "Musical instrument", "Singing", "Guitar", "Piano",
        "Drum", "Violin"
    ],
}

audioset_mapping = {
    "Human": [
        "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
        "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Children shouting", "Screaming",
        "Whispering", "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
        "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Singing", "Choir",
        "Yodeling", "Chant", "Mantra", "Child singing", "Synthetic singing", "Rapping", "Humming",
        "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort",
        "Cough", "Throat clearing", "Sneeze", "Sniff", "Run", "Shuffle", "Walk, footsteps",
        "Chewing, mastication", "Biting", "Gargling", "Stomach rumble", "Burping, eructation",
        "Hiccup", "Fart", "Hands", "Finger snapping", "Clapping", "Heart sounds, heartbeat",
        "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd", "Hubbub, speech noise, speech babble",
        "Children playing"
    ],
    "Sirens/Alarms": [
        "Alarm", "Telephone", "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF",
        "Dial tone", "Busy signal", "Alarm clock", "Siren", "Civil defense siren", "Buzzer",
        "Smoke detector, smoke alarm", "Fire alarm", "Foghorn", "Whistle", "Steam whistle",
        "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Car alarm",
        "Reversing beeps"
    ],
    "Traffic": [
        "Vehicle", "Boat, Water vehicle", "Sailboat, sailing ship", "Rowboat, canoe, kayak",
        "Motorboat, speedboat", "Ship", "Motor vehicle (road)", "Car", "Vehicle horn, car horn, honking",
        "Toot", "Power windows, electric windows", "Skidding", "Tire squeal", "Car passing by",
        "Race car, auto racing", "Truck", "Air brake", "Air horn, truck horn", "Ice cream truck, ice cream van",
        "Bus", "Emergency vehicle", "Motorcycle", "Traffic noise, roadway noise", "Rail transport",
        "Train", "Train whistle", "Train horn", "Railroad car, train wagon", "Train wheels squealing",
        "Subway, metro, underground", "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew",
        "Helicopter", "Fixed-wing aircraft, airplane", "Bicycle", "Skateboard"
    ],
    "Animals": [
        "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling",
        "Whimper (dog)", "Cat", "Purr", "Meow", "Hiss", "Caterwaul", "Livestock, farm animals, working animals",
        "Horse", "Clip-clop", "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink",
        "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
        "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk", "Wild animals", "Roaring cats (lions, tigers)",
        "Roar", "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove",
        "Coo", "Crow", "Caw", "Owl", "Hoot", "Bird flight, flapping wings", "Canidae, dogs, wolves",
        "Rodents, rats, mice", "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly",
        "Buzz", "Bee, wasp, etc.", "Frog", "Croak", "Snake", "Rattle", "Whale vocalization"
    ],
    "Nature": [
        "Wind", "Rustling leaves", "Wind noise (microphone)", "Thunderstorm", "Thunder", "Water",
        "Rain", "Raindrop", "Rain on surface", "Stream", "Waterfall", "Ocean", "Waves, surf",
        "Steam", "Gurgling", "Fire", "Crackle", "Eruption"
    ],
    "Music": [
        "Music", "Musical instrument", "Plucked string instrument", "Guitar", "Electric guitar",
        "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar", "Tapping (guitar technique)",
        "Strum", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", "Piano",
        "Electric piano", "Organ", "Electronic organ", "Hammond organ", "Synthesizer", "Sampler",
        "Harpsichord", "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum", "Rimshot",
        "Drum roll", "Bass drum", "Timpani", "Tabla", "Cymbal", "Hi-hat", "Wood block", "Tambourine",
        "Rattle (instrument)", "Maraca", "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone",
        "Glockenspiel", "Vibraphone", "Steelpan", "Orchestra", "Brass instrument", "French horn",
        "Trumpet", "Trombone", "Bowed string instrument", "String section", "Violin, fiddle",
        "Pizzicato", "Cello", "Double bass", "Wind instrument, woodwind instrument", "Flute",
        "Saxophone", "Clarinet", "Harp", "Bell", "Church bell", "Jingle bell", "Bicycle bell",
        "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)", "Harmonica", "Accordion",
        "Bagpipes", "Didgeridoo", "Shofar", "Theremin", "Singing bowl", "Scratching (performance technique)",
        "Pop music", "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge",
        "Progressive rock", "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music",
        "Reggae", "Country", "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music",
        "Jazz", "Disco", "Classical music", "Opera", "Electronic music", "House music", "Techno",
        "Dubstep", "Drum and bass", "Electronica", "Electronic dance music", "Ambient music",
        "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues", "Music for children",
        "New-age music", "Vocal music", "A capella", "Music of Africa", "Afrobeat", "Christian music",
        "Gospel music", "Music of Asia", "Carnatic music", "Music of Bollywood", "Ska", "Traditional music",
        "Independent music", "Song", "Background music", "Theme music", "Jingle (music)", "Soundtrack music",
        "Lullaby", "Video game music", "Christmas music", "Dance music", "Wedding music", "Happy music",
        "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"
    ]
}


SOUNDS_DICT = {}

for category in audioset_mapping.keys():
    values = audioset_mapping.get(category)
    for sound in values:
        SOUNDS_DICT.update({sound:category})

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
