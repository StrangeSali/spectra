# Spectra AI — Real-Time Environmental Audio Visualizer

Spectra AI is a Python-based real-time environmental sound visualizer engineered for visual accessibility. It captures live ambient audio, computes real-time DSP features, classifies environmental sound events using Google's pre-trained YAMNet model, and maps the output into dynamic, color-coded visual shapes rendered in Pygame.

---

## Technical Features & Performance

* **Target Latency Budget:** <= 50ms pipeline budget for continuous 60 FPS graphics.
* **Multithreaded Architecture:** Completely decouples high-frequency non-blocking audio ingestion, ML inference (~15 Hz), and high-refresh canvas rendering (60 Hz).
* **Smooth Visual Transitions:** Uses Linear Interpolation (LERP) across geometric scaling vectors and RGB color channels to prevent visual jitter.
* **Zero Web Dependencies:** 100% pure Python execution (no JavaScript, no browser runtimes).

---
#TODO update structure
## Directory Architecture

```text
spectra_ai/
├── README.md                  # Project overview and setup instructions
├── requirements.txt           # Python dependency locks
├── config.py                  # Operational constants, sample rates, & color maps
├── main.py                    # Pygame loop, thread orchestration, & UI rendering
├── audio/
│   ├── __init__.py
│   ├── stream.py              # sounddevice microphone input stream thread
│   └── dsp.py                 # Real-time RMS energy & STFT computation
├── models/
│   ├── __init__.py
│   └── classifier.py          # Asynchronous YAMNet inference worker thread
└── graphics/
    ├── __init__.py
    ├── canvas.py              # Pygame surface setup & base shapes
    └── visualizer.py          # LERP interpolation & reactive particle effects
```

---

## Hardware & Environment Requirements

* **Python:** 3.9 to 3.11
* **Audio Hardware:** Working microphone input device.
* **System Library:** PortAudio (Required by sounddevice).

### System PortAudio Setup

* **macOS (Homebrew):** `brew install portaudio`
* **Linux (Debian/Ubuntu):** `sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev`
* **Windows:** Included automatically with standard wheel installation.

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone [https://github.com/your-org/spectra_ai.git](https://github.com/your-org/spectra_ai.git)
cd spectra_ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dependencies (`requirements.txt`)

```text
numpy>=1.23.5,<2.0.0
sounddevice>=0.4.6
scipy>=1.10.1
tensorflow>=2.11.0
tensorflow-hub>=0.13.0
pygame>=2.5.0
```

---

## Configuration (`config.py`)

Key operational parameters configured within the application:

```python
# Audio Constraints
TARGET_SAMPLE_RATE = 16000  # Native YAMNet sampling rate
INPUT_SAMPLE_RATE = 44100   # Hardware default microphone rate
CHUNK_SIZE = 1024           # ~23.2ms buffer window at 44.1kHz
QUEUE_MAXSIZE = 10

# Display Constraints
FPS = 60
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

#TODO create dictionary - category : RGB color
# Class Color Mapping (RGB)
CATEGORY_COLORS = {

}
```

---

## Execution Flow & Controls

1. Connect your input microphone.
2. Launch the application:

```bash
python main.py
```

### Controls
* **ESC / Close Window**: Terminate audio streams, safely stop background threads, and exit Pygame cleanly.

---

## Pipeline Execution Overview

```text
[ Hardware Microphone ]
          │
          ▼  (44.1 kHz Mono Stream Callback)
[ AudioStreamHandler Thread ]
          │  ──> Resample to 16 kHz via scipy.signal.resample_poly
          │  ──> Compute RMS Energy (for geometric scaling)
          ▼
   [ thread-safe queue.Queue ]
          │
          ├──> [ YAMNetInferenceWorker Thread (~15 Hz) ]
          │          │ ──> Continuous 1.0s Rolling Buffer
          │          │ ──> Multi-Label Sound Event Classification
          │          └──> Thread Lock Variable Guard
          │
          └──> [ Main Pygame Loop (60 FPS) ]
                     │ ──> Read Latest State & RMS Energy
                     │ ──> Apply LERP Scaling & Color Vector Shift
                     └──> Render Dynamic Canvas & On-Screen HUD
```

---

## Team & Task Division (8-Day MVP)

| Role | Responsibilities |
| :--- | :--- |
| **Dev 1 (Audio & DSP)** | Non-blocking stream threads and real-time RMS calculation. |
| **Dev 2 (Machine Learning)** | YAMNet integration, sliding window buffers, confidence thresholding, and AudioSet taxonomy mapping. |
| **Dev 3 (Graphics & UX)** | Pygame 60 FPS loop, visual LERP engines, high-contrast color design, and HUD instrumentation. |

---

## License & Acknowledgments

* **YAMNet Model:** Pre-trained model provided by Google via TensorFlow Hub under the Apache 2.0 License.
