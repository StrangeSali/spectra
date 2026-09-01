import io
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional

import av
import numpy as np
import requests

SAMPLE_RATE = 16000


def pcm16_to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:

    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype("<i2", copy=False).tobytes())

    return buffer.getvalue()


def rms_of_pcm16(samples: np.ndarray) -> float:
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)

    if pcm.size == 0:
        return 0.0

    normalised = pcm.astype(np.float32) / 32768.0

    return float(np.sqrt(np.mean(np.square(normalised))))


class AudioDownsampler:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

    def process(self, frames: List[av.AudioFrame]) -> np.ndarray:
        chunks = []

        for frame in frames:
            for resampled in self._resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            return np.zeros(0, dtype=np.int16)

        return np.concatenate(chunks).astype(np.int16, copy=False)


class RollingAudioBuffer:

    def __init__(
        self,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        self._window_size = int(window_seconds * sample_rate)
        self._hop_size = int(hop_seconds * sample_rate)
        self._samples = np.zeros(0, dtype=np.int16)
        self._since_emit = 0

    def push(self, samples: np.ndarray) -> Optional[np.ndarray]:
        pcm = np.asarray(samples, dtype=np.int16).reshape(-1)

        if pcm.size == 0:
            return None

        self._samples = np.concatenate([self._samples, pcm])[-self._window_size :]
        self._since_emit += pcm.size

        if self._samples.size < self._window_size:
            return None

        if self._since_emit < self._hop_size:
            return None

        self._since_emit = 0

        return self._samples.copy()


def latest_recent_entry(payload: Any) -> Optional[Dict[str, Any]]:

    if not isinstance(payload, dict):
        return None

    entries = payload.get("predictions")

    if not isinstance(entries, list):
        return None

    valid = [
        entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("predictions"), list)
    ]

    if not valid:
        return None

    return max(valid, key=lambda entry: entry.get("timestamp") or 0.0)


class UploadError(Exception):
    pass


class LatestChunkUploader(threading.Thread):

    def __init__(
        self,
        url: str,
        post: Callable[..., Any] = requests.post,
        timeout: float = 20.0,
        filename: str = "live_chunk.wav",
    ):
        super().__init__(name="spectra-chunk-uploader", daemon=True)

        self._url = url
        self._post = post
        self._timeout = timeout
        self._filename = filename

        self._cond = threading.Condition()
        self._pending: Optional[bytes] = None
        self._stopped = False

        self._stats: Dict[str, Any] = {
            "uploads": 0,
            "failures": 0,
            "dropped": 0,
            "in_flight": False,
            "last_latency": None,
            "last_error": None,  # error of the most recent attempt; None after a success
            "last_failure": None,  # most recent failure ever, kept after recovery
            "last_predictions": None,
            "last_timestamp": None,
        }

    # --------------------------------------------------

    def submit(self, wav_bytes: bytes) -> None:
        with self._cond:
            if self._stopped:
                return

            if self._pending is not None:
                self._stats["dropped"] += 1

            self._pending = wav_bytes
            self._cond.notify()

    def stop(self, join_timeout: float = 2.0) -> None:
        with self._cond:
            self._stopped = True
            self._pending = None
            self._cond.notify_all()

        if self.is_alive() and threading.current_thread() is not self:
            self.join(join_timeout)

    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            return dict(self._stats)

    # --------------------------------------------------

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()

                if self._stopped:
                    return

                chunk, self._pending = self._pending, None
                self._stats["in_flight"] = True

            self._upload(chunk)

    def _upload(self, chunk: bytes) -> None:
        started = time.monotonic()

        try:
            response = self._post(
                self._url,
                files={"file": (self._filename, chunk, "audio/wav")},
                timeout=self._timeout,
            )

            if response.status_code != 200:
                raise UploadError(
                    f"HTTP {response.status_code}: {str(response.text)[:200]}"
                )

            data = response.json()

            with self._cond:
                self._stats["uploads"] += 1
                self._stats["last_error"] = None
                self._stats["last_predictions"] = data.get("predictions", [])
                self._stats["last_timestamp"] = data.get("timestamp")

        except (
            Exception
        ) as error:  # network, HTTP status, JSON — all must be survivable
            message = f"{type(error).__name__}: {error}"

            with self._cond:
                self._stats["failures"] += 1
                self._stats["last_error"] = message
                self._stats["last_failure"] = message

        finally:
            with self._cond:
                self._stats["last_latency"] = time.monotonic() - started
                self._stats["in_flight"] = False


class RecentPredictionsPoller(threading.Thread):
    def __init__(
        self,
        fetch: Callable[[int], Any],
        interval: float = 0.5,
        n: int = 1,
    ):
        super().__init__(name="spectra-recent-poller", daemon=True)

        self._fetch = fetch
        self._interval = interval
        self._n = n

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._latest: Dict[str, Any] = {
            "timestamp": None,
            "predictions": [],
            "error": None,
            "updated_at": None,  # time.monotonic() when a *new* entry arrived
        }

    def latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()

        if self.is_alive() and threading.current_thread() is not self:
            self.join(join_timeout)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self._interval)

    def _poll_once(self) -> None:
        try:
            entry = latest_recent_entry(self._fetch(self._n))

            with self._lock:
                self._latest["error"] = None

                if (
                    entry is not None
                    and entry.get("timestamp") != self._latest["timestamp"]
                ):
                    self._latest["timestamp"] = entry.get("timestamp")
                    self._latest["predictions"] = entry.get("predictions", [])
                    self._latest["updated_at"] = time.monotonic()

        except Exception as error:  # keep polling; the page shows the error
            with self._lock:
                self._latest["error"] = f"{type(error).__name__}: {error}"
