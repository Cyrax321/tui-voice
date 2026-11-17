"""Base Adapter Interface for Malayalam TTS Benchmark Models."""
from abc import ABC, abstractmethod
import numpy as np


class BaseTTSAdapter(ABC):
    """Abstract interface that all benchmarked TTS models must implement."""

    def __init__(self, name: str, approx_parameters: int, sample_rate: int = 24000):
        self.name = name
        self.approx_parameters = approx_parameters
        self.sample_rate = sample_rate
        self.is_loaded = False

    @abstractmethod
    def load(self, device: str = "cpu"):
        """Load model weights into memory."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text into a mono float32 numpy waveform and return (audio, sample_rate)."""
        pass

    def get_parameter_count(self) -> int:
        """Return exact or approximate parameter count."""
        return self.approx_parameters
