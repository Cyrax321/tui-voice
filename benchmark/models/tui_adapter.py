"""Model Adapter for Tui Voice Mini (Malayalam).

Supports:
1. PyTorch native inference via TinyVoca synthesize API
2. ONNX Runtime inference via onnxruntime
"""
import os
import sys
import json
import numpy as np
import re
from pathlib import Path
from .base_adapter import BaseTTSAdapter

# Import phonemizer frontend
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from tui_voice import phonemes as P
except ImportError:
    P = None


class TuiVoiceAdapter(BaseTTSAdapter):
    """Adapter for Tui Voice Mini Malayalam (4.5M parameters)."""

    def __init__(
        self,
        checkpoint_path: str = None,
        vocab_plan_path: str = None,
        config_path: str = None,
        onnx_path: str = None,
        pitch_shift: float = 0.15,
        pitch_range: float = 0.85,
        speed: float = 1.0,
        voice: int = 0
    ):
        super().__init__(name="Tui Voice Mini Malayalam", approx_parameters=4537291, sample_rate=24000)
        self.checkpoint_path = checkpoint_path
        self.vocab_plan_path = vocab_plan_path
        self.config_path = config_path
        self.onnx_path = onnx_path
        self.pitch_shift = pitch_shift
        self.pitch_range = pitch_range
        self.speed = speed
        self.voice = voice

        self.model = None
        self.ort_session = None
        self.plan = None
        self.cfg = None
        self.upstream_symbols = None

    def load(self, device: str = "cpu"):
        """Load PyTorch checkpoint or ONNX session."""
        # Prefer ONNX if available
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(self.onnx_path)
                self.is_loaded = True
                print(f"Loaded Tui Voice ONNX: {self.onnx_path}")
            except Exception as e:
                print(f"Notice: ONNX load failed ({e}), falling back to PyTorch")

        if not self.is_loaded and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            import torch
            sys.path.insert(0, "/content/tiny-voca/src")
            try:
                from tiny_voca.config import Config
                from tiny_voca.model import TinyVoca

                state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                cfg_dict = state.get("config")
                if not cfg_dict and self.config_path and os.path.exists(self.config_path):
                    cfg_dict = json.load(open(self.config_path))
                self.cfg = Config.from_dict(cfg_dict) if cfg_dict else Config()
                self.model = TinyVoca(self.cfg)

                sd = state.get("model", state.get("system", state))
                self.model.load_state_dict(sd, strict=False)
                self.model.eval().to(device)
                self.is_loaded = True
                print(f"Loaded Tui Voice PyTorch: {self.checkpoint_path}")
            except Exception as e:
                print(f"Notice: PyTorch model load failed: {e}")

        # Load vocab plan
        if self.vocab_plan_path and os.path.exists(self.vocab_plan_path):
            self.plan = json.load(open(self.vocab_plan_path))
            try:
                from tiny_voca.text import SYMBOLS
                self.upstream_symbols = list(SYMBOLS)
            except Exception:
                self.upstream_symbols = self.plan.get("new_symbols", [])

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text into a 24 kHz float32 audio waveform."""
        if not self.is_loaded:
            # Fallback mock for testing environment when running without GPU/weights
            sr = 24000
            dur = max(0.5, len(text) * 0.08)
            t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
            audio = 0.2 * np.sin(2 * np.pi * 220 * t)
            return audio, sr

        import torch
        chunks = re.split(r'([.?!।\n]+)', text)
        sentences = [c.strip() for c in chunks if c.strip() and any(ch.isalnum() for ch in c)]
        if not sentences:
            sentences = [text.strip()]

        all_waves = []
        pause_samples = int(self.sample_rate * 0.18)
        silence = torch.zeros(pause_samples)

        for idx, sent in enumerate(sentences):
            tokens_str = P.phonemize(sent)
            token_ids = P.encode(tokens_str, self.plan, self.upstream_symbols)
            tok_tensor = torch.tensor([token_ids], dtype=torch.long)
            stress = torch.zeros_like(tok_tensor)
            lengths = torch.tensor([len(token_ids)], dtype=torch.long)
            speakers = torch.tensor([self.voice], dtype=torch.long)

            with torch.inference_mode():
                wave, _, _ = self.model.synthesize(
                    tok_tensor, stress, lengths, speakers,
                    speed=self.speed,
                    pitch_shift=self.pitch_shift,
                    pitch_range=self.pitch_range,
                    voiced_vowels=True
                )
            all_waves.append(wave.squeeze(0).cpu())
            if idx < len(sentences) - 1:
                all_waves.append(silence)

        full_wave = torch.cat(all_waves)
        audio = full_wave.numpy()
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0.99:
            audio = audio * (0.95 / peak)

        return audio, self.sample_rate
