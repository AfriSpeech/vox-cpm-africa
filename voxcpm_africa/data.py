"""Data preparation for VoxCPM Africa.

Reads audio-text manifests from the aligned African corpus:
1. Loads audio slices directly using soundfile.
2. Converts transcripts to IPA phonemes via africa-g2p.
3. Exports sliced WAV files and writes JSONL manifests formatted for VoxCPM:
   {"text": "<ipa_transcript>", "audio": "<path_to_wav>"}
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

RULE_ALIASES = {
    "tw": "twi", "ak": "twi", "aka": "twi", "akan": "twi",
    "sw": "swh", "swa": "swh",
    "ha": "hau", "hausa": "hau",
    "yo": "yor", "yoruba": "yor",
    "ig": "ibo", "igbo": "ibo",
    "zu": "zul", "zulu": "zul",
    "en": "eng", "english": "eng",
    "am": "amh", "amharic": "amh",
    "sn": "sna", "shona": "sna",
    "rw": "kin", "kinyarwanda": "kin",
    "ti": "tir", "tigrinya": "tir",
    "so": "som", "somali": "som",
    "ln": "lin", "lingala": "lin",
    "wo": "wol", "wolof": "wol",
    "xh": "xho", "xhosa": "xho",
}


def text_to_ipa(text: str, lang: str) -> str:
    """Convert text to phonetic IPA using africa-g2p with fallback."""
    if not text:
        return ""
    try:
        import africa_g2p
        code = RULE_ALIASES.get(lang.lower(), lang.lower())
        return africa_g2p.g2p(text, code, output="ipa")
    except Exception:
        return text


def load_audio_slice(path: str, start: float, duration: float, target_sr: int = 16000) -> np.ndarray:
    """Load a slice of audio file and resample to target_sr (mono float32)."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = max(0, int(round(start * sr)))
    frames = int(round(duration * sr))
    data, _ = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=True)
    wav = data.mean(1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav)[None], orig_freq=sr, new_freq=target_sr
        )[0].numpy()
    return wav
