#!/usr/bin/env python3
"""
Inference CLI and API for VoxCPM Africa.

Generates speech in African languages and English with voice cloning using
the fine-tuned VoxCPM 0.5B model and Africa-G2P.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from voxcpm_africa.data import text_to_ipa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VoxCPMAfrica:
    def __init__(
        self,
        model_dir: str = "/mnt/volume_d2wey28/models/VoxCPM-0.5B",
        voxcpm_src: str = "/mnt/volume_d2wey28/projects/vox-cpm-africa/VoxCPM/src",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        if os.path.isdir(voxcpm_src):
            sys.path.insert(0, voxcpm_src)

        from voxcpm.model import VoxCPMModel

        logger.info(f"Loading VoxCPM model from {model_dir}...")
        self.model = VoxCPMModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.sampling_rate = 16000

    def generate(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        use_ipa: bool = True,
    ):
        """Synthesize speech audio from text.

        Args:
            text: Input text.
            language: Target African language code.
            ref_audio: Optional reference audio for voice cloning.
            use_ipa: Convert text to IPA via africa-g2p.
        """
        prompt_text = text_to_ipa(text, language) if (use_ipa and language) else text
        with torch.no_grad():
            audio = self.model.generate(
                text=prompt_text,
                prompt_audio=ref_audio,
            )
        return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/mnt/volume_d2wey28/models/VoxCPM-0.5B")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--language", type=str, default="twi", help="Language code")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio clip")
    parser.add_argument("--output", type=str, default="voxcpm_africa_output.wav", help="Output WAV path")
    args = parser.parse_args()

    tts = VoxCPMAfrica(model_dir=args.model)
    audio = tts.generate(text=args.text, language=args.language, ref_audio=args.ref_audio)
    sf.write(args.output, audio, tts.sampling_rate)
    logger.info(f"Generated speech saved to {args.output}")


if __name__ == "__main__":
    main()
