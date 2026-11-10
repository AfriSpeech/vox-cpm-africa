#!/usr/bin/env python3
"""
Prepares the African speech dataset for VoxCPM fine-tuning.

Reads train_final.jsonl and valid_final.jsonl:
- Slices audio at [start, start+duration] and saves 16kHz WAV files
- Converts text to phonetic IPA via africa-g2p
- Writes JSONL manifests formatted for VoxCPM: {"text": "...", "audio": "..."}
"""

import argparse
import json
import logging
import os
from pathlib import Path

import soundfile as sf
from tqdm.auto import tqdm

from voxcpm_africa.data import load_audio_slice, text_to_ipa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", type=str, default="/mnt/volume_d2wey28/data/africa-tts/train_final.jsonl")
    parser.add_argument("--output_manifest", type=str, default="/mnt/volume_d2wey28/data/vox-cpm-africa/train_manifest.jsonl")
    parser.add_argument("--audio_dir", type=str, default="/mnt/volume_d2wey28/data/vox-cpm-africa/wavs")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum samples to slice and export (default: 50,000)")
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.audio_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    logger.info(f"Preparing VoxCPM dataset from {args.input_jsonl} (max_samples: {args.max_samples})...")

    with open(args.input_jsonl, "r", encoding="utf-8") as f, open(out_manifest, "w", encoding="utf-8") as mf:
        pbar = tqdm(desc="Exporting Audio & Manifest for VoxCPM")
        exported = 0
        for idx, line in enumerate(f):
            if args.max_samples and exported >= args.max_samples:
                break
            try:
                d = json.loads(line)
                wav_path = d.get("path")
                start = float(d.get("start", 0.0))
                dur = float(d.get("duration", 0.0))
                if not wav_path or not os.path.exists(wav_path) or dur < 1.0 or dur > 20.0:
                    continue

                text = d.get("transcript") or text_to_ipa(d.get("transcript_orig") or d.get("text", ""), d.get("iso", "eng"))
                if not text or len(text.strip()) < 3:
                    continue

                # Slice audio
                wav_np = load_audio_slice(wav_path, start, dur, target_sr=args.sample_rate)

                out_wav_name = f"sample_{exported:07d}.wav"
                out_wav_path = wav_dir / out_wav_name
                sf.write(out_wav_path, wav_np, args.sample_rate, subtype="PCM_16")

                record = {"audio": str(out_wav_path), "text": text}
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
                pbar.update(1)

            except Exception as e:
                logger.warning(f"Skipping line {idx}: {e}")

        pbar.close()

    logger.info(f"Done! Exported {exported} audio files to {wav_dir} and manifest to {out_manifest}")


if __name__ == "__main__":
    main()
