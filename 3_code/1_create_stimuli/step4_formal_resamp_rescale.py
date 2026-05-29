"""Step 4 for formal-study stimuli: resample and RMS-rescale WAV files.

This step creates the final normalized WAV files from the Step 1 processed
corpus. It does not change TextGrid files.

Default input:
    emo_audio/2_processed/formal_study/untrimmed/*.wav

Default output:
    emo_audio/2_processed/formal_study/untrimmed/48000Hz_80dB/

Example output filename:
    47_01_hap_f_48000Hz_80dB.wav

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step4_formal_resamp_rescale.py --dry-run
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step4_formal_resamp_rescale.py
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step4_formal_resamp_rescale.py --speaker-id 47
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step4_formal_resamp_rescale.py --target-sf 48000 --target-db 80
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step4_formal_resamp_rescale.py --trimmed
"""

import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<sent_idx>\d{2})_(?P<emotion>hap|sad|ang|neu|sca)_(?P<gender>[fm])\.wav$",
    flags=re.IGNORECASE,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def corpus_dir(processed_root: Path, trimmed: bool) -> Path:
    return processed_root / ("trimmed" if trimmed else "untrimmed")


def scale_output_dir(audio_dir: Path, target_sf: int, target_db: int) -> Path:
    return audio_dir / f"{target_sf}Hz_{target_db}dB"


def scaled_filename(wav_path: Path, target_sf: int, target_db: int) -> str:
    return f"{wav_path.stem}_{target_sf}Hz_{target_db}dB{wav_path.suffix}"


def process_wav(
    wav_path: Path,
    out_path: Path,
    target_sf: int,
    target_db: int,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        print(f"Skip already processed file: {out_path.name}")
        return False

    if dry_run:
        print(f"DRY RUN: would resample/rescale {wav_path.name} -> {out_path}")
        return True

    wav_y, wav_sr = librosa.load(wav_path, sr=None)
    if wav_sr != target_sf:
        wav_y = librosa.resample(wav_y, orig_sr=wav_sr, target_sr=target_sf)
        wav_sr = target_sf

    current_rms = float(np.sqrt(np.mean(wav_y**2)))
    if current_rms == 0:
        print(f"WARNING: skipping silent file with zero RMS: {wav_path.name}")
        return False

    target_rms = 2e-5 * (10 ** (target_db / 20))
    wav_y = wav_y * (target_rms / current_rms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav_y, wav_sr)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: resample and rescale formal-study WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Controllable options:
  --processed-root  Root containing Step 1 processed WAV files.
  --speaker-id      Optional speaker number to process, e.g. 47.
  --trimmed         Use the trimmed processed corpus.
  --target-sf       Target sampling frequency in Hz. Default: 48000.
  --target-db       Target RMS intensity level. Default: 80.
  --dry-run         Preview resampled/rescaled output paths without writing files.
  --overwrite       Replace existing scaled WAV files.

Common issue:
  Files with zero RMS are skipped because a gain cannot be computed for silence.
""",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root() / "emo_audio" / "2_processed" / "formal_study",
        help="Formal-study processed audio root.",
    )
    parser.add_argument("--speaker-id", help="Optional speaker number to process, e.g. 47.")
    parser.add_argument("--trimmed", action="store_true", help="Use the trimmed processed corpus.")
    parser.add_argument("--target-sf", type=int, default=48000, help="Target sampling frequency in Hz.")
    parser.add_argument("--target-db", type=int, default=80, help="Target intensity in dB SPL-like RMS scaling.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scaled WAVs.")
    args = parser.parse_args()

    audio_dir = corpus_dir(args.processed_root, args.trimmed)
    if not audio_dir.exists():
        message = f"Processed audio directory not found: {audio_dir}"
        if args.dry_run:
            print(f"DRY RUN WARNING: {message}")
            return
        raise FileNotFoundError(message)

    out_dir = scale_output_dir(audio_dir, args.target_sf, args.target_db)
    processed_count = 0

    for wav_path in sorted(audio_dir.glob("*.wav")):
        match = FILENAME_PATTERN.match(wav_path.name)
        if match is None:
            print(f"WARNING: skipping non-standard filename: {wav_path.name}")
            continue
        if args.speaker_id and match.group("speaker_id") != args.speaker_id:
            continue

        out_path = out_dir / scaled_filename(wav_path, args.target_sf, args.target_db)
        if process_wav(
            wav_path=wav_path,
            out_path=out_path,
            target_sf=args.target_sf,
            target_db=args.target_db,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        ):
            processed_count += 1

    print(f"Done. {'Would create' if args.dry_run else 'Created'} {processed_count} scaled WAV file(s).")


if __name__ == "__main__":
    main()
