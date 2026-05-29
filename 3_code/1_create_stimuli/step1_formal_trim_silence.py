"""Step 1 for formal-study stimuli: copy or trim single-sentence WAV files.

This step prepares raw ADS-only formal-study audio for MFA and later acoustic
processing. It keeps the original filename style, such as:
    47_01_hap_f.wav

Default input:
    emo_audio/1_raw/formal_study/ADS/female/speaker_*/*.wav

Default output when not trimming:
    emo_audio/2_processed/formal_study/untrimmed/*.wav

Default output when trimming:
    emo_audio/2_processed/formal_study/trimmed/*.wav

The script also writes a manifest:
    emo_audio/2_processed/formal_study/formal_audio_manifest.csv
or, when --speaker-id is used:
    emo_audio/2_processed/formal_study/formal_audio_manifest_speaker_47.csv

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step1_formal_trim_silence.py --dry-run
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step1_formal_trim_silence.py
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step1_formal_trim_silence.py --speaker-id 47
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step1_formal_trim_silence.py --trim-silence
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step1_formal_trim_silence.py --trim-silence --top-db 25
"""

import argparse
import csv
import re
import shutil
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<sent_idx>\d{2})_(?P<emotion>hap|sad|ang|neu|sca)_(?P<gender>[fm])\.wav$",
    flags=re.IGNORECASE,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_wav_filename(path: Path) -> dict[str, str] | None:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    return {key: value.lower() for key, value in match.groupdict().items()}


def infer_style(raw_root: Path, wav_path: Path) -> str:
    rel_parts = wav_path.relative_to(raw_root).parts
    return rel_parts[0] if rel_parts else ""


def speaker_id_from_folder(wav_path: Path) -> str | None:
    speaker_folder = wav_path.parent.name
    match = re.match(r"speaker_(\d+)$", speaker_folder, flags=re.IGNORECASE)
    return match.group(1) if match else None


def output_dir(processed_root: Path, trim_silence: bool) -> Path:
    state = "trimmed" if trim_silence else "untrimmed"
    return processed_root / state


def duration_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def write_manifest(manifest_path: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        print(f"DRY RUN: would write manifest with {len(rows)} rows to {manifest_path}")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_wav",
        "output_wav",
        "style",
        "speaker_id",
        "sent_idx",
        "emotion",
        "gender",
        "trim_state",
        "original_duration_sec",
        "output_duration_sec",
        "trim_start_sec",
        "trim_end_sec",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_wav(
    source_path: Path,
    target_path: Path,
    trim_silence: bool,
    top_db: int,
    frame_length: int,
    hop_length: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, str] | None:
    if target_path.exists() and not overwrite:
        print(f"Skip existing file: {target_path.name}")
        return None

    if dry_run:
        action = "trim and write" if trim_silence else "copy"
        print(f"DRY RUN: would {action} {source_path} -> {target_path}")
        return {
            "original_duration_sec": "",
            "output_duration_sec": "",
            "trim_start_sec": "",
            "trim_end_sec": "",
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not trim_silence:
        shutil.copy2(source_path, target_path)
        duration = duration_seconds(target_path)
        return {
            "original_duration_sec": f"{duration:.6f}",
            "output_duration_sec": f"{duration:.6f}",
            "trim_start_sec": "",
            "trim_end_sec": "",
        }

    import librosa
    import soundfile as sf

    wav_y, wav_sr = librosa.load(path=source_path, sr=None)
    trimmed_y, sample_index = librosa.effects.trim(
        wav_y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    sf.write(str(target_path), trimmed_y, wav_sr)

    original_duration = librosa.get_duration(y=wav_y, sr=wav_sr)
    output_duration = librosa.get_duration(y=trimmed_y, sr=wav_sr)
    return {
        "original_duration_sec": f"{original_duration:.6f}",
        "output_duration_sec": f"{output_duration:.6f}",
        "trim_start_sec": f"{sample_index[0] / wav_sr:.6f}",
        "trim_end_sec": f"{sample_index[1] / wav_sr:.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: prepare formal-study WAV files with optional trimming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Controllable options:
  --raw-root        Raw formal-study input root.
  --processed-root  Processed formal-study output root.
  --speaker-id      Optional speaker number to process, e.g. 47.
  --trim-silence    Trim leading/trailing silence. Without this flag, files are copied.
  --top-db          Silence threshold for librosa.effects.trim. Lower is more aggressive.
  --frame-length    Analysis window length for silence detection.
  --hop-length      Step size between silence-detection windows.
  --dry-run         Preview planned copies/trims without writing files.
  --overwrite       Replace existing processed WAV files.

Important:
  Use the matching --trimmed flag in steps 2-4 only if this step was run with
  --trim-silence.
""",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root() / "emo_audio" / "1_raw" / "formal_study",
        help="Formal-study raw audio root.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root() / "emo_audio" / "2_processed" / "formal_study",
        help="Formal-study processed audio root.",
    )
    parser.add_argument("--speaker-id", help="Optional speaker number to process, e.g. 47.")
    parser.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence.")
    parser.add_argument("--top-db", type=int, default=20, help="librosa trim top_db value.")
    parser.add_argument("--frame-length", type=int, default=512, help="librosa trim frame length.")
    parser.add_argument("--hop-length", type=int, default=128, help="librosa trim hop length.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed WAVs.")
    args = parser.parse_args()

    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw formal-study folder not found: {args.raw_root}")

    target_dir = output_dir(args.processed_root, args.trim_silence)
    manifest_rows = []
    seen_outputs = {}
    wav_paths = sorted(args.raw_root.glob("*/*/speaker_*/*.wav"))

    for wav_path in wav_paths:
        metadata = parse_wav_filename(wav_path)
        if metadata is None:
            print(f"WARNING: skipping non-standard filename: {wav_path}")
            continue
        if args.speaker_id and metadata["speaker_id"] != args.speaker_id:
            continue

        folder_speaker_id = speaker_id_from_folder(wav_path)
        if folder_speaker_id and folder_speaker_id != metadata["speaker_id"]:
            print(f"WARNING: speaker folder and filename disagree: {wav_path}")

        style = infer_style(args.raw_root, wav_path)
        if style.upper() != "ADS":
            print(f"WARNING: expected ADS formal-study file, found style '{style}' in {wav_path}")

        target_path = target_dir / wav_path.name
        if target_path.name in seen_outputs and seen_outputs[target_path.name] != wav_path:
            print(f"WARNING: duplicate output filename {target_path.name}: {seen_outputs[target_path.name]} and {wav_path}")
            continue
        seen_outputs[target_path.name] = wav_path

        result = process_wav(
            source_path=wav_path,
            target_path=target_path,
            trim_silence=args.trim_silence,
            top_db=args.top_db,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if result is None:
            continue

        manifest_rows.append(
            {
                "source_wav": str(wav_path),
                "output_wav": str(target_path),
                "style": style,
                "speaker_id": metadata["speaker_id"],
                "sent_idx": metadata["sent_idx"],
                "emotion": metadata["emotion"],
                "gender": metadata["gender"],
                "trim_state": "trimmed" if args.trim_silence else "untrimmed",
                **result,
            }
        )

    manifest_name = "formal_audio_manifest.csv"
    if args.speaker_id:
        manifest_name = f"formal_audio_manifest_speaker_{args.speaker_id}.csv"
    manifest_path = args.processed_root / manifest_name
    write_manifest(manifest_path, manifest_rows, dry_run=args.dry_run)
    print(f"Done. Prepared {len(manifest_rows)} WAV file(s) for {target_dir}.")


if __name__ == "__main__":
    main()
