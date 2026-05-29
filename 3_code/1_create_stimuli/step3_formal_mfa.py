"""Step 3 for formal-study stimuli: run Montreal Forced Aligner.

This step aligns the processed WAV/TXT pairs from Step 2 and writes TextGrid
files. It does not create transcripts; run Step 2 first.

Default input:
    emo_audio/2_processed/formal_study/untrimmed/

Default output:
    emo_audio/3_mfa/formal_study/untrimmed/

The script runs MFA through:
    mamba activate aligner && mfa ...

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step3_formal_mfa.py --dry-run
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step3_formal_mfa.py
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step3_formal_mfa.py --speaker-id 47
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step3_formal_mfa.py --trimmed
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<sent_idx>\d{2})_(?P<emotion>hap|sad|ang|neu|sca)_(?P<gender>[fm])\.wav$",
    flags=re.IGNORECASE,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def corpus_dir(processed_root: Path, trimmed: bool) -> Path:
    return processed_root / ("trimmed" if trimmed else "untrimmed")


def mfa_output_dir(mfa_root: Path, trimmed: bool) -> Path:
    return mfa_root / ("trimmed" if trimmed else "untrimmed")


def speaker_corpus_dir(mfa_root: Path, trimmed: bool, speaker_id: str) -> Path:
    trim_state = "trimmed" if trimmed else "untrimmed"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return mfa_root / "_speaker_corpus" / trim_state / f"speaker_{speaker_id}_{timestamp}"


def build_speaker_corpus(source_dir: Path, target_dir: Path, speaker_id: str, dry_run: bool) -> int:
    """Copy one speaker's complete WAV/TXT pairs into a small MFA corpus."""
    copied_pairs = 0
    for wav_path in sorted(source_dir.glob("*.wav")):
        match = FILENAME_PATTERN.match(wav_path.name)
        if match is None or match.group("speaker_id") != speaker_id:
            continue

        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"WARNING: missing transcript for MFA, skipping {wav_path.name}: {txt_path.name}")
            continue

        if dry_run:
            print(f"DRY RUN: would copy {wav_path.name} and {txt_path.name} to {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, target_dir / wav_path.name)
            shutil.copy2(txt_path, target_dir / txt_path.name)
        copied_pairs += 1

    return copied_pairs


def run_mfa_command(command_list: list[str], dry_run: bool) -> None:
    printable = " ".join(command_list)
    full_command = f"mamba activate aligner && {printable}"

    if dry_run:
        print(f"DRY RUN: would run: {printable}")
        return

    print(f"\nRunning: {printable}")
    result = subprocess.run(
        full_command,
        shell=True,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: run MFA for the formal-study corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Controllable options:
  --processed-root   Root containing processed WAV/TXT pairs.
  --mfa-root         Root where TextGrid outputs should be saved.
  --speaker-id       Optional speaker number to align, e.g. 47.
  --trimmed          Use the trimmed processed corpus.
  --dictionary       MFA dictionary name or path. Default: english_us_arpa.
  --acoustic-model   MFA acoustic model name or path. Default: english_us_arpa.
  --dry-run          Print MFA commands without running them.

Common issue:
  MFA must be installed in a mamba environment named aligner, or the mamba
  activation command needs to be adjusted.

Speaker targeting:
  When --speaker-id is used, the script copies only that speaker's WAV/TXT
  pairs into a timestamped folder under:
    emo_audio/3_mfa/formal_study/_speaker_corpus/
  The TextGrid output still goes to the normal trimmed or untrimmed MFA folder.
""",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root() / "emo_audio" / "2_processed" / "formal_study",
        help="Formal-study processed audio root.",
    )
    parser.add_argument(
        "--mfa-root",
        type=Path,
        default=project_root() / "emo_audio" / "3_mfa" / "formal_study",
        help="Formal-study MFA output root.",
    )
    parser.add_argument("--speaker-id", help="Optional speaker number to align, e.g. 47.")
    parser.add_argument("--trimmed", action="store_true", help="Use the trimmed processed corpus.")
    parser.add_argument("--dictionary", default="english_us_arpa", help="MFA dictionary name or path.")
    parser.add_argument("--acoustic-model", default="english_us_arpa", help="MFA acoustic model name or path.")
    parser.add_argument("--dry-run", action="store_true", help="Print MFA commands without running them.")
    args = parser.parse_args()

    audio_corpus_dir = corpus_dir(args.processed_root, args.trimmed)
    output_dir = mfa_output_dir(args.mfa_root, args.trimmed)

    if not audio_corpus_dir.exists():
        message = f"Processed audio directory not found: {audio_corpus_dir}"
        if args.dry_run:
            print(f"DRY RUN WARNING: {message}")
        else:
            raise FileNotFoundError(message)

    if args.speaker_id and audio_corpus_dir.exists():
        target_corpus_dir = speaker_corpus_dir(args.mfa_root, args.trimmed, args.speaker_id)
        copied_pairs = build_speaker_corpus(
            source_dir=audio_corpus_dir,
            target_dir=target_corpus_dir,
            speaker_id=args.speaker_id,
            dry_run=args.dry_run,
        )
        if copied_pairs == 0 and not args.dry_run:
            raise FileNotFoundError(f"No complete WAV/TXT pairs found for speaker {args.speaker_id} in {audio_corpus_dir}")
        print(
            f"{'DRY RUN: would prepare' if args.dry_run else 'Prepared'} "
            f"{copied_pairs} MFA pair(s) for speaker {args.speaker_id}."
        )
        audio_corpus_dir = target_corpus_dir

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    run_mfa_command(["mfa", "version"], dry_run=args.dry_run)
    run_mfa_command(
        [
            "mfa",
            "validate",
            f'"{audio_corpus_dir}"',
            f'"{args.dictionary}"',
            f'"{args.acoustic_model}"',
        ],
        dry_run=args.dry_run,
    )
    run_mfa_command(
        [
            "mfa",
            "align",
            f'"{audio_corpus_dir}"',
            f'"{args.dictionary}"',
            f'"{args.acoustic_model}"',
            f'"{output_dir}"',
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
