"""Step 5 for formal-study stimuli: create reproducible randomized block CSVs.

Each seed creates one complete block CSV containing every WAV file exactly once.
The seed manifest records how each block was generated.

Default input WAV folder:
    1_studysetup/formal_study/stimuli/

Default input metadata folder:
    emo_audio/1_raw/formal_study/ADS/female/speaker_*/

Default output folder:
    1_studysetup/formal_study/

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step5_formal_rand_order.py --seeds 42 --dry-run --overwrite
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step5_formal_rand_order.py --seeds 42 --overwrite
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step5_formal_rand_order.py --seeds 42 73 --overwrite
"""

import argparse
import csv
import os
import random
import re
import wave
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<sent_idx>\d{2})_(?P<emotion>[a-z]+)_(?P<gender>[fm])_"
    r"(?P<sample_rate>\d+)Hz_(?P<target_db>\d+)dB\.wav$",
    flags=re.IGNORECASE,
)
SUPPORTED_EMOTIONS = {"hap": 1, "neu": 2, "sad": 3}
BLOCK_COLUMNS = [
    "id",
    "speaker",
    "sentence_id",
    "duration",
    "sentence",
    "block",
    "stimulus",
    "correct",
    "trig_value",
    "ITI",
]
SEED_COLUMNS = ["block", "seed", "csv_file"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_wav_filename(wav_path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.match(wav_path.name)
    if match is None:
        raise ValueError(f"Invalid formal-study WAV filename: {wav_path.name}")

    parsed = {key: value.lower() for key, value in match.groupdict().items()}
    if parsed["emotion"] not in SUPPORTED_EMOTIONS:
        raise ValueError(
            f"Unsupported emotion '{parsed['emotion']}' in {wav_path.name}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_EMOTIONS))}"
        )
    return parsed


def load_sentence_metadata(metadata_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    metadata: dict[str, dict[str, dict[str, str]]] = {}
    csv_paths = sorted(metadata_root.glob("speaker_*/sentence_metadata.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No sentence_metadata.csv files found under {metadata_root}")

    for csv_path in csv_paths:
        speaker_id = csv_path.parent.name.removeprefix("speaker_")
        if speaker_id in metadata:
            raise ValueError(f"Duplicate metadata folder for speaker {speaker_id}: {csv_path}")

        rows_by_sentence: dict[str, dict[str, str]] = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sent_idx", "transcript", "emo_order"}
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

            for row in reader:
                sent_idx = row["sent_idx"].strip().zfill(2)
                if sent_idx in rows_by_sentence:
                    raise ValueError(f"Duplicate sent_idx {sent_idx} in {csv_path}")
                rows_by_sentence[sent_idx] = row

        metadata[speaker_id] = rows_by_sentence

    return metadata


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid WAV sample rate in {wav_path}")
        return handle.getnframes() / frame_rate


def relative_stimulus_path(wav_path: Path, output_dir: Path) -> str:
    relative_path = os.path.relpath(wav_path, output_dir).replace(os.sep, "/")
    if not relative_path.startswith("."):
        relative_path = f"./{relative_path}"
    return relative_path


def load_stimulus_rows(
    stimuli_dir: Path,
    metadata: dict[str, dict[str, dict[str, str]]],
    output_dir: Path,
) -> list[dict[str, object]]:
    wav_paths = sorted(stimuli_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {stimuli_dir}")

    rows: list[dict[str, object]] = []
    seen_ids = set()
    for wav_path in wav_paths:
        parsed = parse_wav_filename(wav_path)
        speaker_id = parsed["speaker_id"]
        sent_idx = parsed["sent_idx"]
        emotion = parsed["emotion"]

        speaker_rows = metadata.get(speaker_id)
        if speaker_rows is None:
            raise ValueError(f"No sentence metadata found for speaker {speaker_id}: {wav_path.name}")

        metadata_row = speaker_rows.get(sent_idx)
        if metadata_row is None:
            raise ValueError(f"No sentence metadata found for {speaker_id}_{sent_idx}: {wav_path.name}")

        emo_order = {item.strip().lower() for item in metadata_row["emo_order"].split(";") if item.strip()}
        if emotion not in emo_order:
            raise ValueError(f"Emotion '{emotion}' is not listed in metadata for {wav_path.name}")

        stimulus_id = f"{speaker_id}_{sent_idx}_{emotion}_{parsed['gender']}"
        if stimulus_id in seen_ids:
            raise ValueError(f"Duplicate stimulus ID: {stimulus_id}")
        seen_ids.add(stimulus_id)

        rows.append(
            {
                "id": stimulus_id,
                "speaker": int(speaker_id),
                "sentence_id": int(sent_idx),
                "duration": wav_duration_seconds(wav_path),
                "sentence": metadata_row["transcript"].strip(),
                "stimulus": relative_stimulus_path(wav_path, output_dir),
                "correct": SUPPORTED_EMOTIONS[emotion],
                "trig_value": 1,
            }
        )

    return rows


def iti_seconds(rng: random.Random) -> float:
    return round(min(1.5, max(0.5, rng.gauss(1.0, 0.25))), 3)


def randomized_block_rows(
    stimulus_rows: list[dict[str, object]],
    block_number: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    block_rows = [row.copy() for row in stimulus_rows]
    rng.shuffle(block_rows)
    for row in block_rows:
        row["block"] = block_number
        row["ITI"] = iti_seconds(rng)
    return block_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(output_dir: Path, seeds: list[int]) -> tuple[list[Path], Path]:
    block_paths = [output_dir / f"block{block_number}.csv" for block_number in range(1, len(seeds) + 1)]
    return block_paths, output_dir / "randomization_seeds.csv"


def validate_output_paths(paths: list[Path], overwrite: bool) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        formatted_paths = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "Refusing to overwrite existing output file(s). Re-run with --overwrite:\n"
            f"{formatted_paths}"
        )


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Step 5: create reproducible randomized formal-study block CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Each seed creates one complete randomized block containing every WAV once.
The script stores the seed-to-block mapping in randomization_seeds.csv.

ITI values are seconds sampled from N(1.0, 0.25), clamped to 0.5 through 1.5,
then rounded to three decimal places.
""",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="One random seed per output block.")
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=root / "1_studysetup" / "formal_study" / "stimuli",
        help="Folder containing final formal-study WAV files.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=root / "emo_audio" / "1_raw" / "formal_study" / "ADS" / "female",
        help="Folder containing speaker_*/sentence_metadata.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "1_studysetup" / "formal_study",
        help="Folder where block CSVs and randomization_seeds.csv are written.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview outputs without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output CSV files.")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed because they would create duplicate randomized blocks.")
    if not args.stimuli_dir.is_dir():
        raise FileNotFoundError(f"Stimuli directory not found: {args.stimuli_dir}")
    if not args.metadata_root.is_dir():
        raise FileNotFoundError(f"Metadata root not found: {args.metadata_root}")
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    metadata = load_sentence_metadata(args.metadata_root)
    stimulus_rows = load_stimulus_rows(args.stimuli_dir, metadata, args.output_dir)
    block_paths, seed_manifest_path = output_paths(args.output_dir, args.seeds)
    validate_output_paths([*block_paths, seed_manifest_path], args.overwrite)

    seed_rows = []
    for block_number, (seed, block_path) in enumerate(zip(args.seeds, block_paths), start=1):
        block_rows = randomized_block_rows(stimulus_rows, block_number, seed)
        seed_rows.append({"block": block_number, "seed": seed, "csv_file": block_path.name})
        if args.dry_run:
            print(f"DRY RUN: would write {len(block_rows)} row(s) to {block_path}")
        else:
            write_csv(block_path, BLOCK_COLUMNS, block_rows)

    if args.dry_run:
        print(f"DRY RUN: would write {len(seed_rows)} seed record(s) to {seed_manifest_path}")
    else:
        write_csv(seed_manifest_path, SEED_COLUMNS, seed_rows)

    action = "Would create" if args.dry_run else "Created"
    print(f"Done. {action} {len(block_paths)} block CSV file(s) from {len(stimulus_rows)} stimulus WAV file(s).")


if __name__ == "__main__":
    main()
