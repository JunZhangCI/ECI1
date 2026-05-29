"""Step 2 for formal-study stimuli: create MFA transcript txt files.

This step reads each speaker's sentence_metadata.csv and writes one transcript
txt file beside each processed WAV. The transcript filename matches the WAV stem:
    47_01_hap_f.wav
    47_01_hap_f.txt

Run Step 0 first:
    formal_docx_to_csv.py

Run Step 1 before this step:
    step1_formal_trim_silence.py

Default input metadata:
    emo_audio/1_raw/formal_study/**/speaker_*/sentence_metadata.csv

Default input/output audio folder:
    emo_audio/2_processed/formal_study/untrimmed/

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step2_formal_transcript_txt_creator.py --dry-run
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step2_formal_transcript_txt_creator.py
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step2_formal_transcript_txt_creator.py --speaker-id 47
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\step2_formal_transcript_txt_creator.py --trimmed
"""

import argparse
import csv
import re
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


def corpus_dir(processed_root: Path, trimmed: bool) -> Path:
    return processed_root / ("trimmed" if trimmed else "untrimmed")


def load_sentence_metadata(raw_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    metadata = {}
    for csv_path in sorted(raw_root.glob("*/*/speaker_*/sentence_metadata.csv")):
        speaker_id = csv_path.parent.name.replace("speaker_", "")
        rows_by_sentence = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sent_idx", "transcript", "emo_order"}
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")
            for row in reader:
                row["sent_idx"] = row["sent_idx"].zfill(2)
                row["emo_order"] = row["emo_order"].lower()
                rows_by_sentence[row["sent_idx"]] = row
        metadata[speaker_id] = rows_by_sentence
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: create formal-study transcript txt files for MFA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Controllable options:
  --raw-root        Root containing speaker sentence_metadata.csv files.
  --processed-root  Root where Step 1 wrote processed WAV files.
  --speaker-id      Optional speaker number to process, e.g. 47.
  --trimmed         Use emo_audio/2_processed/formal_study/trimmed instead of untrimmed.
  --dry-run         Preview transcript files without writing them.
  --overwrite       Replace existing txt files.

Validation:
  The script warns if a WAV sentence ID is missing from sentence_metadata.csv,
  or if the filename emotion is not listed in that sentence's emo_order.
""",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root() / "emo_audio" / "1_raw" / "formal_study",
        help="Formal-study raw audio root containing speaker metadata CSVs.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root() / "emo_audio" / "2_processed" / "formal_study",
        help="Formal-study processed audio root.",
    )
    parser.add_argument("--speaker-id", help="Optional speaker number to process, e.g. 47.")
    parser.add_argument("--trimmed", action="store_true", help="Use the trimmed processed corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned transcript outputs without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing txt files.")
    args = parser.parse_args()

    audio_dir = corpus_dir(args.processed_root, args.trimmed)
    if not audio_dir.exists():
        message = f"Processed audio directory not found: {audio_dir}"
        if args.dry_run:
            print(f"DRY RUN WARNING: {message}")
            return
        raise FileNotFoundError(message)

    sentence_metadata = load_sentence_metadata(args.raw_root)
    if not sentence_metadata:
        raise FileNotFoundError(f"No sentence_metadata.csv files found under {args.raw_root}")

    matched_sentences = {speaker_id: set() for speaker_id in sentence_metadata}
    written_count = 0

    for wav_path in sorted(audio_dir.glob("*.wav")):
        parsed = parse_wav_filename(wav_path)
        if parsed is None:
            print(f"WARNING: skipping non-standard filename: {wav_path.name}")
            continue
        if args.speaker_id and parsed["speaker_id"] != args.speaker_id:
            continue

        speaker_rows = sentence_metadata.get(parsed["speaker_id"])
        if speaker_rows is None:
            print(f"WARNING: no sentence_metadata.csv for speaker {parsed['speaker_id']}: {wav_path.name}")
            continue

        row = speaker_rows.get(parsed["sent_idx"])
        if row is None:
            print(f"WARNING: sent_idx {parsed['sent_idx']} missing for speaker {parsed['speaker_id']}: {wav_path.name}")
            continue

        matched_sentences[parsed["speaker_id"]].add(parsed["sent_idx"])
        emo_order = {item.strip() for item in row["emo_order"].split(";") if item.strip()}
        if parsed["emotion"] not in emo_order:
            print(f"WARNING: emotion {parsed['emotion']} not in emo_order for {wav_path.name}: {row['emo_order']}")

        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            continue

        if args.dry_run:
            print(f"DRY RUN: would write {txt_path.name}: {row['transcript']}")
        else:
            txt_path.write_text(row["transcript"], encoding="utf-8")
        written_count += 1

    for speaker_id, rows in sentence_metadata.items():
        if args.speaker_id and speaker_id != args.speaker_id:
            continue
        missing = sorted(set(rows) - matched_sentences.get(speaker_id, set()))
        if missing:
            print(
                f"WARNING: speaker {speaker_id} has {len(missing)} metadata sentence(s) without matching WAVs: "
                f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
            )

    print(f"Done. {'Would create' if args.dry_run else 'Created'} {written_count} transcript txt file(s).")


if __name__ == "__main__":
    main()
