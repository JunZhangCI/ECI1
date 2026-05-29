"""Setup helper: convert formal-study speaker docx files to CSV metadata.

Run this before the four formal-study stimulus steps. Each speaker folder has a
Word document where each paragraph contains one sentence plus the emotion order
used during recording. This script converts those documents into a simple CSV
that the rest of the pipeline can read.

Default input:
    emo_audio/1_raw/formal_study/**/speaker_*/*.docx

Default output, written beside each docx:
    sentence_metadata.csv

Output CSV columns:
    sent_idx     Sentence number inferred from paragraph order, e.g. 01.
    transcript   Sentence text for MFA transcript files.
    emo_order    Semicolon-separated emotion order, e.g. sad;ang;neu;hap;sca.

Common commands:
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\formal_docx_to_csv.py --dry-run
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\formal_docx_to_csv.py
    .\\.venv\\Scripts\\python.exe 3_code\\1_create_stimuli\\formal_docx_to_csv.py --overwrite
"""

import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


EMOTION_MAP = {
    "happy": "hap",
    "sad": "sad",
    "angry": "ang",
    "neutral": "neu",
    "scared": "sca",
}
EMOTION_PATTERN = re.compile(
    r"\b(Happy|Sad|Angry|Neutral|Scared)\b"
    r"\s*/\s*\b(Happy|Sad|Angry|Neutral|Scared)\b"
    r"\s*/\s*\b(Happy|Sad|Angry|Neutral|Scared)\b"
    r"\s*/\s*\b(Happy|Sad|Angry|Neutral|Scared)\b"
    r"\s*/\s*\b(Happy|Sad|Angry|Neutral|Scared)\b",
    flags=re.IGNORECASE,
)
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def docx_paragraphs(docx_path: Path) -> list[str]:
    """Read visible paragraph text from a docx file without requiring python-docx."""
    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.iter(f"{WORD_NAMESPACE}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{WORD_NAMESPACE}t" and node.text:
                parts.append(node.text)
            elif node.tag in {f"{WORD_NAMESPACE}tab", f"{WORD_NAMESPACE}br"}:
                parts.append(" ")
        text = " ".join("".join(parts).split())
        if text:
            paragraphs.append(text)
    return paragraphs


def parse_sentence_row(text: str, row_index: int, source_path: Path) -> dict[str, str] | None:
    match = EMOTION_PATTERN.search(text)
    if not match:
        print(f"WARNING: {source_path.name} row {row_index:02d} has no 5-emotion order: {text}")
        return None

    transcript = " ".join(text[: match.start()].strip().split())
    if not transcript:
        print(f"WARNING: {source_path.name} row {row_index:02d} has no transcript text.")
        return None

    emo_order = ";".join(EMOTION_MAP[item.lower()] for item in match.groups())
    return {
        "sent_idx": f"{row_index:02d}",
        "transcript": transcript,
        "emo_order": emo_order,
    }


def find_docx_files(raw_root: Path) -> list[Path]:
    docx_files = []
    for path in raw_root.rglob("*.docx"):
        if path.name.startswith("~$"):
            continue
        if path.parent.name.startswith("speaker_"):
            docx_files.append(path)
    return sorted(docx_files)


def convert_docx(docx_path: Path, dry_run: bool, overwrite: bool) -> int:
    rows = []
    for row_index, paragraph in enumerate(docx_paragraphs(docx_path), start=1):
        row = parse_sentence_row(paragraph, row_index, docx_path)
        if row is not None:
            rows.append(row)

    output_path = docx_path.parent / "sentence_metadata.csv"
    if dry_run:
        print(f"DRY RUN: would write {len(rows)} rows to {output_path}")
        if rows:
            print(f"  first row: {rows[0]}")
            print(f"  last row:  {rows[-1]}")
        return len(rows)

    if output_path.exists() and not overwrite:
        print(f"Skip existing file: {output_path}")
        return len(rows)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sent_idx", "transcript", "emo_order"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows: {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert formal-study speaker docx sentence lists to sentence_metadata.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Controllable options:
  --raw-root    Formal-study raw root to search for speaker docx files.
  --dry-run     Preview parsed rows and output paths without writing CSVs.
  --overwrite   Replace existing sentence_metadata.csv files.

Notes:
  The script expects five emotion words at the end of each sentence row.
  Emotion names are normalized to hap, sad, ang, neu, and sca.
""",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root() / "emo_audio" / "1_raw" / "formal_study",
        help="Formal-study raw audio root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned CSV outputs without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sentence_metadata.csv files.")
    args = parser.parse_args()

    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw formal-study folder not found: {args.raw_root}")

    docx_files = find_docx_files(args.raw_root)
    if not docx_files:
        print(f"No speaker docx files found under {args.raw_root}")
        return

    total_rows = 0
    for docx_path in docx_files:
        total_rows += convert_docx(docx_path, dry_run=args.dry_run, overwrite=args.overwrite)
    print(f"Done. Parsed {total_rows} sentence rows from {len(docx_files)} docx file(s).")


if __name__ == "__main__":
    main()
