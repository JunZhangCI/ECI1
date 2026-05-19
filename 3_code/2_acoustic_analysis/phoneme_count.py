from pathlib import Path
import re
import pandas as pd

# =========================
# Settings
# =========================
# --- Path setup ---
script_path = Path(__file__)
project_root = script_path.parents[2]
textgrid_dir = project_root / "emo_audio" / "3_mfa" / "pilot" 
selected_emotions =  {"neu", "hap", "sad"}   # e.g., {"ang", "hap", "sad"}
phone_tier_name = "phones"

output_csv = textgrid_dir / "phoneme_duration_summary.csv"


# =========================
# ARPABET phoneme classes
# =========================
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW"
}

STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
APPROXIMANTS = {"L", "R", "W", "Y"}


def remove_stress(phone):
    """
    Remove stress numbers from vowels.
    Example: AH1 -> AH, ER0 -> ER
    """
    return re.sub(r"\d+$", "", phone)


def get_manner(phone):
    """
    Assign phoneme manner/class.
    """
    if phone in VOWELS:
        return "Vowel"
    elif phone in STOPS:
        return "Stop"
    elif phone in FRICATIVES:
        return "Fricative"
    elif phone in AFFRICATES:
        return "Affricate"
    elif phone in NASALS:
        return "Nasal"
    elif phone in APPROXIMANTS:
        return "Approximant"
    else:
        return "Other"


def read_interval_tier(textgrid_path, tier_name="phones"):
    """
    Read intervals from a Praat TextGrid IntervalTier.

    Returns a list of dictionaries:
    [
        {"xmin": ..., "xmax": ..., "text": ...},
        ...
    ]
    """
    text = textgrid_path.read_text(encoding="utf-8", errors="ignore")

    # Find the requested tier block
    tier_pattern = re.compile(
        r'item \[\d+\]:\s*'
        r'class = "IntervalTier"\s*'
        r'name = "' + re.escape(tier_name) + r'"\s*'
        r'xmin = .*?\s*'
        r'xmax = .*?\s*'
        r'intervals: size = \d+\s*'
        r'(.*?)(?=\n\s*item \[\d+\]:|\Z)',
        re.DOTALL
    )

    tier_match = tier_pattern.search(text)

    if tier_match is None:
        raise ValueError(f"Tier '{tier_name}' not found in {textgrid_path.name}")

    tier_block = tier_match.group(1)

    interval_pattern = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin = ([\d.]+)\s*'
        r'xmax = ([\d.]+)\s*'
        r'text = "(.*?)"',
        re.DOTALL
    )

    intervals = []
    for match in interval_pattern.finditer(tier_block):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        label = match.group(3).strip()

        intervals.append({
            "xmin": xmin,
            "xmax": xmax,
            "text": label
        })

    return intervals


# =========================
# Collect phoneme durations
# =========================
rows = []

for tg_path in textgrid_dir.glob("*.TextGrid"):
    filename = tg_path.name
    parts = filename.split("_")

    # Skip files that do not match expected filename structure
    if len(parts) < 3:
        continue

    emotion = parts[2]

    # Only keep selected emotions
    if emotion not in selected_emotions:
        continue

    intervals = read_interval_tier(tg_path, tier_name=phone_tier_name)

    for interval in intervals:
        phone_raw = interval["text"]

        # Skip blank phoneme labels
        if phone_raw == "":
            continue

        # Remove stress number for vowels, e.g., AH1 -> AH
        phone = remove_stress(phone_raw)

        duration = interval["xmax"] - interval["xmin"]
        manner = get_manner(phone)

        rows.append({
            "file": filename,
            "emotion": emotion,
            "phoneme_raw": phone_raw,
            "phoneme": phone,
            "manner": manner,
            "duration": duration
        })


# =========================
# Create table
# =========================
df = pd.DataFrame(rows)

summary = (
    df
    .groupby(["manner", "phoneme"], as_index=False)
    .agg(
        n=("duration", "count"),
        mean_duration=("duration", "mean"),
        sd_duration=("duration", "std")
    )
    .sort_values(["manner", "phoneme"])
)

# Optional: replace NaN SD with 0 when only one token exists
summary["sd_duration"] = summary["sd_duration"].fillna(0)

# =========================
# Export
# =========================
summary.to_csv(output_csv, index=False)

print(f"Saved summary table to: {output_csv}")
print(summary)