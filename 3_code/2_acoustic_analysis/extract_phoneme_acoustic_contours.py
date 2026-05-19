from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

try:
    import parselmouth
except ImportError as exc:
    raise ImportError(
        "This script needs praat-parselmouth. Install it with: "
        "pip install praat-parselmouth"
    ) from exc


# =========================
# Settings
# =========================
# Empty set() means include all groups for that variable.
SELECTED_EMOTIONS = set()  # e.g., {"hap", "sad", "neu"} or set()
SELECTED_GENDERS = set()  # e.g., {"f"} or set()
SELECTED_SPEAKING_STYLES = set()  # e.g., {"CDS"} or set()

PHONE_TIER_NAME = "phones"

# Window around each phoneme onset. Time 0 is the phoneme onset.
WINDOW_START_SEC = -0.200
WINDOW_END_SEC = 0.200

# Extract raw acoustic contours every 10 ms, then sample phoneme windows every 40 ms.
RAW_TIME_STEP_SEC = 0.010
DOWNSAMPLED_TIME_STEP_SEC = 0.040

# Pitch settings for Praat/parselmouth. Adjust if needed for a different speaker set.
PITCH_FLOOR_HZ = 75
PITCH_CEILING_HZ = 600


# =========================
# Path setup
# =========================
SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parents[2]

TEXTGRID_DIR = PROJECT_ROOT / "emo_audio" / "3_mfa" / "pilot"
WAV_DIR = PROJECT_ROOT / "emo_audio" / "2_processed" / "pilot" / "cleaned"

FEATURE_ROOT = PROJECT_ROOT / "emo_audio" / "5_acoustic_features" / "pilot"
PITCH_DIR = FEATURE_ROOT / "pitch"
AMPLITUDE_DIR = FEATURE_ROOT / "amplitude"
OUTPUT_CSV = FEATURE_ROOT / "phoneme_acoustic_windows_wide.csv"


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
        intervals.append({
            "xmin": float(match.group(1)),
            "xmax": float(match.group(2)),
            "text": match.group(3).strip()
        })

    return intervals


def parse_stimulus_tags(textgrid_path):
    """
    Parse tags from filenames like CDS_01_hap_f_cleaned.TextGrid.
    """
    parts = textgrid_path.stem.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {textgrid_path.name}")

    return {
        "speaking_style": parts[0],
        "sentence_index": parts[1],
        "emotion": parts[2],
        "gender": parts[3],
    }


def keep_file(tags):
    """
    Return True if tags match the selected groups.
    Empty selection sets include all groups.
    """
    emotion_ok = not SELECTED_EMOTIONS or tags["emotion"] in SELECTED_EMOTIONS
    gender_ok = not SELECTED_GENDERS or tags["gender"] in SELECTED_GENDERS
    style_ok = (
        not SELECTED_SPEAKING_STYLES
        or tags["speaking_style"] in SELECTED_SPEAKING_STYLES
    )

    return emotion_ok and gender_ok and style_ok


def make_relative_times():
    """
    Build the downsampled relative-time grid, including both window edges.
    """
    n_steps = int(round((WINDOW_END_SEC - WINDOW_START_SEC) / DOWNSAMPLED_TIME_STEP_SEC))
    return WINDOW_START_SEC + np.arange(n_steps + 1) * DOWNSAMPLED_TIME_STEP_SEC


def save_feature_pickle(feature_dir, stem, feature_name, audio_filename, time_sec, values):
    """
    Save one feature contour for one wav file.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = feature_dir / f"{stem}.pkl"

    feature_data = {
        "audio_filename": audio_filename,
        "feature_name": feature_name,
        "time_sec": np.asarray(time_sec, dtype=float),
        "value": np.asarray(values, dtype=float),
        "settings": {
            "raw_time_step_sec": RAW_TIME_STEP_SEC,
            "pitch_floor_hz": PITCH_FLOOR_HZ,
            "pitch_ceiling_hz": PITCH_CEILING_HZ,
        },
    }

    with pickle_path.open("wb") as file:
        pickle.dump(feature_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    return pickle_path


def extract_and_cache_features(wav_path):
    """
    Extract pitch and amplitude contours, save them as separate pickle files,
    and return the in-memory contour dictionaries.
    """
    sound = parselmouth.Sound(str(wav_path))

    pitch = sound.to_pitch(
        time_step=RAW_TIME_STEP_SEC,
        pitch_floor=PITCH_FLOOR_HZ,
        pitch_ceiling=PITCH_CEILING_HZ,
    )
    pitch_time_sec = pitch.xs()
    pitch_hz = pitch.selected_array["frequency"].astype(float)
    pitch_hz[pitch_hz == 0] = np.nan

    intensity = sound.to_intensity(time_step=RAW_TIME_STEP_SEC)
    amplitude_time_sec = intensity.xs()
    amplitude_db = intensity.values[0].astype(float)

    stem = wav_path.stem
    pitch_path = save_feature_pickle(
        PITCH_DIR,
        stem,
        "pitch_hz",
        wav_path.name,
        pitch_time_sec,
        pitch_hz,
    )
    amplitude_path = save_feature_pickle(
        AMPLITUDE_DIR,
        stem,
        "amplitude_db",
        wav_path.name,
        amplitude_time_sec,
        amplitude_db,
    )

    return {
        "pitch": {
            "time_sec": np.asarray(pitch_time_sec, dtype=float),
            "value": np.asarray(pitch_hz, dtype=float),
            "pickle_path": pitch_path,
        },
        "amplitude": {
            "time_sec": np.asarray(amplitude_time_sec, dtype=float),
            "value": np.asarray(amplitude_db, dtype=float),
            "pickle_path": amplitude_path,
        },
    }


def sample_nearest(contour_time_sec, contour_values, target_time_sec):
    """
    Sample a contour at target times using the nearest extracted frame.
    Out-of-range targets and unvoiced pitch frames stay as NaN.
    """
    sampled = np.full(len(target_time_sec), np.nan)

    if len(contour_time_sec) == 0:
        return sampled

    for idx, target_time in enumerate(target_time_sec):
        nearest_idx = int(np.argmin(np.abs(contour_time_sec - target_time)))
        nearest_distance = abs(contour_time_sec[nearest_idx] - target_time)

        if nearest_distance <= RAW_TIME_STEP_SEC / 2:
            sampled[idx] = contour_values[nearest_idx]

    return sampled


def time_column(feature_name, relative_time_ms):
    """
    Make a clear contour column name for one relative time point.
    Example: pitch_-200ms, amplitude_40ms
    """
    return f"{feature_name}_{relative_time_ms}ms"


def make_phoneme_rows(textgrid_path, wav_path, tags, contours, phoneme_instance_start):
    """
    Build one wide-format row per non-empty phoneme in one TextGrid.
    """
    rows = []
    intervals = read_interval_tier(textgrid_path, tier_name=PHONE_TIER_NAME)
    relative_times_sec = make_relative_times()
    relative_times_ms = np.round(relative_times_sec * 1000).astype(int)
    phoneme_instance_id = phoneme_instance_start

    for interval in intervals:
        phoneme_raw = interval["text"]

        if phoneme_raw == "":
            continue

        phoneme = remove_stress(phoneme_raw)
        manner = get_manner(phoneme)
        onset_sec = interval["xmin"]
        offset_sec = interval["xmax"]
        duration_sec = offset_sec - onset_sec
        target_times_sec = onset_sec + relative_times_sec

        pitch_values = sample_nearest(
            contours["pitch"]["time_sec"],
            contours["pitch"]["value"],
            target_times_sec,
        )
        amplitude_values = sample_nearest(
            contours["amplitude"]["time_sec"],
            contours["amplitude"]["value"],
            target_times_sec,
        )

        row = {
            "phoneme_instance_id": phoneme_instance_id,
            "phoneme_label_raw": phoneme_raw,
            "phoneme_label": phoneme,
            "audio_filename": wav_path.name,
            "textgrid_filename": textgrid_path.name,
            "manner": manner,
            "emotion": tags["emotion"],
            "gender": tags["gender"],
            "speaking_style": tags["speaking_style"],
            "sentence_index": tags["sentence_index"],
            "phoneme_onset_sec": onset_sec,
            "phoneme_offset_sec": offset_sec,
            "duration_sec": duration_sec,
        }

        for rel_time_ms, pitch_hz in zip(relative_times_ms, pitch_values):
            row[time_column("pitch", rel_time_ms)] = pitch_hz

        for rel_time_ms, amplitude_db in zip(relative_times_ms, amplitude_values):
            row[time_column("amplitude", rel_time_ms)] = amplitude_db

        rows.append(row)

        phoneme_instance_id += 1

    return rows, phoneme_instance_id


def main():
    """
    Extract acoustic contours and save phoneme-aligned wide-format CSV.
    """
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    PITCH_DIR.mkdir(parents=True, exist_ok=True)
    AMPLITUDE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    phoneme_instance_id = 1
    n_textgrids_included = 0
    n_missing_wavs = 0

    for textgrid_path in sorted(TEXTGRID_DIR.glob("*.TextGrid")):
        try:
            tags = parse_stimulus_tags(textgrid_path)
        except ValueError as exc:
            print(f"Skipping {textgrid_path.name}: {exc}")
            continue

        if not keep_file(tags):
            continue

        wav_path = WAV_DIR / f"{textgrid_path.stem}.wav"

        if not wav_path.exists():
            n_missing_wavs += 1
            print(f"Warning: wav file not found for {textgrid_path.name}: {wav_path}")
            continue

        contours = extract_and_cache_features(wav_path)
        rows, phoneme_instance_id = make_phoneme_rows(
            textgrid_path=textgrid_path,
            wav_path=wav_path,
            tags=tags,
            contours=contours,
            phoneme_instance_start=phoneme_instance_id,
        )

        all_rows.extend(rows)
        n_textgrids_included += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Included TextGrid files: {n_textgrids_included}")
    print(f"Missing wav files skipped: {n_missing_wavs}")
    print(f"Saved pitch pickles to: {PITCH_DIR}")
    print(f"Saved amplitude pickles to: {AMPLITUDE_DIR}")
    print(f"Saved wide-format CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
