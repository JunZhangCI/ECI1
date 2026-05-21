r"""Create shared stimulus predictor pickles for Eelbrain TRF boosting.

The script reads ``order*.xlsx`` from ``1_studysetup/{group}/stimuli/orders``
to discover the stimulus row order. It then saves one group-level predictor
Dataset per acoustic feature under:

    2_data/3_trf/{group}/predictors

Each output Dataset has one row per stimulus trial and can be reused across
subjects in the same group when fitting ``eelbrain.boosting()`` models.

Usage from the project root:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --group pilot --features pitch --overwrite

Show the command-line help:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --help

Choose a group with ``--group``:

    pilot
    NH
    CI

Choose one or more features with ``--features``:

    all
    gammatone
    gammatone_n
    acoustic_onset_spectrogram
    acoustic_onset_spectrogram_n
    mfcc
    pitch

Examples:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --group pilot --features mfcc gammatone_n --overwrite
    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --group NH --features pitch mfcc --overwrite
    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --group CI --features all --overwrite

Use ``--n-bands`` to change the band count for ``gammatone_n`` and
``acoustic_onset_spectrogram_n``. The default is 8:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_stim_predictor_creator.py --group pilot --features gammatone_n --n-bands 16 --overwrite

Command-line options:

    --group
        Which participant/stimulus group to process. This controls both the
        study order folder and output predictor folder.

    --features
        One or more acoustic features to extract. Use ``all`` to generate every
        supported feature. List multiple features separated by spaces.

    --target-fs
        Predictor sampling rate in Hz. The default is 128, matching the current
        processed EEG sampling rate.

    --n-bands
        Number of frequency bands for ``gammatone_n`` and
        ``acoustic_onset_spectrogram_n``. The default is 8.

    --n-mfcc
        Number of MFCC coefficients to extract for the ``mfcc`` feature. The
        default is 13.

    --fill-unvoiced-pitch
        Value used for unvoiced or out-of-range pitch samples. The default is
        0.0, which avoids NaNs in the TRF predictor.

    --overwrite
        Replace existing predictor pickle files. Without this flag, existing
        predictor files are skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import eelbrain
import librosa
import numpy as np
from eelbrain import Dataset, Factor, NDVar, Scalar, UTS, Var

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    parselmouth = None
    call = None


DEFAULT_GROUP = "pilot"
DEFAULT_TARGET_FS = 128.0
DEFAULT_N_BANDS = 8
DEFAULT_N_MFCC = 13

VALID_GROUPS = ("pilot", "NH", "CI")
FEATURE_ALIASES = {
    "gammatone_n": "gammatone_n",
    "acoustic_onset_spectrogram_n": "acoustic_onset_spectrogram_n",
}
FEATURES_ALL = (
    "gammatone",
    "gammatone_n",
    "acoustic_onset_spectrogram",
    "acoustic_onset_spectrogram_n",
    "mfcc",
    "pitch",
)

# Match the pitch settings used in 3_code/2_acoustic_analysis.
PITCH_TIME_STEP_SEC = 0.001
PITCH_FLOOR_HZ = 75
PITCH_TOP_HZ = 800
PITCH_MAX_CANDIDATES = 15
PITCH_VERY_ACCURATE = False
PITCH_ATTENUATION_AT_TOP = 0.03
PITCH_SILENCE_THRESHOLD = 0.09
PITCH_VOICING_THRESHOLD = 0.50
PITCH_OCTAVE_COST = 0.055
PITCH_OCTAVE_JUMP_COST = 0.35
PITCH_VOICED_UNVOICED_COST = 0.14


@dataclass
class StimulusBlock:
    """One stimulus row from the study setup order file."""

    trial_index: int
    stim_name: str
    emotion: str
    speech_style: str
    gender: str
    wav_path: Path
    trigger: float | None = None


@dataclass
class FeatureContext:
    """Shared settings and per-run cache for feature extraction."""

    target_fs: float
    n_bands: int
    n_mfcc: int
    fill_unvoiced_pitch: float
    wav_cache: dict[Path, NDVar] = field(default_factory=dict)
    gt_cache: dict[Path, NDVar] = field(default_factory=dict)
    gt_log_cache: dict[Path, NDVar] = field(default_factory=dict)
    gt_onset_cache: dict[Path, NDVar] = field(default_factory=dict)


def project_root() -> Path:
    """Return the ECI1 project root based on this script location."""
    return Path(__file__).resolve().parents[2]


def xlsx_column_index(cell_ref: str) -> int:
    """Return zero-based column index from an XLSX cell reference such as A1."""
    match = re.match(r"[A-Z]+", cell_ref)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")

    index = 0
    for letter in match.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet from a simple .xlsx file with stdlib XML tools."""
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", namespace):
                text_parts = [node.text or "" for node in item.findall(".//a:t", namespace)]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet_root.findall(".//a:sheetData/a:row", namespace):
        values_by_col: dict[int, str] = {}
        for cell in row.findall("a:c", namespace):
            ref = cell.attrib.get("r", "")
            value_node = cell.find("a:v", namespace)
            if value_node is None or value_node.text is None:
                value = ""
            elif cell.attrib.get("t") == "s":
                value = shared_strings[int(value_node.text)]
            else:
                value = value_node.text
            values_by_col[xlsx_column_index(ref)] = value

        if values_by_col:
            rows.append([values_by_col.get(index, "") for index in range(max(values_by_col) + 1)])

    if not rows:
        return []

    headers = [str(header).strip() for header in rows[0]]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        for row in rows[1:]
    ]


def find_order_workbook(group: str, root: Path) -> Path:
    """Pick the first order*.xlsx workbook from the study setup order folder."""
    order_dir = root / "1_studysetup" / group / "stimuli" / "orders"
    if not order_dir.exists():
        raise FileNotFoundError(f"Missing stimulus order directory: {order_dir}")

    order_files = sorted(
        path for path in order_dir.glob("order*.xlsx") if not path.name.startswith("~$")
    )
    if not order_files:
        raise FileNotFoundError(f"No order*.xlsx files found in {order_dir}")
    return order_files[0]


def parse_stimulus_metadata(stim_name: str) -> tuple[str, str, str]:
    """Parse speech style, emotion, and gender from names like ADS_hap_f_cont_1.wav."""
    stem = Path(stim_name).stem
    stem = re.sub(r"_scaled$", "", stem)
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Could not parse stimulus metadata from filename: {stim_name}")

    speech_style = parts[0]
    emotion = parts[1]
    gender = parts[2]
    return speech_style, emotion, gender


def stimulus_audio_dir(group: str, root: Path) -> Path:
    """Return the continuous-stimulus wav folder for a group."""
    candidates = [
        root / "emo_audio" / "4_rand_concate" / group,
        root / "emo_audio" / "4_random_concate" / group,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Missing stimulus audio directory. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_wav_path(stim_name: str, stim_dir: Path) -> Path:
    """Find the stimulus wav for an ordered stimulus name."""
    direct = stim_dir / stim_name
    if direct.exists():
        return direct

    if stim_name.endswith("_scaled.wav"):
        unscaled = stim_dir / stim_name.replace("_scaled.wav", ".wav")
        if unscaled.exists():
            return unscaled

    raise FileNotFoundError(
        f"Could not find wav for stimulus '{stim_name}' in {stim_dir}. "
        "Tried the exact name and the _scaled.wav -> .wav fallback."
    )


def build_stimulus_blocks(order_path: Path, group: str, root: Path) -> list[StimulusBlock]:
    """Create stimulus block records from the study setup order workbook."""
    order_rows = read_xlsx_first_sheet(order_path)
    required_columns = ("audio_path", "stim_idx")
    missing = [column for column in required_columns if order_rows and column not in order_rows[0]]
    if missing:
        raise KeyError(f"Order workbook is missing columns: {missing}")

    stim_dir = stimulus_audio_dir(group, root)
    included_rows = []
    for row in order_rows:
        stim_idx_raw = str(row.get("stim_idx", "")).strip()
        audio_path_raw = str(row.get("audio_path", "")).strip()
        if not stim_idx_raw or not audio_path_raw:
            continue

        stim_idx = int(float(stim_idx_raw))
        if stim_idx <= 0:
            continue
        included_rows.append((stim_idx, row))

    included_rows.sort(key=lambda item: item[0])

    blocks: list[StimulusBlock] = []
    for stim_idx, row in included_rows:
        stim_name = Path(str(row["audio_path"])).name
        speech_style, emotion, gender = parse_stimulus_metadata(stim_name)
        trigger_raw = str(row.get("trig_value", "")).strip()
        trigger = float(trigger_raw) if trigger_raw else None
        blocks.append(
            StimulusBlock(
                trial_index=stim_idx,
                stim_name=stim_name,
                emotion=emotion,
                speech_style=speech_style,
                gender=gender,
                wav_path=resolve_wav_path(stim_name, stim_dir),
                trigger=trigger,
            )
        )

    if not blocks:
        raise ValueError(f"No usable stimulus rows found in order workbook: {order_path}")

    return blocks


def load_wav(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Load a stimulus wav as a mono Eelbrain NDVar."""
    if block.wav_path in context.wav_cache:
        return context.wav_cache[block.wav_path]

    wav = eelbrain.load.wav(block.wav_path, name=Path(block.stim_name).stem)
    if len(wav.dims) == 2:
        wav = wav.mean("channel")

    context.wav_cache[block.wav_path] = wav
    return wav


def get_gammatone_bank(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Compute or return the cached high-resolution gammatone bank."""
    if block.wav_path not in context.gt_cache:
        wav = load_wav(block, context)
        context.gt_cache[block.wav_path] = eelbrain.gammatone_bank(
            wav,
            80,
            10000,
            128,
            location="left",
            tstep=0.001,
            name="gammatone",
        )
    return context.gt_cache[block.wav_path]


def get_log_gammatone(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Compute or return the cached log-scaled gammatone bank."""
    if block.wav_path not in context.gt_log_cache:
        gt = get_gammatone_bank(block, context)
        context.gt_log_cache[block.wav_path] = (gt + 1).log(name="gammatone_log")
    return context.gt_log_cache[block.wav_path]


def get_acoustic_onset_spectrogram(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Compute or return the cached edge-detected gammatone onset spectrogram."""
    if block.wav_path not in context.gt_onset_cache:
        gt_log = get_log_gammatone(block, context)
        context.gt_onset_cache[block.wav_path] = eelbrain.edge_detector(
            gt_log,
            c=30,
            name="acoustic_onset_spectrogram",
        )
    return context.gt_onset_cache[block.wav_path]


def extract_gammatone_bank(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Extract the full gammatone spectrogram bank."""
    gt = get_gammatone_bank(block, context)
    return eelbrain.resample(gt, context.target_fs, name="gammatone")


def extract_n_band_gammatone(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Extract an n-band log gammatone spectrogram."""
    gt_log = get_log_gammatone(block, context)
    gt_n = gt_log.bin(nbins=context.n_bands, func="sum", dim="frequency")
    return eelbrain.resample(gt_n, context.target_fs, name=f"gammatone_{context.n_bands}")


def extract_acoustic_onset_spectrogram(
    block: StimulusBlock,
    context: FeatureContext,
) -> NDVar:
    """Extract a full acoustic-onset spectrogram."""
    onset = get_acoustic_onset_spectrogram(block, context)
    return eelbrain.resample(onset, context.target_fs, name="acoustic_onset_spectrogram")


def extract_n_band_acoustic_onset_spectrogram(
    block: StimulusBlock,
    context: FeatureContext,
) -> NDVar:
    """Extract an n-band acoustic-onset spectrogram."""
    onset = get_acoustic_onset_spectrogram(block, context)
    onset_n = onset.bin(nbins=context.n_bands, func="sum", dim="frequency")
    return eelbrain.resample(
        onset_n,
        context.target_fs,
        name=f"acoustic_onset_spectrogram_{context.n_bands}",
    )


def extract_mfcc(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Extract MFCC predictor contours."""
    wav = load_wav(block, context)
    sample_rate = 1.0 / wav.time.tstep
    samples = np.asarray(wav.x, dtype=np.float64).squeeze()
    hop_length = max(int(round(sample_rate / context.target_fs)), 1)

    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=int(round(sample_rate)),
        n_mfcc=context.n_mfcc,
        hop_length=hop_length,
    )
    mfcc_fs = sample_rate / hop_length
    if not np.isclose(mfcc_fs, context.target_fs):
        mfcc = librosa.resample(
            mfcc,
            orig_sr=mfcc_fs,
            target_sr=context.target_fs,
            axis=1,
        )

    mfcc_dim = Scalar("mfcc", np.arange(1, context.n_mfcc + 1))
    time_dim = UTS(0.0, 1.0 / context.target_fs, mfcc.shape[1])
    return NDVar(mfcc, dims=(mfcc_dim, time_dim), name="mfcc")


def extract_filtered_autocorrelation_pitch(sound):
    """Extract pitch with Praat's filtered-autocorrelation method."""
    if parselmouth is None or call is None:
        raise ImportError(
            "Pitch extraction needs praat-parselmouth. Install it with: "
            "pip install praat-parselmouth"
        )

    try:
        return call(
            sound,
            "To Pitch (filtered autocorrelation)",
            PITCH_TIME_STEP_SEC,
            PITCH_FLOOR_HZ,
            PITCH_TOP_HZ,
            PITCH_MAX_CANDIDATES,
            PITCH_VERY_ACCURATE,
            PITCH_ATTENUATION_AT_TOP,
            PITCH_SILENCE_THRESHOLD,
            PITCH_VOICING_THRESHOLD,
            PITCH_OCTAVE_COST,
            PITCH_OCTAVE_JUMP_COST,
            PITCH_VOICED_UNVOICED_COST,
        )
    except parselmouth.PraatError as exc:
        raise RuntimeError(
            "Praat filtered autocorrelation is not available in this "
            "praat-parselmouth environment. No fallback was used, so pitch "
            "outputs will not be mislabeled."
        ) from exc


def extract_pitch_contour(block: StimulusBlock, context: FeatureContext) -> NDVar:
    """Extract a continuous pitch contour resampled to the TRF sampling rate."""
    if parselmouth is None:
        raise ImportError(
            "Pitch extraction needs praat-parselmouth. Install it with: "
            "pip install praat-parselmouth"
        )

    wav = load_wav(block, context)
    sound = parselmouth.Sound(str(block.wav_path))
    pitch = extract_filtered_autocorrelation_pitch(sound)

    pitch_time_sec = np.asarray(pitch.xs(), dtype=float)
    pitch_hz = pitch.selected_array["frequency"].astype(float)
    pitch_hz[pitch_hz == 0] = np.nan

    target_n = len(wav.bin(1.0 / context.target_fs, dim="time", label="start").time)
    target_time_sec = np.arange(target_n, dtype=float) / context.target_fs
    valid = np.isfinite(pitch_hz)

    pitch_resampled = np.full(target_n, np.nan, dtype=float)
    if np.any(valid):
        pitch_resampled = np.interp(
            target_time_sec,
            pitch_time_sec[valid],
            pitch_hz[valid],
            left=np.nan,
            right=np.nan,
        )

    if context.fill_unvoiced_pitch is not None:
        pitch_resampled = np.nan_to_num(
            pitch_resampled,
            nan=context.fill_unvoiced_pitch,
        )

    time_dim = UTS(0.0, 1.0 / context.target_fs, target_n)
    return NDVar(pitch_resampled, dims=time_dim, name="pitch")


def feature_output_name(feature: str, context: FeatureContext) -> str:
    """Return the output feature name and predictor column name."""
    if feature == "gammatone_n":
        return f"gammatone_{context.n_bands}"
    if feature == "acoustic_onset_spectrogram_n":
        return f"acoustic_onset_spectrogram_{context.n_bands}"
    return feature


def feature_extractor(feature: str) -> Callable[[StimulusBlock, FeatureContext], NDVar]:
    """Map a normalized feature option to its extraction function."""
    extractors = {
        "gammatone": extract_gammatone_bank,
        "gammatone_n": extract_n_band_gammatone,
        "acoustic_onset_spectrogram": extract_acoustic_onset_spectrogram,
        "acoustic_onset_spectrogram_n": extract_n_band_acoustic_onset_spectrogram,
        "mfcc": extract_mfcc,
        "pitch": extract_pitch_contour,
    }
    return extractors[feature]


def make_base_dataset(blocks: list[StimulusBlock], name: str) -> Dataset:
    """Create the shared metadata columns for one predictor Dataset."""
    ds = Dataset(name=name)
    ds["trial_index"] = Var([block.trial_index for block in blocks], name="trial_index")
    ds["stim_name"] = Factor([block.stim_name for block in blocks], name="stim_name")
    ds["emotion"] = Factor([block.emotion for block in blocks], name="emotion")
    ds["speech_style"] = Factor(
        [block.speech_style for block in blocks],
        name="speech_style",
    )
    ds["gender"] = Factor([block.gender for block in blocks], name="gender")
    if any(block.trigger is not None for block in blocks):
        ds["trigger"] = Var(
            [np.nan if block.trigger is None else block.trigger for block in blocks],
            name="trigger",
        )
    return ds


def make_predictor_dataset(
    blocks: list[StimulusBlock],
    feature: str,
    context: FeatureContext,
) -> Dataset:
    """Extract one predictor for all stimulus rows."""
    output_name = feature_output_name(feature, context)
    extractor = feature_extractor(feature)
    predictors: list[NDVar] = []
    durations: list[float] = []

    for row_index, block in enumerate(blocks, start=1):
        print(f"  [{row_index:03d}/{len(blocks):03d}] {output_name}: {block.stim_name}")
        predictor = extractor(block, context).copy(name=output_name)
        predictors.append(predictor)
        durations.append(float(predictor.time.tstop))

    ds = make_base_dataset(blocks, name=output_name)
    ds["duration"] = Var(durations, name="duration")
    ds[output_name] = predictors
    return ds


def predictor_info(
    feature: str,
    context: FeatureContext,
    order_path: Path,
    root: Path,
) -> dict:
    """Build metadata saved with each predictor Dataset."""
    info = {
        "feature": feature_output_name(feature, context),
        "feature_option": feature,
        "target_fs": context.target_fs,
        "n_bands": context.n_bands,
        "n_mfcc": context.n_mfcc,
        "source_order_workbook": str(order_path),
        "project_root": str(root),
        "eelbrain_version": eelbrain.__version__,
        "librosa_version": librosa.__version__,
    }

    if parselmouth is not None:
        info["parselmouth_version"] = parselmouth.__version__
        info["praat_version"] = parselmouth.PRAAT_VERSION

    return info


def save_feature_dataset(
    blocks: list[StimulusBlock],
    feature: str,
    context: FeatureContext,
    out_dir: Path,
    overwrite: bool,
    order_path: Path,
    root: Path,
) -> Path | None:
    """Create and save one feature Dataset."""
    output_name = feature_output_name(feature, context)
    out_path = out_dir / f"{output_name}.pickle"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing predictor: {out_path}")
        return None

    print(f"Extracting predictor: {output_name}")
    ds = make_predictor_dataset(blocks, feature, context)
    ds.info.update(
        predictor_info(
            feature=feature,
            context=context,
            order_path=order_path,
            root=root,
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    eelbrain.save.pickle(ds, out_path)
    print(f"Saved {out_path}")
    return out_path


def normalize_features(features: list[str]) -> list[str]:
    """Expand and validate CLI feature options."""
    if not features or "all" in features:
        return list(FEATURES_ALL)

    normalized = []
    valid = set(FEATURES_ALL)
    for feature in features:
        feature = FEATURE_ALIASES.get(feature, feature)
        if feature not in valid:
            raise ValueError(
                f"Unknown feature '{feature}'. Valid values are: all, "
                + ", ".join(FEATURES_ALL)
            )
        if feature not in normalized:
            normalized.append(feature)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared group-level stimulus predictor pickles for Eelbrain."
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        choices=VALID_GROUPS,
        help="Group under 1_studysetup and emo_audio/4_rand_concate.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        help=(
            "Feature(s) to create: all, gammatone, gammatone_n, "
            "acoustic_onset_spectrogram, acoustic_onset_spectrogram_n, mfcc, pitch."
        ),
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=DEFAULT_TARGET_FS,
        help="Target predictor sampling rate in Hz.",
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=DEFAULT_N_BANDS,
        help="Number of frequency bands for n-band spectrogram predictors.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=DEFAULT_N_MFCC,
        help="Number of MFCC coefficients.",
    )
    parser.add_argument(
        "--fill-unvoiced-pitch",
        type=float,
        default=0.0,
        help="Value used for unvoiced/out-of-range pitch samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictor pickle files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    features = normalize_features(args.features)

    order_path = find_order_workbook(args.group, root)
    print(f"Using stimulus order workbook: {order_path}")
    blocks = build_stimulus_blocks(order_path, args.group, root)
    out_dir = root / "2_data" / "3_trf" / args.group / "predictors"

    context = FeatureContext(
        target_fs=args.target_fs,
        n_bands=args.n_bands,
        n_mfcc=args.n_mfcc,
        fill_unvoiced_pitch=args.fill_unvoiced_pitch,
    )

    saved_paths = []
    for feature in features:
        saved_path = save_feature_dataset(
            blocks=blocks,
            feature=feature,
            context=context,
            out_dir=out_dir,
            overwrite=args.overwrite,
            order_path=order_path,
            root=root,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

    print(f"Finished. Saved {len(saved_paths)} predictor file(s).")


if __name__ == "__main__":
    main()
