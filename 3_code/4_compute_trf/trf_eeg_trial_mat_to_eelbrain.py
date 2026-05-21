"""Convert trial-level EEG .mat files to Eelbrain TRF-ready pickles.

Input files are expected under:
    2_data/2_processed/{group}/{subject}/ref_down_filt_chRej/ica/by_trial

Each input .mat file should contain a top-level ``trial_EEG`` structure from
``3_code/3_preprocessing/step4_restruct_data.m``. The script saves one
session-specific Eelbrain Dataset pickle per input file under:
    2_data/3_trf/{group}/eegs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import mat73
import numpy as np
from eelbrain import Dataset, Factor, NDVar, Sensor, UTS, Var
import eelbrain


DEFAULT_GROUP = "pilot"
DEFAULT_SUBJECTS: list[str] = []
DEFAULT_SESSIONS = "both"

VALID_GROUPS = ("pilot", "NH", "CI")
VALID_SESSIONS = ("ses-1", "ses-2", "both")


def project_root() -> Path:
    """Return the ECI1 project root based on this script location."""
    return Path(__file__).resolve().parents[2]


def unwrap_singleton(value):
    """Convert MATLAB one-item cells/arrays to their contained Python value."""
    while True:
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            return value


def as_label(value) -> str:
    """Normalize MATLAB cell/list labels like ['hap'] to plain strings."""
    value = unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def as_float(value) -> float:
    """Normalize MATLAB numeric cells/arrays to a Python float."""
    return float(unwrap_singleton(value))


def selected_sessions(session_arg: str) -> list[str]:
    """Expand the CLI session selection to concrete filename tokens."""
    if session_arg == "both":
        return ["ses-1", "ses-2"]
    return [session_arg]


def detect_subjects(group_dir: Path) -> list[str]:
    """Find subject folders for a group when the user did not list subjects."""
    if not group_dir.exists():
        raise FileNotFoundError(f"Missing group directory: {group_dir}")

    subjects = sorted(path.name for path in group_dir.glob("sub*") if path.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No subject folders found in {group_dir}")
    return subjects


def find_trial_mat_files(subject_dir: Path, sessions: Iterable[str]) -> list[Path]:
    """Find by-trial .mat files for one subject and one or more sessions."""
    by_trial_dir = subject_dir / "ref_down_filt_chRej" / "ica" / "by_trial"
    if not by_trial_dir.exists():
        print(f"Skipping missing by-trial folder: {by_trial_dir}")
        return []

    mat_files: list[Path] = []
    for session in sessions:
        mat_files.extend(sorted(by_trial_dir.glob(f"*_{session}_*_by_trial.mat")))

    return sorted(set(mat_files))


def parse_file_metadata(mat_path: Path) -> tuple[str, str]:
    """Extract session and acquisition labels from a by-trial filename."""
    session_match = re.search(r"ses-\d+", mat_path.name)
    acq_match = re.search(r"acq-\d+", mat_path.name)

    session = session_match.group(0) if session_match else "unknown"
    acq = acq_match.group(0) if acq_match else "unknown"
    return session, acq


def load_trial_eeg(mat_path: Path) -> dict:
    """Load the top-level trial_EEG variable from a MATLAB v7.3 file."""
    mat = mat73.loadmat(str(mat_path))
    if "trial_EEG" not in mat:
        raise KeyError(
            f"'trial_EEG' not found in {mat_path}. Top-level keys: {list(mat.keys())}"
        )

    trial_eeg = mat["trial_EEG"]
    if not isinstance(trial_eeg, dict):
        raise TypeError(f"'trial_EEG' in {mat_path} is not a MATLAB struct/dict")
    return trial_eeg


def make_sensor(chanlocs: list[dict]) -> Sensor:
    """Create an Eelbrain Sensor from EEGLAB chanlocs X/Y/Z and labels."""
    labels: list[str] = []
    xyz_mm: list[list[float]] = []

    for idx, chan in enumerate(chanlocs):
        if not isinstance(chan, dict):
            raise TypeError(f"chanlocs[{idx}] is not a dict-like channel location")

        labels.append(as_label(chan.get("labels", f"ch{idx + 1:02d}")))
        xyz_mm.append(
            [
                as_float(chan["X"]),
                as_float(chan["Y"]),
                as_float(chan["Z"]),
            ]
        )

    xyz_m = np.asarray(xyz_mm, dtype=float) / 1000.0
    return Sensor(xyz_m, names=labels)


def normalize_trial_data(trial_data, n_channels: int) -> np.ndarray:
    """Return EEG data as time x channel for one trial."""
    data = np.asarray(trial_data, dtype=float).squeeze()

    if data.ndim != 2:
        raise ValueError(
            f"Expected trial EEG data to be 2D after squeezing, got shape {data.shape}"
        )

    if data.shape[0] == n_channels:
        return data.T
    if data.shape[1] == n_channels:
        return data

    raise ValueError(
        "Could not identify channel axis in trial data. "
        f"Shape={data.shape}, expected one axis to match {n_channels} channels."
    )


def get_trial_field(trial_eeg: dict, key: str, index: int, default=None):
    """Safely read one value from a trial_EEG per-trial field."""
    values = trial_eeg.get(key)
    if values is None:
        return default
    return values[index]


def build_dataset(mat_path: Path) -> Dataset:
    """Convert one trial_EEG .mat file into an Eelbrain Dataset."""
    trial_eeg = load_trial_eeg(mat_path)

    required_keys = ["data", "chanlocs", "srate"]
    missing = [key for key in required_keys if key not in trial_eeg]
    if missing:
        raise KeyError(f"{mat_path} is missing required trial_EEG keys: {missing}")

    trial_data = trial_eeg["data"]
    n_trials = len(trial_data)
    srate = as_float(trial_eeg["srate"])
    subject = as_label(trial_eeg.get("subject", mat_path.name))
    session, acq = parse_file_metadata(mat_path)

    sensor = make_sensor(trial_eeg["chanlocs"])
    n_channels = len(sensor.names)

    eeg_trials: list[NDVar] = []
    stim_names: list[str] = []
    emotions: list[str] = []
    speech_styles: list[str] = []
    genders: list[str] = []
    triggers: list[float] = []

    for trial_index in range(n_trials):
        trial_time_by_channel = normalize_trial_data(
            trial_data[trial_index],
            n_channels=n_channels,
        )
        time_dim = UTS(0.0, 1.0 / srate, trial_time_by_channel.shape[0])

        eeg_trials.append(
            NDVar(
                trial_time_by_channel,
                dims=(time_dim, sensor),
                name="eeg",
                info={"unit": "uV"},
            )
        )
        stim_names.append(as_label(get_trial_field(trial_eeg, "audio_files", trial_index, "")))
        emotions.append(as_label(get_trial_field(trial_eeg, "emotions", trial_index, "")))
        speech_styles.append(
            as_label(get_trial_field(trial_eeg, "speech_styles", trial_index, ""))
        )
        genders.append(as_label(get_trial_field(trial_eeg, "genders", trial_index, "")))
        triggers.append(as_float(get_trial_field(trial_eeg, "triggers", trial_index, np.nan)))

    ds = Dataset(name=f"{mat_path.stem.replace('_by_trial', '')}_eeg")
    ds["subject"] = Factor([subject] * n_trials, name="subject")
    ds["session"] = Factor([session] * n_trials, name="session")
    ds["acq"] = Factor([acq] * n_trials, name="acq")
    ds["trial_index"] = Var(np.arange(1, n_trials + 1), name="trial_index")
    ds["stim_name"] = Factor(stim_names, name="stim_name")
    ds["emotion"] = Factor(emotions, name="emotion")
    ds["speech_style"] = Factor(speech_styles, name="speech_style")
    ds["gender"] = Factor(genders, name="gender")
    ds["trigger"] = Var(triggers, name="trigger")
    ds["eeg"] = eeg_trials

    ds.info["source_mat"] = str(mat_path)
    ds.info["source_file"] = as_label(trial_eeg.get("source_file", ""))
    ds.info["subject"] = subject
    ds.info["session"] = session
    ds.info["acq"] = acq
    ds.info["srate"] = srate

    return ds


def output_path_for(mat_path: Path, out_dir: Path) -> Path:
    """Return the session-specific pickle path for one input .mat file."""
    out_stem = mat_path.stem.replace("_by_trial", "_eeg")
    return out_dir / f"{out_stem}.pickle"


def convert_files(
    group: str,
    subjects: list[str],
    sessions: list[str],
    overwrite: bool,
    root: Path,
) -> list[Path]:
    """Convert all selected files and return saved pickle paths."""
    group_dir = root / "2_data" / "2_processed" / group
    out_dir = root / "2_data" / "3_trf" / group / "eegs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not subjects:
        subjects = detect_subjects(group_dir)

    saved_paths: list[Path] = []
    for subject in subjects:
        subject_dir = group_dir / subject
        mat_files = find_trial_mat_files(subject_dir, sessions)
        if not mat_files:
            print(f"No matching by-trial .mat files found for {subject}")
            continue

        for mat_path in mat_files:
            out_path = output_path_for(mat_path, out_dir)
            if out_path.exists() and not overwrite:
                print(f"Skipping existing output: {out_path}")
                continue

            print(f"Converting {mat_path}")
            ds = build_dataset(mat_path)
            eelbrain.save.pickle(ds, out_path)
            saved_paths.append(out_path)
            print(f"Saved {out_path}")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert trial-level EEG .mat files to Eelbrain Dataset pickles."
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        choices=VALID_GROUPS,
        help="Participant group folder under 2_data/2_processed.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=DEFAULT_SUBJECTS,
        help="Subject folders to process. If omitted, all sub* folders are detected.",
    )
    parser.add_argument(
        "--sessions",
        default=DEFAULT_SESSIONS,
        choices=VALID_SESSIONS,
        help="Session selector. Use both to process ses-1 and ses-2 separately.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output pickle files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    sessions = selected_sessions(args.sessions)

    saved_paths = convert_files(
        group=args.group,
        subjects=args.subjects,
        sessions=sessions,
        overwrite=args.overwrite,
        root=root,
    )

    print(f"Finished. Saved {len(saved_paths)} file(s).")


if __name__ == "__main__":
    main()
