r"""Estimate Eelbrain TRF models for selected talker-emotion conditions.

The script reads selected subject/session EEG Dataset pickles from:

    2_data/3_trf/{group}/eegs

and one or more predictor Dataset pickles from:

    2_data/3_trf/{group}/predictors

By default, the script attempts all combinations of gender, emotion, and
speech style. Use ``--genders``, ``--emotions``, and ``--speech-styles`` to
run a subset, or pass legacy condition codes with ``--conditions`` (for
example ``CDS_hap_f``). Use ``--models pitch mfcc pitch+mfcc`` to estimate
multiple model definitions in one run. Matching EEG and predictor trials are
cropped to the shortest trial length, concatenated along time, and used to
estimate TRF models with ``eelbrain.boosting()``. Results are saved under:

    2_data/3_trf/{group}/results

Usage from the project root:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --models pitch mfcc pitch+mfcc

Show the command-line help:

    .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --help
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import eelbrain
import numpy as np
from eelbrain import NDVar, UTS


VALID_GROUPS = ("pilot", "NH", "CI")
VALID_SESSIONS = ("ses-1", "ses-2")
VALID_GENDERS = ("f", "m")
VALID_EMOTIONS = ("neu", "sad", "hap")
VALID_SPEECH_STYLES = ("ADS", "CDS")
DEFAULT_TSTART = -0.1
DEFAULT_TSTOP = 0.5
DEFAULT_BASIS = 0.05
DEFAULT_PARTITIONS = 4
DEFAULT_ERROR = "l1"


@dataclass(frozen=True)
class Condition:
    """Speech-style, emotion, and gender parsed from a condition code."""

    code: str
    speech_style: str
    emotion: str
    gender: str


@dataclass(frozen=True)
class PredictorSpec:
    """One predictor file and the NDVar column to use from it."""

    name: str
    path: Path
    key: str


@dataclass(frozen=True)
class ModelSpec:
    """One TRF model definition."""

    name: str
    predictor_names: list[str]
    predictor_specs: list[PredictorSpec]


@dataclass(frozen=True)
class SubjectRun:
    """One subject/session/acquisition EEG file to process."""

    subject: str
    session: str
    acq: str
    eeg_path: Path


@dataclass(frozen=True)
class TrfSettings:
    """Settings passed through to eelbrain.boosting()."""

    tstart: float
    tstop: float
    basis: float
    partitions: int
    error: str


def project_root() -> Path:
    """Return the ECI1 project root based on this script location."""
    return Path(__file__).resolve().parents[2]


def parse_condition_code(code: str) -> Condition:
    """Parse condition codes like CDS_hap_f."""
    parts = code.split("_")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Invalid condition '{code}'. Expected format: speechStyle_emotion_gender, "
            "for example CDS_hap_f."
        )

    speech_style, emotion, gender = parts
    invalid_parts = []
    if speech_style not in VALID_SPEECH_STYLES:
        invalid_parts.append(f"speech style '{speech_style}'")
    if emotion not in VALID_EMOTIONS:
        invalid_parts.append(f"emotion '{emotion}'")
    if gender not in VALID_GENDERS:
        invalid_parts.append(f"gender '{gender}'")
    if invalid_parts:
        raise ValueError(
            f"Invalid condition '{code}': " + ", ".join(invalid_parts) + ". "
            f"Valid speech styles: {', '.join(VALID_SPEECH_STYLES)}; "
            f"emotions: {', '.join(VALID_EMOTIONS)}; "
            f"genders: {', '.join(VALID_GENDERS)}."
        )

    return Condition(
        code=code,
        speech_style=speech_style,
        emotion=emotion,
        gender=gender,
    )


def make_condition(speech_style: str, emotion: str, gender: str) -> Condition:
    """Create a Condition from separate CLI filter values."""
    return Condition(
        code=f"{speech_style}_{emotion}_{gender}",
        speech_style=speech_style,
        emotion=emotion,
        gender=gender,
    )


def build_conditions(args: argparse.Namespace) -> list[Condition]:
    """Build conditions from explicit codes or variable filters."""
    if args.conditions:
        return [parse_condition_code(code) for code in args.conditions]

    return [
        make_condition(speech_style, emotion, gender)
        for speech_style in args.speech_styles
        for emotion in args.emotions
        for gender in args.genders
    ]


def parse_acq_label(path: Path) -> str:
    """Extract an acquisition label from an EEG pickle filename."""
    match = re.search(r"acq-\d+", path.name)
    return match.group(0) if match else "unknown"


def parse_session_label(path: Path) -> str:
    """Extract a session label from an EEG pickle filename."""
    match = re.search(r"ses-\d+", path.name)
    return match.group(0) if match else "unknown"


def parse_subject_label(path: Path) -> str:
    """Extract the subject label from an EEG pickle filename."""
    match = re.match(r"(.+?)_ses-\d+_", path.name)
    if match is None:
        raise ValueError(f"Could not parse subject from EEG filename: {path.name}")
    return match.group(1)


def find_subject_runs(
    root: Path,
    group: str,
    subject: str | None,
    session: str | None,
    acq: str | None,
) -> list[SubjectRun]:
    """Find EEG files matching the selected subject, session, and acquisition."""
    eeg_dir = root / "2_data" / "3_trf" / group / "eegs"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"Missing EEG directory: {eeg_dir}")

    pattern = f"{subject or '*'}_{session or 'ses-*'}_*_eeg.pickle"
    matches = sorted(eeg_dir.glob(pattern))

    subject_runs = [
        SubjectRun(
            subject=parse_subject_label(path),
            session=parse_session_label(path),
            acq=parse_acq_label(path),
            eeg_path=path,
        )
        for path in matches
        if acq is None or parse_acq_label(path) == acq
    ]
    if not subject_runs:
        selectors = []
        if subject is not None:
            selectors.append(f"subject={subject}")
        if session is not None:
            selectors.append(f"session={session}")
        if acq is not None:
            selectors.append(f"acq={acq}")
        selector_text = ", ".join(selectors) if selectors else "all subjects/sessions/acqs"
        raise FileNotFoundError(f"No EEG pickles found for {selector_text} in {eeg_dir}")

    return sorted(subject_runs, key=lambda run: (run.subject, run.session, run.acq))


def load_dataset(path: Path):
    """Load an Eelbrain pickle with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input pickle: {path}")
    return eelbrain.load.unpickle(path)


def predictor_key_from_dataset(name: str, ds) -> str:
    """Find the predictor NDVar column in a predictor Dataset."""
    info_key = getattr(ds, "info", {}).get("feature")
    for candidate in (info_key, name, Path(name).stem):
        if candidate and candidate in ds:
            return str(candidate)

    available = ", ".join(str(key) for key in ds.keys())
    raise KeyError(
        f"Could not find predictor NDVar column for '{name}'. "
        f"Tried ds.info['feature'] and '{name}'. Available columns: {available}"
    )


def resolve_predictor_spec(root: Path, group: str, predictor: str) -> PredictorSpec:
    """Resolve one predictor name to its pickle path and signal column."""
    predictor_dir = root / "2_data" / "3_trf" / group / "predictors"
    if not predictor_dir.exists():
        raise FileNotFoundError(f"Missing predictor directory: {predictor_dir}")

    path = predictor_dir / f"{predictor}.pickle"
    ds = load_dataset(path)
    return PredictorSpec(
        name=predictor,
        path=path,
        key=predictor_key_from_dataset(predictor, ds),
    )


def resolve_predictor_specs(root: Path, group: str, predictors: list[str]) -> list[PredictorSpec]:
    """Resolve predictor names to pickle paths and signal columns."""
    return [resolve_predictor_spec(root, group, predictor) for predictor in predictors]


def model_name_for(predictors: list[str]) -> str:
    """Return a filename-safe model name from one or more predictor names."""
    return "-".join(predictors)


def parse_model_token(token: str) -> list[str]:
    """Parse a model token like pitch+mfcc into predictor names."""
    predictor_names = token.split("+")
    if any(not name for name in predictor_names):
        raise ValueError(
            f"Invalid model '{token}'. Use predictor names joined with '+', "
            "for example pitch+mfcc."
        )
    if len(set(predictor_names)) != len(predictor_names):
        raise ValueError(f"Invalid model '{token}': predictor names are repeated.")
    return predictor_names


def build_model_specs(root: Path, group: str, args: argparse.Namespace) -> list[ModelSpec]:
    """Build one or more model definitions from CLI arguments."""
    if args.predictors:
        model_predictors = [args.predictors]
    else:
        model_predictors = [parse_model_token(token) for token in args.models]

    predictor_cache: dict[str, PredictorSpec] = {}
    models: list[ModelSpec] = []
    seen_model_names: set[str] = set()

    for predictor_names in model_predictors:
        model_name = model_name_for(predictor_names)
        if model_name in seen_model_names:
            raise ValueError(f"Duplicate model definition: {model_name}")
        seen_model_names.add(model_name)

        predictor_specs = []
        for predictor in predictor_names:
            if predictor not in predictor_cache:
                predictor_cache[predictor] = resolve_predictor_spec(root, group, predictor)
            predictor_specs.append(predictor_cache[predictor])

        models.append(
            ModelSpec(
                name=model_name,
                predictor_names=list(predictor_names),
                predictor_specs=predictor_specs,
            )
        )

    return models


def require_columns(ds, columns: list[str], label: str) -> None:
    """Raise a readable error when a Dataset misses required columns."""
    missing = [column for column in columns if column not in ds]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def value_at(ds, column: str, index: int) -> str:
    """Return a Dataset cell as a plain string for comparisons."""
    return str(ds[column][index])


def check_time_dimension(nd: NDVar, label: str) -> None:
    """Ensure an NDVar has an Eelbrain time dimension."""
    if not hasattr(nd, "time"):
        raise ValueError(f"{label} has no time dimension")


def validate_trial_order(eeg_ds, predictor_datasets: dict[str, Any]) -> None:
    """Verify EEG and predictor datasets have the same trial count and order."""
    require_columns(
        eeg_ds,
        ["stim_name", "emotion", "speech_style", "gender", "eeg"],
        "EEG dataset",
    )

    n_cases = eeg_ds.n_cases
    for name, ds in predictor_datasets.items():
        require_columns(
            ds,
            ["stim_name", "emotion", "speech_style", "gender"],
            f"Predictor dataset '{name}'",
        )

        if ds.n_cases != n_cases:
            raise ValueError(
                f"Case count mismatch for predictor '{name}': "
                f"EEG={n_cases}, predictor={ds.n_cases}"
            )

        for index in range(n_cases):
            eeg_name = value_at(eeg_ds, "stim_name", index)
            pred_name = value_at(ds, "stim_name", index)
            if eeg_name != pred_name:
                raise ValueError(
                    f"Stimulus order mismatch at row {index + 1} for predictor '{name}': "
                    f"EEG={eeg_name}, predictor={pred_name}"
                )


def condition_indices(eeg_ds, condition: Condition) -> list[int]:
    """Return trial indices matching a condition."""
    indices = []
    for index in range(eeg_ds.n_cases):
        if (
            value_at(eeg_ds, "speech_style", index) == condition.speech_style
            and value_at(eeg_ds, "emotion", index) == condition.emotion
            and value_at(eeg_ds, "gender", index) == condition.gender
        ):
            indices.append(index)

    return indices


def crop_ndvar_to_n_samples(nd: NDVar, n_samples: int) -> NDVar:
    """Crop an NDVar along time to exactly n_samples."""
    cropped = nd.sub(time=nd.time[:n_samples])
    if len(cropped.time) != n_samples:
        raise ValueError(
            f"Cropping failed for {nd.name}: requested {n_samples}, "
            f"got {len(cropped.time)}"
        )
    return cropped


def concat_ndvars_time(ndvars: list[NDVar]) -> NDVar:
    """Concatenate compatible NDVars along their time dimension."""
    if not ndvars:
        raise ValueError("Cannot concatenate an empty NDVar list")

    first = ndvars[0]
    first_dim_names = [dim.name for dim in first.dims]
    if "time" not in first_dim_names:
        raise ValueError(f"{first.name} has no time dimension")
    time_axis = first_dim_names.index("time")

    for index, nd in enumerate(ndvars[1:], start=2):
        dim_names = [dim.name for dim in nd.dims]
        if dim_names != first_dim_names:
            raise ValueError(
                f"Dimension order mismatch in {first.name} item {index}: "
                f"{dim_names} vs {first_dim_names}"
            )

        for axis, (dim, first_dim) in enumerate(zip(nd.dims, first.dims)):
            if axis == time_axis:
                continue
            if len(dim) != len(first_dim):
                raise ValueError(
                    f"Non-time dimension mismatch in {first.name} item {index}, "
                    f"dimension '{dim.name}': {len(dim)} vs {len(first_dim)}"
                )

    concatenated = np.concatenate([nd.x for nd in ndvars], axis=time_axis)
    total_n = sum(len(nd.time) for nd in ndvars)
    new_time = UTS(first.time.tmin, first.time.tstep, total_n)
    new_dims = list(first.dims)
    new_dims[time_axis] = new_time
    return NDVar(concatenated, tuple(new_dims), name=first.name, info=first.info)


def align_and_concatenate(
    eeg_ds,
    predictor_datasets: dict[str, Any],
    predictor_specs: list[PredictorSpec],
    condition: Condition,
) -> tuple[NDVar, NDVar | list[NDVar], list[dict[str, Any]]]:
    """Select, crop, and concatenate trials for one condition."""
    indices = condition_indices(eeg_ds, condition)
    eeg_trials: list[NDVar] = []
    predictor_trials: dict[str, list[NDVar]] = {spec.name: [] for spec in predictor_specs}
    length_summary: list[dict[str, Any]] = []

    for index in indices:
        eeg_nd = eeg_ds["eeg"][index]
        check_time_dimension(eeg_nd, f"EEG trial {index + 1}")

        trial_lengths = {"eeg": len(eeg_nd.time)}
        predictor_ndvars: dict[str, NDVar] = {}

        for spec in predictor_specs:
            pred_nd = predictor_datasets[spec.name][spec.key][index]
            check_time_dimension(pred_nd, f"Predictor '{spec.name}' trial {index + 1}")
            predictor_ndvars[spec.name] = pred_nd
            trial_lengths[spec.name] = len(pred_nd.time)

        target_n = min(trial_lengths.values())
        eeg_trials.append(crop_ndvar_to_n_samples(eeg_nd, target_n))

        for spec in predictor_specs:
            predictor_trials[spec.name].append(
                crop_ndvar_to_n_samples(predictor_ndvars[spec.name], target_n)
            )

        length_summary.append(
            {
                "trial_index": int(index + 1),
                "stim_name": value_at(eeg_ds, "stim_name", index),
                "original_lengths": trial_lengths,
                "used_length": int(target_n),
            }
        )

    y = concat_ndvars_time(eeg_trials)
    x_by_predictor = [
        concat_ndvars_time(predictor_trials[spec.name]) for spec in predictor_specs
    ]
    x: NDVar | list[NDVar] = x_by_predictor[0] if len(x_by_predictor) == 1 else x_by_predictor
    return y, x, length_summary


def output_path(
    root: Path,
    group: str,
    subject: str,
    session: str,
    acq: str,
    condition: str,
    model_name: str,
) -> Path:
    """Build the output result path for one model."""
    out_dir = root / "2_data" / "3_trf" / group / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{subject}_{session}_{acq}_{condition}_{model_name}_trf.pickle"


def result_payload(result, metadata: dict[str, Any]) -> dict[str, Any]:
    """Wrap a TRF result with reproducibility metadata."""
    return {
        "result": result,
        "metadata": metadata,
    }


def fit_condition(
    *,
    root: Path,
    group: str,
    subject: str,
    session: str,
    acq: str,
    eeg_path: Path,
    model: ModelSpec,
    condition: Condition,
    settings: TrfSettings,
    overwrite: bool,
    parallel: bool,
) -> Path | None:
    """Load data, fit one condition TRF, and save the result."""
    out_path = output_path(
        root=root,
        group=group,
        subject=subject,
        session=session,
        acq=acq,
        condition=condition.code,
        model_name=model.name,
    )
    if out_path.exists() and not overwrite:
        print(f"Skipping existing result: {out_path}")
        return None

    eeg_ds = load_dataset(eeg_path)
    if not condition_indices(eeg_ds, condition):
        print(f"Skipping {condition.code}: no matching trials")
        return None

    predictor_datasets = {spec.name: load_dataset(spec.path) for spec in model.predictor_specs}
    validate_trial_order(eeg_ds, predictor_datasets)
    y, x, length_summary = align_and_concatenate(
        eeg_ds=eeg_ds,
        predictor_datasets=predictor_datasets,
        predictor_specs=model.predictor_specs,
        condition=condition,
    )

    print(f"Estimating TRF: {subject} {session} {acq} {condition.code} {model.name}")
    result = eelbrain.boosting(
        y=y,
        x=x,
        tstart=settings.tstart,
        tstop=settings.tstop,
        basis=settings.basis,
        partitions=settings.partitions,
        test=1,
        selective_stopping=True,
        scale_data=True,
        error=settings.error,
    )

    metadata = {
        "group": group,
        "subject": subject,
        "session": session,
        "acq": acq,
        "condition": condition.code,
        "speech_style": condition.speech_style,
        "emotion": condition.emotion,
        "gender": condition.gender,
        "model_name": model.name,
        "predictor_names": model.predictor_names,
        "eeg_path": str(eeg_path),
        "predictors": [
            {"name": spec.name, "path": str(spec.path), "key": spec.key}
            for spec in model.predictor_specs
        ],
        "settings": settings.__dict__,
        "parallel": parallel,
        "n_trials": len(length_summary),
        "length_summary": length_summary,
        "eelbrain_version": eelbrain.__version__,
    }

    eelbrain.save.pickle(result_payload(result, metadata), out_path)
    print(f"Saved {out_path}")
    return out_path


def run_sequential(
    *,
    root: Path,
    group: str,
    subject_runs: list[SubjectRun],
    models: list[ModelSpec],
    conditions: list[Condition],
    settings: TrfSettings,
    overwrite: bool,
) -> list[Path]:
    """Fit all subject-model-condition pairs in the current process."""
    saved_paths: list[Path] = []
    for subject_run in subject_runs:
        for model in models:
            for condition in conditions:
                saved_path = fit_condition(
                    root=root,
                    group=group,
                    subject=subject_run.subject,
                    session=subject_run.session,
                    acq=subject_run.acq,
                    eeg_path=subject_run.eeg_path,
                    model=model,
                    condition=condition,
                    settings=settings,
                    overwrite=overwrite,
                    parallel=False,
                )
                if saved_path is not None:
                    saved_paths.append(saved_path)
    return saved_paths


def run_parallel(
    *,
    root: Path,
    group: str,
    subject_runs: list[SubjectRun],
    models: list[ModelSpec],
    conditions: list[Condition],
    settings: TrfSettings,
    overwrite: bool,
) -> list[Path]:
    """Fit subject-model-condition pairs in separate processes."""
    max_workers = os.cpu_count() or 1
    saved_paths: list[Path] = []
    print(f"Running in parallel with {max_workers} worker(s)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fit_condition,
                root=root,
                group=group,
                subject=subject_run.subject,
                session=subject_run.session,
                acq=subject_run.acq,
                eeg_path=subject_run.eeg_path,
                model=model,
                condition=condition,
                settings=settings,
                overwrite=overwrite,
                parallel=True,
            )
            for subject_run in subject_runs
            for model in models
            for condition in conditions
        ]

        for future in as_completed(futures):
            saved_path = future.result()
            if saved_path is not None:
                saved_paths.append(saved_path)

    return sorted(saved_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate TRF models for selected subject/session conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""Examples:
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --help
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --models pitch mfcc pitch+mfcc
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --session ses-1 --acq acq-1 --models pitch mfcc pitch+mfcc
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --subject sub-pilot_1 --session ses-1 --acq acq-1 --predictors mfcc
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --subject sub-pilot_1 --session ses-1 --acq acq-1 --models pitch mfcc pitch+mfcc --genders f --emotions hap --speech-styles ADS
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --subject sub-pilot_1 --session ses-1 --acq acq-1 --predictors mfcc --genders f --emotions hap sad --speech-styles ADS
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --subject sub-pilot_1 --session ses-1 --acq acq-1 --predictors mfcc --conditions ADS_hap_f
  .\.venv\Scripts\python.exe 3_code\4_compute_trf\trf_computation.py --group pilot --subject sub-pilot_1 --session ses-1 --acq acq-1 --predictors mfcc gammatone_8 --conditions ADS_hap_f
""",
    )
    parser.add_argument(
        "--group",
        choices=VALID_GROUPS,
        required=True,
        help="Participant group under 2_data/3_trf.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Optional subject ID, for example sub-pilot_1. "
            "If omitted, all matching subjects are processed."
        ),
    )
    parser.add_argument(
        "--session",
        choices=VALID_SESSIONS,
        default=None,
        help="Optional session label. If omitted, all matching sessions are processed.",
    )
    parser.add_argument(
        "--acq",
        default=None,
        help="Optional acquisition label. If omitted, all matching acquisitions are processed.",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--predictors",
        nargs="+",
        help="Predictor pickle stem(s), for example mfcc gammatone_8.",
    )
    model_group.add_argument(
        "--models",
        nargs="+",
        help=(
            "One or more model definitions. Join predictors in one model with '+', "
            "for example pitch mfcc pitch+mfcc."
        ),
    )
    parser.add_argument(
        "--genders",
        nargs="+",
        choices=VALID_GENDERS,
        default=VALID_GENDERS,
        help="Gender value(s) to include. Defaults to f m.",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=VALID_EMOTIONS,
        default=VALID_EMOTIONS,
        help="Emotion value(s) to include. Defaults to neu sad hap.",
    )
    parser.add_argument(
        "--speech-styles",
        nargs="+",
        choices=VALID_SPEECH_STYLES,
        default=VALID_SPEECH_STYLES,
        help="Speaking style value(s) to include. Defaults to ADS CDS.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "Optional explicit condition code(s), for example CDS_hap_f ADS_sad_m. "
            "When provided, this overrides --genders, --emotions, and --speech-styles."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model-condition TRF fits in parallel with all available CPU cores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result pickle files.",
    )
    parser.add_argument(
        "--tstart",
        type=float,
        default=DEFAULT_TSTART,
        help="TRF lag start in seconds.",
    )
    parser.add_argument(
        "--tstop",
        type=float,
        default=DEFAULT_TSTOP,
        help="TRF lag stop in seconds.",
    )
    parser.add_argument(
        "--basis",
        type=float,
        default=DEFAULT_BASIS,
        help="Basis width passed to eelbrain.boosting().",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=DEFAULT_PARTITIONS,
        help="Number of partitions passed to eelbrain.boosting().",
    )
    parser.add_argument(
        "--error",
        choices=("l1", "l2"),
        default=DEFAULT_ERROR,
        help="Error metric passed to eelbrain.boosting().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    subject_runs = find_subject_runs(
        root=root,
        group=args.group,
        subject=args.subject,
        session=args.session,
        acq=args.acq,
    )
    models = build_model_specs(root, args.group, args)
    conditions = build_conditions(args)
    settings = TrfSettings(
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        partitions=args.partitions,
        error=args.error,
    )

    for subject_run in subject_runs:
        print(
            f"Using EEG: {subject_run.subject} {subject_run.session} "
            f"{subject_run.acq} -> {subject_run.eeg_path}"
        )
    for model in models:
        predictor_text = ", ".join(
            f"{spec.name} [{spec.key}]" for spec in model.predictor_specs
        )
        print(f"Using model: {model.name} = {predictor_text}")

    if args.parallel:
        saved_paths = run_parallel(
            root=root,
            group=args.group,
            subject_runs=subject_runs,
            models=models,
            conditions=conditions,
            settings=settings,
            overwrite=args.overwrite,
        )
    else:
        saved_paths = run_sequential(
            root=root,
            group=args.group,
            subject_runs=subject_runs,
            models=models,
            conditions=conditions,
            settings=settings,
            overwrite=args.overwrite,
        )

    print(f"Finished. Saved {len(saved_paths)} result file(s).")


if __name__ == "__main__":
    main()
