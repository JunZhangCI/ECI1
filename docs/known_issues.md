# Known Code Issues To Review

This file records pilot-stage code issues found during repository setup. It is not a substitute for testing.

## PRP Extraction

File: `3_code/5_extract_prp/emo_prp_extract.m`

- Contains an incomplete line: `concat_order_path = fullfile`
- Uses `epoch_first` before it is defined.
- Still has pilot-specific assumptions around subject IDs and preprocessing folder structure.

## TRF Utility Script

File: `3_code/4_compute_trf/split_by_emo.m`

- Uses old hard-coded paths under `C:\projects\emo_EEG`.
- Should be changed to resolve paths from the current `ECI1` project root or shared config.

## Notebooks

Files:

- `3_code/4_compute_trf/*.ipynb`
- `3_code/6_analysis/**/*.ipynb`

Action:

- Strip outputs before committing.
- Consider moving reusable notebook logic into `.py` scripts as the workflow stabilizes.

