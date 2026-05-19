# ECI1 Workflow

This document summarizes the intended analysis workflow. It is written for a beginner-friendly, reproducible pipeline where each stage has clear inputs, outputs, and manual steps.

## 1. Stimulus Creation

Code location: `3_code/1_create_stimuli`

Purpose:

- Prepare emotional speech stimuli from the Chatterjee et al. corpus.
- Trim or preserve silence.
- Create transcript text files.
- Run forced alignment.
- Resample/rescale audio.
- Randomize and concatenate trials.

Local-only inputs and outputs:

- Original audio in `emo_audio/`
- Generated stimuli in `emo_audio/` or `1_studysetup/.../stimuli/`
- Generated WAV files and alignment outputs

Git policy:

- Track stimulus creation scripts.
- Do not track generated audio or TextGrid files.

## 2. Acoustic And Phoneme Analysis

Code location: `3_code/2_acoustic_analysis`

Purpose:

- Summarize phoneme durations and phoneme classes.
- Support later TRF and PRP predictor construction.

Current status:

- Adapted from pilot work and should be reviewed after stimulus creation is finalized.

## 3. EEG Preprocessing

Code location: `3_code/3_preprocessing`

Purpose:

- Calculate trigger latency using recorded audio.
- Import EEG data.
- Apply montage and reference.
- Downsample to 128 Hz.
- Filter between 1 and 30 Hz.
- Manually reject noisy channels.
- Run ICA for ocular and CI-related artifacts.
- Restructure data for trial-level analysis.

Local-only inputs and outputs:

- Raw EEG in `2_data/1_raw/`
- Processed EEG and preprocessing logs in `2_data/2_processed/`

Manual steps:

- Visual channel rejection.
- ICA component review.
- CI artifact decisions.

Git policy:

- Track scripts and helper functions.
- Do not track raw data, processed data, or logs generated during participant processing.

## 4. TRF Modeling

Code location: `3_code/4_compute_trf`

Purpose:

- Build acoustic and phoneme-level predictors.
- Estimate temporal response functions.
- Compare model prediction accuracy across predictor sets, emotions, and groups.

Current status:

- Pilot notebooks and scripts need path/config cleanup before being treated as reproducible.

Git policy:

- Track source notebooks without outputs.
- Do not track predictor pickles, model results, or generated figures.

## 5. PRP Extraction

Code location: `3_code/5_extract_prp`

Purpose:

- Extract phoneme-related EEG segments from continuous speech.
- Apply detrending, optional baseline correction, optional RMS rejection, and optional standardization.
- Save PRP data for downstream analysis.

Current status:

- The script should be checked before use with new data because it still contains pilot-stage assumptions.

Git policy:

- Track source code.
- Do not track generated PRP `.mat` files.

## 6. Analysis And Visualization

Code location: `3_code/6_analysis`

Purpose:

- Analyze behavioral accuracy, TRF model performance, and PRP responses.
- Generate figures for presentations, manuscripts, and lab updates.

Git policy:

- Track source notebooks without outputs.
- Do not track generated figures unless they are intentionally small, final, and non-sensitive.

