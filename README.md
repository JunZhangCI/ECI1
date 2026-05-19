# ECI1

ECI1 is a research code repository for a continuous-speech EEG project on phoneme-level neural encoding of emotional speech in normal-hearing listeners and cochlear implant listeners.

The repository is intended to track code, notebooks without outputs, documentation, and small configuration files. Raw data, processed EEG data, generated audio, model outputs, and participant-related files stay local and are excluded from Git.

## Project Folders

| Folder | Purpose | Git policy |
| --- | --- | --- |
| `1_studysetup/` | Study setup files and stimulus order metadata | Track only small source metadata; generated stimuli are ignored |
| `2_data/` | Raw and processed EEG data | Ignored |
| `3_code/` | MATLAB, Python, and Jupyter analysis code | Tracked |
| `4_presentation/` | Presentation source and figures | Track source selectively; rendered large files are ignored |
| `5_writing/` | Writing drafts and manuscript notes | Prefer Markdown docs in `docs/`; Word/PDF files are ignored |
| `emo_audio/` | Original and generated emotional speech audio | Ignored |
| `resources/` | MATLAB project metadata | Tracked if small and useful |
| `docs/` | Project documentation and development notes | Tracked |
| `config/` | Example shared pipeline configuration | Tracked |

## Current Workflow

The project workflow follows the research plan:

1. Create and prepare emotional speech stimuli.
2. Run acoustic and phoneme-level analyses.
3. Preprocess continuous-speech EEG data.
4. Estimate TRF models.
5. Extract PRPs.
6. Run statistical analysis and visualization.

The currently revised code sections are:

- `3_code/1_create_stimuli`
- `3_code/3_preprocessing`

Later stages are still being adapted from the pilot project.

## Python Environment Setup

This project uses a project-level Python virtual environment in `.venv/`. The environment is local to this folder and is ignored by Git.

From the project root, activate the environment:

```powershell
cd C:\projects\ECI1
.\.venv\Scripts\Activate.ps1
```

Install or refresh the project packages from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

After activation, run Python scripts from the project root. For example:

```powershell
python 3_code\2_acoustic_analysis\extract_phoneme_acoustic_contours.py
```

VS Code is configured to use `.venv\Scripts\python.exe` for this workspace. If VS Code does not pick it up immediately, run `Python: Select Interpreter` and choose the interpreter inside `C:\projects\ECI1\.venv`.

## Git Setup

Initialize local Git from the project root:

```powershell
git init
git branch -M main
git status --short
```

Create an empty private repository named `ECI1` on GitHub, then connect it:

```powershell
git remote add origin <your-private-github-repo-url>
git push -u origin main
```

Before each commit, check what will be tracked:

```powershell
git status --short
git status --ignored
```

Only code, documentation, small config files, and output-stripped notebooks should appear as tracked files.

## Notebook Output Policy

Jupyter notebooks are tracked, but notebook outputs should not be tracked.

Install the notebook stripping tools in the Python environment you use for this project:

```powershell
python -m pip install pre-commit nbstripout
pre-commit install
nbstripout --install
```

Then test the setup:

```powershell
nbstripout --status
pre-commit run --all-files
```

If a notebook is large, clear output before committing.

## Data Safety

Do not commit:

- Raw participant EEG files
- Processed EEG files
- Generated `.mat`, `.set`, `.fdt`, `.pkl`, or `.pickle` files
- Original or generated audio
- Grant applications, drafts, consent documents, or participant-identifying material
- Notebook outputs

See `docs/data_management.md` for details.
