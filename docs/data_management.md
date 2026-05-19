# Data Management

This repository is for source code and project documentation, not for research data storage.

## Keep Out Of Git

Do not commit:

- Raw EEG data: `.bdf`, `.edf`
- EEGLAB and MATLAB outputs: `.set`, `.fdt`, `.mat`
- Model outputs: `.pkl`, `.pickle`, `.h5`, `.hdf5`
- Audio files: `.wav`, `.mp3`, `.flac`, `.aiff`
- Forced-alignment outputs: `.TextGrid`
- Participant-specific logs or quality-control outputs
- Grant PDFs, consent files, or other sensitive administrative documents
- Notebook outputs

These files should remain on local storage, OneDrive, lab storage, or another approved research-data storage system.

## Safe To Track

Usually safe to commit:

- MATLAB scripts: `.m`
- Python scripts: `.py`
- Markdown documentation: `.md`
- Small YAML/JSON configuration files
- Jupyter notebooks after outputs are stripped
- Small study setup metadata that does not contain participant-identifying information

## Before Every Commit

Run:

```powershell
git status --short
git status --ignored
```

Check that no raw data, generated data, audio, or notebook outputs are staged.

If Git shows a file that looks too large or sensitive, do not add it. Update `.gitignore` first.

## GitHub Visibility

The GitHub repository should be private during active development. Public release can be considered later after:

- Data paths are cleaned.
- Notebook outputs are stripped.
- No participant or grant material is present.
- Example/demo data are intentionally prepared.
- The lab has approved sharing.

