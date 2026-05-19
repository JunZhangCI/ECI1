import sys
from pathlib import Path
import subprocess
import os

# --- Configuration ---
trim_silence = False  # Set to True to go through trimmed files, False to go through original files

# --- Path setup ---
script_path = Path(__file__)
project_root = script_path.parents[2]
corpus_dir = project_root / "emo_audio" / "2_processed"
mfa_dir = project_root / "emo_audio" / "3_mfa" 
if trim_silence:   
    audio_corpus_dir = corpus_dir / "trimmed"
    mfa_out_dir = mfa_dir / "trimmed"
else:
    audio_corpus_dir = corpus_dir / "untrimmed"
    mfa_out_dir = mfa_dir / "untrimmed"
if not audio_corpus_dir.exists():
    raise FileNotFoundError(f"Audio directory not found: {audio_corpus_dir}")
os.makedirs(mfa_out_dir, exist_ok=True)

# --- Set the MFA Arguments ---
DICTIONARY_PATH = "english_us_arpa"
ACOUSTIC_MODEL = "english_us_arpa"

def run_command(command_list):
    """Executes a shell command inside the aligner environment."""
    # Combine activation and your target command using '&&'
    # This ensures mfa is fully loaded into the correct environment before running the command
    full_command = f"mamba activate aligner && {' '.join(command_list)}"
    
    print(f"\nRunning: {' '.join(command_list)}")
    
    result = subprocess.run(
        full_command,
        shell=True,
        text=True,
        stdout=sys.stdout, # Redirects the standard output or any error messages 
        stderr=sys.stderr  # of the command straight to your current terminal screen in real time.
    )
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

# --- Processing ---
# 1. validate that MFA is installed and available
try:
    run_command(["mfa", "version"])
except subprocess.CalledProcessError:
    print("Error: MFA is not installed or not available in the environment.")
    sys.exit(1)

# 2. validate the dataset structure and files
validate_cmd = [
    "mfa", "validate",
    f'"{audio_corpus_dir}"',
    f'"{DICTIONARY_PATH}"',
    f'"{ACOUSTIC_MODEL}"'
]
run_command(validate_cmd)

# 3. run MFA alignment
align_cmd = [
    "mfa", "align",
    f'"{audio_corpus_dir}"',
    f'"{DICTIONARY_PATH}"',
    f'"{ACOUSTIC_MODEL}"',
    f'"{mfa_out_dir}"'
]
run_command(align_cmd)