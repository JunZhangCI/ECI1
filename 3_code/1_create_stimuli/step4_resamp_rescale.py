import librosa
from pathlib import Path
import os
import numpy as np
import soundfile as sf

#--- Configuration ---
TARGET_SF = 48000
TARGET_SCALE = 80  # Praat-like intensity target in dB
trim_silence = False  # Set to True to process trimmed files, False to process original files

# --- Path setup ---
script_path = Path(__file__)
project_root = script_path.parents[2]
raw_audio_dir = project_root / "emo_audio" / "2_processed"
if trim_silence:
    raw_audio_dir = raw_audio_dir / "trimmed"
else:
    raw_audio_dir = raw_audio_dir / "untrimmed"
if not raw_audio_dir.exists():
    raise FileNotFoundError(f"Audio directory not found: {raw_audio_dir}")
out_audio_dir = raw_audio_dir/ f"{TARGET_SF}Hz_{TARGET_SCALE}dB"
os.makedirs(out_audio_dir, exist_ok=True)

wav_files = librosa.util.find_files(raw_audio_dir, ext='wav')

for wav_path in wav_files:
    name, ext = os.path.splitext(os.path.basename(wav_path))
    out_filename = f"{name}_{TARGET_SF}Hz_{TARGET_SCALE}dB{ext}"
    out_file_path = os.path.join(out_audio_dir, out_filename)

    if os.path.exists(out_file_path):
        print(f"Skip already processed file: {out_file_path}")
        continue

    # keep original sampling rate
    wav_y, wav_sr = librosa.load(wav_path, sr=None)

    # 1) resample
    if wav_sr != TARGET_SF:
        wav_y = librosa.resample(wav_y, orig_sr=wav_sr, target_sr=TARGET_SF)
        wav_sr = TARGET_SF

    # 2) scale like Praat intensity
    current_rms = np.sqrt(np.mean(wav_y ** 2))
    target_rms = 2e-5 * (10 ** (TARGET_SCALE / 20))
    gain = target_rms / current_rms
    wav_y = wav_y * gain

    sf.write(out_file_path, wav_y, wav_sr)

    