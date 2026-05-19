import librosa
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# --- Configuration ---
raw_audio_conditions = ['Male CDS', 'Male ADS', 'Fem CDS', 'Fem ADS']
audio_sets = ['Test', 'Training']

trim_silence = True  # Set to True to enable trimming silence from audio files
top_db = 20  # Threshold in decibels for trimming silence (lower means more aggressive trimming)
frame_length = 512  # Frame length for trimming (in samples)
hop_length = 128  # Hop length for trimming (in samples)

# --- Path setup ---
script_path = Path(__file__)
project_root = script_path.parents[2]
raw_audio_dir = project_root / "emo_audio" / "1_MC_2015"
if trim_silence:
    out_audio_dir = project_root / "emo_audio" / "2_processed" / "trimmed"
else:
    out_audio_dir = project_root / "emo_audio" / "2_processed" / "untrimmed"
os.makedirs(out_audio_dir, exist_ok=True)

# Create text file to store boundary times
if trim_silence:
    boundary_filename = "trim_boundaries.txt"
    boundary_file_path = os.path.join(out_audio_dir, boundary_filename)

# --- Processing ---
for condition in raw_audio_conditions:  
    style = condition.split()[1]  

    for audio_set in audio_sets:
        condition_audio_path = f"{raw_audio_dir}/{style} Stimuli/{condition}/{audio_set}"   
        wav_files = librosa.util.find_files(condition_audio_path, ext='wav')  
        print(f"Processing {condition} - {audio_set}: Found {len(wav_files)} files.")

        for wav_path in wav_files:
            # Extract filename components
            path_obj = Path(wav_path)
            name = path_obj.stem   # Filename without extension
            ext = path_obj.suffix  # Extension (e.g., '.wav')
            # Reconstruct filename component
            components = name.split("_")
            components[0] = style
            new_name = "_".join(components)
            # Setup output filename base and folder
            if trim_silence:
                out_filename = f"{new_name}_trim{ext}"
            else:
                out_filename = f"{new_name}{ext}"  
            out_file_path = out_audio_dir / out_filename

            # Skip duplicate processing
            if out_file_path.exists():
                print(f"Skip already existing file: {out_file_path.name}")
                continue

            # Load audio array
            wav_y, wav_sr = librosa.load(path=wav_path, sr=None)

            # Trim silence if enabled
            if trim_silence:
                yt, index = librosa.effects.trim(
                    wav_y, 
                    top_db=top_db,
                    frame_length = frame_length,
                    hop_length = hop_length
                    )
            
                # Print original and cleaned durations
                original_duration = librosa.get_duration(y=wav_y, sr=wav_sr)
                trimmed_duration = librosa.get_duration(y=yt, sr=wav_sr)
                # Get boundary times in seconds
                start_time = index[0] / wav_sr 
                end_time = index[1] / wav_sr
                # Save durations and boundary times to a text file
                with open(boundary_file_path, 'a') as f:
                    f.write(f"{out_filename}: \n - Original Duration = {original_duration:.2f} s, "
                            f"Cleaned Duration = {trimmed_duration:.2f} s, Start Time = {start_time:.2f} s, End Time = {end_time:.2f} s\n")
                sf.write(str(out_file_path), yt, wav_sr)
            else:
                sf.write(str(out_file_path), wav_y, wav_sr)