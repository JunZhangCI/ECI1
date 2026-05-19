import librosa
from pathlib import Path
import os
import soundfile as sf
import random
import numpy as np
import csv

#--- Configuration ---
TARGET_SF = 48000
TARGET_SCALE = 80  # Praat-like intensity target in dB
trim_silence = False  # Set to True to process trimmed files, False to process original files
emotions = ['neu'] # possible values: 'hap', 'sad', 'ang', 'neu', 'sca'
genders = ['f', 'm']  # possible values: 'f', 'm'
styles = ['CDS', 'ADS']  # possible values: 'CDS', 'ADS'
recordings = 4  # number of times to create the concatenated file
repeatition = 2  # number of times to repeat the senetences in the concatenated file
ISI = 0  # inter-stimulus interval in seconds

# --- Path setup ---
script_path = Path(__file__)
project_root = script_path.parents[2]
raw_audio_dir = project_root / "emo_audio" / "2_processed"
if trim_silence:
    raw_audio_dir = raw_audio_dir / "trimmed"
    out_audio_dir = project_root / "emo_audio" / "4_rand_concate" / "trimmed"
else:
    raw_audio_dir = raw_audio_dir / "untrimmed"
    out_audio_dir = project_root / "emo_audio" / "4_rand_concate" / "untrimmed"
raw_audio_dir = raw_audio_dir / f"{TARGET_SF}Hz_{TARGET_SCALE}dB"
if not raw_audio_dir.exists():
    raise FileNotFoundError(f"Audio directory not found: {raw_audio_dir}")
out_audio_dir = out_audio_dir / f"{TARGET_SF}Hz_{TARGET_SCALE}dB_ISI{int(ISI*1000)}ms"
os.makedirs(out_audio_dir, exist_ok=True)

#--- log file setup ---
csv_filename = "random_cont_order.csv"
csv_file_path = os.path.join(out_audio_dir, csv_filename)
if not os.path.exists(csv_file_path):
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "recording_index", "output_filename", "file_order"])
wav_files = librosa.util.find_files(raw_audio_dir, ext='wav', recurse=False)

#--- Processing ---
for gender in genders:
    for style in styles:
        for emotion in emotions:
            selected_files = []
            # 1. match files based on emotion, style, and gender
            for wav_file in wav_files:
                name = Path(wav_file).stem
                parts = name.split('_')
                # Verify filename has enough components to avoid bugs
                if len(parts) >= 4:
                    if parts[0] == style and parts[2] == emotion and parts[3] == gender:
                        selected_files.append(wav_file)
            # Skip if no matches found on disk for this combo
            if not selected_files:
                continue
            # Repeat the selected files as specified
            selected_files = selected_files * repeatition
            # 2. check existing output files to avoid overwriting
            prefix_match = f"{style}_{emotion}_{gender}_concate"
            existing_files = [f for f in os.listdir(out_audio_dir) if f.startswith(prefix_match) and f.endswith(".wav")]
            # determine the next index
            start_index = len(existing_files) + 1

            # 3. Create concatenated recordings with random order
            for i in range(recordings):
                # Shuffle the selected files randomly
                random_files = random.sample(selected_files, len(selected_files))
                # Concatenate the audio files   
                concatenated_audio = []
                for wav_file in random_files:
                    y, sr = librosa.load(wav_file, sr = None)
                    concatenated_audio.append(y)
                    concatenated_audio.append(np.zeros(int(ISI * sr)))  # Add ISI of silence
                final_audio = np.concatenate(concatenated_audio)
                # Save the concatenated audio file  
                out_filename = f"{style}_{emotion}_{gender}_concate_{start_index + i}.wav" 
                out_file_path = os.path.join(out_audio_dir, out_filename)
                sf.write(out_file_path, final_audio, sr)  
                # Write the order of files to the CSV
                with open(csv_file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        emotion,
                        start_index + i,
                        out_filename,
                        ";\n".join(os.path.basename(f) for f in random_files)
                    ])