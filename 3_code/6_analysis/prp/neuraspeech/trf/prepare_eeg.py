import os
import mne
import eelbrain
import warnings
import numpy as np
import pandas as pd

from .utils import resolve_path, load_config, resolve_param

def prepare_eeg(
    config = None,
    *,
    eeg_data_path = None,
    metadata_path = None,
    output_path = None,
    stim_path = None,
    metadata_filename_col = None,
    metadata_stim_col = None,
    channels_to_drop = None,
    duration_mode = None,
    fixed_duration = None,
    ):
    """
    Prepare EEG data for TRF analysis. Accept either a configuration file or keyword arguments. Note that kwargs override config values.

    Parameters
    ----------
    config : Config or None
        Config object with a .data attribute.
    eeg_data_path : str or Path or None
        Path to EEG .fif files.
    metadata_path : str or Path or None
        Path to metadata CSV.
    output_path : str or Path or None
        Output folder.
    stim_path : str or Path or None
        Path to stimuli.
    metadata_filename_col : str or None
        Metadata column containing EEG filenames.
    metadata_stim_col : str or None
        Metadata column containing stimulus names.
    channels_to_drop : list[str] or None
        Channels to drop.
    duration_mode : str or None
        Duration handling mode.
    fixed_duration : float or None
        Fixed duration, if applicable.
    """
    # Load paths and variables from config or kwargs
    cfgs = config.data if config is not None else {}
    eeg_data_path = resolve_path(resolve_param(eeg_data_path, cfgs, ['EEG', 'EEG_path']))
    metadata_path = resolve_path(resolve_param(metadata_path, cfgs, ['EEG', 'metadata', 'path']))
    output_path = resolve_path(resolve_param(output_path, cfgs, ['EEG', 'output_path']))
    stim_path = resolve_path(resolve_param(stim_path, cfgs, ['stim_path']))

    metadata_filename_col = resolve_param(metadata_filename_col, cfgs, ['EEG', 'metadata', 'filename_col'])
    metadata_stim_col = resolve_param(metadata_stim_col, cfgs, ['EEG', 'metadata', 'stim_col'])
    channels_to_drop = resolve_param(channels_to_drop, cfgs, ['EEG', 'channels_to_drop'], required = False, default = [])
    duration_mode = resolve_param(duration_mode, cfgs, ['EEG', 'duration_mode'])
    fixed_duration = resolve_param(fixed_duration, cfgs, ['EEG', 'fixed_duration'], required = False, default = None)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    # Get list of EEG data files and read metadata
    eeg_data_files = list(eeg_data_path.glob('*.fif'))
    metadata = pd.read_csv(metadata_path)

    # Suppress MNE info messages
    mne.set_log_level('ERROR')   # or "WARNING"

    # Suppress the RuntimeWarning about filename conventions
    warnings.filterwarnings(
        'ignore',
        message = ".*does not conform to MNE naming conventions.*",
        category = RuntimeWarning,
    )

    # Now, iterate through each EEG data file
    # Dictionary to keep track of files that were not processed and the reasons why
    files_not_processed = {}
    stim_min_durations = {}

    for eeg_file in eeg_data_files:
        print('Processing file:', eeg_file)

        # Extract the base filename and look up its metadata
        eeg_file_basename = os.path.basename(eeg_file)
        eeg_file_metadata = metadata.loc[metadata[metadata_filename_col] == eeg_file_basename]
        if eeg_file_metadata.empty:
            print(f'  ** Metadata for {eeg_file_basename} not found. Skipping this file. **')
            files_not_processed[eeg_file_basename] = 'Metadata not found'
            continue

        # Extract stimulus file(s) from metadata. Assuming the 'stim' column contains one or more stimulus filenames separated by semicolons.
        stimuli_files_str = eeg_file_metadata[metadata_stim_col].values[0]
        stimulus_files = stimuli_files_str.split(';')
        stimulus_files = [x.strip() for x in stimulus_files]
        n_stim = len(stimulus_files)
        if n_stim == 0:
            print(f'  ** No stimulus files listed for {eeg_file_basename}. Skipping this file. **')
            files_not_processed[eeg_file_basename] = 'No stimulus files listed'
            continue

        # Load the raw EEG data (.fif) using MNE
        epochs = mne.read_epochs(eeg_file, verbose = 'ERROR')

        # Pick only EEG channels, drop the specified channels, and crop the epochs to exclude the baseline period
        epochs = epochs.pick(picks = ['eeg'])
        epochs = epochs.drop_channels(channels_to_drop)
        epochs = epochs.crop(tmin = 0.0)

        # Extract channel names and their corresponding 3D locations to create an eelbrain Sensor object
        channels = [x['ch_name'] for x in epochs.info['chs']]
        xyz = np.array([x['loc'][0:3] for x in epochs.info['chs']])
        sensor = eelbrain.Sensor(xyz, channels)

        # Make NDVar for the EEG data.
        eeg = epochs.get_data() # shape: (n_epochs, n_channels, n_times)
        eeg = np.transpose(eeg, (0, 2, 1)) # shape: (n_epochs, n_times, n_channels)
        if eeg.shape[0] != n_stim:
            print(f'  ** Number of epochs in {eeg_file_basename} ({eeg.shape[0]}) does not match number of stimulus files listed in metadata ({n_stim}). Skipping this file. **')
            files_not_processed[eeg_file_basename] = 'Number of epochs does not match number of stimulus files'
            continue

        sr = epochs.info['sfreq']
        n_epochs, n_timesamples, n_channels = eeg.shape
        case = eelbrain.Case(n_epochs)
        time = eelbrain.UTS(0, 1/sr, n_timesamples)
        eeg_ndvar = eelbrain.NDVar(eeg, (case, time, sensor))

        if (duration_mode == 'min') or (duration_mode == 'fixed'):
            # Find the minimum duration across stimulus files of all epochs
            # To speed up processing, we can keep track of the minimum durations for unique sets of stimulus files.
            # If we encounter the same set of stimulus files again, we can reuse the previously computed minimum duration 
            # instead of reloading and reprocessing the audio files.
            if stimuli_files_str in stim_min_durations:
                min_stim_duration = stim_min_durations[stimuli_files_str]
            else:
                min_stim_duration = np.inf
                for stim_file in stimulus_files:
                    wav = eelbrain.load.wav(os.path.join(stim_path, stim_file))
                    stim_duration = wav.time[-1] - wav.time[0]
                    if stim_duration < min_stim_duration:
                        min_stim_duration = stim_duration
                stim_min_durations[stimuli_files_str] = min_stim_duration

            # Next, if duration_mode is "min", we will crop the EEG data to the minimum stimulus duration.
            if duration_mode == 'min':
                eeg_ndvar = eeg_ndvar.sub(time = (0.0, min_stim_duration))
            
            # If duration_mode is "fixed", we will crop the EEG data to the user-provided fixed duration, 
            # but only if the minimum stimulus duration is at least as long as the fixed duration. If not, we will skip this file and log it in files_not_processed.
            else:
                if min_stim_duration < fixed_duration:
                    print(f'  ** Minimum stimulus duration for {eeg_file_basename} ({min_stim_duration:.2f} seconds) is shorter than the fixed duration ({fixed_duration:.2f} seconds). Skipping this file. **')
                    files_not_processed[eeg_file_basename] = 'Minimum stimulus duration is shorter than fixed duration'
                    continue
                eeg_ndvar = eeg_ndvar.sub(time = (0.0, fixed_duration))

        else:
            raise ValueError(f'Invalid duration_mode: {duration_mode}. Must be either "min" or "fixed".')

        # Now contruct eelbrain Dataset
        eeg_ds = eelbrain.Dataset()
        eeg_ds['eeg'] = eeg_ndvar
        for c in eeg_file_metadata.columns:
            if c != 'stim':  # Already extracted stimulus info and filename is not needed as a variable in the dataset
                eeg_ds[c] = eelbrain.Factor([eeg_file_metadata[c].iloc[0]] * n_epochs)

        # Add info about how duration was handled
        eeg_ds['duration_mode'] = eelbrain.Factor([duration_mode] * n_epochs)

        # Add stimulus information as a Factor variable in the dataset. 
        # This will be needed later to find the corresponding predictors in the TRF analysis.
        eeg_ds['stim'] = eelbrain.Factor(stimulus_files)

        # Save the EEG dataset object as a pickle file. The filename will be the same as the original EEG file but with a .pickle extension.
        out_filename = eeg_file_basename.split('.')[0] + '.pickle'
        eelbrain.save.pickle(eeg_ds, os.path.join(output_path, out_filename))

    print()
    print('Done. Successfully prepared EEG datasets have been saved to:', output_path)
    if len(files_not_processed) > 0:
        print('The following files were not processed due to the reasons indicated:')
        for f, reason in files_not_processed.items():
            print(f'- {f}: {reason}')

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    default_conf = Path(__file__).resolve().parent / 'conf' / 'conf.yaml'
    parser.add_argument(
        '-c', '--conf_dir',
        default = str(default_conf),
        help = 'Path to the configuration file',
    )
    args = parser.parse_args()

    config = load_config(args.conf_dir)
    prepare_eeg(config)