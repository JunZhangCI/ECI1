import os
import re
import eelbrain
import numpy as np
import textgrids as tg
from .utils import resolve_path, load_config, resolve_param


def make_predictors_from_textgrids(
    config = None,
    *,
    eeg_sr = None,
    textgrid_path = None,
    stim_path = None,
    predictor_set = None,
    predictor_name = None,
    target_tier = None,
    select_events = None,
    apply_before_select = None,
    remove_pattern = None,
    alignment = None
    ):
    """
    Make predictors from TextGrid annotations of the stimuli. Accept either a configuration file or keyword arguments. Note that keyword arguments
    override config values.

    Parameters
    ----------
    config : Config or None
        Config object with a .data attribute.
    eeg_sr : int or float or None
        EEG sampling rate to which predictors will be aligned.
    textgrid_path : str or Path or None
        Path to the TextGrid annotation files.
    stim_path : str or Path or None
        Path to the stimulus audio files.
    predictor_set : str or None
        Name of the predictor set under which outputs will be organized.
    predictor_name : str or None
        Name of the TextGrid-based predictor to generate.
    target_tier : str or int or None
        Target tier in the TextGrid from which events or intervals will be extracted.
    select_events : list[str] or None
        Labels of events to include.
    apply_before_select : bool or None
        Whether regex-based label processing should be applied before event selection.
    remove_pattern : str or None
        Regex pattern used to modify or remove parts of event labels.
    alignment : str or None
        Alignment rule used to place events in time. Options include 'start', 'end', 'mid', 'fill'.
    """
    # Load paths and variables from config
    cfgs = config.data if config is not None else {}
    eeg_sr = resolve_param(eeg_sr, cfgs, ['EEG', 'sr'])
    textgrid_path = resolve_path(resolve_param(textgrid_path, cfgs, ['predictor', 'textgrid', 'textgrid_path']))
    stim_path = resolve_path(resolve_param(stim_path, cfgs, ['stim_path']))
    predictor_set = resolve_param(predictor_set, cfgs, ['predictor', 'predictor_set'])
    predictor_name = resolve_param(predictor_name, cfgs, ['predictor', 'textgrid', 'predictor_name'])
    target_tier = resolve_param(target_tier, cfgs, ['predictor', 'textgrid', 'target_tier'])
    select_events = resolve_param(select_events, cfgs, ['predictor', 'textgrid', 'select_events'], required = False, default = None)
    apply_before_select = resolve_param(apply_before_select, cfgs, ['predictor', 'textgrid', 'event_label_regex', 'apply_before_select'])
    remove_pattern = resolve_param(remove_pattern, cfgs, ['predictor', 'textgrid', 'event_label_regex', 'remove_pattern'], required = False, default = None)
    alignment = resolve_param(alignment, cfgs, ['predictor', 'textgrid', 'alignment'])

    # Create output folder to save the predictors
    outpath = os.path.join('predictors', predictor_set, predictor_name)
    outpath = resolve_path(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print(f'Generating predictors using TextGrids in:\n  {textgrid_path}')
    print(f'Output path:\n {outpath}')

    n_events = len(select_events)
    print(f"Number of events for building the predictors: {n_events}")

    # Loop through all audio files and make predictors using the corresponding TextGrids
    files_not_processed = {}
    wavs = list(stim_path.glob('*.wav'))
    for w in wavs:
        stim_id = os.path.basename(w).split('.')[0]
        tg_path = os.path.join(textgrid_path, f'{stim_id}.TextGrid')

        # Try loading the TextGrid file
        try:
            textgrid = tg.TextGrid(tg_path)
        except FileNotFoundError:
            print(f" ** TextGrid file not found for {stim_id}: {tg_path}. Skipping this file. **")
            files_not_processed[stim_id] = 'TextGrid not found'
            continue

        # Extract events from the target tier
        try:
            events = textgrid[target_tier]
        except KeyError:
            print(f" ** Target tier not found in the TextGrid {tg_path} for {stim_id}. Skipping this file. **")
            files_not_processed[stim_id] = 'target_tier not found in TextGrid'
            continue

        is_point_tier = textgrid[target_tier].is_point_tier
        tier_type = 'PointTier' if is_point_tier else 'Interval'
        print(f'Processing {os.path.basename(w)} (target tier: {target_tier}; tier type: {tier_type})...')

        # Process the labels by removing the specified pattern, if this should be done before selecting the events of interest
        if apply_before_select and (remove_pattern is not None):
            new_events = []
            for e in events:
                e.text = re.sub(remove_pattern, '', e.text)
                new_events.append(e)
            events = new_events
        
        # Then, select the events of interest
        events = [e for e in events if e.text in select_events]

        # If the regex should be applied after selecting the events of interest, apply it now
        if (not apply_before_select) and (remove_pattern is not None):
            new_events = []
            for e in events:
                e.text = re.sub(remove_pattern, '', e.text)
                new_events.append(e)
            events = new_events

        # Build the predictor
        # Load the stimulus file, resample it to match the EEG sampling rate, and get the time dimension
        wav = eelbrain.load.wav(w)
        time_dim = wav.bin(step = 1/eeg_sr, label = 'start').time # Set label to 'start' so that time UTS begins at 0
        n_timesamples = time_dim.nsamples

        # Descriptor for the dimension of the included events
        event_dim = eelbrain.Categorial(predictor_name, values = select_events)

        # Create an NDVar with the time dimension and event dimension, and populate it with 0s
        pred = eelbrain.NDVar.zeros((event_dim, time_dim), name = predictor_name)

        # Loop through the events and set the corresponding time points in the predictor to 1
        for e in events:
            event_idx = event_dim.values.index(e.text)

            # The following applies only if events are an Interval tier.
            if not is_point_tier:
                if alignment in ['start', 'end', 'mid']:
                    if alignment == 'start':
                        # Find the index of the time point closest to the event start time (e.xmin) and set that to 1
                        time_idx = np.argmin(np.abs(time_dim.times - e.xmin))
                    elif alignment == 'end':
                        # Find the index of the time point closest to the event end time (e.xmax) and set that to 1
                        time_idx = np.argmin(np.abs(time_dim.times - e.xmax))
                    else:
                        # Find the index of the time point closest to the event midpoint time and set that to 1
                        event_midpoint = (e.xmin + e.xmax) / 2
                        time_idx = np.argmin(np.abs(time_dim.times - event_midpoint))
                    pred.x[event_idx, time_idx] = 1

                elif alignment == 'fill':
                    # Find the indices of the time points that fall within the event duration and set those to 1
                    time_indices = np.where((time_dim.times >= e.xmin) & (time_dim.times < e.xmax))[0]
                    pred.x[event_idx, time_indices] = 1
                else:
                    raise ValueError(f"Invalid alignment option: {alignment}. Please choose from 'start', 'end', 'mid', or 'fill'.")
            
            # If PointTier
            else:
                time_idx = np.argmin(np.abs(time_dim.times - e.xpos))
                pred.x[event_idx, time_idx] = 1
        
        # Save predictors to pickle
        eelbrain.save.pickle(pred, os.path.join(outpath, f'{stim_id}~{predictor_name}-{n_events}.pickle'))

    print()
    print('All done.')
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
    make_predictors_from_textgrids(config)