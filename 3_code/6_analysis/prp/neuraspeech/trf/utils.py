import os
import re
import mne
import yaml
import eelbrain
import numpy as np
import textgrids as tg
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt

TRF_DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    data: dict
    path: Path

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)
    
def resolve_path(path_like, base_dir=TRF_DIR):
    p = Path(path_like)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()

def load_config(conf_path=None):
    """
    Load the configuration file.

    Parameters
    ----------
    conf_path : str or Path or None
        Path to the YAML configuration file. If None, use
        TRF_DIR / 'conf' / 'conf.yaml'.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if conf_path is None:
        conf_path = TRF_DIR / 'conf' / 'conf.yaml'
    else:
        conf_path = resolve_path(conf_path)

    conf_path = Path(conf_path)

    with open(conf_path, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return Config(data = config_data, path = conf_path)

def get_nested(cfgs, keys, required = True, default = None):
    """
    Retrieve a nested value from a config-like dictionary.

    Parameters
    ----------
    cfgs : dict
        Configuration dictionary to search.
    keys : list[str]
        Sequence of nested keys specifying the path to the target value.
        For example, ['EEG', 'metadata', 'path'] will return
        cfgs['EEG']['metadata']['path'].
    required : bool, default=True
        If True, raise a ValueError when the key path does not exist.
        If False, return ``default`` instead.
    default : any, default=None
        Value to return if the key path is missing and ``required`` is False.

    Returns
    -------
    any
        The value found at the specified nested key path.

    Raises
    ------
    ValueError
        If the key path does not exist and ``required`` is True.

    Notes
    -----
    This function is useful for safely retrieving deeply nested values from configuration dictionaries without manually checking each level.
    """
    value = cfgs
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        if required:
            key_str = " -> ".join(keys)
            raise ValueError(f"Missing config value at {key_str}.")
        return default

def resolve_param(arg_value, cfgs, keys, required = True, default = None):
    """
    Resolve a parameter value from either an explicit argument or a config.

    Parameters
    ----------
    arg_value : any
        Explicit value passed directly to the calling function. If not None,
        this value is returned without consulting the config.
    cfgs : dict
        Configuration dictionary to search if ``arg_value`` is None.
    keys : list[str]
        Sequence of nested keys specifying where to find the value in
        ``cfgs``.
    required : bool, default=True
        If True, raise a ValueError when neither ``arg_value`` nor the
        config entry is available.
        If False, return ``default`` instead.
    default : any, default=None
        Value to return if ``arg_value`` is None and the config entry is
        missing while ``required`` is False.

    Returns
    -------
    any
        The resolved parameter value. Explicit argument values take
        precedence over config values.

    Raises
    ------
    ValueError
        If ``arg_value`` is None, the config entry is missing, and
        ``required`` is True.

    Notes
    -----
    This function is intended for functions that support both direct
    keyword arguments and config-based parameter specification.
    """
    if arg_value is not None:
        return arg_value
    return get_nested(cfgs, keys, required = required, default = default)

def select(data_files, metadata, subset_configs):
    """
    Get metadata about the data files and subset based on the subsetting configurations.
    """
    df_info = [metadata.loc[metadata['filename'] == df].to_dict(orient='records')[0] for df in data_files]
    include = subset_configs['include']
    exclude = subset_configs['exclude']

    # Apply filters to subset data files
    if include:
        for v in include:
            if include[v]:
                df_info = [d for d in df_info if d[v] in include[v]]
    if exclude:
        for v in exclude:
            if exclude[v]:
                df_info = [d for d in df_info if d[v] not in exclude[v]]
    # return [d['filename'] for d in df_info]
    return {d['filename']: {k: v for k, v in d.items() if k != 'filename'} for d in df_info}


def get_sensor_info(montage_path):
    """
    Get sensor information.
    """
    ext = os.path.basename(montage_path).split('.')[-1]
    if ext != 'fif':
        chan_info = mne.channels.read_custom_montage(montage_path)
    else:
        chan_info = mne.channels.read_dig_fif(montage_path)
        
    ch_pos = chan_info.get_positions()['ch_pos']
    channels, locs = list(ch_pos.keys()), np.array(list(ch_pos.values()))
    return eelbrain.Sensor(locs, names = channels)

def plot_textgrid_events(
    config,
    apply_regex = True,
    normalize = False,
    threshold = None,
    sort_by_count = True,
    figsize = (12, 4),
    return_counts = False
    ):
    """
    Plot the distribution of all event labels found in the target tier 
    across all stimulus TextGrids, and also performs a check to see if 
    all files can be processed correctly.

    Parameters
    ----------
    config : Config
        Loaded config object with .data
    apply_regex : bool
        If True, apply the regex-based label transformation defined in the config
        before counting events. If False, use the original labels from the TextGrids.
    normalize : bool
        If True, plot proportions instead of raw counts.
    threshold : float or int or None
        If not None, draw a red horizontal line at this threshold and print all
        event labels with values >= threshold. For normalize=False, this is a count
        threshold. For normalize=True, this is a proportion threshold.
    sort_by_count : bool
        If True, sort bars by descending count. Otherwise sort by label.
    figsize : tuple
        Matplotlib figure size.
    return_counts : bool
        If True, also return the Counter of event counts.

    Returns
    -------
    counts : collections.Counter, optional
        Returned only if return_counts=True.
    """
    cfgs = config.data

    textgrid_path = resolve_path(cfgs['predictor']['textgrid']['textgrid_path'])
    stim_path = resolve_path(cfgs['stim_path'])
    target_tier = cfgs['predictor']['textgrid']['target_tier']
    remove_pattern = cfgs['predictor']['textgrid']['event_label_regex']['remove_pattern']

    print(f'Stimuli in: {stim_path}')
    print(f'TextGrids in: {textgrid_path}')
    print(f'Target tier: {target_tier}')

    counts = Counter()
    files_not_processed = {}

    wavs = list(stim_path.glob('*.wav'))
    for w in wavs:
        stim_id = os.path.basename(w).split('.')[0]
        tg_path = os.path.join(textgrid_path, f'{stim_id}.TextGrid')

        try:
            textgrid = tg.TextGrid(tg_path)
        except FileNotFoundError:
            files_not_processed[stim_id] = 'TextGrid not found'
            continue

        try:
            events = textgrid[target_tier]
        except KeyError:
            files_not_processed[stim_id] = 'target_tier not found in TextGrid'
            continue

        # Apply regex transformation only if requested
        if apply_regex and (remove_pattern is not None):
            new_events = []
            for e in events:
                e.text = re.sub(remove_pattern, '', e.text)
                new_events.append(e)
            events = new_events

        # Count all non-empty event labels
        for e in events:
            label = e.text.strip()
            if label != '':
                counts[label] += 1

    if len(counts) == 0:
        print('No events found.')
        if return_counts:
            return counts
        return

    items = list(counts.items())
    if sort_by_count:
        items = sorted(items, key = lambda x: x[1], reverse = True)
    else:
        items = sorted(items, key = lambda x: x[0])

    labels = [x[0] for x in items]
    raw_values = [x[1] for x in items]

    if normalize:
        total = sum(raw_values)
        values = [v / total if total > 0 else 0 for v in raw_values]
        ylabel = 'Proportion'
        if (threshold is not None) and not (0 <= threshold <= 1):
            raise ValueError("When normalize = True, threshold should be between 0 and 1.")
    else:
        values = raw_values
        ylabel = 'Count'
        if (threshold is not None) and (threshold < 0):
            raise ValueError("When normalize = False, threshold should be >= 0.")

    plt.figure(figsize = figsize)
    plt.bar(labels, values, color = 'gray')

    if threshold is not None:
        plt.axhline(
            y = threshold,
            color = 'red',
            linestyle = '--',
            linewidth = 1.5,
            label = f'{ylabel} = {threshold}'
        )
        plt.legend()

    plt.ylabel(ylabel)
    plt.xlabel('Event label')
    plt.title('Event distribution across TextGrids')
    plt.xticks(ha = 'center')
    plt.tight_layout()
    plt.show()

    if threshold is not None:
        passing = [lab for lab, val in zip(labels, values) if val >= threshold]
        scale_name = 'proportion' if normalize else 'count'
        print(f'Events with {scale_name} >= {threshold}:')
        if len(passing) == 0:
            print('None')
        else:
            print(passing)

    if files_not_processed:
        print('The following files were not processed:')
        for f, reason in files_not_processed.items():
            print(f'- {f}: {reason}')
    else:
        print('No issues found. All files processed correctly.')

    if return_counts:
        return counts