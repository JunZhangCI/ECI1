import os
import eelbrain
import numpy as np
from .utils import resolve_path, load_config, resolve_param

def make_gammatone_predictors(
    config = None,
    *,
    eeg_sr = None,
    stim_path = None,
    predictor_set = None,
    gammatone_bank_cfgs = None,
    c = None,
    n_bands = None,
    scale = None,
    save_raw_gt_spec = None,
    ):
    """
    Make predictors based on gammatone spectorgrams of the audio. Accept either a configuration file or keyword arguments. Note that keyword arguments override config values.

    Parameters
    ----------
    config : Config or None
        Config object with a .data attribute.
    eeg_sr : int or float or None
        EEG sampling rate to which predictors will be resampled.
    stim_path : str or Path or None
        Path to the stimulus audio files.
    predictor_set : str or list[str] or None
        Name of the output predictor set.
    gammatone_bank_cfgs : dict or None
        Configuration settings for the gammatone filterbank.
    c : float or None
        Saturation model for auditory edge-detection model.
    n_bands : int or None
        How many frequency bands there will be in the final multiband predictors?
    scale : str or None
        Scale the gammatone spectrogram? Either 'log', 'linear', 'power', or None.
    save_raw_gt_spec : bool or None
        Whether to save the raw gammatone spectrograms in addition to the processed predictors.
    """
    # Load paths and variables from config
    cfgs = config.data if config is not None else {}
    eeg_sr = resolve_param(eeg_sr, cfgs, ['EEG', 'sr'])
    stim_path = resolve_path(resolve_param(stim_path, cfgs, ['stim_path']))
    predictor_set = resolve_param(predictor_set, cfgs, ['predictor', 'predictor_set'])
    gammatone_bank_cfgs = resolve_param(gammatone_bank_cfgs, cfgs, ['predictor', 'gammatone', 'gammatone_bank_cfgs'])
    c = resolve_param(c, cfgs, ['predictor', 'gammatone', 'c'])
    n_bands = resolve_param(n_bands, cfgs, ['predictor', 'gammatone', 'n_bands'])
    scale = resolve_param(scale, cfgs, ['predictor', 'gammatone', 'scale'])
    save_raw_gt_spec = resolve_param(save_raw_gt_spec, cfgs, ['predictor', 'gammatone', 'save_raw_gt_spec'])

    # Create output folder to save the gammatone predictors
    outpath = os.path.join('predictors', predictor_set, 'gammatone')
    outpath = resolve_path(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print(f'Generating predictors for stimuli in:\n  {stim_path}')
    print(f'Output path:\n {outpath}')

    # Loop through all audio files and extract gammatone envelope and onset features
    wavs = list(stim_path.glob('*.wav'))
    for w in wavs:
        print(f'Processing {os.path.basename(w)}...')
        # track_id = int(re.search(r'\d+', os.path.basename(w)).group())
        stim_id = os.path.basename(w).split('.')[0]
        wav = eelbrain.load.wav(w)
        wav.x = wav.x.astype(np.int16)

        # Apply a gammatone filterbank, producing a high resolution spectrogram
        gt = eelbrain.gammatone_bank(wav, **gammatone_bank_cfgs)

        # Apply transformation if needed
        if scale == 'log':
            gt = (gt + 1).log() # Log-transform to simulate peripheral auditory processing
        elif scale == 'power':
            gt = gt ** 0.6
        elif scale == 'linear':
            gt = gt
        else:
            raise ValueError('Invalid scale.')

        # Apply the edge detector model to generate an acoustic onset spectrogram
        gt_on = eelbrain.edge_detector(gt, c = c, name = 'onset')

        # Create the gammatone envelope and onset spectrogram predictors
        # (binning the frequency axis into n_bands bands and one single band, i.e., broadband envelope)
        gt_nband = gt.bin(nbins = n_bands, func = 'sum', dim = 'frequency')
        gt_nband = gt_nband.bin(1/eeg_sr, dim = 'time', label = 'start')

        gt_1band = gt.sum('frequency')
        gt_1band = gt_1band.bin(1/eeg_sr, dim = 'time', label = 'start')

        gt_on_nband = gt_on.bin(nbins = n_bands, func = 'sum', dim = 'frequency')
        gt_on_nband = gt_on_nband.bin(1/eeg_sr, dim = 'time', label = 'start')

        gt_on_1band = gt_on.sum('frequency')
        gt_on_1band = gt_on_1band.bin(1/eeg_sr, dim = 'time', label = 'start')

        # Save predictors to pickle
        eelbrain.save.pickle(gt_nband, os.path.join(outpath, f'{stim_id}~gammatone-{n_bands}.pickle'))
        eelbrain.save.pickle(gt_1band, os.path.join(outpath, f'{stim_id}~gammatone-1.pickle'))
        eelbrain.save.pickle(gt_on_nband, os.path.join(outpath, f'{stim_id}~gammatone-on-{n_bands}.pickle'))
        eelbrain.save.pickle(gt_on_1band, os.path.join(outpath, f'{stim_id}~gammatone-on-1.pickle'))

        # Save raw gammatone spectrograms if needed
        if save_raw_gt_spec:
            p = os.path.join(outpath, 'raw_gammatone_specs')
            if not os.path.exists(p):
                os.makedirs(p)
            eelbrain.save.pickle(gt, os.path.join(p, f'{stim_id}~raw-gammatone-spectrogram.pickle'))
            eelbrain.save.pickle(gt_on, os.path.join(p, f'{stim_id}~raw-gammatone-onset-spectrogram.pickle'))

    print('All done.')

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
    make_gammatone_predictors(config)