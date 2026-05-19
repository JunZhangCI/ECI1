import os
import time
import random
import shutil
import eelbrain
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from .utils import resolve_path, resolve_param

def check_duration_mode(eeg):
    dur_mode = list(set(eeg['duration_mode']))
    if len(dur_mode) != 1:
        raise ValueError('Multiple duration modes found in the EEG data. Please check the EEG data and ensure that all have the same duration mode.')
    dur_mode = dur_mode[0]
    if dur_mode not in ['fixed', 'min']:
        raise ValueError('Invalid duration mode found in the EEG data. Currently, it should be either "fixed" or "min".')
    tstop = eeg['eeg'].time.tstop
    return dur_mode, tstop

def get_predictors(predictor_set_path, model, stims, tstop = None):
    """
    Load in predictors of a specified model.
    """
    model_preds = []
    for predictor in model:
        predictor_type = predictor.split('-')[0]
        all_preds = []
        for stim in stims:
            pred = eelbrain.load.unpickle(os.path.join(predictor_set_path, predictor_type, f'{stim}~{predictor}.pickle'))
            if tstop is not None:
                pred = pred.sub(time = (0, tstop))
            all_preds.append(pred)
        all_preds = eelbrain.combine(all_preds, 'case')
        model_preds.append(all_preds)
    return model_preds

def check_time_match(eeg, predictors):
    """
    Check if the time points of the EEG data and predictors match.
    """
    eeg_time_nsamples = eeg['eeg'].time.nsamples
    match = []
    for pred in predictors:
        pred_times_nsamples = pred.time.nsamples
        match.append(eeg_time_nsamples == pred_times_nsamples)
    return all(match)

def estimate_trfs(
    *,
    eeg_path,
    predictor_set_path,
    results_path,
    base_model,
    estimation_mode,
    boosting_cfgs,
    random_seed,
    parallelproc = False,
    data_file = None
    ):
    """
    Actual function running TRF model.
    """
    start_time = time.time()

    # Create results folder it doesn't exist
    os.makedirs(results_path, exist_ok = True)

    # Check if we're running in multiprocessing mode
    if parallelproc and (data_file is not None):
        selected_data_files = [data_file]
    else:
        print('Not using parallel processing.')
        # Select data files for which TRFs will estimated
        selected_data_files = [os.path.basename(f) for f in list(eeg_path.glob('*.pickle'))]

    # Prefix for the printed message
    if parallelproc and (len(selected_data_files) == 1):
        prefix = selected_data_files[0].split('.')[0]
        prefix = f'[{prefix}] '
    else:
        prefix = ''
    print(prefix + 'No. data files to run:', len(selected_data_files))

    # Loop through all selected data files
    for df in selected_data_files:

        random.seed(random_seed)
        np.random.seed(random_seed)

        results_path_df = os.path.join(results_path, df.split('.')[0])
        if not os.path.exists(results_path_df):
            os.makedirs(results_path_df)

        # Load EEG and trim/filter/resample if needed
        eeg = eelbrain.load.unpickle(os.path.join(eeg_path, df))

        # Use this function to check how the duration of EEG was handled during preparation. 
        # This will determine how we handle predictors during TRF estimation.
        dur_mode, tstop = check_duration_mode(eeg)

        # Get predictors for the base model and check if they match with the EEG data along the time dimension.
        stims = [x.split('.')[0] for x in list(eeg['stim'])] # Remove the extension from the stim names to match with predictor file names
        base_model_preds = get_predictors(predictor_set_path = predictor_set_path,
                                        model = base_model, 
                                        stims = stims,
                                        tstop = tstop)
        all_match = check_time_match(eeg, base_model_preds)
        if not all_match:
            print(f'  ** At least one predictor does not match the EEG data ({df}; {eeg["eeg"].time.nsamples} samples) along the time dimension. Skip this file. **')
            continue

        # Estimate base TRF model
        print(prefix + 'Estimating TRFs for base model...')
        base_boosting_res = eelbrain.boosting(y = eeg['eeg'], x = base_model_preds, **boosting_cfgs)
        eelbrain.save.pickle(base_boosting_res, os.path.join(results_path_df, 'BoostingResults-base.pickle'))
        
        # If estimation mode is 'unique_contribution', estimate additional models each leaving out one predictor from the base model. 
        # If estimation mode is 'base_only', skip this step.
        if estimation_mode == 'unique_contribution':
            test_models = {p: [x for x in base_model if x != p] for p in base_model}
            test_models_preds = {}
            for x in test_models:
                test_model = test_models[x]
                test_models_preds[x] = get_predictors(predictor_set_path = predictor_set_path, 
                                                    model = test_model,
                                                    stims = stims,
                                                    tstop = tstop)
            print(prefix + '\nEstimating TRFs for test models, each time leaving out one predictor from the base model...')
            for test_model_name, test_preds in tqdm(test_models_preds.items(), desc = prefix):
                test_boosting_res = eelbrain.boosting(y = eeg['eeg'], x = test_preds, **boosting_cfgs)
                eelbrain.save.pickle(test_boosting_res, os.path.join(results_path_df, f'BoostingResults-test_excl-{test_model_name}.pickle'))
        elif estimation_mode == 'base_only':
            print(prefix + 'Done estimating the base model.')
        else:
            raise ValueError('Invalid model estimation mode.')
        
        exec_time = time.time() - start_time
        print(prefix + f'Finished estimating all models for {df} ({exec_time:.2f} seconds).\n')

def estimate_trfs_task(args):
    result = estimate_trfs(**args)
    return result

def get_max_workers():
    # Detect the maximum usable worker count from the environment
    slurm_cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_cpus_on_node = os.environ.get('SLURM_CPUS_ON_NODE')
    if slurm_cpus_per_task is not None:
        max_workers = int(slurm_cpus_per_task)
    elif slurm_cpus_on_node is not None:
        max_workers = int(slurm_cpus_on_node)
    else:
        max_workers = os.cpu_count() or 1
    return max_workers

def run_boosting(
    config = None,
    *,
    parallelproc = None,
    eeg_path = None,
    predictor_set_path = None,
    results_path = None,
    base_model = None,
    estimation_mode = None,
    boosting_cfgs = None,
    random_seed = None,
    n_workers = None
    ):
    """
    Run boosting for TRF estimation. Accept either a configuration file or keyword arguments. Note that keyword arguments override config values.

    Parameters
    ----------
    config : Config or None
        Config object with a .data attribute.
    parallelproc : bool or None
        Whether to use parallel processing.
    eeg_path : str or Path or None
        Path to prepared EEG pickle files.
    predictor_set_path : str or Path or None
        Path to the predictor set directory.
    results_path : str or Path or None
        Output directory for boosting results.
    base_model : list[str] or None
        List of predictors included in the base TRF model.
    estimation_mode : str or None
        Estimation mode: 'base_only' or 'unique_contribution'.
    boosting_cfgs : dict or None
        Keyword arguments passed to ``eelbrain.boosting``.
    random_seed : int or None
        Random seed used for reproducibility.
    n_workers : int or None
        Number of workers for parallel processing. If None, use the maximum available workers.
    """
    # Load paths and variables from config or kwargs
    cfgs = config.data if config is not None else {}
    parallelproc = resolve_param(parallelproc, cfgs, ['TRF_estimation', 'use_parallel_processing'])
    eeg_path = resolve_path(resolve_param(eeg_path, cfgs, ['TRF_estimation', 'eeg_path']))
    predictor_set_path = resolve_path(resolve_param(predictor_set_path, cfgs, ['TRF_estimation', 'predictor_set_path']))
    results_path = resolve_path(resolve_param(results_path, cfgs, ['TRF_estimation', 'results_path']))
    base_model = resolve_param(base_model, cfgs, ['TRF_estimation', 'base_model'])
    estimation_mode = resolve_param(estimation_mode, cfgs, ['TRF_estimation', 'estimation_mode'])
    boosting_cfgs = resolve_param(boosting_cfgs, cfgs, ['TRF_estimation', 'boosting_cfgs'])
    random_seed = resolve_param(random_seed, cfgs, ['random_seed'])
    n_workers = resolve_param(n_workers, cfgs, ['TRF_estimation', 'n_workers'], required = False, default = None)

    os.makedirs(results_path, exist_ok = True)
    if config is not None and hasattr(config, 'path'):
        shutil.copyfile(config.path, os.path.join(results_path, os.path.basename(config.path)))
        print(f'Using configuration file: {config.path} (copied to {results_path})')
    else:
        print('Running without a configuration file; using resolved keyword arguments.')

    if parallelproc:
        max_workers = get_max_workers()

        # Resolve final n_workers
        if n_workers is None:
            n_workers = max_workers
        elif isinstance(n_workers, int) and n_workers > 0:
            n_workers = min(n_workers, max_workers)
        else:
            raise ValueError(
                "Invalid number of workers specified. "
                "It should be a positive integer or None."
            )

        print(
            f'Using parallel processing with {n_workers} workers '
            f'(max available: {max_workers}).'
        )

        data_files = [os.path.basename(f) for f in list(eeg_path.glob('*.pickle'))]

        param_list = []
        for df in data_files:
            args_mp = {
                'eeg_path': eeg_path,
                'predictor_set_path': predictor_set_path,
                'results_path': results_path,
                'base_model': base_model,
                'estimation_mode': estimation_mode,
                'boosting_cfgs': boosting_cfgs,
                'random_seed': random_seed,
                'parallelproc': True,
                'data_file': df,
            }
            param_list.append(args_mp)

        with Pool(processes = n_workers) as pool:
            result = pool.map(estimate_trfs_task, param_list)

    else:
        estimate_trfs(
            eeg_path = eeg_path,
            predictor_set_path = predictor_set_path,
            results_path = results_path,
            base_model = base_model,
            estimation_mode = estimation_mode,
            boosting_cfgs = boosting_cfgs,
            random_seed = random_seed,
            parallelproc = False,
            data_file = None,
        )