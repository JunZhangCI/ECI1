import os
import mne
import glob
import copy
import scipy
import mat73
import pickle
import numbers
import eelbrain
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.formula.api import ols
from mne.viz import plot_topomap
from scipy.spatial.distance import pdist
from typing import Optional, Union, Tuple, Callable, Dict, List, Any

from neuraspeech.utils import parse_bids

from .config import *

# Type hints for .get_values()
TimeInput = Union[
    float,
    int,
    Tuple[float, float],
    List[Union[float, int, Tuple[float, float]]]
]

class DatasetBase:
    def __init__(self):
        self.data: Optional[eelbrain.Dataset] = None
        self.val_col: Optional[str] = None

    def subset_dataset(self,
                       filters: Dict[str, List[str]], 
                       exclude: bool = False, 
                       return_copy: bool = False, 
                       data_to_subset: Optional[eelbrain.Dataset] = None,
                       sort_by: Optional[str] = None) -> Optional[eelbrain.Dataset]:

        # If no external data (data_to_subset) is supplied, will copy data from self.data
        if data_to_subset is None:
            if self.data is None:
                raise ValueError("No data to subset.")
            new_data = self.data.copy()
        else:
            new_data = data_to_subset.copy()

        # Apply filters. If exclude = True, then EXCLUDE the levels in the list for each factor
        for f in filters:
            keep_or_exclude = filters[f]
            tmp = []
            for x in keep_or_exclude:
                if exclude:
                    new_data = new_data.sub(f"{f} != '{x}'")
                else:
                    subset = new_data.sub(f"{f} == '{x}'")
                    tmp.append(subset)
            if not exclude:
                new_data = eelbrain.combine(tmp)

        # Try sorting by filename or any factor specified by the user
        if sort_by is None:
            if 'filename' in new_data.keys():
                new_data.sort('filename')
        else:
            new_data.sort(sort_by)

        if new_data.n_cases <= 0:
            new_data = None
            warnings.warn("No data left after subsetting. Setting data to None.")
        
        # Return a subsetted data without chaning self.data or assign new, filtered data to self.data
        if return_copy:
            return new_data
        else:
            self.data = new_data

    def subset_sensors(self,
                        var: str,
                        subset: List[str], 
                        exclude: bool = False, 
                        return_copy: bool = False, 
                        data_to_subset: Optional[eelbrain.Dataset] = None) -> Optional[eelbrain.Dataset]:
        
        # If no external data (data_to_subset) is supplied, will copy data from self.data
        if data_to_subset is None:
            if self.data is None:
                raise ValueError("No data to subset.")
            new_data = self.data.copy()
        else:
            new_data = data_to_subset.copy()

        # Extract the NDVar
        new_ndvar = new_data[var]

        if exclude:
            sensors_to_keep = [x for x in new_ndvar.sensor.names if x not in subset]
        else:
            sensors_to_keep = subset[:]

        new_ndvar = new_ndvar.sub(sensor = sensors_to_keep)

        n_sensors = len(new_ndvar.sensor.names)
        if n_sensors <= 0:
            new_ndvar = None
            warnings.warn("No data left after subsetting. Setting data to None.")

        new_data[var] = new_ndvar

        # Return a subsetted data without changing self.data or assign new, filtered data to self.data
        if return_copy:
            return new_data
        else:
            self.data = new_data

    def recode(self, var: str, 
               replacements: dict[str, str]) -> None:
        """
        Recode categorical values in a column of the dataset.

        Parameters:
        - var: Name of the variable/column to recode.
        - replacements: Dictionary mapping old values (keys) to new values (values). Only values found in the dictionary will be changed.
        """
        col = self.data[var]

        if isinstance(col, eelbrain.Factor):
            # Build new labels per case, then rebuild Factor (re-encodes codes cleanly)
            new_labels = [replacements.get(str(v), str(v)) for v in col]
            self.data[var] = eelbrain.Factor(new_labels)
        else:
            # Non-Factor column: fall back to your original behavior
            for i in range(self.data.n_cases):
                value = self.data[i, var]
                new_value = replacements.get(value)
                if new_value is not None:
                    self.data[i, var] = new_value
    
    def get_data(self) -> eelbrain.Dataset:
        return self.data.copy()
    
    def get_values(
        self,
        times: TimeInput,
        subset_sensors: Optional[List[str]] = None,
        summarize_sensors: bool = True,
        summarize_time: bool = True,
        summary_func: str = 'mean',
        output_col_labels: Optional[List[str]] = None,
        return_dataframe: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Extract values from an NDVar at one or more time points or time windows.

        Parameters:
        - times: Time specification. Can be a single time point (float or int), a single time window as a tuple (start, end), or a list containing time points and/or time windows.
        - subset_sensors: Optional list of sensors to include.
        - summarize_sensors: If True, summarize values across sensors using summary_func. If False, return values separately for each sensor.
        - summarize_time: If True, summarize values within each time window using summary_func. If False, return values separately for each sampled time point within the window.
        - summary_func: Summary function to apply. Currently support: 'mean', 'max', 'min', or 'sum'.
        - output_col_labels: Custom labels for columns of the extracted value in the output dataframe.
        - return_dataframe: If True, return the output as a pandas DataFrame. If False, add the extracted values to self.data and return None.

        Returns:
        - A pandas DataFrame if return_dataframe is True; otherwise None.
        """
        if self.val_col is None:
            raise ValueError("This class does not define `val_col`.")

        return self._get_values_from_col(
            val_col=self.val_col,
            times=times,
            subset_sensors=subset_sensors,
            summarize_sensors=summarize_sensors,
            summarize_time=summarize_time,
            summary_func=summary_func,
            output_col_labels=output_col_labels,
            return_dataframe=return_dataframe
        )

    def _get_values_from_col(
            self,
            val_col: str,
            times: TimeInput,
            subset_sensors: Optional[List[str]] = None,
            summarize_sensors: bool = True,
            summarize_time: bool = True,
            summary_func: str = 'mean',
            output_col_labels: Optional[List[str]] = None,
            return_dataframe: bool = True
        ) -> Optional[pd.DataFrame]:
            
            if self.data is None:
                raise ValueError("No data available.")

            # Make a copy of the data first
            new_data = copy.deepcopy(self.data)

            # Extract NDVar
            val_ndvar = new_data[val_col]

            if subset_sensors is not None:
                val_ndvar = val_ndvar.sub(sensor=subset_sensors)

            sensors = val_ndvar.sensor.names
            times = self._normalize_times(times)

            if output_col_labels is not None:
                assert len(times) == len(output_col_labels), 'Length of `output_col_labels` must match length of `times` (as a list).'

            for i, t in enumerate(times):

                # Use the custom output_col_labels if not None
                if output_col_labels is not None:
                    col_label = output_col_labels[i]
                else:
                    col_label = f'x{i}'

                # Single time point
                if self._is_scalar_time(t):
                    val_ndvar_t = val_ndvar.sub(time=t)

                    if summarize_sensors:
                        new_data[col_label] = self._summarize(
                            val_ndvar_t, summary_func, dim='sensor'
                        )
                    else:
                        for s in sensors:
                            new_data[f'{col_label}_{s}'] = val_ndvar_t.sub(sensor=s)

                # Time window
                elif self._is_time_window(t):
                    start, end = t
                    val_ndvar_t = val_ndvar.sub(time=(start, end))

                    if summarize_time:
                        val_ndvar_t = self._summarize(
                            val_ndvar_t, summary_func, dim='time'
                        )

                        if summarize_sensors:
                            new_data[col_label] = self._summarize(
                                val_ndvar_t, summary_func, dim='sensor'
                            )
                        else:
                            for s in sensors:
                                new_data[f'{col_label}_{s}'] = val_ndvar_t.sub(sensor=s)

                    else:
                        time_arr = val_ndvar_t.time.times
                        for j, tt in enumerate(time_arr):
                            val_ndvar_tt = val_ndvar_t.sub(time=tt)

                            if summarize_sensors:
                                new_data[f'{col_label}_t{j}'] = self._summarize(
                                    val_ndvar_tt, summary_func, dim='sensor'
                                )
                            else:
                                for s in sensors:
                                    new_data[f'{col_label}_t{j}_{s}'] = val_ndvar_tt.sub(sensor=s)

                else:
                    raise ValueError("Some elements in `times` are invalid.")

            if return_dataframe:
                return new_data.as_dataframe()
            else:
                self.data = new_data
                print('Adding extracted values to the data object.')
                return None

    def save(self, path: str) -> None:
        """
        Save internal attributes to a pickle file.
        """
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved to {path}")

    def load(self, path: str) -> None:
        """
        Load internal attributes from a pickle file.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        print(f"Loaded attributes from {path}")

    @classmethod
    def from_pickle(cls, path: str):
        obj = cls.__new__(cls)  # create instance without calling __init__
        obj.load(path)
        return obj

    def _summarize(
        self,
        ndvar: eelbrain.NDVar,
        summary_func: str,
        dim: str
    ) -> eelbrain.NDVar:
        """
        Summarize an NDVar across a specified dimension.

        Parameters:
        - ndvar: The NDVar to summarize.
        - summary_func: Summary function to apply. Must be one of 'mean', 'max',
          'min', or 'sum'.
        - dim: Name of the dimension over which to summarize.

        Returns:
        - A summarized NDVar.
        """
        if summary_func == 'mean':
            out = ndvar.mean(dim)
        elif summary_func == 'max':
            out = ndvar.max(dim)
        elif summary_func == 'min':
            out = ndvar.min(dim)
        elif summary_func == 'sum':
            out = ndvar.sum(dim)
        else:
            raise ValueError("Invalid summary function.")
        return out

    def _is_scalar_time(self, x: object) -> bool:
        """
        Check whether an object is a valid scalar time value.

        Parameters:
        - x: Object to check.

        Returns:
        - True if x is a real number (but not a bool), otherwise False.
        """
        return isinstance(x, numbers.Real) and not isinstance(x, bool)

    def _is_time_window(self, x: object) -> bool:
        """
        Check whether an object is a valid time window.

        Parameters:
        - x: Object to check.

        Returns:
        - True if x is a tuple of two scalar time values, otherwise False.
        """
        return (
            isinstance(x, tuple)
            and len(x) == 2
            and all(self._is_scalar_time(xx) for xx in x)
        )

    def _normalize_times(
        self,
        times: TimeInput
    ) -> List[Union[float, Tuple[float, float]]]:
        """
        Normalize time input into a standard internal format.

        Parameters:
        - times: A single time point, a single time window, or a list containing
          time points and/or time windows.

        Returns:
        - A list whose elements are either floats or (float, float) tuples.
        """
        if self._is_scalar_time(times):
            return [float(times)]

        if self._is_time_window(times):
            start, end = float(times[0]), float(times[1])
            if start > end:
                raise ValueError("Time window start must be <= end.")
            return [(start, end)]

        if isinstance(times, list):
            out: List[Union[float, Tuple[float, float]]] = []
            for t in times:
                if self._is_scalar_time(t):
                    out.append(float(t))
                elif self._is_time_window(t):
                    start, end = float(t[0]), float(t[1])
                    if start > end:
                        raise ValueError("Time window start must be <= end.")
                    out.append((start, end))
                else:
                    raise ValueError(
                        "Each element in `times` must be either a number "
                        "or a tuple like (start, end)."
                    )
            return out

        raise ValueError(
            "`times` must be a number, a tuple (start, end), "
            "or a list containing numbers and/or tuples."
        )

    def _plot_time_course(self,
                          split_by: List[str],
                          line_by: Optional[str] = None,
                          var: str = None,
                          subset_dataset: Optional[Dict[str, List[str]]] = None,
                          subset_dataset_exlcude: bool = False,
                          subset_sensors: Optional[List[str]] = None,
                          subset_sensors_exclude: bool = False,
                          equal_y_scale: Union[bool, str] = False,
                          line_colors: Optional[Dict[str, str]] = None,
                          mark: Optional[Dict[str, Any]] = None,
                          title_prefix: Optional[str] = None,
                          show_legend: bool = True,
                          show_topo: bool = True,
                          fig_path: Optional[str] = None,
                          **plot_kwargs):
        """
        Plot NDVar data in a multi-panel grid.

        Parameters:
        - split_by: factors used to define each subplot (e.g., ['sub', 'cond'])
        - line_by: a factor whose levels are shown as separate lines in each plot
        - var: name of the NDVar variable in self.data
        - equal_y_scale: False (no equalization), True (global equal), or a str (equal within levels of a specific factor)
        - line_colors: a dictionary specifying the color for each line in the line_by factor
        - mark: Optional dict with keys:
            - 'time': list of floats (time points to mark)
            - 'label': list of strings (same length as 'time')
            - 'line_color': line color (default = 'gray')
            - 'label_color': label color (default = 'black')
            - 'alpha': transparency (default = 0.7)
            - fontsize: fontsize of the label
        - title_prefix: optional string prefix for subplot titles
        - plot_kwargs: passed to the Eelbrain NDVar.plot() method
        """
        if self.data is None:
            raise ValueError("No data to plot.")
        
        # Subset data
        if subset_dataset is not None:
            ds = self.subset_dataset(filters = subset_dataset, exclude = subset_dataset_exlcude, return_copy = True)
        else:
            ds = self.data.copy()

        ndvar_copy = ds[var].copy() # Keep a copy for the topo plot later

        if subset_sensors is not None:
            ds = self.subset_sensors(
                var = var,
                subset = subset_sensors, 
                exclude = subset_sensors_exclude, 
                data_to_subset = ds, 
                return_copy = True)
            included_sensors_msg = f'Included sensors: {list(ds[var].sensor.names)}'
        else:
            included_sensors_msg = 'Included sensors: all'
        included_sensors = list(ds[var].sensor.names)
        print(included_sensors_msg)

        # Determine aggregation
        group_factors = split_by + ([line_by] if line_by else [])
        agg_by = ' % '.join(group_factors)
        drop_vars = [x for x in ds.keys() if x not in group_factors + [var]]
        ds = ds.aggregate(agg_by, drop=drop_vars)

        # Get levels
        levels = {f: ds[f].cells for f in split_by}
        n_levels = {f: len(v) for f, v in levels.items()}

        # Layout: use factor with most levels as columns
        col_factor = max(n_levels, key=n_levels.get)
        row_factors = [f for f in split_by if f != col_factor]

        n_cols = n_levels[col_factor]
        n_rows = int(np.prod([n_levels[f] for f in row_factors])) if row_factors else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

        # Track y-lims for optional equal_y_scale
        y_lims = {}

        # Initialized legend handles
        legend_handles = []
        legend_labels = []

        row_keys_all = list(product(*[levels[f] for f in row_factors])) if row_factors else [()]
        for i, row_keys in enumerate(row_keys_all):
            for j, col_key in enumerate(levels[col_factor]):
                cond = {f: k for f, k in zip(row_factors, row_keys)} if row_factors else {}
                cond[col_factor] = col_key
                sub_ds = ds.copy()
                for k, v in cond.items():
                    sub_expr = f"{k} == '{v}'"
                    sub_ds = sub_ds.sub(sub_expr)

                ax = axes[i, j]

                if sub_ds.n_cases == 0:
                    ax.axis('off')
                    continue
                
                for _, line_by_level in enumerate(sub_ds[line_by]):
                    tmp = sub_ds.sub(f"{line_by} == '{line_by_level}'")
                    time = tmp[var].time
                    time_arr = np.linspace(time.tmin, time.tmax, time.nsamples)
                    y = tmp[var].get_data()
                    if y.shape[-1] > 1: # If there are more than 1 sensor
                        y = np.mean(y, axis = -1) # Average across the sensor dimension
                    y = np.squeeze(y) # Squeeze out dimensios with length 1

                    if line_colors is not None:
                        line, = ax.plot(time_arr, y, label = line_by_level, color = line_colors.get(line_by_level), **plot_kwargs)
                    else:
                        line, = ax.plot(time_arr, y, label = line_by_level, **plot_kwargs)
                    if line_by_level not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(line_by_level)

                
                # Mark vertical lines
                if mark is not None:
                    times = mark.get('time', [])
                    labels = mark.get('label', [])
                    line_color = mark.get('line_color', 'gray')
                    label_color = mark.get('label_color', 'black')
                    fontsize =  mark.get('fontsize', 12)
                    alpha = mark.get('alpha', 0.7)

                    if len(times) != len(labels):
                        raise ValueError("mark['time'] and mark['label'] must have the same length.")

                    for t, label in zip(times, labels):
                        ax.axvline(t, color = line_color, alpha = alpha, linestyle = '-')
                        ax.text(t, 1.01, label,
                                transform = ax.get_xaxis_transform(),
                                color = label_color,
                                fontsize = fontsize, ha = 'center', va = 'bottom')
                        
                title = f"{title_prefix or ''} " + ' | '.join([f"{k}={v}" for k, v in cond.items()])
                ax.set_xlabel('Time (s)', fontsize = 14)

                if mark:
                    label_fontsize = mark.get('fontsize', 12)
                    ax.set_title(title.strip(), pad = label_fontsize + 6, fontsize = 14)
                else:
                    ax.set_title(title.strip(), fontsize = 14)

                # Only track y-limits for non-empty plots
                if equal_y_scale:
                    if isinstance(equal_y_scale, str):
                        key = cond[equal_y_scale]
                    else:
                        key = 'all'
                    cur_ylim = ax.get_ylim()
                    if key not in y_lims:
                        y_lims[key] = [cur_ylim[0], cur_ylim[1]]
                    else:
                        y_lims[key][0] = min(y_lims[key][0], cur_ylim[0])
                        y_lims[key][1] = max(y_lims[key][1], cur_ylim[1])
                
        # Apply equal y-scaling
        if equal_y_scale:
            for i, row_keys in enumerate(row_keys_all):
                for j, col_key in enumerate(levels[col_factor]):
                    cond = {f: k for f, k in zip(row_factors, row_keys)} if row_factors else {}
                    cond[col_factor] = col_key
                    key = cond[equal_y_scale] if isinstance(equal_y_scale, str) else 'all'
                    axes[i, j].set_ylim(y_lims[key])

        fig.tight_layout()

        if show_legend and legend_handles:
            fig.legend(legend_handles, legend_labels,
                       loc = 'lower center', bbox_to_anchor = (0.5, -0.07),
                       ncol = len(legend_labels), frameon = False, prop = {'size': 14})
            fig.subplots_adjust(bottom = 0.2)  # add space for the legend

        if fig_path is not None:
            plt.savefig(fig_path, bbox_inches = 'tight')

        plt.show()

        if show_topo:
            topo_ndvar = eelbrain.NDVar(np.zeros(ndvar_copy.x.shape[-1]), ndvar_copy.sensor)
            eelbrain.plot.Topomap(topo_ndvar, mark = included_sensors, 
                                  mcolor = '#F2545B', clip = 'circle', msize= 20)
    
    def _plot_topomap(self,
                  split_by: List[str],
                  times: List[float],
                  var: str,
                  subset_dataset: Optional[Dict[str, List[str]]] = None,
                  subset_dataset_exlcude: bool = False,
                  mark: Optional[List[str]] = None,
                  title_prefix: Optional[str] = None,
                  vlim: Tuple = (0.0, 0.5),
                  show_colorbar: bool = True,
                  fig_path: Optional[str] = None,
                  **plot_kwargs):
        """
        Plot NDVar topomaps arranged in a grid.

        Parameters:
        - split_by: list of factor names for rows (combinations will form each row)
        - times: list of time points (in seconds) for columns
        - var: name of the NDVar variable in self.data
        - vlim: color scale of the topomap
        - mark: list of sensor names to highlight with yellow circle
        - plot_kwargs: extra args passed to eelbrain.plot.Topomap
        """
        if self.data is None:
            raise ValueError("No data to plot.")

        # Subset data
        if subset_dataset is not None:
            ds = self.subset_dataset(filters = subset_dataset, exclude = subset_dataset_exlcude, return_copy = True)
        else:
            ds = self.data.copy()

        # Determine grid layout
        row_keys_all = list(product(*[ds[f].cells for f in split_by]))
        n_rows = len(row_keys_all)
        n_cols = len(times)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

        images = []

        for i, row_keys in enumerate(row_keys_all):
            cond = {f: v for f, v in zip(split_by, row_keys)}
            sub_ds = ds.copy()
            for k, v in cond.items():
                sub_expr = f"{k} == '{v}'"
                sub_ds = sub_ds.sub(sub_expr)

            if sub_ds.n_cases == 0:
                for j in range(n_cols):
                    axes[i, j].axis('off')
                continue

            for j, t in enumerate(times):
                ax = axes[i, j]
                avg = sub_ds[var].mean('case')
                ndvar = avg.sub(time=t)

                # Get data and sensor positions
                data = ndvar.x  # shape (n_sensors,)
                sensor_pos = np.array([ndvar.sensor.locs[s_idx] for s_idx, s in enumerate(ndvar.sensor.names)])[:, :2]

                # Plot the topomap
                im, _ = plot_topomap(data, sensor_pos, axes=ax, show=False, vlim = vlim, **plot_kwargs)
                images.append(im)

                # Mark sensors (if provided)
                if mark:
                    for sensor in mark:
                        if sensor in ndvar.sensor.names:
                            idx = ndvar.sensor.names.index(sensor)
                            pos = sensor_pos[idx]
                            ax.plot(*pos, 'o', markersize = 8,
                                    markerfacecolor = 'yellow', markeredgecolor = 'black')

                # Title for each subplot
                ax.set_title(f"{title_prefix or ''} {', '.join(f'{k}={v}' for k, v in cond.items())}\n{t:.3f}s", fontsize = 14)

        # Show shared colorbar
        if show_colorbar:
            # Shorter and lower bar
            cbar_ax = fig.add_axes([0.35, 0.02, 0.25, 0.02])  # [left, bottom, width, height]
            cbar = plt.colorbar(images[0], cax=cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize = 14)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, wspace=0.05, hspace=0.4)

        if fig_path is not None:
            plt.savefig(fig_path, bbox_inches = 'tight')

        plt.show()

class PRPData(DatasetBase):
    def __init__(self,
                 data_path: Optional[str] = None,
                 sr: int = 128,
                 time_win: Tuple[float, float] = (0.0, 0.5),
                 eeg_fieldname: str = 'eegs',
                 phoneme_fieldname: str = 'phones',
                 montage_path: Optional[str] = None,
                 add_phoneme_manner: Union[bool, Dict[str, List[str]]] = True,
                 metadata_fn: Optional[Callable[[str], Dict]] = None) -> None:
        super().__init__()
        self.data_path: Optional[str] = data_path
        self.sr: int = sr
        self.time_win: Tuple[float, float] = time_win
        self.eeg_fieldname: str = eeg_fieldname
        self.phoneme_fieldname: str = phoneme_fieldname
        self.montage_path: Optional[str] = montage_path
        self.add_phoneme_manner: Union[bool, Dict[str, List[str]]] = add_phoneme_manner
        self.metadata_fn: Callable[[str], Dict] = metadata_fn or parse_bids

        if time_win[0] >= time_win[1]:
            raise ValueError(f"Invalid time window: {time_win}. Start must be < end.")
        
        if self.montage_path is None:
            warnings.warn("Montage is not supplied. Will create a dummy montage for PRP data and topomaps will not be shown.")

        # If add_phoneme_manner is True, will add manner class for each phoneme using the default phoneme-to-manner mapping from config
        if add_phoneme_manner is True:
            self.phoneme2manner = PHON2MANNER.copy()
        # Alternatively, the user can supply a custom "manner-to-phoneme" mapping and the manner information will be added accordingly
        elif isinstance(add_phoneme_manner, dict):
            self.phoneme2manner = self._make_custom_phon2man_mapping(self.add_phoneme_manner)
            self.add_phoneme_manner = True
        else:
            self.phoneme2manner = None
            self.add_phoneme_manner = False
        
        # Compute PRPs
        self.val_col = 'PRP'
        self.data = self._compute_prps()

    def _compute_prps(self) -> None:
        if self.data_path is None:
            raise ValueError("data_path is not set. Cannot load data.")

        # Get all .mat files in data_path and compute PRPs (currently support only .mat format)
        data_files = glob.glob(self.data_path + '/*.mat')

        # Initialize a list to store all PRP datasets
        prp_datasets = []

        for df in tqdm(data_files):
            df_basename = os.path.basename(df)
            try:
                data = scipy.io.loadmat(df)
            except:
               # For v7.3 files
                data = mat73.loadmat(df)

            # Get data file meta data
            if self.metadata_fn is not None:
                info = self.metadata_fn(df_basename)
            else:
                info = None            
            
            # EEG time-locked to each phoneme instance
            eegs = data[self.eeg_fieldname]
            if eegs.ndim != 3:
                raise ValueError(f"Expected 3D EEG array (n_phoneme_instances, n_channels, n_timepoints), but got shape {eegs.shape}")
            
            # Get phoneme labels and make sure that they are a 1-D numpy array
            phones = data[self.phoneme_fieldname]
            phones = np.array([x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in np.squeeze(phones)])

            # Get phoneme-related potentials
            unique_phonemes = np.unique(phones)
            prps = []
            phonemes = []
            for p in unique_phonemes:
                idx = np.where(phones == p)[0]
                prp = np.mean(eegs[idx, :, :], axis = 0) 
                phonemes.append(p)
                prps.append(prp)

            # Store as eelbrain NDVar
            prps = np.array(prps)
            prps = prps.transpose((0, 2, 1)) # (phonemes, time, channels)
            case = eelbrain.Case(prps.shape[0])
            time = eelbrain.UTS(self.time_win[0], 1.0/self.sr, prps.shape[1])

            # Make sensor
            if self.montage_path is not None:
                sensor = self._make_sensor(self.montage_path)
            else:
                sensor = self._make_dummy_sensor(n_channels = prps.shape[-1])
            prp_ndvar = eelbrain.NDVar(prps, (case, time, sensor))

            # PRP dataset for a single data file
            prp_ds = self._make_dataset(prp_ndvar, df_basename, phonemes, info)
            prp_datasets.append(prp_ds)
        
        # Combine all datasets and assign it to self.data
        try:
            out = eelbrain.combine(prp_datasets)
        except:
            warnings.warn('Some NDVars have mismatching dimensions. Retry combining the PRP datasets with dim_intersection=True to discard dimensions not present in all datasets.')
            out = eelbrain.combine(prp_datasets, dim_intersection = True)

        print('Done.')
        return out
            

    def data_summary(self) -> None:
        if self.data is None:
            print('No PRP data.')
        else:
            print('== PRP dataset summary ==')
            print(self.data.summary())
            
    def plot_manner_prps(self,
                         split_by: List[str],
                         line_by: str = 'manner',
                         line_colors: Optional[Dict[str, str]] = None,
                         mark: Optional[Dict[str, Any]] = None,
                         equal_y_scale: Union[bool, str] = False,
                         title_prefix: Optional[str] = None,
                         show_legend: bool = True,
                         fig_path: Optional[str] = None,
                         **plot_kwargs) -> None:
        """
        Plot PRPs with lines for each manner category, split across panels defined by split_by.
        """
        if self.data is None:
            raise ValueError("No PRP data to plot.")
        
        plot_kwargs.setdefault('lw', 3)

        if line_by == 'manner' and line_colors is None:
            line_colors = MANNER_COLORS
        
        self._plot_time_course(
            split_by=split_by,
            line_by=line_by,
            var=self.val_col,
            equal_y_scale = equal_y_scale,
            line_colors = line_colors,
            mark = mark,
            title_prefix = title_prefix,
            show_legend = show_legend,
            fig_path = fig_path,
            **plot_kwargs
        )
    
    def get_avg_amplitude(self, time_intervals: List[Tuple[float, float]]) -> eelbrain.Dataset:
        """
        Get the average amplitudes in a list of time intervals from the PRPs.
        """
        amplitude_data = []
        time_int_labels = []
        for i, ti in enumerate(time_intervals):
            avg_amp = self.data[self.val_col].mean(time = ti).get_data()
            amplitude_data.append(avg_amp)
            time_int_labels.append(f't{i+1}')

        amplitude_data = np.stack(amplitude_data, axis = -1)
        dims = (
            eelbrain.Case(self.data.n_cases),
            self.data[self.val_col].sensor, 
            eelbrain.Categorial('time_interval', values = time_int_labels)
            )
        amplitude_data = eelbrain.NDVar(amplitude_data, dims)
        out = self.data.copy()
        out['Amplitude'] = amplitude_data

        # Drop PRP column
        cols = [x for x in out.keys() if x != self.val_col]
        return out[cols]
    
    # def _make_default_phon2man_mapping(self) -> Dict[str, str]:
    #     # Note that by default it is assumed that the phoneme labels use the ARPAbet symbol system
    #     manner2phon = {
    #         'Vowel': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'],
    #         'Nasal-approximant': ['L', 'M', 'N', 'NG', 'R', 'W', 'Y'],
    #         'Fricative': ['DH', 'F', 'HH', 'S', 'SH', 'V', 'Z', 'TH', 'ZH'],
    #         'Stop': ['B', 'D', 'G', 'K', 'P', 'T', 'CH', 'JH', 'Q']
    #                    }
    #     # Reverse mapping from phoneme to manner
    #     phon2manner = {phoneme: manner for manner, phonemes in manner2phon.items() for phoneme in phonemes}
    #     return phon2manner
    
    def _make_custom_phon2man_mapping(self, manner2phon: Dict[str, List[str]]) -> Dict[str, str]:
        # Reverse mapping from phoneme to manner
        phon2manner = {phoneme: manner for manner, phonemes in manner2phon.items() for phoneme in phonemes}
        return phon2manner
    
    def _make_sensor(self, montage_path: str) -> eelbrain.Sensor:
        ext = montage_path.split('.')[-1]
        # Read montage depending on the file type
        if ext == 'csv':
            montage = pd.read_csv(montage_path)
            ch_names = montage['ch_name'].tolist()
            xyz = montage[['x', 'y', 'z']].to_numpy()
        elif ext == 'locs':
            montage = mne.channels.read_custom_montage(montage_path)
            ch_pos = montage.get_positions()['ch_pos']
            ch_names = [x for x in ch_pos]
            xyz = np.array([ch_pos[x] for x in ch_names])
        elif ext == 'pkl':
            with open(montage_path, 'rb') as f:
                chan_info = pickle.load(f)
                try:
                    ch_names = chan_info['ch_name']
                except:
                    try:
                        ch_names = chan_info['ch_names']
                    except:
                        raise ValueError('Invalid key for channel names in the provided montage file.')
                xyz = chan_info['xyz']
        else:
            raise ValueError('Montage path is either invalid or the montage data format is currently not supported.')
        
        sensor = eelbrain.Sensor(xyz, names = ch_names)
        return sensor
        
    def _make_dummy_sensor(self, n_channels: int) -> eelbrain.Sensor:
        xyz = np.zeros((n_channels, 3))
        ch_names = [f'e{x}' for x in range(1, n_channels + 1)]
        sensor = eelbrain.Sensor(xyz, names = ch_names)
        return sensor
    
    def _make_dataset(self, 
                       prpndvar: eelbrain.Sensor, 
                       data_filename: str,
                       phonemes: List[str], 
                       info: Optional[Dict[str, Any]]) -> eelbrain.Dataset:
        # Filename
        d = {'filename': eelbrain.Factor([data_filename] * len(prpndvar))}

        # If metadata is not None
        if info is not None:
            d.update({k: eelbrain.Factor([info[k]] * len(prpndvar)) for k in info})

        # Add phoneme, PRP data, and manner information if (self.add_phoneme_manner is True)
        if self.add_phoneme_manner:
            d.update({'phoneme': eelbrain.Factor(phonemes),
                      'manner': eelbrain.Factor([self.phoneme2manner[p] for p in phonemes]),
                      self.val_col: prpndvar})
            
        d.update({'phoneme': eelbrain.Factor(phonemes), self.val_col: prpndvar})
        d = eelbrain.Dataset(d)
        return d

    def __repr__(self) -> str:
        return (f"PRPData(data_path={self.data_path}, sr={self.sr}, "
                f"time_win={self.time_win}, eeg_fieldname='{self.eeg_fieldname}', "
                f"phoneme_fieldname='{self.phoneme_fieldname}')")

class FStatistic(DatasetBase):
    def __init__(self,
                 prp_data: PRPData,
                 by: str = 'filename',
                 category: str = 'manner') -> None:
        super().__init__()

        if prp_data.data is None:
            raise ValueError("The provided PRPData object has no data. Please run .compute_prps() first.")

        if by not in prp_data.data.keys():
            raise ValueError(f"'by' factor '{by}' not found in PRPData dataset.")
        if category not in prp_data.data.keys():
            raise ValueError(f"'category' factor '{category}' not found in PRPData dataset.")

        self.by = by
        self.category = category

        # Compute F-stat for each level of the 'by' factor
        self.val_col = 'F'
        self.data = self._compute_f_statistics(prp_data.data)

    def _compute_f_statistics(self, ds: eelbrain.Dataset, var: str = 'PRP') -> eelbrain.Dataset:
        """
        Compute F-statistic (between-category / within-category variance ratio) for each level of 'by'.
        Stores the result in self.data.
        """
        all_datasets = []

        for level in tqdm(ds[self.by].cells):
            sub_ds = ds.sub(f"{self.by} == '{level}'")
            labels = list(sub_ds[self.category])
            prp_ndvar = sub_ds[var]

            data = prp_ndvar.get_data()  # shape: (n_obs, n_time, n_chans)

            f_stats = np.zeros((data.shape[1], data.shape[2]))

            for t in range(data.shape[1]):
                for c in range(data.shape[2]):
                    df = pd.DataFrame({
                        self.category: labels,
                        'value': data[:, t, c]
                    })
                    model = ols('value ~ C(' + self.category + ')', data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    f = anova_table.F.iloc[0]
                    f_stats[t, c] = f

            case = eelbrain.Case(1)
            time_dim = prp_ndvar.get_dim('time')
            sensor_dim = prp_ndvar.get_dim('sensor')
            f_ndvar = eelbrain.NDVar(f_stats[np.newaxis, :, :], (case, time_dim, sensor_dim))

            # Get metadata from the first row (assuming consistent per level)
            meta = {k: eelbrain.Factor([sub_ds[k][0]]) for k in sub_ds.keys() if k not in [self.category, 'phoneme', var]}

            f_ds = eelbrain.Dataset({**meta, 
                                     'stat': eelbrain.Factor(['F-stat']),
                                     self.val_col: f_ndvar})
            all_datasets.append(f_ds)

        out = eelbrain.combine(all_datasets)
        print('Done.')
        return out

    def plot_f_stats(self,
                     split_by: List[str],
                     line_by: str = 'stat',
                     line_colors: Optional[str] = None,
                     mark: Optional[Dict[str, Any]] = None,
                     equal_y_scale: Union[bool, str] = False,
                     title_prefix: Optional[str] = None,
                     show_legend: bool = True,
                     fig_path: Optional[str] = None,
                     **plot_kwargs):
        """
        Plot PRPs with lines for each manner category, split across panels defined by split_by.
        """
        if self.data is None:
            raise ValueError("No F-statistic data to plot.")
        
        plot_kwargs.setdefault('lw', 3)

        if line_by == 'stat':
            if isinstance(line_colors, str):
                line_colors = {'F-stat': line_colors}
            else:
                line_colors = {'F-stat': '#F2545B'} # Default line color

        self._plot_time_course(
            split_by = split_by,
            line_by = line_by,
            var = self.val_col,
            equal_y_scale = equal_y_scale,
            line_colors = line_colors,
            mark = mark,
            title_prefix = title_prefix,
            show_legend = show_legend,
            fig_path = fig_path,
            **plot_kwargs
        )

    def plot_f_topo(self,
                split_by: List[str],
                times: List[float],
                mark: Optional[List[str]] = None,
                title_prefix: Optional[str] = None,
                vlim: Tuple = (0.0, 5.0), 
                fig_path: Optional[str] = None,
                **plot_kwargs):
        """
        Plot F-statistics as topomaps at given time points, split across panels.
        """
        self._plot_topomap(
            split_by = split_by,
            times = times,
            var = self.val_col,
            mark = mark,
            title_prefix = title_prefix,
            vlim = vlim,
            fig_path = fig_path,
            **plot_kwargs
        )

class PhonemeDistance(DatasetBase):
    def __init__(self,
                 prp_data: PRPData,
                 by: str = 'filename') -> None:
        super().__init__()

        if prp_data.data is None:
            raise ValueError("The provided PRPData object has no data. Please run .compute_prps() first.")

        if by not in prp_data.data.keys():
            raise ValueError(f"'by' factor '{by}' not found in PRPData dataset.")
        if 'phoneme' not in prp_data.data.keys():
            raise ValueError(f"'phoneme' column not found in PRPData dataset.")

        self.by = by

        # Compute phoneme distance
        self.val_col = 'D'
        self.data = self._compute_phoneme_dist(prp_data.data)

    def _compute_phoneme_dist(self, ds: eelbrain.Dataset, var: str = 'PRP') -> eelbrain.Dataset:
        """
        Compute average pairwise phoneme (Euclidean) distance.
        Stores the result in self.data.
        """
        all_datasets = []

        for level in tqdm(ds[self.by].cells):
            sub_ds = ds.sub(f"{self.by} == '{level}'")
            labels = list(sub_ds['phoneme'])
            prp_ndvar = sub_ds[var]

            data = prp_ndvar.get_data()  # shape: (n_obs, n_time, n_chans)

            dists = np.zeros((data.shape[1], data.shape[2]))

            for t in range(data.shape[1]):
                for c in range(data.shape[2]):

                    # Compute pairwise Euclidean distances
                    pairwise_dists = pdist(data[:, t, c].reshape(-1, 1), # Reshape to 2D
                                           metric='euclidean')
                    # Average distance
                    avg_d = np.mean(pairwise_dists)

                    dists[t, c] = avg_d

            case = eelbrain.Case(1)
            time_dim = prp_ndvar.get_dim('time')
            sensor_dim = prp_ndvar.get_dim('sensor')
            d_ndvar = eelbrain.NDVar(dists[np.newaxis, :, :], (case, time_dim, sensor_dim))

            # Get metadata from the first row (assuming consistent per level)
            meta = {k: eelbrain.Factor([sub_ds[k][0]]) for k in sub_ds.keys() if k not in ['manner', 'phoneme', var]}

            d_ds = eelbrain.Dataset({**meta, 
                                     'stat': eelbrain.Factor(['Phoneme-dist']),
                                     self.val_col: d_ndvar})
            all_datasets.append(d_ds)

        out = eelbrain.combine(all_datasets)
        print('Done.')
        return out
    
    def plot_dists(self,
                   split_by: List[str],
                   line_by: str = 'stat',
                   line_colors: Optional[str] = None,
                   mark: Optional[Dict[str, Any]] = None,
                   equal_y_scale: Union[bool, str] = False,
                   title_prefix: Optional[str] = None,
                   show_legend: bool = True,
                   fig_path: Optional[str] = None,
                   **plot_kwargs):
        """
        Plot phoneme distances.
        """
        if self.data is None:
            raise ValueError("No phoneme distance data to plot.")
        
        plot_kwargs.setdefault('lw', 3)

        if isinstance(line_colors, str):
            line_colors = {'Phoneme-dist': line_colors}
        else:
            line_colors = {'Phoneme-dist': '#0072C8'} # Default line color

        self._plot_time_course(
            split_by = split_by,
            line_by = line_by,
            var = self.val_col,
            equal_y_scale = equal_y_scale,
            line_colors = line_colors,
            mark = mark,
            title_prefix = title_prefix,
            show_legend = show_legend,
            fig_path = fig_path,
            **plot_kwargs
        )

    def plot_dists_topo(self,
                        split_by: List[str],
                        times: List[float],
                        mark: Optional[List[str]] = None,
                        title_prefix: Optional[str] = None,
                        vlim: Optional[Tuple] = None,
                        fig_path: Optional[str] = None,
                        **plot_kwargs):
        """
        Plot phoneme distances as topomaps at given time points, split across panels.
        """
        self._plot_topomap(
            split_by = split_by,
            times = times,
            var = self.val_col,
            mark = mark,
            title_prefix = title_prefix,
            vlim = vlim,
            fig_path = fig_path,
            **plot_kwargs
        )