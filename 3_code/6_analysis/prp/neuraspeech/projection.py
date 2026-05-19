import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import eelbrain
from typing import Optional, List, Union
import matplotlib.pyplot as plt
from .prp import PRPData, DatasetBase
from .config import *


class TemporalLDA(DatasetBase):
    """
    Apply Linear Discriminant Analysis (LDA) to PRP data across time points (per sensor),
    optionally grouped by a column (e.g., filename or group).
    """

    def __init__(self,
                 prp_data: PRPData,
                 by: Optional[str] = None,
                 category: str = 'manner',
                 n_dim: int = 2):
        super().__init__()

        if prp_data.data is None:
            raise ValueError("The provided PRPData object has no data.")

        if category not in prp_data.data.keys():
            raise ValueError(f"'{category}' not found in PRP dataset.")

        self.by = by
        self.category = category
        self.n_dim = n_dim

        self.data = self._run_temporallda(prp_data.data)

    def _run_temporallda(self, ds: eelbrain.Dataset, var: str = 'PRP') -> eelbrain.Dataset:
        """
        Run temporal LDA for each group and sensor, store LDA results and distances.
        """
        if self.by is not None:
            levels = ds[self.by].cells
        else:
            levels = [None]

        output_datasets = []

        for level in tqdm(levels):
            sub_ds = ds if level is None else ds.sub(f"{self.by} == '{level}'")
            labels = np.array(sub_ds[self.category])
            prp_ndvar = sub_ds[var]
            eeg = prp_ndvar.get_data()  # shape: (n_cases, n_time, n_sensors)
            n_cases, n_time, n_sensors = eeg.shape

            # Store results
            reduced = np.zeros((n_cases, n_sensors, self.n_dim))
            own_d = np.zeros((n_cases, n_sensors))
            other_d = np.zeros((n_cases, n_sensors))

            for s in range(n_sensors):
                X = eeg[:, :, s]  # (n_cases, n_time)
                lda = LinearDiscriminantAnalysis(n_components=self.n_dim)
                try:
                    X_lda = lda.fit_transform(X, labels)
                except Exception:
                    # In case of degenerate LDA, fill with NaNs
                    X_lda = np.full((n_cases, self.n_dim), np.nan)

                reduced[:, s, :] = X_lda

                # Compute category centroids and distances
                centroids = {
                    c: X_lda[labels == c].mean(axis=0)
                    for c in np.unique(labels)
                }

                for i, (vec, c) in enumerate(zip(X_lda, labels)):
                    if np.any(np.isnan(vec)):
                        own_d[i, s] = np.nan
                        other_d[i, s] = np.nan
                        # print('NaN found.')
                        continue

                    own = np.linalg.norm(vec - centroids[c])
                    others = np.mean([
                        np.linalg.norm(vec - centroids[c2])
                        for c2 in centroids if c2 != c
                    ])
                    own_d[i, s] = own
                    other_d[i, s] = others

            # Wrap as NDVar
            case = eelbrain.Case(n_cases)
            sensor = prp_ndvar.get_dim('sensor')
            dim = eelbrain.Categorial('lda_dim', values=[f'd{i+1}' for i in range(self.n_dim)])

            ndvar_reduc = eelbrain.NDVar(reduced, (case, sensor, dim))
            ndvar_own = eelbrain.NDVar(own_d, (case, sensor))
            ndvar_other = eelbrain.NDVar(other_d, (case, sensor))
            ndvar_ratio = ndvar_other / ndvar_own

            # Keep all metadata (excluding existing NDVars)
            meta = {
                k: sub_ds[k] for k in sub_ds.keys()
                if not isinstance(sub_ds[k], eelbrain.NDVar)
            }

            lda_ds = eelbrain.Dataset({
                **meta,
                'lda_reduc': ndvar_reduc,
                'own_dist': ndvar_own,
                'other_dist': ndvar_other,
                'other_own_ratio': ndvar_ratio
            })
            output_datasets.append(lda_ds)

        combined = eelbrain.combine(output_datasets)
        print("Done.")
        return combined

    def plot_phoneme_scatter(self,
                            x: str,
                            y: str,
                            split_by: Optional[str] = None,
                            subset_sensors: Optional[Union[str, List[str]]] = None,
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            title_prefix: Optional[str] = '',
                            save_path: Optional[str] = None,
                            show_topo: bool = True,
                            **kwargs):
        """
        Scatter plot of phonemes in 2D space (e.g., own_dist vs other_dist), grouped optionally.

        Parameters
        ----------
        x : str
            Column in self.data for x-axis (must be NDVar of shape n_rows x n_sensors).
        y : str
            Column in self.data for y-axis (same requirement).
        split_by : str or None
            Grouping variable to split panels.
        subset_sensors : str, list of str, or None
            Sensors to include; averaged if multiple.
        xlabel : str or None
            Label of the x-axis.
        ylabel : str or None
            Label of the y-axis.
        title_prefix : str
            Prefix for plot title.
        save_path : str or None
            If set, saves figure to this path.
        show_topo : bool
            Whether to show topomap of selected sensors.
        """
        import matplotlib.pyplot as plt
        from neuraspeech.config import MANNER_COLORS

        data = self.get_data()
        x_vals = data[x]
        y_vals = data[y]

        if subset_sensors is not None:
            x_vals = x_vals.sub(sensor=subset_sensors).mean('sensor')
            y_vals = y_vals.sub(sensor=subset_sensors).mean('sensor')
        else:
            x_vals = x_vals.mean('sensor')
            y_vals = y_vals.mean('sensor')

        df = data.as_dataframe()
        df['x'] = x_vals.x
        df['y'] = y_vals.x

        # Average across subjects for each phoneme
        group_cols = ['phoneme', 'manner']
        if split_by:
            group_cols.insert(0, split_by)
        df_avg = df.groupby(group_cols, as_index=False)[['x', 'y']].mean()
        df_avg = df_avg.dropna(subset = ['x', 'y'])


        panels = df_avg[split_by].unique() if split_by else [None]
        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharex=True, sharey=True)
        if n_panels == 1:
            axes = [axes]

        # Global axis limits
        x_min, x_max = df_avg['x'].min(), df_avg['x'].max()
        y_min, y_max = df_avg['y'].min(), df_avg['y'].max()
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        xlim = (x_min - margin_x, x_max + margin_x)
        ylim = (y_min - margin_y, y_max + margin_y)

        # Set axis labels
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        for ax, panel in zip(axes, panels):
            plot_df = df_avg[df_avg[split_by] == panel] if panel is not None else df_avg
            for _, row in plot_df.iterrows():
                # ax.scatter(row['x'], row['y'],
                #         color=MANNER_COLORS.get(row['manner'], 'gray'),
                #         s=50, edgecolor='black', linewidth=0.5)
                ax.text(row['x'], row['y'], row['phoneme'], 
                        color=MANNER_COLORS.get(row['manner'], 'gray'), fontsize = 10,
                        ha='center', va='center', fontweight='bold')

            title = f"{title_prefix}{panel}" if panel is not None else f"{title_prefix}"
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.axhline(0, color='gray', linewidth=1, linestyle='--')
            ax.axvline(0, color='gray', linewidth=1, linestyle='--')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

        # Optional topomap for sensor selection
        if show_topo:
            topo_ndvar = eelbrain.NDVar(np.zeros(data[x].x.shape[-1]), data[x].sensor)
            eelbrain.plot.Topomap(topo_ndvar, mark = subset_sensors, 
                                  mcolor = '#F2545B', clip = 'circle', msize= 20)
            
        return


