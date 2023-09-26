import jax

# jax.config.update("jax_platform_name", "cpu")
import optax
from jax import jit, vmap

import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde

import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import flowio

from typing import List, Dict, Tuple
import logging


### {{{                     --     calibration class     --


class Calibration:
    """
    The Calibration class handles the calibration of the fluorescence values of a given experiment
    using color controls, blank controls, and some beads of known fluorescence values.
    """

    ### {{{                           --     init     --

    def __init__(
        self,
        color_controls: Dict[Tuple[str], str],
        beads_file: str,
        reference_protein: str = 'mkate',
        reference_channel: str = 'PE_TEXAS_RED',
        beads_mef_values: Dict[str, List[float]] = SPHEROTECH_RCP_30_5a,
        channel_to_unit: Dict[str, str] = FORTESSA_CHANNELS,
        random_seed: int = 42,
        reference_channels=None,
        channels_enabled_for_unmixing: List[str] = None,
        channels_enabled_for_beads: List[str] = None,
        max_value=FORTESSA_MAX_V,
        offset=FORTESSA_OFFSET,
        remove_after_fluo_value=None,
        verbose=True,
    ):
        """
        Parameters
        ----------
        color_controls : dict[tuple[str], str] - dictionary {control protein : path to fcs or csv, or dataframe}
            e.g. {'eYFP': 'path/to/eyfp-control.fcs',
                  'eBFP': eBFPdataframe,
                  'all OR allcolor OR allcolors OR full': 'path/to/allcolor-control.fcs'
                  'blank OR empty OR inert OR cntl ': 'path/to/allcolor-control.fcs' }

        beads_file : str - path to the beads control fcs file

        reference_protein : str - name of the reference protein that will be used to align all
                            other proteins quantities to, i.e they will be expressed in the same unit
                            The channel is picked as the brightest channel of the reference protein

        reference_channel : str - name of the reference channel that will be used to convert
                            to standatd MEF units

        beads_mef_vlaues : dict[str, list[float]] - dictionary associating the unit name
            to the list of reference values for the beads in this unit
            e.g : {'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549], 'MEPE': ...}

        channel_to_unit : dict[str, str] - dictionary associating the channel name to the unit name
            e.g. {'Pacific Blue-A': 'PacBlue', 'AmCyan-A': 'MEAMCY', ...}

        random_seed : int - random seed for any resampling that might be done

        use_channels : list[str] - list of channels to use when computing protein quantity
                        from channel values (aka "bleedthrough correction").
                        None means all available channels

        max_value : int - maximum value of the fluorescence values, used to detect saturation

        offset : int - offset to add to the fluorescence values to avoid negative values

        """
        self.color_controls = color_controls
        self.beads_file = beads_file
        self.reference_protein = escape(reference_protein)
        self.reference_channel = escape(reference_channel)
        self.beads_mef_values = {escape(k): v for k, v in beads_mef_values.items()}
        self.channel_to_unit = {escape(k): escape(v) for k, v in channel_to_unit.items()}
        self.unit_to_channel = {v: k for k, v in self.channel_to_unit.items()}
        self.random_seed = random_seed
        self.fitted = False

        if channels_enabled_for_unmixing is None:
            self.channel_names = sorted(list(self.channel_to_unit.keys()))
        else:
            self.channel_names = escape(channels_enabled_for_unmixing)
            print(f'channels enabled for unmixing: {self.channel_names}')

        self.user_provided_reference_channels = (
            {escape(p): escape(c) for p, c in reference_channels.items()}
            if reference_channels is not None
            else None
        )

        self.controls = {
            astuple(escape(k)): load_to_df(v, self.channel_names) for k, v in color_controls.items()
        }

        if channels_enabled_for_beads is None:
            self.beads_channel_order = sorted(list(self.channel_to_unit.keys()))
        else:
            self.beads_channel_order = escape(channels_enabled_for_beads)
            # remove channels that are not in the channel_to_unit dict
            ignored_channels = [
                c for c in self.beads_channel_order if c not in escape(self.channel_to_unit.keys())
            ]
            if len(ignored_channels) > 0:
                print(
                    f'{ignored_channels} not in the channel_to_unit dict. Ignoring for beads calibration'
                )
                self.beads_channel_order = [
                    c for c in self.beads_channel_order if c not in ignored_channels
                ]

            print(f'channels enabled for beads: {self.beads_channel_order}')

        self.beads = load_to_df(self.beads_file, self.beads_channel_order)

        self.max_value = max_value
        self.offset = offset
        self.verbose = verbose

        VALID_BLANK_NAMES = ['BLANK', 'EMPTY', 'INERT', 'CNTL']
        VALID_ALL_NAMES = ['ALL', 'ALLCOLOR', 'ALLCOLORS', 'FULL']

        controls_order = sorted(list(self.controls.keys()))
        self.protein_names = sorted(
            list(
                set(
                    [
                        p
                        for c in controls_order
                        for p in c
                        if p not in VALID_BLANK_NAMES + VALID_ALL_NAMES
                    ]
                )
            )
        )

        assert self.reference_protein in self.protein_names, (
            f'Unknown reference protein {self.reference_protein}. '
            f'Available proteins are {self.protein_names}'
        )

        controls_values, controls_masks = [], []
        NPROTEINS = len(self.protein_names)
        has_blank, has_all = False, False
        for m in controls_order:  # m is a tuple of protein names
            vals = self.controls[m].values
            if remove_after_fluo_value is not None:
                vals = vals[np.all(vals < remove_after_fluo_value, axis=1)]

            if m in [astuple(escape(n)) for n in VALID_BLANK_NAMES]:
                base_mask = np.zeros((NPROTEINS,))
                has_blank = True
            elif m in [astuple(escape(n)) for n in VALID_ALL_NAMES]:
                base_mask = np.ones((NPROTEINS,))
                has_all = True
            else:
                for p in m:
                    assert p in self.protein_names, f'Unknown protein {p}'
                base_mask = np.array([p in m for p in self.protein_names])

            masks = np.tile(base_mask, (vals.shape[0], 1))
            controls_values.append(vals)
            controls_masks.append(masks)

        self.controls_values = jnp.vstack(controls_values)
        self.controls_masks = jnp.vstack(controls_masks)

        assert has_blank, f'BLANK control not found. Should be named any of {VALID_BLANK_NAMES}'
        assert has_all, f'ALLCOLOR control not found. Should be named any of {VALID_ALL_NAMES}'

    ##────────────────────────────────────────────────────────────────────────────}}}




    def apply_to_array(self, Y):  # returns: MEF_X, AU_X, selection_mask
        assert self.fitted, 'You must fit the calibration first'

        Y = Y - self.autofluorescence
        X_AU = self.unmix(Y)  # abundances in arbitrary units

        logX = logtransform(X_AU, self.log_scale_factor, self.offset)
        logX_refAU = self.cmap_transform(logX)  # abundances in arbitraty units of the ref proein
        logX_MEF = self.apply_bead_calibration(logX_refAU)
        X_MEF = inverse_logtransform(logX_MEF, self.log_scale_factor, self.offset)

        selection = self.filter_saturated(Y, only_in_reference_channel=False)

        return X_MEF, X_AU, selection

    def apply(
        self,
        df,
        preserve_columns=False,
        include_arbitrary_units=False,
        include_unselected=False,
    ):
        """
        Apply the calibration to a dataframe of values.
        Options:
        - preserve_columns: if True, the columns of the dataframe are preserved, and the calibrated
        values are appended to the dataframe. Otherwise, the dataframe is replaced by the calibrated
        values. If a list of columns is provided, only these columns are preserved.
        - include_arbitraty_units: if True, the calibrated values are also returned in original arbitrary units
        after bleedthrough correction. Otherwise, the calibrated values are only returned in MEF units.
        """
        odf = df.copy()
        odf = odf.reset_index(drop=True)
        print(f'Applying calibration to {len(odf)} cells')
        odf.columns = escape(list(odf.columns))
        print(f'Columns: {odf.columns}')
        odf = odf[self.channel_names]
        X, au_X, to_keep = self.apply_to_array(
            odf.values, with_extra=True
        )  # to_keep is a boolean mask for the rows
        print(f'Kept {to_keep.sum()} cells. X = {X.shape}, au_X = {au_X.shape}')
        assert X.shape[1] == len(self.protein_names)
        assert X.shape == au_X.shape, f'X.shape={X.shape}, au_X.shape={au_X.shape}'
        assert len(to_keep) == len(df)

        # xdf is the dataframe with calibrated values
        xdf = pd.DataFrame(X, columns=self.protein_names)
        if include_arbitrary_units:
            # we append au_X, with column names of fluo_proteins with _AU appended
            au_xdf = pd.DataFrame(au_X, columns=[p + '_AU' for p in self.protein_names])
            xdf = pd.concat([xdf, au_xdf], axis=1)
            print(
                f'Added {len(au_xdf.columns)} columns of arbitrary units, total rows = {len(xdf)}'
            )

        if preserve_columns:
            xdf = pd.concat([odf, xdf], axis=1)
            print(
                f'Added {len(xdf.columns) - len(odf.columns)} original columns, total rows = {len(xdf)}'
            )

        if include_unselected:
            xdf['selected'] = to_keep
            print(f'Added column "selected", total rows = {len(xdf)}')
        else:
            xdf = xdf[to_keep].reset_index(drop=True)
            print(f'Removed unselected cells, total rows = {len(xdf)}')

        return xdf

    def apply_to_cytoflow_xp(self, xp, **kwargs):
        from copy import deepcopy

        xp_copy = deepcopy(xp)
        calibrated = self.apply(xp_copy.data, preserve_columns=xp_copy.channels, **kwargs)
        xp_copy.data.loc[:, xp_copy.channels] = calibrated[xp_copy.channels]
        xp_copy.data = xp_copy.data.loc[calibrated.index]  # remove rows that were filtered out
        for col in calibrated.columns:
            if col not in xp_copy.channels:
                xp_copy.add_channel(col, calibrated[col])
        return xp_copy





##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                     --     diagnostic plots     --
def plot_bead_peaks_diagnostics(
    peaks,
    densities,
    vmat,
    observations,
    color_channels,
    max_obs=10000,
    peaks_min_x=CALIBRATION_PEAKS_MIN_X,
    peaks_max_x=CALIBRATION_PEAKS_MAX_X,
):
    """
    Plot diagnostics for the bead peaks computation:
    - An assignment chart showing the proportion of observations assigned to each bead
    - 1 plot per channel showing both bead assignment, peak detection and final standardized densities
    """

    if observations.shape[0] > max_obs:
        observations = observations[np.random.choice(observations.shape[0], max_obs, replace=False)]

    mainfig = plt.figure(constrained_layout=True, figsize=(12, 14))
    subfigs = mainfig.subfigures(1, 2, wspace=-0.3, width_ratios=[0.4, 10])
    assignment = jnp.sort(jnp.argmax(vmat, axis=1))[:, None]
    ax = subfigs[0].subplots(1, 1)
    cmap = plt.get_cmap('tab10')
    cmap.set_bad(color='white')
    centroids = [
        (jnp.arange(len(assignment))[assignment[:, 0] == i]).mean() for i in range(vmat.shape[1])
    ]
    ax.imshow(
        assignment,
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        vmin=0,
        vmax=vmat.shape[1],
        origin='lower',
    )
    for i, c in enumerate(centroids):
        ax.text(
            0, c, f'bead {i}', rotation=90, verticalalignment='center', horizontalalignment='center'
        )
    ax.set_ylabel('Observations, sorted by bead assignment')
    ax.set_xticks([])
    ax.set_yticks([0, len(assignment)])

    resolution = densities.shape[2]
    x = jnp.linspace(peaks_min_x, peaks_max_x, resolution)
    axes = subfigs[1].subplots(len(color_channels), 1, sharex=True)
    if len(color_channels) == 1:
        axes = [axes]
    weights = vmap(w_function)(x)
    beadcolors = [plt.get_cmap('tab10')(1.0 * i / 10) for i in range(10)]
    for c in range(len(color_channels)):
        ax = axes[c]
        for b in range(peaks.shape[0]):
            dens = densities[b, c]
            dens /= jnp.max(densities[b, c])
            # plot a gradient to show the confidence threshold
            ax.plot(x, dens, label=f'bead {b}', linewidth=1.25, color=beadcolors[b])
            # also plot the actual distribution
            kde = gaussian_kde(observations[:, c], bw_method=0.01)
            d2 = kde(x)
            d2 /= jnp.max(d2) * 1.25
            ax.plot(x, d2, color='k', linewidth=0.75, alpha=1, label='_nolegend_')
            ax.fill_between(x, d2, 0, color='k', alpha=0.01)

            ax.imshow(
                weights[::5][None, :],
                extent=(peaks_min_x, peaks_max_x, 0, 1.5),
                aspect='auto',
                cmap='Greys_r',
                alpha=0.05,
            )
            # vline at peak
            ax.axvline(peaks[b, c], color='k', linewidth=0.5, dashes=(3, 3))
            # write bead number
            ax.text(peaks[b, c] + 0.01, 0.3 + 0.1 * b, f'{b}', fontsize=8, ha='center', va='center')
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_xlim(peaks_min_x, 1.1)
            title = f'{color_channels[c]}'
            ax.set_ylabel(title)

        axes[0].set_title(
            'Density of observations across channel range (grey background = out of range)'
        )
        # color: density of observations per bead assignment
        # --: estimated peak position\n
        # dark curve: original distribution of observations in channel
    subfigs[1].suptitle("Bead peak diagnostics")





def plot_beads_after_correction(
    transform,
    peaks,
    calib,
    observations,
    color_channels,
    channel_to_unit,
    max_obs=10000,
    title=None,
    fname=None,
):
    from matplotlib import cm

    if observations.shape[0] > max_obs:
        key = jax.random.PRNGKey(0)
        reorder = jax.random.permutation(key, np.arange(max_obs))
        observations = observations[reorder]

    NBEADS, NCHAN = peaks.shape
    fig, axes = plt.subplots(1, NCHAN, figsize=(1.3 * NCHAN, 9))
    tdata = jnp.array([t(o) for t, o in zip(transform, observations.T)]).T
    tpeaks = jnp.array([t(p) for t, p in zip(transform, peaks.T)]).T
    for c in range(NCHAN):
        ax = axes[c]
        ax.set_title(f'{color_channels[c]} \n(to {channel_to_unit[color_channels[c]]})', pad=40)
        ym, yM = 0.3, 1.4
        ax.set_ylim(ym, yM)
        ax.set_yticks([])
        ax.set_xlim(-0.5, 0.5)
        plot.remove_axis_and_spines(ax)
        has_nans = np.mean(np.isnan(tdata[:, c])) > 0.5
        if has_nans:
            # could not calibrate this channel
            ax.axhspan(0, yM, color='k', alpha=0.05)
            ax.plot([-0.5, 0.5], [ym, yM], color='k', alpha=0.5, lw=0.5, dashes=(3, 3))
            ax.plot([-0.5, 0.5], [yM, ym], color='k', alpha=0.5, lw=0.5, dashes=(3, 3))
            for yt in [0.2, 0.8]:
                ax.text(
                    0,
                    yt * (ym + yM),
                    'Channel\ncould not\nbe calibrated',
                    fontsize=8,
                    color='#999999',
                    horizontalalignment='center',
                    verticalalignment='center',
                )

        else:
            peak_min, peak_max = np.min(tpeaks[:, c]) * 0.99, np.max(tpeaks[:, c]) * 1.01
            print(f'Channel = {color_channels[c]}')
            error = np.where(
                (calib[1:, c] >= peak_min) & (calib[1:, c] <= peak_max),
                np.abs(tpeaks[1:, c] - calib[1:, c]),
                np.nan,
            )
            # then compute the mean where it's not nan
            avg_dist_btwn_peaks = np.nanmean(np.diff(calib[:, c]))
            error = np.nanmean(error) / avg_dist_btwn_peaks
            # add beads as horizontal lines
            for b in range(0, NBEADS):
                alpha = 0.7
                ax.axhline(tpeaks[b, c], color='k', lw=1, xmin=0, xmax=0.5)
                ax.text(
                    -0.07 - (0.05 * b),
                    tpeaks[b, c] + 0.01,
                    f'{b}',
                    fontsize=8,
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center',
                )
                color = 'k'
                if calib[b, c] <= peak_min or calib[b, c] >= peak_max:
                    color = 'r'
                    alpha = 0.35
                ax.axhline(
                    calib[b, c], alpha=alpha, color=color, lw=1, dashes=(3, 3), xmin=0.5, xmax=1
                )
                ax.text(
                    0.45,
                    calib[b, c] + 0.01,
                    f'{b}',
                    fontsize=8,
                    color=color,
                    alpha=alpha,
                    horizontalalignment='center',
                    verticalalignment='center',
                )

            plot.plot_fluo_distribution(ax, tdata[:, c])
            ax.text(
                0.05,
                1.4,
                f'TARGET',
                fontsize=8,
                color='#999999',
                horizontalalignment='left',
                verticalalignment='center',
            )
            ax.text(
                -0.05,
                1.4,
                f'REAL',
                fontsize=8,
                color='#999999',
                horizontalalignment='right',
                verticalalignment='center',
            )
            precision = np.clip(1.0 - error, 0, 1)
            cmap = cm.get_cmap('RdYlGn')
            color = cmap(precision)
            ax.text(
                0,
                0.2,
                f'quality: {100.0*precision:.1f}%',
                fontsize=9,
                color=color,
                horizontalalignment='center',
            )

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname, dpi=150)
        print('saved', fname)
        plt.close(fig)

##────────────────────────────────────────────────────────────────────────────}}}
