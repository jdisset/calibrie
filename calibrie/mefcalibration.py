# Copyright (c) 2025 Jean Disset
# MIT License - see LICENSE file for details.

from jaxopt import GaussNewton
import matplotlib.patches as patches
from calibrie.utils import Escaped
from .pipeline import Task, DiagnosticFigure
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial
from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

from jax.scipy.stats import gaussian_kde
import pandas as pd

from scipy.interpolate import UnivariateSpline
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

from .utils import Context, LoadedData

from dracon.utils import DictLike
from dracon.loader import LoadedConfig

from pydantic import Field


class MEFBeadsCalibration(Task):
    """
    beads_data: np.ndarray - array of shape (n_events, n_channels) containing the beads data
    """

    beads_data: LoadedData
    channel_units: LoadedConfig[Escaped[DictLike]] = (
        'pkg:calibrie:config/beads/fortessa/spherotech_urcp-100-2H@channel_units'
    )
    beads_mef_values: LoadedConfig[Escaped[DictLike[str, List[float]]]] = (
        'pkg:calibrie:config/beads/fortessa/spherotech_urcp-100-2H@mef_values'
    )
    use_channels: Optional[Escaped[List[str]]] = None
    ignore_channels_with_missing_units: bool = False
    density_resolution: int = 1000
    density_bw_method: float = 0.2
    relative_density_threshold: float = 0.8
    resample_observations_to: int = 30000
    log_poly_threshold: float = 200
    log_poly_scale: float = 0.5
    model_resolution: int = 7
    model_degree: int = 2

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        self._tr = partial(
            utils.logtransform,
            threshold=self.log_poly_threshold,
            compression=self.log_poly_scale,
            base=10,
        )

        self._itr = partial(
            utils.inv_logtransform,
            threshold=self.log_poly_threshold,
            compression=self.log_poly_scale,
            base=10,
        )

    def initialize(self, ctx: Context):

        self._calibration_channel_name = ctx.channel_names[
            ctx.reference_channels[ctx.reference_protein_id]
        ]

        if self.use_channels is None:
            self.use_channels = ctx.channel_names

        if self.channel_units[self._calibration_channel_name] is None:
            raise ValueError(f'{self._calibration_channel_name} has no associated units')

        self._log.debug(
            f'Calibrating with channel {self._calibration_channel_name} and protein {ctx.reference_protein_name}'
        )

        if self._calibration_channel_name not in ctx.channel_names:
            raise ValueError('Calibration channel not found in the main channels')
        if self._calibration_channel_name not in self.use_channels:
            raise ValueError(
                'Calibration channel not found in the channels used for beads (use_channels)'
            )

        self._calibration_units_name = self.channel_units[self._calibration_channel_name]
        if self._calibration_units_name not in self.beads_mef_values:
            raise ValueError(f'No beads values for {self._calibration_units_name}')

        remove_chans = []

        assert self.use_channels is not None

        for c in self.use_channels:
            if c not in self.channel_units:
                if not self.ignore_channels_with_missing_units:
                    raise ValueError(
                        f"""No MEF units for {c}.
                        Pass 'ignore_channels_with_missing_units=True' to ignore"""
                    )
                else:
                    remove_chans.append(c)
            elif self.channel_units[c] not in self.beads_mef_values:
                raise ValueError(f'No unit values for channel {c} ({self.channel_units[c]})')

        if len(remove_chans) > 0:
            self._log.debug(f'No beads units for {remove_chans}. Removing them from use_channels')
            self.use_channels = [c for c in self.use_channels if c not in remove_chans]

        all_chans = list(set(ctx.channel_names + self.use_channels))
        self._beads_data_array = utils.load_to_df(self.beads_data, all_chans).values

        if ctx.get('colinearizer', None) is not None:
            chan_ids = [all_chans.index(c) for c in ctx.channel_names]
            self._beads_data_array[:, chan_ids] = ctx.colinearizer.process(
                Context(abundances_mapped=self._beads_data_array[:, chan_ids])
            ).observations_raw

        self._beads_data_array = self._beads_data_array[
            :, [all_chans.index(c) for c in self.use_channels]
        ]

        self._beads_mef_array = np.array(
            [self.beads_mef_values[self.channel_units[c]] for c in self.use_channels]
        ).T

        self._saturation_thresholds = []
        for i, c in enumerate(self.use_channels):
            if c in ctx.channel_names:
                idc = ctx.channel_names.index(c)
                self._saturation_thresholds.append(ctx.saturation_thresholds[idc])
            else:
                self._saturation_thresholds.append(
                    np.quantile(self._beads_data_array[:, i], [0.001, 0.999])
                )

        self._calibration_channel_id = self.use_channels.index(self._calibration_channel_name)

        self._saturation_thresholds = np.array(self._saturation_thresholds)

        self._log.debug(
            f'Calibrated abundances will be expressed in {self._calibration_units_name}'
        )

        self._log.debug(f'Computing beads locations')
        self.compute_peaks(self._beads_data_array, self._beads_mef_array)
        self._log.debug(f'Locations: {self._bead_peak_locations}')
        self.fit_regressions()
        self._log.debug(f'Applying calibration to controls')
        self._controls_abundances_MEF = self.process(
            Context(abundances_mapped=ctx.controls_abundances_mapped)
        ).abundances_MEF

        return Context(
            controls_abundances_MEF=self._controls_abundances_MEF,
            calibration_units_name=self._calibration_units_name,
        )


    def process(self, ctx: Context):
        self._log.debug(f'Calibrating values to {self._calibration_units_name}')
        params = self._params[self._calibration_channel_id]
        tr_abundances = self._tr(ctx.abundances_mapped)
        tr_calibrated = vmap(utils.evaluate_stacked_poly, in_axes=(1, None))(tr_abundances, params)
        abundances_MEF = self._itr(tr_calibrated).T
        self._log.debug(f'Calibrated values mef: {abundances_MEF}')
        return Context(abundances_MEF=abundances_MEF)

    def compute_peaks(
        self,
        beads_observations,
        beads_mef_values,
    ):
        """Compute coordinates of each bead's peak in all channels"""

        assert beads_observations.shape[1] == beads_mef_values.shape[1]

        if beads_observations.shape[0] > self.resample_observations_to:
            reorder = np.random.choice(
                beads_observations.shape[0], self.resample_observations_to, replace=False
            )
            beads_observations = beads_observations[reorder]

        not_saturated = (
            (beads_observations > self._saturation_thresholds[:, 0])
            & (beads_observations < self._saturation_thresholds[:, 1])
        ).T

        self._obs_tr = self._tr(beads_observations)
        self._mef_tr = self._tr(beads_mef_values)
        self._saturation_thresholds_tr = self._tr(self._saturation_thresholds)

        self._log.debug(f'Computing votes')

        @jit
        @partial(vmap, in_axes=(1, 1))
        def vote(chan_observations, chan_mef):
            """Compute the vote matrix"""
            # it's a (CHANNEL, OBSERVATIONS, BEAD) matrix
            # where each channel tells what affinity each observation has to each bead, from the channel's perspective.
            # This is computed using optimal transport (it's the OT matrix)
            # High values mean it's obvious in this channel that the observation should be paired with a certain bead.
            # Low values mean it's not so obvious, usually because the observation is not in the valid range
            # so there's a bunch of points around this one that could be paired with any of the remaining beads
            # This is much more robust than just computing OT for all channels at once
            return Sinkhorn()(
                LinearProblem(PointCloud(chan_observations[:, None], chan_mef[:, None]))
            ).matrix

        votes = vote(self._obs_tr, self._mef_tr)  # (CHANNELS, OBSERVATIONS, BEADS)
        valid_votes = votes * not_saturated[:, :, None] + 1e-12  # (CHANNELS, OBSERVATIONS, BEADS)
        vmat = (
            np.sum(valid_votes, axis=0) / np.sum(not_saturated, axis=0)[:, None]
        )  # weighted average

        # Use these votes to decide which beads are the most likely for each observation
        # Tried with a softer version of this just in case, but I couldn't see any improvement
        self._vmat = np.argmax(vmat, axis=1)[:, None] == np.arange(vmat.shape[1])[None, :]

        # We add some tiny random normal noise to avoid singular matrix errors when computing the KDE
        # on a bead that would have only the exact same value (which can happen when out of range)
        self._peaks_max_x = self._obs_tr.max() * 1.05
        self._peaks_min_x = self._obs_tr.min() * 0.95
        noise_std = (self._peaks_max_x - self._peaks_min_x) / (self.density_resolution * 5)
        obs = self._obs_tr + np.random.normal(0, noise_std, self._obs_tr.shape)

        # Now we can compute the densities for each bead in each channel in order to locate the peaks
        self._log.debug(f'Computing densities')
        x = np.linspace(self._peaks_min_x, self._peaks_max_x, self.density_resolution)
        w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=self.density_bw_method)(x)
        densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(obs, self._vmat)
        densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)
        self._beads_densities = densities / np.max(densities, axis=2)[:, :, None]

        # find the most likely peak positions for each bead using
        # the average intensity weighted by density

        unsaturated_density = (x > self._saturation_thresholds_tr[:, 0][:, None]) & (
            x < self._saturation_thresholds_tr[:, 1][:, None]
        )

        rel_mask = self._beads_densities > self.relative_density_threshold
        # forget about the unsaturated_density mask, it's not useful

        w = self._beads_densities * rel_mask

        # w.shape is (BEADS, CHANNELS, RESOLUTION)

        is_zero = np.sum(w, axis=2) == 0
        w = np.where(is_zero[:, :, None], 1, w)

        self._weighted_densities = w
        self._log.debug(f'w sum: {np.sum(w)}')

        xx = np.tile(x, (self._beads_densities.shape[0], self._beads_densities.shape[1], 1))
        self._bead_peak_locations = np.average(
            xx, axis=2, weights=w
        )  # peaks.shape is (BEADS, CHANNELS)

        # compute the overlap matrix
        @jit
        def overlap(X):
            def overlap_1v1(x, y):
                s = jnp.sum(jnp.minimum(x, y), axis=1)
                return jnp.clip(s / jnp.minimum(jnp.sum(x, axis=1), jnp.sum(y, axis=1)), 0, 1)

            return vmap(vmap(overlap_1v1, (0, None)), (None, 0))(X, X)

        self._log.debug(f'Computing overlap matrix')

        self._overlap = overlap(self._weighted_densities)

    @partial(jax.jit, static_argnums=(0,))
    def _regression(self, x, y, w, xbounds):
        # EXPECTS VALUES ALREADY IN LOG SPACE
        params = utils.fit_stacked_poly_uniform_spacing(
            x, y, w, *xbounds, self.model_resolution, self.model_degree, endpoint=True
        )
        return params

    def fit_regressions(self, ignore_first_bead=True, **_):
        # we weight the points by how much they *don't* overlap
        diag = np.eye(self._overlap.shape[0], self._overlap.shape[1], dtype=bool)
        W = np.array(1 - self._overlap.at[diag].set(0).max(axis=0))
        if ignore_first_bead:  # almost...
            W[0, :] = 1e-6 * W[0, :]
        self._peak_weights = W
        # we fit a regression for each channel
        self._params = []
        for i in range(len(self.use_channels)):
            peaks_au = self._bead_peak_locations[:, i]
            beads_mef = self._mef_tr[:, i]
            self._params.append(
                self._regression(
                    peaks_au,
                    beads_mef,
                    w=W[:, i],
                    xbounds=self._saturation_thresholds_tr[i],
                )
            )


## {{{                        --     diagnostics     --
    def diagnostics(self, ctx: Context, **kw):
        controls_abundances, std_models = utils.generate_controls_dict(
            self._controls_abundances_MEF,
            ctx.controls_masks,
            ctx.protein_names,
            **kw,
        )

        f, _ = plots.unmixing_plot(
            controls_abundances,
            ctx.protein_names,
            ctx.channel_names,
            **kw,
        )
        f.suptitle(f'Unmixing after calibration to {self._calibration_units_name}', y=1.05)

        peakfig = self.plot_bead_peaks_diagnostics()

        return [
            DiagnosticFigure(fig=f, name='Unmixing after mef calibration'),
            DiagnosticFigure(fig=peakfig, name='Bead peaks'),
        ]


    def plot_bead_peaks_diagnostics(self):
        """
        Plot diagnostics for the bead peaks computation using subplots instead of subfigures,
        with correct scaling for multiple axes.
        """
        svmat = np.array(self._vmat)
        nbeads = self._bead_peak_locations.shape[0]
        NCHAN = len(self.use_channels)

        fig = plt.figure(figsize=(18, 25))
        gs = fig.add_gridspec(
            ncols=NCHAN + 1,
            nrows=NCHAN + 4,
            width_ratios=[0.2] + [1] * NCHAN,
            height_ratios=[1] * NCHAN + [0.1, 1.2, 0.1, 2],
            hspace=0.4,
            wspace=0.3,
        )

        # Assignment plot
        ax_assignment = fig.add_subplot(gs[:, 0])
        self.plot_assignment(ax_assignment)

        axes_densities = []
        for i in range(NCHAN):
            if i == 0:
                ax = fig.add_subplot(gs[i, 1 : NCHAN])
            else:
                ax = fig.add_subplot(gs[i, 1 : NCHAN], sharex=axes_densities[0])
            axes_densities.append(ax)
        self.plot_densities(axes_densities)

        # Overlap plot
        axes_overlap = [fig.add_subplot(gs[i, -1]) for i in range(NCHAN)]
        self.plot_overlap(axes_overlap)

        # Regressions plot
        axes_regressions = [fig.add_subplot(gs[NCHAN+1, 1 + i]) for i in range(NCHAN)]
        self.plot_regressions(axes_regressions)

        # After correction plot
        axes_after = [fig.add_subplot(gs[-1, 1 + i]) for i in range(NCHAN)]
        self.plot_beads_after_correction(axes_after)

        # Function to add framed title
        def add_framed_title(axes, title, padding=0.0125):
            # Get the positions of the first and last axes in the group
            first_ax = axes[0]
            last_ax = axes[-1]

            # Calculate the bounding box for the group of axes
            bbox = first_ax.get_position()
            bbox.y0 = min(ax.get_position().y0 for ax in axes)
            bbox.y1 = max(ax.get_position().y1 for ax in axes)
            bbox.x1 = last_ax.get_position().x1

            # Add padding
            bbox.x0 -= padding
            bbox.x1 += padding
            bbox.y0 -= padding
            bbox.y1 += padding

            # Add the title
            fig.text(
                bbox.x0 + bbox.width / 2,
                bbox.y1,
                title,
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
                transform=fig.transFigure,
            )

        fig.tight_layout()
        # Add framed titles
        add_framed_title([ax_assignment], "Assignment")
        add_framed_title(
            axes_densities, "Density and assignment of observations to bead number\n(hatched = out of range)"
        )
        add_framed_title(axes_overlap, "Bead overlap matrices")
        add_framed_title(axes_regressions, "Mapping of Arbitrary Units to MEF")
        add_framed_title(axes_after, "Observed peaks alignment to targets after correction")
        # Adjust layout

        return fig

    def plot_assignment(self, ax):
        assignment = np.sort(np.argmax(self._vmat, axis=1))[:, None]
        cmap = plt.get_cmap('tab10')
        cmap.set_bad(color='white')
        centroids = [
            (np.arange(len(assignment))[assignment[:, 0] == i]).mean()
            for i in range(self._vmat.shape[1])
        ]
        ax.imshow(
            assignment,
            aspect='auto',
            cmap=cmap,
            interpolation='none',
            vmin=0,
            vmax=self._vmat.shape[1],
            origin='lower',
        )
        for i, c in enumerate(centroids):
            ax.text(
                0,
                c,
                f'bead {i}',
                rotation=90,
                verticalalignment='center',
                horizontalalignment='center',
            )
        ax.set_ylabel('Observations, sorted by bead assignment')
        ax.set_xticks([])
        ax.set_yticks([0, len(assignment)])

    def plot_densities(self, axes):
        resolution = self._beads_densities.shape[2]
        x = np.linspace(self._peaks_min_x, self._peaks_max_x, resolution)
        nbeads = self._bead_peak_locations.shape[0]

        weights = (x > self._saturation_thresholds_tr[:, 0][:, None]) & (
            x < self._saturation_thresholds_tr[:, 1][:, None]
        )

        beadcolors = [plt.get_cmap('tab10')(1.0 * i / 10) for i in range(10)]
        for c, ax in enumerate(axes):
            for b in range(nbeads):
                dens = self._beads_densities[b, c]
                dens /= np.max(self._beads_densities[b, c])
                ax.plot(x, dens, label=f'bead {b}', linewidth=1.25, color=beadcolors[b])

                # also plot the actual distribution
                kde = gaussian_kde(self._obs_tr[:, c], bw_method=0.01)
                d2 = kde(x)
                d2 /= np.max(d2) * 1.25
                ax.plot(x, d2, color='k', linewidth=0.75, alpha=1, label='_nolegend_')
                ax.fill_between(x, d2, 0, color='k', alpha=0.01)

                # red hatches where weights are 0, on grey bg
                ax.fill_between(
                    x,
                    0,
                    1.3,
                    where=~weights[c],
                    hatch='///',
                    facecolor='#dddddd',
                    edgecolor='#ff0000',
                    alpha=0.05,
                    linewidth=0.0,
                    zorder=-10,
                )

                # vline at peak
                ax.axvline(self._bead_peak_locations[b, c], color='k', linewidth=0.5, dashes=(3, 3))
                # write bead number
                ax.text(
                    self._bead_peak_locations[b, c] + 0.01,
                    0.3 + 0.1 * b,
                    f'{b}',
                    fontsize=8,
                    ha='center',
                    va='center',
                )
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])

            tr, invtr, xlims_tr, ylims_tr = plots.make_symlog_ax(
                ax,
                [self._itr(self._peaks_min_x), self._itr(self._peaks_max_x * 1.02)],
                None,
                linthresh=self.log_poly_threshold,
                linscale=self.log_poly_scale,
            )

            title = f'{self.use_channels[c]}'
            ax.set_ylabel(title)

    def plot_overlap(self, axes):
        nbeads = self._bead_peak_locations.shape[0]
        for i, ax in enumerate(axes):
            im = ax.imshow(
                self._overlap[:, :, i], cmap='Reds', extent=[0, nbeads, 0, nbeads], origin='lower'
            )
            ax.set_xticks(np.arange(nbeads) + 0.5)
            ax.set_yticks(np.arange(nbeads) + 0.5)
            ax.set_xticklabels(np.arange(nbeads))
            ax.set_yticklabels(np.arange(nbeads))

    def plot_regressions(self, axes):
        NBEADS, NCHAN = self._mef_tr.shape
        for c, ax in enumerate(axes):
            w = self._peak_weights[:, c]
            ax.scatter(
                self._bead_peak_locations[:, c],
                self._mef_tr[:, c],
                c=np.clip(w, 0, 1),
                cmap='RdYlGn',
            )
            x = np.linspace(
                self._saturation_thresholds_tr[c, 0], self._saturation_thresholds_tr[c, 1], 300
            )
            y = utils.evaluate_stacked_poly(x, self._params[c])
            ax.plot(x, y, color='k', ls='--', alpha=0.75)
            # add name of bead
            for i in range(NBEADS):
                ax.text(
                    self._bead_peak_locations[i, c],
                    self._mef_tr[i, c] + 0.1,
                    f'{i}',
                    fontsize=8,
                    ha='center',
                    va='bottom',
                )

            xlims = self._itr(ax.get_xlim())
            ylims = self._itr(ax.get_ylim())

            plots.make_tr_ax(ax, tr=self._tr, invtr=self._itr, xlims=xlims, ylims=ylims)
            # no units or ticks
            ax.set_title(self.use_channels[c])
            ax.set_xlabel('AU')
            ax.set_ylabel('MEF')

    def plot_beads_after_correction(self, axes):
        from matplotlib import cm

        NBEADS, NCHAN = self._mef_tr.shape
        tdata = np.array(
            [utils.evaluate_stacked_poly(o, p) for o, p in zip(self._obs_tr.T, self._params)]
        ).T
        tpeaks = np.array(
            [
                utils.evaluate_stacked_poly(o, p)
                for o, p in zip(self._bead_peak_locations.T, self._params)
            ]
        ).T

        for c, ax in enumerate(axes):
            chan = self.use_channels[c]
            ym, yM = self._peaks_min_x * 0.8, self._peaks_max_x * 1.5
            ax.set_ylim(ym, yM)
            ax.set_yticks([])
            ax.set_xlim(-0.5, 0.5)
            # ax.axis('off')
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel(f'{chan} \n(to {self.channel_units[chan]})')

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
                error = np.where(
                    (self._mef_tr[1:, c] >= peak_min) & (self._mef_tr[1:, c] <= peak_max),
                    np.abs(tpeaks[1:, c] - self._mef_tr[1:, c]),
                    np.nan,
                )
                # then compute the mean where it's not nan
                avg_dist_btwn_peaks = np.nanmean(np.diff(self._mef_tr[:, c]))
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
                    if self._mef_tr[b, c] <= peak_min or self._mef_tr[b, c] >= peak_max:
                        color = 'r'
                        alpha = 0.35
                    ax.axhline(
                        self._mef_tr[b, c],
                        alpha=alpha,
                        color=color,
                        lw=1,
                        dashes=(3, 3),
                        xmin=0.5,
                        xmax=1,
                    )
                    ax.text(
                        0.45,
                        self._mef_tr[b, c] + 0.01,
                        f'{b}',
                        fontsize=8,
                        color=color,
                        alpha=alpha,
                        horizontalalignment='center',
                        verticalalignment='center',
                    )

                plots.plot_fluo_distribution(ax, tdata[:, c])
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

##────────────────────────────────────────────────────────────────────────────}}}
