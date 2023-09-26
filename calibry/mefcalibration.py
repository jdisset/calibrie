from jaxopt import GaussNewton
from .pipeline import Task
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial
from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from jax.scipy.stats import gaussian_kde

from scipy.interpolate import UnivariateSpline
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

# remember: the first bead is not reliable and should'nt contribute to the calibration
MEF_VALUES_SPHEROTECH_RCP_30_5a = {
    'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549],
    'MEPE': [196, 2800, 8770, 25174, 74335, 219816, 548646, 1600005, 4255375],
    'MEPTR': [1034, 17518, 53950, 153641, 450901, 1283877, 3254513, 9431807, 24840372],
    'MEPerCP': [605, 4354, 10032, 22473, 51739, 115599, 256091, 562684, 1201350],
    'MEPCY5.5': [180, 2088, 6705, 20441, 66215, 211174, 645020, 2478405, 10603147],
    'MEeF710': [258, 2259, 6862, 20129, 63316, 196680, 609247, 2451473, 11687960],
    'MEPCY7': [88, 534, 1555, 4600, 14826, 47575, 161926, 706536, 3262715],
    'MEPTR': [1142, 17518, 53950, 153641, 450901, 1283877, 3254513, 9431807, 24840372],
    'MEPCY5': [1026, 4354, 10032, 22473, 51739, 115599, 256091, 562684, 1201350],
    'MEPCY5.5': [185, 1999, 6228, 20393, 69124, 220232, 777840, 2521966, 8948283],
    'MEAX700': [480, 6625, 17113, 58590, 199825, 629666, 2289301, 6504723, 17637305],
    'MEPCY7': [25, 457, 1334, 4666, 17500, 58774, 230324, 724800, 2057002],
    'MEAPC': [743, 1170, 1970, 4669, 13757, 36757, 119744, 293242, 638909],
    'MEAX680': [945, 6844, 17166, 56676, 195246, 622426, 2333985, 6617776, 17561028],
    'MEAX700': [495, 6625, 17113, 58590, 199825, 629666, 2289301, 6504723, 17637305],
    'MEAPCCY7': [73, 1385, 3804, 13066, 47512, 151404, 542987, 1305924, 2540123],
    'PacBlue': [979, 4450, 8342, 17587, 38906, 89281, 179989, 408481, 822214],
    'MEAMCY': [1987, 5974, 10513, 21623, 46727, 105630, 213273, 494395, 1072308],
    'MEPO': [148, 391, 753, 1797, 4766, 13937, 39280, 156244, 652221],
    'MEQ605': [1718, 3133, 4774, 8471, 16359, 34465, 71375, 189535, 517591],
    'MEQ655': [1060, 1859, 2858, 5598, 11928, 27542, 66084, 202508, 650000],
    'MEQ705': [840, 1695, 2858, 5598, 11928, 27542, 66084, 202508, 650000],
    'MEBV711': [1345, 1564, 3234, 5516, 12249, 29651, 71051, 197915, 596714],
    'MEQ800': [857, 1358, 2085, 4301, 10037, 23446, 64511, 186279, 644779],
}

FORTESSA_CHANNELS_UNITS = {
    'Pacific_Blue': 'PacBlue',
    'AmCyan': 'MEAMCY',
    'FITC': 'MEFL',
    'PerCP_Cy5_5': 'MEPCY5.5',
    'PE': 'MEPE',
    'PE_Texas_Red': 'MEPTR',
    'APC': 'MEAPC',
    'APC_Alexa_700': 'MEAX700',
    'APC_Cy7': 'MEAPCCY7',
}


class MEFBeadsCalibration(Task):
    def __init__(
        self,
        beads_data,
        beads_mef_values: Dict[str, List[float]] = MEF_VALUES_SPHEROTECH_RCP_30_5a,
        channel_units: Dict[str, str] = FORTESSA_CHANNELS_UNITS,
        use_channels: List[str] = None,
        density_resolution: int = 1000,
        density_bw_method=0.2,
        density_threshold: float = 0.25,
        resample_observations_to: int = 20000,
        log_poly_threshold=100,
        log_poly_scale=0.5,
        model_resolution: int = 7,
        model_degree: int = 1,
        **kwargs,
    ):
        """
        beads_data: np.ndarray - array of shape (n_events, n_channels) containing the beads data
        beads_mef_vlaues : dict[str, list[float]] - dictionary associating the unit name
            to the list of reference values for the beads in this unit
            e.g : {'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549], 'MEPE': ...}

        channel_units : dict[str, str] - dictionary associating the channel name to the unit name
            e.g. {'Pacific Blue-A': 'PacBlue', 'AmCyan-A': 'MEAMCY', ...}"""

        super().__init__(**kwargs)
        self.beads_data = beads_data
        self.beads_mef_values = utils.escape(beads_mef_values)
        self.channel_units = utils.escape(channel_units)
        self.density_resolution = density_resolution
        self.density_bw_method = density_bw_method
        self.density_threshold = density_threshold
        self.resample_observations_to = resample_observations_to
        if use_channels is not None:
            raise NotImplementedError(
                'MEF\'s use_channels is not implemented yet, issues with colinearizer'
            )
        self.use_channels = utils.escape(use_channels)
        self.model_resolution = model_resolution
        self.model_degree = model_degree
        self.tr = partial(
            utils.logtransform, threshold=log_poly_threshold, compression=log_poly_scale, base=10
        )
        self.itr = partial(
            utils.inv_logtransform,
            threshold=log_poly_threshold,
            compression=log_poly_scale,
            base=10,
        )
        self.log_poly_threshold = log_poly_threshold
        self.log_poly_scale = log_poly_scale

    def initialize(
        self,
        channel_names,
        saturation_thresholds,
        reference_channels,
        reference_protein_name,
        reference_protein_id,
        controls_abundances_mapped,
        colinearizer=None,
        **_,
    ):

        self.calibration_channel_name = channel_names[reference_channels[reference_protein_id]]
        if self.use_channels is None:
            self.use_channels = channel_names

        if self.channel_units[self.calibration_channel_name] is None:
            raise ValueError(f'{self.calibration_channel_name} has no associated units')

        self.log.debug(
            f'Calibrating with {self.calibration_channel_name} from {reference_protein_name}'
        )

        if self.calibration_channel_name not in channel_names:
            raise ValueError('Calibration channel not found in the main channels')
        if self.calibration_channel_name not in self.use_channels:
            raise ValueError(
                'Calibration channel not found in the channels used for beads (use_channels)'
            )

        self.calibration_channel_id = self.use_channels.index(self.calibration_channel_name)
        self.calibration_units_name = self.channel_units[self.calibration_channel_name]
        if self.calibration_units_name not in self.beads_mef_values:
            raise ValueError(f'No beads values for {self.calibration_units_name}')

        self.beads_data_array = utils.load_to_df(self.beads_data, self.use_channels).values
        if colinearizer is not None:
            self.beads_data_array = list(colinearizer.process(self.beads_data_array).values())[0]

        self.beads_mef_array = np.array(
            [self.beads_mef_values[self.channel_units[c]] for c in self.use_channels]
        ).T

        self.saturation_thresholds = []
        for i, c in enumerate(self.use_channels):
            if c in channel_names:
                idc = channel_names.index(c)
                self.saturation_thresholds.append(saturation_thresholds[idc])
            else:
                self.saturation_thresholds.append(
                    np.quantile(self.beads_data_array[:, i], [0.001, 0.999])
                )

        self.saturation_thresholds = np.array(self.saturation_thresholds)

        self.log.debug(f'Calibrated abundances will be expressed in {self.calibration_units_name}')

        self.log.debug(f'Computing beads locations')
        self.bead_peak_locations = self.compute_peaks(
            self.beads_data_array,
            self.beads_mef_array,
        )
        self.log.debug(f'Locations: {self.bead_peak_locations}')
        self.fit_regressions()
        self.log.debug(f'Applying calibration to controls')
        self.controls_abundances_MEF = self.process(controls_abundances_mapped)['abundances_MEF']

        return {
            'controls_abundances_MEF': self.controls_abundances_MEF,
            'calibration_units_name': self.calibration_units_name,
        }

    def diagnostics(self, controls_masks, protein_names, channel_names, **kw):

        controls_abundances, std_models = utils.generate_controls_dict(
            self.controls_abundances_MEF,
            controls_masks,
            protein_names,
            **kw,
        )

        f, _ = plots.unmixing_plot(
            controls_abundances,
            protein_names,
            channel_names,
            **kw,
        )
        f.suptitle(f'Unmixing after calibration to {self.calibration_units_name}', y=1.05)

        self.plot_bead_peaks_diagnostics()
        self.plot_regressions()

    def process(self, abundances_mapped, **_):
        self.log.debug(f'Calibrating values to {self.calibration_units_name}')
        params = self.params[self.calibration_channel_id]
        tr_abundances = self.tr(abundances_mapped)
        tr_calibrated = vmap(utils.evaluate_stacked_poly, in_axes=(1, None))(tr_abundances, params)
        abundances_MEF = self.itr(tr_calibrated).T
        return {'abundances_MEF': abundances_MEF}

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
            (beads_observations > self.saturation_thresholds[:, 0])
            & (beads_observations < self.saturation_thresholds[:, 1])
        ).T

        self.obs_tr = self.tr(beads_observations)
        self.mef_tr = self.tr(beads_mef_values)
        self.saturation_thresholds_tr = self.tr(self.saturation_thresholds)

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

        votes = vote(self.obs_tr, self.mef_tr)  # (CHANNELS, OBSERVATIONS, BEADS)
        valid_votes = votes * not_saturated[:, :, None] + 1e-12  # (CHANNELS, OBSERVATIONS, BEADS)
        vmat = (
            np.sum(valid_votes, axis=0) / np.sum(not_saturated, axis=0)[:, None]
        )  # weighted average

        # Use these votes to decide which beads are the most likely for each observation
        # Tried with a softer version of this just in case, but I couldn't see any improvement
        self.vmat = np.argmax(vmat, axis=1)[:, None] == np.arange(vmat.shape[1])[None, :]

        # We add some tiny random normal noise to avoid singular matrix errors when computing the KDE
        # on a bead that would have only the exact same value (which can happen when out of range)
        self.peaks_max_x = self.obs_tr.max() * 1.05
        self.peaks_min_x = self.obs_tr.min() * 0.95
        noise_std = (self.peaks_max_x - self.peaks_min_x) / (self.density_resolution * 5)
        obs = self.obs_tr + np.random.normal(0, noise_std, self.obs_tr.shape)

        # Now we can compute the densities for each bead in each channel in order to locate the peaks
        x = np.linspace(self.peaks_min_x, self.peaks_max_x, self.density_resolution)
        w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=self.density_bw_method)(x)
        densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(obs, self.vmat)
        densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)
        self.beads_densities = densities / np.max(densities, axis=2)[:, :, None]

        # find the most likely peak positions for each bead using
        # the average intensity weighted by density
        unsaturated_density = (x > self.saturation_thresholds_tr[:, 0][:, None]) & (
            x < self.saturation_thresholds_tr[:, 1][:, None]
        )

        w = (
            np.clip(self.beads_densities - self.density_threshold, 0, 1)
            * unsaturated_density[None, :]
        )

        w = np.where(
            self.beads_densities > self.density_threshold, np.maximum(w, 1e-9), 0
        )  # still some tiny weight when OOR

        # w.shape is (BEADS, CHANNELS, RESOLUTION)

        is_zero = np.sum(w, axis=2) == 0
        w = np.where(is_zero[:, :, None], 1, w)

        self.weighted_densities = w

        xx = np.tile(x, (self.beads_densities.shape[0], self.beads_densities.shape[1], 1))
        peaks = np.average(xx, axis=2, weights=w)  # peaks.shape is (BEADS, CHANNELS)

        # compute the overlap matrix
        @jit
        def overlap(X):
            def overlap_1v1(x, y):
                s = jnp.sum(jnp.minimum(x, y), axis=1)
                return jnp.clip(s / jnp.minimum(jnp.sum(x, axis=1), jnp.sum(y, axis=1)), 0, 1)

            return vmap(vmap(overlap_1v1, (0, None)), (None, 0))(X, X)

        self.overlap = overlap(self.weighted_densities)

        return peaks

    @partial(jax.jit, static_argnums=(0,))
    def regression(self, x, y, w, xbounds):
        # EXPECTS VALUES ALREADY IN LOG SPACE
        params = utils.fit_stacked_poly_uniform_spacing(
            x, y, w, *xbounds, self.model_resolution, self.model_degree, endpoint=True
        )
        return params

    def fit_regressions(self, ignore_first_bead=True, **_):
        # we weight the points by how much they *don't* overlap
        diag = np.eye(self.overlap.shape[0], self.overlap.shape[1], dtype=bool)
        W = np.array(1 - self.overlap.at[diag].set(0).max(axis=0))
        if ignore_first_bead:  # almost...
            W[0, :] = 1e-6 * W[0, :]
        self.peak_weights = W
        # we fit a regression for each channel
        self.params = []
        for i in range(len(self.use_channels)):
            peaks_au = self.bead_peak_locations[:, i]
            beads_mef = self.mef_tr[:, i]
            self.params.append(
                self.regression(
                    peaks_au,
                    beads_mef,
                    w=W[:, i],
                    xbounds=self.saturation_thresholds_tr[i],
                )
            )

    def plot_regressions(self):
        NBEADS, NCHAN = self.mef_tr.shape
        fig, axes = plt.subplots(1, NCHAN, figsize=(5 * NCHAN, 5))
        for c in range(NCHAN):
            ax = axes[c]
            w = self.peak_weights[:, c]
            ax.scatter(
                self.bead_peak_locations[:, c],
                self.mef_tr[:, c],
                c=np.clip(w, 0, 1),
                cmap='RdYlGn',
            )
            x = np.linspace(
                self.saturation_thresholds_tr[c, 0], self.saturation_thresholds_tr[c, 1], 300
            )
            y = utils.evaluate_stacked_poly(x, self.params[c])
            ax.plot(x, y, color='k', ls='--', alpha=0.75)
            ax.set_title(self.use_channels[c])
            ax.set_xlabel('AU')
            ax.set_ylabel('MEF')

    def plot_bead_peaks_diagnostics(self):
        """
        Plot diagnostics for the bead peaks computation:
        - An assignment chart showing the proportion of observations assigned to each bead
        - 1 plot per channel showing both bead assignment, peak detection and final standardized densities
        """
        svmat = np.array(self.vmat)
        nbeads = self.bead_peak_locations.shape[0]

        mainfig = plt.figure(constrained_layout=True, figsize=(15, 15))
        subfigs = mainfig.subfigures(1, 3, wspace=0, width_ratios=[0.4, 10, 2])
        assignment = np.sort(np.argmax(self.vmat, axis=1))[:, None]
        ax = subfigs[0].subplots(1, 1)
        cmap = plt.get_cmap('tab10')
        cmap.set_bad(color='white')
        centroids = [
            (np.arange(len(assignment))[assignment[:, 0] == i]).mean()
            for i in range(self.vmat.shape[1])
        ]
        ax.imshow(
            assignment,
            aspect='auto',
            cmap=cmap,
            interpolation='none',
            vmin=0,
            vmax=self.vmat.shape[1],
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

        resolution = self.beads_densities.shape[2]
        x = np.linspace(self.peaks_min_x, self.peaks_max_x, resolution)
        axes = subfigs[1].subplots(len(self.use_channels), 1, sharex=True)
        if len(self.use_channels) == 1:
            axes = [axes]

        weights = (x > self.saturation_thresholds_tr[:, 0][:, None]) & (
            x < self.saturation_thresholds_tr[:, 1][:, None]
        )

        print(f'weights.shape: {weights.shape}')

        beadcolors = [plt.get_cmap('tab10')(1.0 * i / 10) for i in range(10)]
        for c in range(len(self.use_channels)):
            ax = axes[c]
            for b in range(nbeads):

                dens = self.beads_densities[b, c]
                dens /= jnp.max(self.beads_densities[b, c])
                ax.plot(x, dens, label=f'bead {b}', linewidth=1.25, color=beadcolors[b])

                # also plot the actual distribution
                kde = gaussian_kde(self.obs_tr[:, c], bw_method=0.01)
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
                ax.axvline(self.bead_peak_locations[b, c], color='k', linewidth=0.5, dashes=(3, 3))
                # write bead number
                ax.text(
                    self.bead_peak_locations[b, c] + 0.01,
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
                    [self.itr(self.peaks_min_x),
                    self.itr(self.peaks_max_x * 1.02)],
                    None,
                    linthresh=self.log_poly_threshold,
                    linscale=self.log_poly_scale,
                )

                title = f'{self.use_channels[c]}'
                ax.set_ylabel(title)

            axes[0].set_title('Density and assignment of observations (hatched = out of range)')
        subfigs[1].suptitle("Bead peak diagnostics")

        # plot overlap matrix:
        axes = subfigs[2].subplots(self.overlap.shape[2], 1, sharex=True)
        for i, ax in enumerate(axes):
            im = ax.imshow(
                self.overlap[:, :, i], cmap='Reds', extent=[0, nbeads, 0, nbeads], origin='lower'
            )
            ax.set_xticks(np.arange(nbeads) + 0.5)
            ax.set_yticks(np.arange(nbeads) + 0.5)
            ax.set_xticklabels(np.arange(nbeads))
            ax.set_yticklabels(np.arange(nbeads))
        axes[0].set_title("Bead overlap matrices")

    def plot_beads_after_correction(self):
        from matplotlib import cm

        NBEADS, NCHAN = self.mef_tr.shape
        fig, axes = plt.subplots(1, NCHAN, figsize=(1.3 * NCHAN, 9))
        tdata = np.array(
            [utils.evaluate_stacked_poly(o, p) for o, p in zip(self.obs_tr.T, self.params)]
        ).T
        tpeaks = np.array(
            [
                utils.evaluate_stacked_poly(o, p)
                for o, p in zip(self.bead_peak_locations.T, self.params)
            ]
        ).T
        for c in range(NCHAN):
            ax = axes[c]
            chan = self.use_channels[c]
            ax.set_title(f'{chan} \n(to {self.channel_units[chan]})', pad=40)
            ym, yM = self.peaks_min_x * 0.8, self.peaks_max_x * 1.5
            ax.set_ylim(ym, yM)
            ax.set_yticks([])
            ax.set_xlim(-0.5, 0.5)
            ax.axis('off')
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

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
                print(f'Channel = {self.use_channels[c]}')
                error = np.where(
                    (self.mef_tr[1:, c] >= peak_min) & (self.mef_tr[1:, c] <= peak_max),
                    np.abs(tpeaks[1:, c] - self.mef_tr[1:, c]),
                    np.nan,
                )
                # then compute the mean where it's not nan
                avg_dist_btwn_peaks = np.nanmean(np.diff(self.mef_tr[:, c]))
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
                    if self.mef_tr[b, c] <= peak_min or self.mef_tr[b, c] >= peak_max:
                        color = 'r'
                        alpha = 0.35
                    ax.axhline(
                        self.mef_tr[b, c],
                        alpha=alpha,
                        color=color,
                        lw=1,
                        dashes=(3, 3),
                        xmin=0.5,
                        xmax=1,
                    )
                    ax.text(
                        0.45,
                        self.mef_tr[b, c] + 0.01,
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

        fig.tight_layout()


# def plot_beads_diagnostics(self):

# plot_bead_peaks_diagnostics(
# self.__log_beads_peaks,
# self.__beads_densities,
# self.__beads_vmat,
# self.__log_beads_data,
# self.beads_channel_order,
# )

# plot_beads_after_correction(
# self.beads_transform,
# self.__log_beads_peaks,
# self.__log_beads_mef,
# self.__log_beads_data,
# self.beads_channel_order,
# self.channel_to_unit,
# title="""Beads-based MEF calibration
# right side shows beads intensities in their respective MEF units
# left side shows the real beads data after calibration\n\n""",
# )
# plt.show()
