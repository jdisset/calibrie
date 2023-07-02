
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde

from scipy.interpolate import UnivariateSpline
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

from . import plot
import matplotlib.pyplot as plt


### {{{                        --     beads utils     --

CALIBRATION_PEAKS_MIN_X = -0.025
CALIBRATION_PEAKS_MAX_X = 1.025


# weights observations: 0 if out of range, 1 if in range
def w_function(x, a=0, b=0.99, lsteepness=10, rsteepness=1000):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


@partial(jit, static_argnums=(2, 3, 4))
def compute_peaks(
    observations,
    beads,
    resolution=1000,
    bw_method=0.2,
    max_obs=20000,
    peaks_min_x=CALIBRATION_PEAKS_MIN_X,
    peaks_max_x=CALIBRATION_PEAKS_MAX_X,
):
    """Compute coordinates of each bead's peak in all channels"""

    # First we compute the vote matrix:
    # it's a (CHANNEL, OBSERVATIONS, BEAD) matrix
    # where each channel tells what affinity each observation has to each bead, from the channel's perspective.
    # This is computed using optimal transport (it's the OT matrix)
    # High values mean it's obvious in this channel that the observation should be paired with a certain bead.
    # Low values mean it's not so obvious, usually because the observation is not in the valid range
    # so there's a bunch of points around this one that could be paired with any of the remaining beads
    # This is much more robust than just computing OT for all channels at once

    @jit
    @partial(vmap, in_axes=(1, 1))
    def vote(s, t):
        return Sinkhorn()(LinearProblem(PointCloud(s[:, None], t[:, None]))).matrix

    if observations.shape[0] > max_obs:
        key = jax.random.PRNGKey(0)
        reorder = jax.random.permutation(key, jnp.arange(max_obs))
        observations = observations[reorder]

    votes = vote(observations, beads)  # (CHANNELS, OBSERVATIONS, BEADS)

    # votes already intrinsically contain a notion of confidence in the pairing, but I want to
    # make it even more explicit by discrediting votes for observations that are clearly out of range.
    # Weight each observation in each channel: 0 = out of range, 1 = in range (+ smooth at the edges)
    weights = vmap(w_function)(observations).T  # (CHANNELS, OBSERVATIONS) -> "validity" from range
    weighted_votes = votes * weights[:, :, None] + 1e-12  # (CHANNELS, OBSERVATIONS, BEADS)
    vmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]  # weighted average

    # Use these votes to decide which beads are the most likely for each observation
    # Tried with a softer version of this just in case, but I couldn't see any improvement
    vmat = jnp.argmax(vmat, axis=1)[:, None] == jnp.arange(vmat.shape[1])[None, :]

    # We add some tiny random normal noise to avoid singular matrix errors when computing the KDE
    # on a bead that would have only the exact same value (which can happen when out of range)
    noise_std = (peaks_max_x - peaks_min_x) / (resolution * 5)
    obs = observations + jax.random.normal(jax.random.PRNGKey(0), observations.shape) * noise_std

    # Now we can compute the densities for each bead in each channel
    x = jnp.linspace(peaks_min_x, peaks_max_x, resolution)
    w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=bw_method)(x)
    densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(obs, vmat)
    densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)

    # find the most likely peak positions for each bead using the average intensity weighted by density
    DENSITY_TH = (
        0.25  # min threshold (as fraction of max) for density to be included in the average
    )
    RANGE_TH = (
        0.2  # min weight from the validity range (to avoid giving importance to saturated values)
    )
    normalized_densities = densities / jnp.max(densities, axis=2)[:, :, None]
    w = jnp.where(
        (normalized_densities > DENSITY_TH) & (w_function(x) > RANGE_TH)[None, None, :],
        normalized_densities - DENSITY_TH,
        0,
    )
    w = jnp.where(
        normalized_densities > DENSITY_TH, jnp.maximum(w, 1e-9), 0
    )  # still some tiny weight when OOR
    xx = jnp.tile(x, (densities.shape[0], densities.shape[1], 1))
    peaks = jnp.average(xx, axis=2, weights=w)  # peaks.shape is (BEADS, CHANNELS)

    # let's also estimate a confidence score for each peak
    # we will compute that by how much overlap ther is between each bead's distributions

    return peaks, (densities, vmat)


def beads_fit_spline(
    logpeaks, logcalib, spline_degree=2, smooth_percent=1, CF=100, ignore_first_bead=True, **_
):
    X = logpeaks * CF  # (NBEADS, NCHAN)
    Y = logcalib * CF
    NBEADS, NCHAN = X.shape

    # it's necessary that the function is monotonous for invertibility
    ydiff = np.diff(Y, axis=0)
    assert np.all(ydiff >= 0)

    # any distance between peaks that is less than MIN_PERCENT_DIST%
    # of the range will be given a weight that's inversely proportional to the distance
    # Peaks so close together probably means the bead data is overlapping in this region
    # and we can't be that confident in the fit
    MIN_PERCENT_DIST = 3
    dist_fwd = np.abs(X[1:] - X[:-1])
    dist_fwd = np.concatenate([dist_fwd[0:1], dist_fwd], axis=0)
    dist_bwd = np.abs(X[:-1] - X[1:])
    dist_bwd = np.concatenate([dist_bwd, dist_bwd[-1:]], axis=0)
    avg_dist_to_neigh = (dist_fwd + dist_bwd) / 2
    w = np.clip(avg_dist_to_neigh / MIN_PERCENT_DIST, 0, 1)
    # check that X is ordered
    order = np.argsort(X, axis=0)
    # detect if there are any beads that are out of order and put their weight to a very small value
    w = np.where(order != np.arange(0, NBEADS)[:, None], 1e-6, w)

    # reorder, but only X
    X = X[order, np.arange(0, NCHAN)]

    if ignore_first_bead:  # almost...
        w[0, :] = 1e-6

    splines = [
        UnivariateSpline(
            X[:, c], Y[:, c], k=spline_degree, s=smooth_percent, check_finite=True, w=w[:, c]
        )
        for c in range(0, NCHAN)
    ]
    inv_splines = [
        UnivariateSpline(
            Y[:, c], X[:, c], k=spline_degree, s=smooth_percent, check_finite=True, w=w[:, c]
        )
        for c in range(0, NCHAN)
    ]

    def make_transform(s):
        def transform(x):
            return s(x * CF) / CF

        return transform

    transforms = [make_transform(s) for s in splines]
    inv_transforms = [make_transform(s) for s in inv_splines]
    return transforms, inv_transforms

##────────────────────────────────────────────────────────────────────────────}}}

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

def plot_beads_diagnostics(self):

    plot_bead_peaks_diagnostics(
        self.__log_beads_peaks,
        self.__beads_densities,
        self.__beads_vmat,
        self.__log_beads_data,
        self.beads_channel_order,
    )

    plot_beads_after_correction(
        self.beads_transform,
        self.__log_beads_peaks,
        self.__log_beads_mef,
        self.__log_beads_data,
        self.beads_channel_order,
        self.channel_to_unit,
        title="""Beads-based MEF calibration
        right side shows beads intensities in their respective MEF units
        left side shows the real beads data after calibration\n\n""",
    )
    plt.show()



