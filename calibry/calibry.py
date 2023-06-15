import jax

# jax.config.update("jax_platform_name", "cpu")
import optax
from jax import jit, vmap

import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

from scipy.interpolate import UnivariateSpline

import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import flowio

from typing import List, Dict, Tuple
import logging


### {{{             --     some default bead configurations     --

# remember: the first bead is not reliable and should'nt contribute to the calibration
SPHEROTECH_RCP_30_5a = {
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


FORTESSA_CHANNELS = {
    'Pacific_Blue_A': 'PacBlue',
    'AmCyan_A': 'MEAMCY',
    'FITC_A': 'MEFL',
    'PerCP_Cy5_5_A': 'MEPCY5.5',
    'PE_A': 'MEPE',
    'PE_Texas_Red_A': 'MEPTR',
    'APC_A': 'MEAPC',
    'APC_Alexa_700_A': 'MEAX700',
    'APC_Cy7_A': 'MEAPCCY7',
}

FORTESSA_MAX_V = 2**18 - 1

FORTESSA_OFFSET = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{              --     small tools     --
def escape_name(name):
    return name.replace('-', '_').replace(' ', '_').upper().removesuffix('_A')


def escape(names):
    if isinstance(names, str):
        return escape_name(names)
    if isinstance(names, list):
        return [escape_name(name) for name in names]
    if isinstance(names, tuple):
        return tuple([escape_name(name) for name in names])
    if isinstance(names, dict):
        return {escape_name(k): escape_name(v) for k, v in names.items()}
    else:
        return names


def astuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


# weights observations: 0 if out of range, 1 if in range
def w_function(x, a=0, b=0.99, lsteepness=10, rsteepness=1000):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


def load_fcs_to_df(fcs_file):
    fcs_data = flowio.FlowData(fcs_file.as_posix())
    channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]
    original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    return pd.DataFrame(original_data, columns=escape(channels))


def load_to_df(data, column_order=None):
    # data can be either a pandas dataframe of a path to a file
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = escape(list(df.columns))
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        df = pd.DataFrame(data)
    else:
        data = Path(data)
        assert data.exists(), f'File {data} does not exist'
        if data.suffix == '.fcs':
            df = load_fcs_to_df(data)
        elif data.suffix == '.csv':
            df = pd.read_csv(data)
            df.columns = escape(list(df.columns))
        else:
            raise ValueError(f'File {data} has unknown suffix {data.suffix}')
    if column_order is not None:
        df = df[escape(column_order)]
    # reset index so that it is ordered
    df = df.reset_index(drop=True)
    return df


def logtransform(x, scale, offset=0):
    x = jnp.clip(x, 1e-9, None)
    return jnp.log10(x + offset) / scale


def inverse_logtransform(x, scale, offset=0):
    return 10 ** (x * scale) - offset


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     beads related functions     --


CALIBRATION_PEAKS_MIN_X = -0.025
CALIBRATION_PEAKS_MAX_X = 1.025


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


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fluo_distribution(ax, data, res=2000):
    from jax.scipy.stats import gaussian_kde

    xmax = np.max(data)
    XX = np.linspace(0.0, 1.1 * xmax, res)
    kde = gaussian_kde(data.T, bw_method=0.01)
    densities = kde(XX.T)
    ldensities = np.log10(1.0 + densities)
    densities = (densities / densities.max()) * 0.4
    ldensities = (ldensities / ldensities.max()) * 0.4
    ax.fill_betweenx(XX, -ldensities, 0, color='k', alpha=0.2, lw=0)
    ax.fill_betweenx(XX, -densities, 0, color='k', alpha=0.2, lw=0)
    ax.plot(-densities, XX, color='k', alpha=0.5, lw=0.7)
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)


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
        remove_axis_and_spines(ax)
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

            plot_fluo_distribution(ax, tdata[:, c])
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

### {{{                 --     simple optimization functions   --


def compute_spectral_signature(Y, M, max_iterations=50, jax_seed=0, verbose=False):
    """
    Spectral signature estimation using Alternating Least Squares (ALS) method.

    Args:
        Y (array): Observed fluorescence intensities (cells x channels).
        M (array): Masks matrix from single-color control samples (proteins x channels).
        max_iterations (int, optional): Maximum number of iterations for ALS. Defaults to 50.
        jax_seed (int, optional): Seed for JAX random number generator. Defaults to 0.
        verbose (bool, optional): Display progress information. Defaults to False.

    Returns:
        S (array): Spectral signature matrix (proteins x channels).
        K (array): Estimated protein quantities (proteins x cells).
    """
    # model: Y = XM.S
    # Y: observations
    # M: masks (from controls: single color = (1,0,...))
    # S: spectral signature
    # X: estimated quantity of protein.

    normalize = lambda x: x / (jnp.maximum(jnp.max(x, axis=1)[:, None], 1e-12))
    K = jax.random.uniform(jax.random.PRNGKey(jax_seed), (M.shape[0],), minval=0.1)

    @jit
    def alsq(K):  # one iteration of alternating least squares
        S = jnp.linalg.lstsq(K[:, None] * M, Y, rcond=None)[0]
        S = normalize(S)  # find S from K
        K = Y @ jnp.linalg.pinv(S)  # find K from S
        K = jnp.average(K, axis=1, weights=M)
        K = jnp.maximum(jnp.nan_to_num(K, nan=0), 0)
        return S, K

    Sprev = 0
    for _ in range(max_iterations):
        S, K = alsq(K)
        error = ((Y - (K[:, None] * M) @ S) ** 2).mean() / jnp.linalg.norm(Y)
        if verbose:
            print(f'error: {error:.2e}, update: {jnp.mean((Sprev - S)**2):.2e}, S: {S}')
        Sprev = S

    return S, K


def affine_opt(Ylog, ref_channel):
    # Ylog is a matrix of log-transformed fluorescence values
    # we need to find the best affine mapping to the reference channel
    # Ylog[:, i] = a_i * Ylog[:, refchanid] + b_i
    # so we solve the linear system
    # Ylog[:, i] = A @ [a_i, b_i] (with A = [Ylog[:, refchanid], 1])

    ref = Ylog[:, ref_channel]

    def solve_column(y):
        A = jnp.stack([y, jnp.ones_like(y)], axis=1)
        return jnp.linalg.lstsq(A, ref, rcond=None)[0]

    res = vmap(solve_column)(Ylog.T)
    a, b = res[:, 0], res[:, 1]
    return a, b


def affine_gd(Y, ref_channel, num_iter=1000, learning_rate=0.1, max_N=500000, verbose=False):
    # gradient descent version of the above affine fit

    NCHAN = Y.shape[1]
    if len(Y) > max_N:
        choice = jax.random.choice(jax.random.PRNGKey(0), len(Y), (min(max_N, len(Y)),))
        Y = Y[choice]

    params = {
        'a': jnp.ones((NCHAN,)),
        'b': jnp.zeros((NCHAN,)),
        'c': jnp.zeros((NCHAN,)),
    }

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    Yref = Y[:, ref_channel]

    @jax.value_and_grad
    def lossf(params):
        yhat = params['a'] * Y + params['b']
        yhat = jnp.maximum(yhat, 0)
        err = (yhat - Yref[:, None]) ** 2
        err = err
        return jnp.mean(err**2)

    @jit
    def update(params, opt_state):
        loss, grad = lossf(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i in range(0, num_iter):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if verbose:
            print(f'iter {i}: loss = {loss:.2e}')

    a, b = params['a'], params['b']
    return a, b


from tqdm import tqdm


from scipy.interpolate import LSQUnivariateSpline, interp1d

DEFAULT_CMAP_Q = np.array([0.7])  # where to put spline knots
DEFAULT_CLAMP_Q = np.array([0, 1])  # where to clamp values


def cmap_spline(
    X,
    Y,
    spline_order=3,
    CF=100,
    spline_knots_at_quantile=DEFAULT_CMAP_Q,
    switch_to_linear_at_quantile=[0.02, 0.98],
):
    X, Y = X * CF, Y * CF
    x_order = np.argsort(X, axis=0)
    Xx, Yx = np.take_along_axis(X, x_order, axis=0), Y[x_order]
    w = 0.25 * np.ones_like(Y)

    xknots = [np.quantile(X[:, c], spline_knots_at_quantile) for c in range(X.shape[1])]
    print(f'xknots = {xknots}')

    x_linear_transition_lower = np.quantile(
        X, switch_to_linear_at_quantile[0], axis=0
    )  # switch to linear extrapolation at this point
    x_linear_transition_upper = np.quantile(
        X, switch_to_linear_at_quantile[1], axis=0
    )  # switch to linear extrapolation at this point

    splines = []

    for c in range(X.shape[1]):
        try:
            spline = LSQUnivariateSpline(Xx[:, c], Yx[:, c], k=spline_order, t=xknots[c], w=w)
            splines.append(spline)
        except ValueError as e:
            msg = f'Failed to fit spline for channel {c}, with X = {Xx[:, c]}, Y = {Yx[:, c]}, knots = {xknots[c]}: {e}'
            raise ValueError(msg)

    def linear_extrapolation_up(x, spline, x_end, y_end):
        slope = spline.derivative()(x_end)
        return y_end + slope * (x - x_end)

    def linear_extrapolation_down(x, spline, x_start, y_start):
        slope = max(spline.derivative()(x_start), 0)
        return y_start - slope * (x_start - x)

    def combined_spline(x, spline, x_start, y_start, x_end, y_end):
        up = np.where(x > x_end, linear_extrapolation_up(x, spline, x_end, y_end), spline(x))
        return np.where(x < x_start, linear_extrapolation_down(x, spline, x_start, y_start), up)

    y_transition_up = np.array(
        [spline(x_trans) for spline, x_trans in zip(splines, x_linear_transition_upper)]
    )

    y_transition_down = np.array(
        [spline(x_trans) for spline, x_trans in zip(splines, x_linear_transition_lower)]
    )

    transform = lambda X: np.array(
        [
            combined_spline(x * CF, s, xd, yd, xu, yu) / CF
            for s, x, xd, yd, xu, yu in zip(
                splines,
                X.T,
                x_linear_transition_lower,
                y_transition_down,
                x_linear_transition_upper,
                y_transition_up,
            )
        ]
    ).T

    # TODO
    # y_order = np.argsort(Y, axis=0)
    # Xy, Yy = X[y_order], Y[y_order]
    # yknots = np.linspace(
    # np.quantile(Yy, Q[0], axis=0), np.quantile(Yy, Q[-1], axis=0), n_knots, endpoint=True
    # ).T
    # inv_splines = []
    # for c in range(X.shape[1]):
    # try:
    # spline = LSQUnivariateSpline(Yy, Xy[:, c], k=spline_order, t=yknots, w=w)
    # inv_splines.append(spline)
    # except:
    # msg = f'failed to fit spline for channel {c}, with Y = {Yy}'
    # logging.warning(msg)
    # inv_splines.append(lambda x: x)

    inv_transform = lambda Y: None

    return transform, inv_transform


##────────────────────────────────────────────────────────────────────────────}}}

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

    def fit_beads(self, verbose=True, **kw):

        if verbose:
            print('Computing peaks assignment...')

        # normalizing beads values to [0;1] (fo the observations, not necessarily the reference values)
        self.log_scale_factor = jnp.log10(self.max_value + self.offset)
        calib_values = []
        for c in self.beads_channel_order:
            if c in self.channel_to_unit:
                # check that every channel has a MEF value
                if self.channel_to_unit[c] not in self.beads_mef_values:
                    raise ValueError(
                        f'No MEF values provided for channel {c} ({self.channel_to_unit[c]})'
                    )
                calib_values.append(self.beads_mef_values[self.channel_to_unit[c]])
            else:
                raise ValueError(f'Channel {c} not found in channel_to_unit dict')

        calib_values = jnp.array(calib_values).T
        NBEADS, NCHAN = calib_values.shape

        if verbose:
            print(f'Computing channel calibration to MEF using {NBEADS} beads in {NCHAN} channels')

        self.__log_beads_mef = logtransform(calib_values, self.log_scale_factor, 0)
        self.__log_beads_data = logtransform(self.beads.values, self.log_scale_factor, self.offset)
        self.__log_beads_peaks, (self.__beads_densities, self.__beads_vmat) = compute_peaks(
            self.__log_beads_data, self.__log_beads_mef
        )

        self.beads_transform, self.beads_inv_transform = beads_fit_spline(
            self.__log_beads_peaks, self.__log_beads_mef, **kw
        )

    def pick_reference_channels(self, max_sat_proportion=0.001, min_dyn_range=3):
        # for each protein ctrl, we'll pick a reference channel as the one with the highest dynamic range
        if self.user_provided_reference_channels is not None:
            user_provided = []
            for pid, p in enumerate(self.protein_names):
                if p in self.user_provided_reference_channels.keys():
                    ch = self.user_provided_reference_channels[p]
                    assert ch in self.channel_names, f'Unknown channel {ch} in reference_channels'
                    user_provided.append(self.channel_names.index(ch))
                else:
                    user_provided.append(None)
            unknown_provided = [
                p for p in self.user_provided_reference_channels if p not in self.protein_names
            ]
            if len(unknown_provided) > 0:
                print(
                    f'Unknown proteins {unknown_provided} in user_provided_reference_channels. Ignoring them.'
                )
        else:
            user_provided = [None] * len(self.protein_names)
        single_masks = (self.controls_masks.sum(axis=1) == 1).astype(bool)
        ref_channels = []
        for pid, prot in enumerate(self.protein_names):
            if user_provided[pid] is not None:
                ref_channels.append(user_provided[pid])
                continue
            prot_mask = (single_masks) & (self.controls_masks[:, pid]).astype(bool)
            prot_values = self.controls_values[prot_mask]
            prot_sat = (prot_values <= self.saturation_thresholds['lower']) | (
                prot_values >= self.saturation_thresholds['upper']
            )
            sat_proportion = prot_sat.sum(axis=0) / prot_sat.shape[0]
            dyn_range = np.log10(np.quantile(prot_values, [0.001, 0.999], axis=0).ptp(axis=0))
            effective_range = dyn_range.copy()
            effective_range[sat_proportion > max_sat_proportion] = 0
            effective_range[effective_range < min_dyn_range] = 0
            if effective_range.sum() == 0:
                ref_channels.append(None)
                raise ValueError(
                    f"""
                    No reference channel found for {prot}!
                    Try increasing max_sat_proportion and/or decreasing min_dyn_range'
                    Saturation proportions per channel: {sat_proportion}
                    Log10 dynamic range per channel: {dyn_range}
                    """
                )
            else:
                ref_channels.append(effective_range.argmax())
        self.reference_channels = ref_channels

    def estimate_autofluorescence(self):
        if self.verbose:
            print('Estimating autofluorescence...')
        zero_masks = self.controls_masks.sum(axis=1) == 0
        self.autofluorescence = np.median(self.controls_values[zero_masks], axis=0)

    def compute_saturation_thresholds(self):
        self.saturation_thresholds = {
            'lower': np.quantile(self.controls_values, 0.0001, axis=0),
            'upper': np.quantile(self.controls_values, 0.9999, axis=0),
        }

    def compute_linear_spectral_signature(self):
        if self.verbose:
            print('Computing spectral signature with the linear model...')
        single_masks = self.controls_masks.sum(axis=1) == 1
        masks = self.controls_masks[single_masks]
        Y = self.controls_values[single_masks] - self.autofluorescence
        self.spillover_matrix, _ = compute_spectral_signature(Y, masks)

    def pick_calibration_units(self):
        reference_protein_id = self.protein_names.index(self.reference_protein)
        reference_channel_id = self.reference_channels[reference_protein_id]
        reference_channel_name = self.channel_names[reference_channel_id]
        # check that the reference protein's reference channel is available
        if self.channel_to_unit[reference_channel_name] is None:
            raise ValueError(
                f'Cannot calibrate with {reference_channel_name} because its unit is unknown'
            )
        if reference_channel_name not in self.beads_channel_order:
            raise ValueError(
                f'Cannot calibrate with {reference_channel_name} because it is not selected for beads calibration'
            )
        self.calibration_units = self.channel_to_unit[self.channel_names[reference_channel_id]]
        self.calibration_channel_name = reference_channel_name
        self.calibration_channel_id = self.beads_channel_order.index(reference_channel_name)

        if self.verbose:
            print(f'Calibrated abundances will be expressed in {self.calibration_units}')

    def filter_saturated(self, Y, only_in_reference_channel=True):
        if only_in_reference_channel:
            Y_rc = Y[:, self.reference_channels]
            sat_rc_low = self.saturation_thresholds['lower'][self.reference_channels]
            sat_rc_high = self.saturation_thresholds['upper'][self.reference_channels]
            selection = ((Y_rc > sat_rc_low) & (Y_rc < sat_rc_high)).all(axis=1)
            return selection
        else:
            selection = (
                (Y > self.saturation_thresholds['lower'])
                & (Y < self.saturation_thresholds['upper'])
            ).all(axis=1)
            return selection

    def map_proteins_to_reference(self, unmix_method):
        if self.verbose:
            print(f'Computing all protein mapping to {self.reference_protein}...')
        all_masks = self.controls_masks.sum(axis=1) > 1
        Y = self.controls_values[all_masks] - self.autofluorescence
        selection = self.filter_saturated(Y, only_in_reference_channel=True)
        Y = Y[selection]
        X = unmix_method(Y) + self.offset
        refprotid = self.protein_names.index(self.reference_protein)
        X = X[jnp.all(X > 1, axis=1)]
        logX = logtransform(X, self.log_scale_factor)
        self.cmap_transform, self.cmap_inv_transform = cmap_spline(logX, logX[:, refprotid])

    def unmix(self, Y):
        return Y @ jnp.linalg.pinv(self.spillover_matrix)  # apply bleedthrough

    def unmix_nnls(self, Y):
        from scipy.optimize import nnls
        from tqdm import tqdm

        X = np.empty((Y.shape[0], self.spillover_matrix.shape[1]))
        for i in tqdm(range(Y.shape[0])):
            X[i, :], _ = nnls(self.spillover_matrix.T, Y[i, :])
        return X

    def fit(self, verbose=True, **_):
        self.verbose = verbose
        t0 = time.time()

        self.compute_saturation_thresholds()
        self.pick_reference_channels()
        self.pick_calibration_units()

        self.estimate_autofluorescence()

        self.fit_beads()

        self.compute_linear_spectral_signature()

        self.map_proteins_to_reference(self.unmix)

        print('Done fitting in {:.2f} seconds'.format(time.time() - t0))
        self.fitted = True

    def apply_bead_calibration(self, X):
        calibratedX = jnp.array(
            [self.beads_transform[self.calibration_channel_id](X[:, i]) for i in range(X.shape[1])]
        ).T
        return calibratedX

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


    def plot_bleedthrough_diagnostics(self, figsize=None):
        NCHANNELS = len(self.channel_names)
        NPROTS = len(self.protein_names)
        figsize = figsize or (NCHANNELS * 1.2, NPROTS * 1.2)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.spillover_matrix, cmap='Greys', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(self.channel_names)))
        ax.set_xticklabels(self.channel_names, rotation=45)
        ax.set_yticks(range(len(self.protein_names)))
        ax.set_yticklabels(self.protein_names)
        ax.set_xlabel('Channels')
        ax.set_ylabel('Proteins')
        # no grid lines
        ax.grid(False)
        # write values in the center of each square
        for i in range(len(self.protein_names)):
            for j in range(len(self.channel_names)):
                color = 'w' if self.spillover_matrix[i, j] > 0.5 else 'k'
                text = ax.text(
                    j,
                    i,
                    '{:.3f}'.format(self.spillover_matrix[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                )
        # fig.colorbar(im, ax=ax)
        plt.show()

    def plot_mapping_diagnostics(
        self, scatter_size=20, scatter_alpha=0.2, max_n_points=15000, figsize=(10, 10)
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plot transform
        NP = len(self.protein_names)
        all_masks = self.controls_masks.sum(axis=1) > 1
        Y = self.controls_values[all_masks] - self.autofluorescence
        X = Y @ jnp.linalg.pinv(self.spillover_matrix)  # apply bleedthrough
        X += self.offset  # add offset
        logX = logtransform(X, self.log_scale_factor, 0)
        # logX = logX[jnp.all(logX > self.clamp_values[0], axis=1)]
        # logX = logX[jnp.all(logX < self.clamp_values[1], axis=1)]
        refid = self.protein_names.index(self.reference_protein)
        target = logX[:, refid]

        scatter_x = []
        scatter_y = []
        scatter_color = []

        for i in range(NP):
            fpname = self.protein_names[i]
            color = get_bio_color(fpname)
            xrange = (logX[:, i].min(), logX[:, i].max())
            xx = np.linspace(*xrange, 200)
            yy = self.cmap_transform(np.tile(xx, (NP, 1)).T)[:, i]
            scatter_x.append(logX[:, i].flatten())
            scatter_y.append(target.flatten())
            scatter_color.append([color] * len(logX))
            ax.plot(xx, yy, alpha=1, lw=5, color='w')
            if i == refid:
                fpname += ' (reference)'
            ax.plot(xx, yy, label=fpname, alpha=1, lw=1.5, color=color, zorder=10)

        scatter_x = np.concatenate(scatter_x)
        scatter_y = np.concatenate(scatter_y)
        scatter_color = np.concatenate(scatter_color)
        NPOINTS = min(max_n_points, len(scatter_x))
        order = np.random.permutation(len(scatter_x))[:NPOINTS]

        ax.scatter(
            scatter_x[order],
            scatter_y[order],
            c=scatter_color[order],
            s=scatter_size,
            alpha=scatter_alpha,
            lw=0,
            zorder=0,
        )

        yrange = (target.min(), target.max())
        xrange = (scatter_x.min(), scatter_x.max())
        ax.set_ylim(yrange)
        ax.set_xlim(xrange)
        ax.legend()
        ax.set_xlabel('source')
        ax.set_ylabel('target')
        # no grid, grey background
        ax.grid(False)
        ax.set_title(f'Color mapping to {self.reference_protein}')
        plt.show()

    def plot_unmixing_diagnostics(self, use_func=None):
        NPROT = len(self.protein_names)
        FSIZE = 5
        fig, axes = plt.subplots(NPROT, NPROT, figsize=(NPROT * FSIZE, NPROT * FSIZE))
        xlims = np.array([-1e6, 1e6])
        ylims = np.array([-1e3, 1e6])
        single_masks = (self.controls_masks.sum(axis=1) == 1).astype(bool)
        for ctrl_id, ctrl_name in tqdm(list(enumerate(self.protein_names))):
            prot_mask = (single_masks) & (self.controls_masks[:, ctrl_id]).astype(bool)
            Y = self.controls_values[prot_mask]
            if use_func is not None:
                X = use_func(Y)
            else:
                X_MEF, X_AU, selected = self.apply_to_array(Y)
                X = X_AU
            for pid, prot_name in enumerate(self.protein_names):
                ax = axes[ctrl_id, pid]
                if pid == ctrl_id:
                    ax.text(
                        0.5,
                        0.5,
                        f'unmixing of\n'
                        f'{prot_name} control\n'
                        f'(linear model, {len(self.channel_names)} channels)',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        fontsize=14,
                    )
                    ax.axis('off')
                    continue
                ax.scatter(X[:, pid], X[:, ctrl_id], s=0.1, alpha=0.2, c='k')
                ax.set_ylabel(f'{ctrl_name}')
                ax.set_xlabel(f'{prot_name}')
                ax.set_xscale('symlog', linthresh=(50), linscale=0.4)
                ax.set_yscale('symlog', linthresh=(50), linscale=0.2)
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
        fig.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}##


### {{{                    --     plot styling tools     --

CHAN_WAVELENGTHS = {
    'apc': 650,
    'apc_cy7': 780,
    'fitc': 500,
    'amcyan': 490,
    'pe': 578,
    'pe_texas_red': 615,
    'apc_alexa_700': 700,
    'pacific_blue': 452,
}


def wave2rgb(wave):
    # This is a port of javascript code from  http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 0.8
    wave = np.clip(wave, 385, 775)
    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0
    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0

    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow(c * factor, gamma)

    return f(red), f(green), f(blue)


def get_bio_color(name, default='k'):
    import difflib

    colors = {
        'ebfp': '#529edb',
        'eyfp': '#fbda73',
        'mkate': '#f75a5a',
        'neongreen': '#33f397',
        'irfp720': '#97582a',
    }

    # colors['fitc'] = colors['neongreen']
    # colors['gfp'] = colors['neongreen']
    # colors['pe_texas_red'] = colors['mkate']
    # colors['mcherry'] = colors['mkate']
    # colors['pacific_blue'] = colors['ebfp']
    # colors['tagbfp'] = colors['ebfp']
    # colors['AmCyan'] = '#00a4c7'
    # colors['apc'] = '#f75a5a'
    # colors['percp_cy5_5'] = colors['mkate']

    for k, v in CHAN_WAVELENGTHS.items():
        colors[k] = wave2rgb(v)

    closest = difflib.get_close_matches(name.lower(), colors.keys(), n=1)
    if len(closest) == 0:
        color = default
    else:
        color = colors[closest[0]]
    return color


from mpl_toolkits.axes_grid1 import make_axes_locatable


def mkfig(rows, cols, size=(7, 7), **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]), **kw)
    return fig, ax


def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_topright_spines(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_size(w, h, axes, fig):
    """w, h: width, height in inches"""
    l = min([ax.figure.subplotpars.left for ax in axes])
    r = max([ax.figure.subplotpars.right for ax in axes])
    t = max([ax.figure.subplotpars.top for ax in axes])
    b = min([ax.figure.subplotpars.bottom for ax in axes])
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


def style_violin(parts):
    for pc in parts['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
        pc.set_linewidth(1)
    parts['cbars'].set_linewidth(0)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.5)


import string


class ShortScientificFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 'm':
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), '')
                else:
                    return super().format_field(value, '.1f')
            else:
                if value == int(value):
                    return super().format_field(value, '.0e').replace('e+0', 'e').replace('e+', 'e')
                else:
                    return super().format_field(value, '.1e').replace('e+0', 'e').replace('e+', 'e')
        else:
            return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     plotting tools     --

#


def fluo_scatter(
    rawx,
    pnames,
    xmin=0,
    xmax=None,
    title=None,
    types=None,
    fname=None,
    logscale=True,
    alpha=None,
    s=5,
    **_,
):
    NPOINTS = rawx.shape[0]
    if alpha is None:
        alpha = np.clip(1 / np.sqrt(NPOINTS / 50), 0.01, 1)
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)
    if len(pnames) == 1:
        axes = [axes]
    if types is None:
        types = [''] * len(pnames)

    xmin = rawx.min() if xmin is None else 10**xmin
    xmax = rawx.max() if xmax is None else 10**xmax

    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        xcoords = jax.random.normal(jax.random.PRNGKey(0), (rawx.shape[0],)) * 0.1
        ax.scatter(xcoords, rawx[:, xid], color=color, alpha=alpha, s=s, zorder=10, lw=0)
        if logscale:
            ax.set_yscale('symlog')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
        remove_spines(ax)
        ax.set_xticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)

    return fig, axes


def scatter_matrix(
    Y,
    dim_names,
    subsample=50000,
    log=False,
    plot_size=4,
    dpi=200,
    color='k',
    s=0.1,
    alpha=0.2,
    c=None,
    xlims=[-1e4, 1e4],
    ylims=[-1e2, 1e5],
    **kwargs,
):
    n_channels = Y.shape[1]
    idx = np.random.choice(Y.shape[0], min(subsample, Y.shape[0]), replace=False)
    Ysub = Y[idx, :]
    if c is not None:
        c = c[idx]
    # plot each axis against each axis (with scatter plots)
    fig, axes = plt.subplots(
        n_channels, n_channels, figsize=(n_channels * plot_size, n_channels * plot_size), dpi=dpi
    )
    for row in range(n_channels):
        for col in range(n_channels):
            ax = axes[row, col]
            sc = ax.scatter(
                Ysub[:, col], Ysub[:, row], s=s, alpha=alpha, color=color, c=c, **kwargs
            )

            if row == n_channels - 1:
                ax.set_xlabel(dim_names[col])

            if col == 0:
                ax.set_ylabel(dim_names[row])

            if log:
                ax.set_xscale('symlog', linthresh=(50), linscale=0.4)
                ax.set_yscale('symlog', linthresh=(50), linscale=0.2)
            # ax.set_xlim(xlims)
            # ax.set_ylim(ylims)

    return fig, axes


def unmixing_plot(
    controls_observations,
    controls_abundances,
    protein_names,
    channel_names,
    xlims=[-1e6, 1e6],
    ylims=[-1e2, 1e6],
    **_,
):
    # adding the 'all' to protein names:
    NPROT = len(protein_names)
    FSIZE = 5
    fig, axes = plt.subplots(NPROT, NPROT, figsize=(NPROT * FSIZE, NPROT * FSIZE))
    for ctrl_id, ctrl_name in enumerate(protein_names):
        Y = controls_observations[ctrl_name]
        X = controls_abundances[ctrl_name]
        for pid, prot_name in enumerate(protein_names):
            ax = axes[ctrl_id, pid]
            if pid == ctrl_id:
                ax.text(
                    0.5,
                    0.5,
                    f'unmixing of\n'
                    f'{prot_name} control\n'
                    f'(linear model, {len(channel_names)} channels)',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis('off')
                continue
            ax.scatter(X[:, pid], X[:, ctrl_id], s=0.1, alpha=0.2, c='k')
            ax.set_ylabel(f'{ctrl_name}')
            ax.set_xlabel(f'{prot_name}')
            ax.set_xscale('symlog', linthresh=(50), linscale=0.4)
            ax.set_yscale('symlog', linthresh=(50), linscale=0.2)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
    fig.tight_layout()
    return fig, axes


def spillover_matrix_plot(spillover_matrix, channel_names, protein_names, figsize=None, ax=None, **_):
    print(spillover_matrix)
    NCHANNELS = len(channel_names)
    NPROTS = len(protein_names)
    figsize = figsize or (NCHANNELS * 1.2, NPROTS * 1.2)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    im = ax.imshow(spillover_matrix, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45)
    ax.set_yticks(range(len(protein_names)))
    ax.set_yticklabels(protein_names)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Proteins')
    # no grid lines
    ax.grid(False)
    # write values in the center of each square
    for i in range(len(protein_names)):
        for j in range(len(channel_names)):
            color = 'w' if spillover_matrix[i, j] > 0.5 else 'k'
            text = ax.text(
                j,
                i,
                '{:.3f}'.format(spillover_matrix[i, j]),
                ha="center",
                va="center",
                color=color,
            )
    return fig, ax

##────────────────────────────────────────────────────────────────────────────}}}
