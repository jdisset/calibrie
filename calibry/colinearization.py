from joblib import Memory
cachedir = '/tmp/'
memory = Memory(cachedir, verbose=0)
from jaxopt import GaussNewton
from .pipeline import Task
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial
from tqdm import tqdm

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


### {{{                        --     utils     --
def partition_uniform_resample(x, N, partition, qrange=None, logspace=True):
    """
    Partition x into partitions (either int or array-like of boundaries), and resample each partition
    to have a number of samples proportional to the range it covers, (N samples in total).
    """

    transform = lambda x: x
    if logspace:
        transform = lambda x: np.log10(x - x.min() + 50)
    xtr = transform(x)
    if qrange is None:
        qrange = np.quantile(xtr, [0.001, 0.99])

    if isinstance(partition, int):
        partition_bounds = np.linspace(*qrange, partition + 1)[1:-1]
    else:
        partition_bounds = transform(np.array(partition))

    partition_sizes = np.diff(np.concatenate([[qrange[0]], partition_bounds, [qrange[1]]]))
    partition_ratios = partition_sizes / (qrange[1] - qrange[0])

    npartitions = len(partition_ratios)

    partition_assignment = np.digitize(xtr, partition_bounds)
    actual_partition_counts = np.bincount(partition_assignment)

    target_partition_counts = (N * partition_ratios).astype(int)

    missing = N - np.sum(target_partition_counts)
    target_partition_counts[np.argmin(target_partition_counts)] += missing

    need_replacement = target_partition_counts > actual_partition_counts

    selected_indices = [
        np.random.choice(np.where(partition_assignment == i)[0], size=tc, replace=r)
        for i, tc, r in zip(np.arange(npartitions), target_partition_counts, need_replacement)
    ]

    ids = np.random.permutation(np.concatenate(selected_indices))

    return ids, actual_partition_counts, target_partition_counts


def resampled_singleprt_ctrls(
    controls_values, protein_names, controls_masks, reference_channels, N, npartitions=10
):
    """
    Returns a hopefully slightly more uniformly distributed subsampled array of control values,
    with shape (nprot, N, nchan)
    """

    # let's isolate all the available single controls
    single_masks = np.eye(len(protein_names)).astype(bool)  # one hot encoding
    null_observations = controls_values[np.all(controls_masks == 0, axis=1)]
    singlectrl_observations = [
        controls_values[np.all(controls_masks == sc, axis=1)] for sc in single_masks
    ]
    if npartitions <= 1:
        return np.array(
            [x[np.random.choice(x.shape[0], N, replace=True)] for x in singlectrl_observations]
        )

    singlectrl_ref_observations = [
        sc[:, rc].squeeze() for sc, rc in zip(singlectrl_observations, reference_channels)
    ]
    null_ref_std = null_observations[:, reference_channels].std(axis=0)
    null_ref_mean = null_observations[:, reference_channels].mean(axis=0)
    normed_singlectrl_ref_observations = [
        (x - m) / s for x, m, s in zip(singlectrl_ref_observations, null_ref_mean, null_ref_std)
    ]
    Ysingle_ids = [
        partition_uniform_resample(x, N, npartitions, logspace=True)
        for x in normed_singlectrl_ref_observations
    ]

    Ysingle = np.array([x[ids[0]] for x, ids in zip(singlectrl_observations, Ysingle_ids)])

    return Ysingle


def draw_path(
    linearization_path,
    best_range_prot_per_channel,
    protein_names,
    channel_names,
    absolute_ranges,
    relative_ranges,
    Wsingle,
    dpi=100,
):
    nchannels = len(channel_names)
    nproteins = len(protein_names)
    # draw the path:
    fig, ax, coords2d = plots.range_plots(
        protein_names, channel_names, absolute_ranges, relative_ranges, Wsingle, dpi=dpi
    )

    for ngrp, p in enumerate(linearization_path):
        group_letter = chr(ord("A") + ngrp)
        group_label = f'{group_letter}: ' if len(linearization_path) > 1 else ''
        group_color = plt.cm.tab10(ngrp)
        for nbatch, lbatch in enumerate(p):
            for source_chan, dest_chan, quality in lbatch:
                source_chan = int(source_chan)
                dest_chan = int(dest_chan)
                # draw arrow from source to dest
                prot_id = best_range_prot_per_channel[source_chan]
                source_xy = coords2d[prot_id * nchannels + source_chan]
                dest_xy = coords2d[prot_id * nchannels + dest_chan]
                # we need a curved arrow so they don't overlap
                # we can use the upper arc of a circle
                h = 0.3
                offset = np.array([0, 0.3])
                a = source_xy + offset
                b = dest_xy + offset
                if source_chan < dest_chan:
                    h = -h
                lw = 2
                l = np.sum(np.abs((a - b)))

                color = plt.cm.RdYlGn(quality)

                a3 = patches.FancyArrowPatch(
                    a,
                    b,
                    connectionstyle=f"arc3,rad={h/l**0.9}",
                    arrowstyle='->',
                    mutation_scale=10,
                    lw=lw,
                    color=color,
                )
                ax.add_patch(a3)
                # write batchn at the axis label level of the soure_chan
                ax.text(
                    source_xy[0],
                    nproteins,
                    f"{group_label}[{nbatch+1}/{len(p)}]",
                    ha='center',
                    va='center',
                    fontsize=8,
                    c=group_color,
                    alpha=0.75,
                )

    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     linearization core tools    --


@memory.cache
def find_colinearization_sequence(
    start_channel: int,
    remaining_channel_ids: np.ndarray,
    absolute_ranges: np.ndarray,
    best_range_prot_per_channel: np.ndarray,
    range_threshold: float,
):

    """
    Finds a sequence of linearization steps.
    A linearization step is a list of (source channel, destination channel, quality) tuples
    that specifies which (source) channel should be colinearized with which (destination) one. The quality is a heuristic for the
    quality of the regression (expressing source_chan as a function of dest_chan).
    A full sequence can be composed of multiple steps: each step has to be performed on the data transformed by all the previous steps.
    The algorithm here is greedy so potentially not optimal: it tries to linearize as many channels as possible at each step.
    Should be good enough for most realistic cases.
    """

    norm_ranges = absolute_ranges / absolute_ranges.max()

    def _impl(
        linearized_channels_ids: np.ndarray,
        linearized_channels_quality: np.ndarray,
        remaining_channel_ids: np.ndarray,
    ):

        # protein id of the best range for each channel
        allpsources = best_range_prot_per_channel[remaining_channel_ids]
        # shape of *_dest_ranges_in_source_prot should be (nremaining_channels, nlinearized_channels),
        norm_dest_ranges_in_source_prot = norm_ranges[allpsources[:, None], linearized_channels_ids]

        absolute_dest_ranges_in_source_prot = absolute_ranges[
            allpsources[:, None], linearized_channels_ids
        ]
        absolute_source_ranges = absolute_ranges[
            allpsources, remaining_channel_ids
        ]  # shape (nremaining_channels,)

        allpsources
        norm_dest_ranges_in_source_prot
        absolute_source_ranges
        absolute_dest_ranges_in_source_prot

        # mapping quality is a heuristic for the quality of the regression (expressing source_chan as a function of dest_chan)
        # as a first approximation, range(x) / range(y) is not too bad to favor mapping to channels with a lot of signal
        mapping_qualities = (
            np.clip(
                absolute_dest_ranges_in_source_prot
                / np.clip(absolute_source_ranges[:, None], 1e-6, None),
                0,
                1,
            )
            * linearized_channels_quality  # compounds (dimnishes) quality when mapping to a channel linearized in prev step
        )

        qualities = np.where(
            norm_dest_ranges_in_source_prot > range_threshold, mapping_qualities, -1
        )
        best_dests = linearized_channels_ids[np.argmax(qualities, axis=1)]
        best_qualities = np.max(qualities, axis=1)
        valid = best_qualities > 0
        next_round = np.stack(
            [remaining_channel_ids[valid], best_dests[valid], best_qualities[valid]], axis=1
        )
        next_remaining_channel_ids = remaining_channel_ids[~valid]

        if len(next_remaining_channel_ids) == 0 or len(next_round) == 0:
            return [next_round]

        next_linearized_ids = np.concatenate(
            [linearized_channels_ids, remaining_channel_ids[valid]]
        )
        next_linearized_quality = np.concatenate(
            [linearized_channels_quality, best_qualities[valid]]
        )

        next_linearized_ids
        next_remaining_channel_ids

        return [next_round] + _impl(
            next_linearized_ids,
            next_linearized_quality,
            next_remaining_channel_ids,
        )

    remaining_minus_start = np.setdiff1d(remaining_channel_ids, [start_channel])
    return [np.array([[start_channel, start_channel, 1.0]])] + _impl(
        np.array([start_channel]),
        np.array([1.0]),
        remaining_minus_start,
    )


def extract_linearized(group):
    conc = np.concatenate(group)
    return conc[:, [0, 2]]


@memory.cache
def compute_possible_linearization_steps(
    non_linearized_channel_ids,
    absolute_ranges,
    relative_ranges,
    best_range_prot_per_channel,
    range_threshold,
):
    """
    returns the possible linearization groups that could be formed
    by starting a linearization step from each of the non_linearized_channel_ids. For each possible start channel,
    it will return the resulting linearization steps (i.e a group),
    the remaining non linearized channels, and the newly linearized channels from this group.
    """

    # in many cases we can't linearize all channels together as some channels will be disconnected,
    # i.e there's no way, even indirectly, to map 2 such channels because
    # they don't share any path of connected channels.
    # For example, irfp720 and mneongreen don't have any common channel so FITC and APC would have to be independently
    # "linearized". However, if we introduce another "bridge" molecule such as mkate,
    # which has some APC in common with irfp720 as well as percp in common with mneongreen,
    # then we have a way to make all channels colinearized (although in 2 steps,
    # but that's still fine and will be found by the make_path function)

    relative_norm_ranges = relative_ranges / relative_ranges.max()

    # filter out channels that have essentially no signal
    best_norm_ranges = np.take_along_axis(
        relative_norm_ranges, best_range_prot_per_channel[None, :], axis=0
    ).squeeze()
    has_above_threshold_range = best_norm_ranges[non_linearized_channel_ids] > range_threshold
    remaining = non_linearized_channel_ids[has_above_threshold_range].astype(int)

    if len(remaining) == 0:
        return None, None, None

    mask = np.eye(len(remaining), dtype=bool)

    possible_next_non_linearized_channel_ids = np.repeat(
        remaining[None, :], len(remaining), axis=0
    )[~mask].reshape(len(remaining), -1)

    possible_groups = [
        find_colinearization_sequence(
            i,
            nl,
            absolute_ranges,
            best_range_prot_per_channel,
            range_threshold,
        )
        for i, nl in zip(
            remaining,
            possible_next_non_linearized_channel_ids,
        )
    ]

    newly_linearized_ids = [extract_linearized(g)[:, 0].astype(int) for g in possible_groups]
    # number_of_linearized = np.array([len(g) for g in newly_linearized_ids])

    # remove from non_linearized_channel_ids the ones that have been linearized
    remaining_non_linearized_channel_ids = [
        np.setdiff1d(nl, l).astype(int)
        for nl, l in zip(possible_next_non_linearized_channel_ids, newly_linearized_ids)
    ]

    return possible_groups, remaining_non_linearized_channel_ids, newly_linearized_ids


@memory.cache
def compute_group_dependency(channel_group_a, channel_group_b, norm_ranges):
    if len(channel_group_a) == 0 or len(channel_group_b) == 0:
        return 0.0
    # We quantify dependence as the max of (min of pairwise ranges across all channels of a group)
    minranges = np.minimum(
        norm_ranges[:, channel_group_a, None], norm_ranges[:, None, channel_group_b]
    )
    return minranges.max()


@memory.cache
def compute_worst_group_dependency(groups, absolute_ranges):
    # max group_dependency between any 2 groups
    if len(groups) < 2:
        return 0
    norm_ranges = absolute_ranges / absolute_ranges.max()
    all_dependencies = np.array(
        [
            compute_group_dependency(g1, g2, norm_ranges)
            for g1, g2 in itertools.combinations(groups, 2)
        ]
    )
    max_dependency = all_dependencies.max()
    return max_dependency


def linearization_group_quality(path):
    # avg quality of the linearization path
    conc = np.concatenate(path)
    return np.mean(conc[:, 2])


def find_best_linearization_path(
    nchannels,
    absolute_ranges,
    relative_ranges,
    best_range_prot_per_channel,
    range_threshold,
    indep_power=2,
):

    # Completely independent groups of channels are not much of an issue, (even though not ideal).
    # What is much more annoying, is when there are some proteins for which some of the channels
    # of different group have a simultanous signal. That's bad. We want to find a set of groups that will maximize
    # group independence (ideally there'd be only one group)
    # best_paths will return a list of paths, aka a list of linearization groups, that maximizes independence.

    # leaves = 0

    def _impl(remaining_channel_ids, all_prev_qualities, all_prev_groups):
        grps, remaining, new_lin = compute_possible_linearization_steps(
            remaining_channel_ids,
            absolute_ranges,
            relative_ranges,
            best_range_prot_per_channel,
            range_threshold,
        )

        if grps is None:
            worst_dep = compute_worst_group_dependency(all_prev_groups, absolute_ranges)
            worst_indep = (1.0 - worst_dep) ** indep_power
            score = np.array(all_prev_qualities).mean() * worst_indep
            return [], score

        qualities = [linearization_group_quality(g) for g in grps]

        next_groups_and_scores = [
            _impl(r, all_prev_qualities + [q], all_prev_groups + [nl])
            for r, q, nl in zip(remaining, qualities, new_lin)
        ]

        scores = [s for _, s in next_groups_and_scores]
        best_score = np.argmax(scores)

        next_groups, _ = next_groups_and_scores[best_score]

        return [grps[best_score]] + next_groups, scores[best_score]

    return _impl(np.arange(nchannels), [], [])


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     linearization tasks     --
import itertools


# @jit
# def logtransform(x, scale, offset):
    # return (jnp.log(jnp.clip(x + offset, 1, None)) - jnp.log(offset)) * scale
# @jit
# def inv_logtransform(x, scale, offset):
    # return jnp.exp(x / scale + jnp.log(offset)) - offset


class Colinearization(Task):
    def __init__(
        self,
        information_range_threshold=0.02,
        transform_resolution=7,
        transform_degree=2,
        linearization_path=None,
        params=None,
        resample_N=50000,
        logspace=False,
        log_poly_threshold=200,
        log_poly_scale=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params = params
        self.linearization_path = linearization_path
        self.resample_N = resample_N
        self.information_range_threshold = information_range_threshold
        self.transform_resolution = transform_resolution
        self.transform_degree = transform_degree
        if logspace:
            self.tr = lambda x: partial(utils.logtransform, threshold=log_poly_threshold, compression=log_poly_scale)(x+100)*100
            self.itr = lambda x: partial(utils.inv_logtransform, threshold=log_poly_threshold, compression=log_poly_scale)(x/100)-100

        else:
            self.tr = lambda x: x
            self.itr = lambda x: x

    @partial(jax.jit, static_argnums=(0,))
    def regression(self, x, y, w, xbounds):
        xbounds = self.tr(xbounds)
        x_tr, y_tr = self.tr(x), self.tr(y)
        params = utils.fit_stacked_poly_uniform_spacing(
            x_tr, y_tr, w, *xbounds, self.transform_resolution, self.transform_degree
        )
        return params

    def find_linearization_params(
        self, channel_names, compute_coef_at_percentile=80, progress=False
    ):
        assert self.linearization_path is not None

        params = [
            self.generate_single_identity_transform(yb)
            for yb in self.corrected_saturation_thresholds
        ]

        Ysingle_prime = self.singleprt_observations

        total_needed_linearizations = sum(
            len(extract_linearized(p)) for p in self.linearization_path
        )

        if progress:
            pbar = tqdm(total=total_needed_linearizations)
            msg = 'Linearization'

        for group in self.linearization_path:
            for step in group:
                for src_chan, dest_chan, _ in step.astype(int):
                    if src_chan != dest_chan:
                        prot = self.best_range_prot_per_channel[src_chan]
                        x = self.singleprt_observations[prot, :, src_chan]
                        y = Ysingle_prime[prot, :, dest_chan]
                        w = (
                            self.not_saturated[prot, :, src_chan]
                            * self.not_saturated[prot, :, dest_chan]
                        )
                        # compute rough coefficient to express y as x * coef
                        percentile_bounds = np.percentile(
                            x[w], [compute_coef_at_percentile - 1, compute_coef_at_percentile + 1]
                        )
                        inbounds_x = (x[w] > percentile_bounds[0]) & (x[w] < percentile_bounds[1])
                        matching_x = x[w][inbounds_x]
                        matching_y = y[w][inbounds_x]
                        coef = np.nanmean(matching_x / matching_y)
                        xbounds = self.corrected_saturation_thresholds[src_chan]
                        params[src_chan] = self.regression(x, y * coef, w, xbounds)
                        rmse_before = np.sqrt(
                            np.nanmean((Ysingle_prime[prot, :, src_chan] - y) ** 2)
                        )
                        Ysingle_prime[prot, :, src_chan] = self.transform_chan(x, params[src_chan])
                        rmse_after = np.sqrt(
                            np.nanmean((Ysingle_prime[prot, :, src_chan] - y * coef) ** 2)
                        )
                        msg = f'Linearizing {channel_names[src_chan]} -> {channel_names[dest_chan]}: {rmse_before:.2f} -> {rmse_after:.2f}'
                        self.log.debug(msg)
                    if progress:
                        pbar.set_description(msg)
                        pbar.update(1)
        return params

    def initialize(
        self,
        controls_values,
        controls_masks,
        channel_names,
        protein_names,
        reference_channels,
        **_,
    ):
        t0 = time.time()

        self.autofluorescence = utils.estimate_autofluorescence(controls_values, controls_masks)
        self.corrected_controls = controls_values - self.autofluorescence
        self.corrected_saturation_thresholds = utils.estimate_saturation_thresholds(
            self.corrected_controls
        )

        self.log.debug(f"Resampling observations for linearization")
        # let's resample the single prot ctrls and put them all in a single array
        self.singleprt_observations = resampled_singleprt_ctrls(
            self.corrected_controls,
            protein_names,
            controls_masks,
            reference_channels,
            self.resample_N,
            npartitions=0,
        )

        assert self.singleprt_observations.shape == (
            len(protein_names),
            self.resample_N,
            len(channel_names),
        ), f"{self.singleprt_observations.shape} != {(len(protein_names), self.resample_N, len(channel_names))}"

        # null_observations is the array of unstained cells observations
        self.null_observations = self.corrected_controls[np.all(controls_masks == 0, axis=1)]

        self.not_saturated = (
            self.singleprt_observations > self.corrected_saturation_thresholds[:, 0]
        ) & (self.singleprt_observations < self.corrected_saturation_thresholds[:, 1])

        # we need to know the range of values for each protein in each channel for 2 reasons:
        # - knowing how much of the pmt should and can be linearized
        # - as a proxy for the amount of information in each channel (the more the better)
        self.absolute_ranges = np.subtract(
            *np.quantile(self.singleprt_observations, [0.999, 0.001], axis=1)
        )

        # the relative range is the range of values relative to the noise of the null (unstainde) observations.
        # it should be a better measure of information than just the pure absolute ranges
        self.relative_ranges = self.absolute_ranges / self.null_observations.std(axis=0)[None, :]
        self.unsaturated_proportion = np.mean(self.not_saturated, axis=1)

        # pick the highest range
        self.biggest_range_value_per_channel = np.max(self.absolute_ranges, axis=0)
        # now to find the actual "best", we pick the prot with the least amount of saturation that's still within 0.1% of the max range
        self.acceptable_range = self.absolute_ranges > 0.999 * self.biggest_range_value_per_channel
        self.best_range_prot_per_channel = np.argmax(
            self.acceptable_range * self.unsaturated_proportion, axis=0
        )

        nchannels, nproteins = len(channel_names), len(protein_names)

        if self.linearization_path is None and self.params is None:
            self.log.debug(f"Looking for a good linearization path")
            self.linearization_path, _ = find_best_linearization_path(
                nchannels,
                self.absolute_ranges,
                self.relative_ranges,
                self.best_range_prot_per_channel,
                self.information_range_threshold,
            )

        if self.params is None:
            self.log.debug("Using linearization plan:")
            self.log.debug(self.linearization_path)
            self.log.debug(f"Computing linearization params")
            self.params = self.find_linearization_params(channel_names)

        self.log.debug(f"Computing new controls values")
        self.new_controls_values = self.process(controls_values)['observations_raw']
        self.new_autofluorescence = utils.estimate_autofluorescence(
            self.new_controls_values, controls_masks
        )
        self.new_saturation_thresholds = utils.estimate_saturation_thresholds(
            self.new_controls_values
        )


        self.log.debug(f"Linearization initialization done in {time.time() - t0:.1f}s")
        self.log.debug(f"newcontrols_values shape: {self.new_controls_values.shape}")

        return {
            'controls_values': self.new_controls_values,
            'autofluorescence': self.new_autofluorescence,
            'saturation_thresholds': self.new_saturation_thresholds,
            'linearization_path': self.linearization_path,
            'linearization_params': self.params,
            'colinearizer': self,
        }

    @partial(jit, static_argnums=(0,))
    def generate_single_identity_transform(self, xbounds):
        xbounds = self.tr(xbounds)
        spacing = (xbounds[1] - xbounds[0]) / self.transform_resolution
        mticks = jnp.linspace(*xbounds, self.transform_resolution, endpoint=False) + spacing / 2
        stds = jnp.ones_like(mticks) * spacing
        coeffs = jnp.zeros((self.transform_resolution, self.transform_degree + 1))
        if self.transform_degree > 0:
            coeffs = coeffs.at[:, -2].set(1)
        return coeffs, mticks, stds

    @partial(jit, static_argnums=(0,))
    def generate_identity_transforms(self, saturation_thresholds):
        return vmap(self.generate_single_identity_transform)(saturation_thresholds)

    @partial(jit, static_argnums=(0,))
    def transform_chan(self, y, t):
        return self.itr(utils.evaluate_stacked_poly(self.tr(y), t))

    def compute_yprime_from_list(self, y, need_update=None, progress=False):
        if need_update is None:
            need_update = np.ones(len(self.params), dtype=bool)
        res = []
        it = tqdm(list(zip(y, self.params, need_update))) if progress else zip(y, self.params, need_update)
        for yi, pi, nu in it:
            if nu:
                res.append(self.transform_chan(yi, pi))
            else:
                res.append(yi)
        return np.stack(res, axis=0)


    def diagnostics(self, saturation_thresholds, channel_names, **_):
        assert self.params is not None, "You need to run initialize() first"

        intervals = vmap(jnp.linspace, (0, 0, None))(
            saturation_thresholds[:, 0], saturation_thresholds[:, 1], 500
        ).T
        interval_reg = self.compute_yprime_from_list(intervals.T, self.params).T
        fig, axes = plt.subplots(1, len(channel_names), figsize=(len(channel_names) * 5, 5))
        for i, ax in enumerate(axes):
            ax.plot(intervals[:, i], interval_reg[:, i], c='r', lw=1)
            ax.set_xlabel(channel_names[i])
            ax.set_ylabel(channel_names[i] + "'")
        fig.tight_layout()

    def process(self, observations_raw, **kw):
        assert self.params is not None, "You need to run initialize() first"
        self.log.debug(f"Linearizing observations of shape {observations_raw.shape}")
        x = self.compute_yprime_from_list(observations_raw.T).T
        self.log.debug(f"Linearization done, output shape: {x.shape}")
        return {'observations_raw': x}





##────────────────────────────────────────────────────────────────────────────}}}


