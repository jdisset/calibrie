# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

"""
This module implements a complex procedure for "colinearizing" multi-channel fluorescence measurements,
particularly in situations where a single fluorescent protein emits in multiple channels but they do not
appear perfectly linearly related due to real-world measurement artifacts. It attempts to find and apply
a sequence of transformations (including polynomial and optional log-based transformations) to ensure
all channels become more linearly comparable and thus improve downstream compensation and unmixing.

The main points are:
- Identification of a "linearization path" that groups and orders channels to be iteratively aligned.
- Resampling of single-protein control data to ensure broad coverage over signal intensity ranges.
- Use of polynomial fits, combined with optional nonlinear (log) transformations, to adjust channel
  values so that each channel can be expressed as a linear or near-linear function of another.
- Heuristic measures of channel "information content" and "range quality" guide the selection of
  reference channels and order of linearization.
- Caching and iterative refinement steps allow scalable and repeatable linearization, even for large
  datasets.

This approach is useful when linear unmixing is complicated by imperfectly linear channel responses,
autofluorescence residuals, or spectral bleedthrough that simple linear models fail to address directly.
After linearization, subsequent tasks like spillover compensation or protein abundance estimation should
yield more stable and interpretable results.
"""

from joblib import Memory

cachedir = '/tmp/'
memory = Memory(cachedir, verbose=0)
from .utils import Context
from jaxopt import GaussNewton
from copy import deepcopy
from .pipeline import Task, DiagnosticFigure
import jax
from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.patches as patches
import numpy as np
import time
import pandas as pd
import lineax as lx
from functools import partial
import itertools
from tqdm import tqdm
from pydantic import Field, ConfigDict

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, Optional


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
    exclude_channels=None,
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

    start_chans = np.arange(nchannels)
    if exclude_channels is not None:
        start_chans = np.setdiff1d(start_chans, exclude_channels)
    return _impl(start_chans, [], [])


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     linearization tasks     --


class Colinearization(Task):
    """Task for colinearizing multi-channel fluorescence measurements."""

    # Configuration fields with descriptions
    information_range_threshold: float = Field(
        default=0.005,
        description="Threshold for considering channel information content significant",
    )
    transform_resolution: int = Field(
        default=10, description="Resolution of the transformation grid"
    )
    transform_degree: int = Field(default=1, description="Degree of polynomial transformation")
    resample_N: int = Field(default=50000, description="Number of samples for resampling")
    logspace: bool = Field(
        default=False, description="Whether to perform transformations in log space"
    )
    log_poly_threshold: int = Field(
        default=200, description="Threshold for log-polynomial transformation"
    )
    log_poly_scale: float = Field(default=0.4, description="Scale factor for log transformation")
    exclude_channels: List[str] = Field(
        default_factory=list, description="List of channel names to exclude from linearization"
    )
    diagnostics_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Settings for diagnostic visualization"
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        # Set up transformations based on logspace setting
        if self.logspace:
            self._tr = (
                lambda x: partial(
                    utils.logtransform,
                    threshold=self.log_poly_threshold,
                    compression=self.log_poly_scale,
                )(x + 100)
                * 100
            )
            self._itr = (
                lambda x: partial(
                    utils.inv_logtransform,
                    threshold=self.log_poly_threshold,
                    compression=self.log_poly_scale,
                )(x / 100)
                - 100
            )
        else:
            self._tr = lambda x: x
            self._itr = lambda x: x

        self._linearization_path = []
        self._params = None

    def initialize(self, ctx: Context) -> Context:
        """Initialize the colinearization task with control data."""
        t0 = time.time()

        # Extract required data from context
        self._controls_values = ctx.controls_values
        self._controls_masks = ctx.controls_masks
        self._channel_names = ctx.channel_names
        self._protein_names = ctx.protein_names
        self._reference_channels = ctx.reference_channels

        # Initialize processing pipeline
        autofluorescence = utils.estimate_autofluorescence(
            self._controls_values, self._controls_masks
        )

        # in initialize method, after corrected_controls calculation:
        corrected_controls = self._controls_values - autofluorescence
        self._corrected_saturation_thresholds = utils.estimate_saturation_thresholds_qtl(
            corrected_controls
        )

        self._null_observations = corrected_controls[np.all(self._controls_masks == 0, axis=1)]

        self._log.debug("Resampling observations for linearization")
        self._singleptr_observations = resampled_singleprt_ctrls(
            corrected_controls,
            self._protein_names,
            self._controls_masks,
            self._reference_channels,
            self.resample_N,
            npartitions=0,
        )

        self._not_saturated = (
            self._singleptr_observations > self._corrected_saturation_thresholds[:, 0]
        ) & (self._singleptr_observations < self._corrected_saturation_thresholds[:, 1])

        # Calculate ranges and quality metrics
        self._compute_ranges_and_metrics()

        # Find linearization path if not provided
        if not self._linearization_path and self._params is None:
            self._find_linearization_path()

        # Compute transformation parameters
        if self._params is None:
            self._log.debug("Computing linearization params")
            self._params = self.find_linearization_params(self._channel_names)

        new_loader = self.make_loader(ctx.cell_data_loader)
        self._log.debug(f"Linearization initialization done in {time.time() - t0:.1f}s")

        return Context(
            cell_data_loader=new_loader,
            saturation_thresholds=self._corrected_saturation_thresholds,
        )

    def find_linearization_params(
        self, channel_names, compute_coef_at_percentile=80, progress=False
    ):
        """Find the parameters for linearization transformations."""
        assert self._linearization_path is not None

        params = [
            self._generate_single_identity_transform(yb)
            for yb in self._corrected_saturation_thresholds
        ]

        Ysingle_prime = self._singleptr_observations

        total_needed_linearizations = sum(
            len(extract_linearized(p)) for p in self._linearization_path
        )

        if progress:
            pbar = tqdm(total=total_needed_linearizations)
            msg = 'Linearization'

        for group in self._linearization_path:
            for step in group:
                for src_chan, dest_chan, _ in step.astype(int):
                    if src_chan != dest_chan:
                        prot = self._best_range_prot_per_channel[src_chan]
                        x = self._singleptr_observations[prot, :, src_chan]
                        y = Ysingle_prime[prot, :, dest_chan]
                        w = (
                            self._not_saturated[prot, :, src_chan]
                            * self._not_saturated[prot, :, dest_chan]
                        )
                        # compute rough coefficient to express y as x * coef
                        percentile_bounds = np.percentile(
                            x[w], [compute_coef_at_percentile - 1, compute_coef_at_percentile + 1]
                        )
                        inbounds_x = (x[w] > percentile_bounds[0]) & (x[w] < percentile_bounds[1])
                        matching_x = x[w][inbounds_x]
                        matching_y = y[w][inbounds_x]
                        coef = np.nanmean(matching_x / matching_y)
                        xbounds = self._corrected_saturation_thresholds[src_chan]
                        params[src_chan] = self._regression(x, y * coef, w, xbounds)
                        rmse_before = np.sqrt(
                            np.nanmean((Ysingle_prime[prot, :, src_chan] - y) ** 2)
                        )
                        Ysingle_prime[prot, :, src_chan] = self._transform_chan(x, params[src_chan])
                        rmse_after = np.sqrt(
                            np.nanmean((Ysingle_prime[prot, :, src_chan] - y * coef) ** 2)
                        )
                        msg = f'Linearizing {channel_names[src_chan]} -> {channel_names[dest_chan]}: {rmse_before:.2f} -> {rmse_after:.2f}'
                        self._log.debug(msg)
                    if progress:
                        pbar.set_description(msg)
                        pbar.update(1)
        return params

    @partial(jax.jit, static_argnums=(0,))
    def _regression(self, x, y, w, xbounds):
        """Perform regression with optional log transformation."""
        xbounds = self._tr(xbounds)
        x_tr, y_tr = self._tr(x), self._tr(y)
        params = utils.fit_stacked_poly_uniform_spacing(
            x_tr, y_tr, w, *xbounds, self.transform_resolution, self.transform_degree
        )
        return params

    def _compute_ranges_and_metrics(self):
        """Compute ranges and quality metrics for channels."""
        self._absolute_ranges = np.subtract(
            *np.quantile(self._singleptr_observations, [0.999, 0.001], axis=1)
        )
        self._relative_ranges = self._absolute_ranges / self._null_observations.std(axis=0)[None, :]
        self._unsaturated_proportion = np.mean(self._not_saturated, axis=1)
        self._biggest_range_value_per_channel = np.max(self._absolute_ranges, axis=0)
        self._acceptable_range = (
            self._absolute_ranges > 0.999 * self._biggest_range_value_per_channel
        )
        self._best_range_prot_per_channel = np.argmax(
            self._acceptable_range * self._unsaturated_proportion, axis=0
        )

    def _find_linearization_path(self):
        """Find optimal linearization path."""
        exclude_list = None
        if self.exclude_channels:
            exclude_list = [self._channel_names.index(c) for c in self.exclude_channels]

        self._linearization_path, _ = find_best_linearization_path(
            len(self._channel_names),
            self._absolute_ranges,
            self._relative_ranges,
            self._best_range_prot_per_channel,
            self.information_range_threshold,
            exclude_channels=exclude_list,
        )

        self._log.debug("Using linearization plan:")
        self._log.debug(str(self._linearization_path))

    def make_loader(self, current_loader: Callable):
        """Create a new loader that applies colinearization transformation."""
        old_loader = deepcopy(current_loader)

        def loader(data: Any, column_order: Optional[List[str]] = None) -> pd.DataFrame:
            df = old_loader(data, None)  # Pass None to get all columns

            if df is not None and self._params is not None:
                self._log.debug(f"Applying colinearization to data of shape {df.shape}")
                print(f"Applying colinearization to data of shape {df.shape}")

                # Get the list of channels that should be transformed
                channels_to_transform = self._channel_names

                # Get indices of columns to transform
                transform_indices = [
                    i for i, col in enumerate(df.columns) if col in channels_to_transform
                ]

                # Only transform the relevant columns
                values_to_transform = df.iloc[:, transform_indices].values
                self._log.debug(f"Values shape before transformation: {values_to_transform.shape}")
                transformed_values = self.compute_yprime_from_list(values_to_transform.T).T
                self._log.debug(f"Values shape after transformation: {transformed_values.shape}")

                # Create new dataframe with all original data
                transformed_df = df.copy()
                # Update only the transformed columns
                for idx, col_idx in enumerate(transform_indices):
                    transformed_df.iloc[:, col_idx] = transformed_values[:, idx]

                # Filter to requested columns if specified
                if column_order is not None:
                    try:
                        return transformed_df[column_order]
                    except KeyError as e:
                        raise ValueError(
                            f"When loading cells, asked for unknown column {e}. Available: {list(transformed_df.columns)}"
                        )

                return transformed_df

            return df

        return loader

    # Additional helper methods remain unchanged...
    @partial(jit, static_argnums=(0,))
    def _generate_single_identity_transform(self, xbounds):
        xbounds = self._tr(xbounds)
        spacing = (xbounds[1] - xbounds[0]) / self.transform_resolution
        mticks = jnp.linspace(*xbounds, self.transform_resolution, endpoint=False) + spacing / 2
        stds = jnp.ones_like(mticks) * spacing
        coeffs = jnp.zeros((self.transform_resolution, self.transform_degree + 1))
        if self.transform_degree > 0:
            coeffs = coeffs.at[:, -2].set(1)
        return coeffs, mticks, stds

    @partial(jit, static_argnums=(0,))
    def _generate_identity_transforms(self, saturation_thresholds):
        return vmap(self._generate_single_identity_transform)(saturation_thresholds)

    @partial(jit, static_argnums=(0,))
    def _transform_chan(self, y, t):
        return self._itr(utils.evaluate_stacked_poly(self._tr(y), t))

    def compute_yprime_from_list(self, y, need_update=None, progress=False):
        if need_update is None:
            need_update = np.ones(len(self._params), dtype=bool)
        res = []
        it = (
            tqdm(list(zip(y, self._params, need_update)))
            if progress
            else zip(y, self._params, need_update)
        )
        for yi, pi, nu in it:
            if nu:
                res.append(self._transform_chan(yi, pi))
            else:
                res.append(yi)
        return np.stack(res, axis=0)

    def diagnostics(self, ctx: Context, **kwargs) -> Optional[List[DiagnosticFigure]]:
        """Generate diagnostic visualizations showing before/after distributions."""
        saturation_thresholds = ctx.saturation_thresholds
        channel_names = ctx.channel_names

        assert self._params is not None, "You need to run initialize() first"

        intervals = vmap(jnp.linspace, (0, 0, None))(
            saturation_thresholds[:, 0], saturation_thresholds[:, 1], 500
        ).T
        interval_reg = self.compute_yprime_from_list(intervals.T, self._params).T

        fig = plt.figure(figsize=(len(channel_names) * 8, 12))
        gs = plt.GridSpec(1, len(channel_names), figure=fig)

        for i in range(len(channel_names)):
            # Top plot: transformation curve
            ax_top = fig.add_subplot(gs[0, i])

            # Plot transformation curves (top)
            ax_top.plot(intervals[:, i], interval_reg[:, i], c='r', lw=2, label='Transformation')
            ax_top.plot(
                [intervals[:, i].min(), intervals[:, i].max()],
                [intervals[:, i].min(), intervals[:, i].max()],
                'k--',
                alpha=0.5,
                label='Identity',
            )

            # Add saturation thresholds
            for threshold in self._corrected_saturation_thresholds[i]:
                ax_top.axvline(threshold, color='orange', linestyle='--', alpha=0.5)

            # Style plots
            for ax in [ax_top]:
                ax.set_xlabel(channel_names[i])
                if self.logspace:
                    ax.set_xscale('symlog', linthresh=50)
                plots.remove_topright_spines(ax)
                ax.legend(fontsize='small')

            ax_top.set_ylabel(f"{channel_names[i]}' (transformed)")

        fig.suptitle("Channel Transformations and Distributions", y=1.02)
        fig.tight_layout()

        figs = [DiagnosticFigure(fig=fig, name="Channel Transformations and Distributions")]

        # Add linearization path visualization
        if hasattr(self, '_absolute_ranges') and hasattr(self, '_relative_ranges'):
            path_fig = self._visualize_linearization_path()
            if path_fig:
                figs.append(DiagnosticFigure(fig=path_fig, name="Linearization Path"))

        return figs

    def _visualize_linearization_path(self) -> Optional[plt.Figure]:
        """Create visualization of the linearization path"""
        try:
            fig, ax, coords2d = plots.range_plots(
                self._protein_names,
                self._channel_names,
                self._absolute_ranges,
                self._relative_ranges,
                self._not_saturated,
                dpi=100,
            )

            for ngrp, p in enumerate(self._linearization_path):
                group_letter = chr(ord("A") + ngrp)
                group_label = f'{group_letter}: ' if len(self._linearization_path) > 1 else ''
                group_color = plt.cm.tab10(ngrp)
                for nbatch, lbatch in enumerate(p):
                    for source_chan, dest_chan, quality in lbatch:
                        source_chan = int(source_chan)
                        dest_chan = int(dest_chan)
                        # draw arrow from source to dest
                        prot_id = self._best_range_prot_per_channel[source_chan]
                        source_xy = coords2d[prot_id * len(self._channel_names) + source_chan]
                        dest_xy = coords2d[prot_id * len(self._channel_names) + dest_chan]
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
                            len(self._protein_names),
                            f"{group_label}[{nbatch+1}/{len(p)}]",
                            ha='center',
                            va='center',
                            fontsize=8,
                            c=group_color,
                            alpha=0.75,
                        )

            fig.suptitle("Linearization Path")
            fig.tight_layout()

            return fig

        except Exception as e:
            self._log.warning(f"Failed to create linearization path visualization: {e}")
            return None


##────────────────────────────────────────────────────────────────────────────}}}
