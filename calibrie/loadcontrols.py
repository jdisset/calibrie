# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

"""
This module provides the LoadControls task, which loads single, multi and no-protein controls from a list of files.
"""

from .pipeline import Task, DiagnosticFigure
from .metrics import TaskMetrics
from typing import List, Dict, Tuple, Optional
from . import utils as ut
from calibrie.utils import LoadedData, Context
import numpy as np
from . import plots, utils
import re
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging

_VALID_BLANK_NAMES = ['BLANK', 'EMPTY', 'INERT', 'CNTL']
_VALID_ALL_NAMES = ['ALL', 'ALLCOLOR', 'ALLCOLORS', 'FULL']

EPSILON = 1e-6


def normalized_metrics(scores):
    # normalize scores so they're comparable
    result = {}
    for metric in scores:
        result[metric] = (scores[metric] - np.min(scores[metric])) / (
            np.max(scores[metric]) - np.min(scores[metric])
        )
    return result


class ChannelMetrics:
    """Helper class to compute and store channel quality metrics with validation."""

    # Validation thresholds
    BLANK_MIN_EVENTS = 1000  # Minimum number of blank control events
    SIGNAL_WARNING_THRESHOLD = -0.25  # Only warn if signal z-score is significantly negative
    SNR_WARNING_THRESHOLD = 0.5  # Warn about very low SNR that could indicate problems
    BLANK_CV_WARNING = 20.0  # Coefficient of variation threshold for blank controls
    MACHINE_MAX = 2**18  # Typical max value for most cytometers, used for range calculations

    def __init__(self, values, masks, channel_names, protein_names, validation_warnings=True):
        self.values = values
        self.masks = masks
        self.channel_names = channel_names
        self.protein_names = protein_names
        self.validation_warnings = validation_warnings
        self._log = logging.getLogger(__name__)

        # Pre-compute basic masks and stats
        self.single_masks = masks.sum(axis=1) == 1
        self.blank_mask = np.logical_not(np.any(masks, axis=1))
        self.blank_values = values[self.blank_mask]

        # Pre-compute blank statistics
        self.blank_mean = np.mean(self.blank_values, axis=0)
        self.blank_std = np.std(self.blank_values, axis=0)

        # Validate blank controls (only critical issues)
        self._validate_blank_controls()

        # Initialize metric storage
        self.signal_strengths = {}
        self.signal_to_noise = {}
        self.specificities = {}
        self.crosstalks = {}
        self.correlations = {}
        self.dynamic_ranges = {}

        self._compute_metrics()

    def _validate_blank_controls(self):
        if len(self.blank_values) < self.BLANK_MIN_EVENTS:
            self._log.warning(
                f"Very few blank control events ({len(self.blank_values)} < {self.BLANK_MIN_EVENTS})"
            )

    def _compute_single_protein_metrics(self, pid, prot, prot_values, other_values=None):
        """Compute all metrics for a single protein."""
        # Store correlation matrix
        self.correlations[prot] = np.corrcoef(prot_values.T)
        self.signal_strengths[prot] = {}
        self.signal_to_noise[prot] = {}
        self.specificities[prot] = {}
        self.crosstalks[prot] = {}
        self.dynamic_ranges[prot] = {}

        for cid, chan in enumerate(self.channel_names):
            # Signal strength (z-score relative to blank)
            mean_signal = np.mean(prot_values[:, cid])
            signal = (mean_signal - self.blank_mean[cid]) / self.blank_std[cid]

            if signal < self.SIGNAL_WARNING_THRESHOLD and self.validation_warnings:
                self._log.warning(f"Negative signal for {prot} in {chan}: {signal:.2f} z-scores")

            if other_values is not None:
                max_other_signal = np.max(np.abs(other_values[:, cid] - self.blank_mean[cid]))
                crosstalk = max_other_signal / self.blank_std[cid]
            else:
                crosstalk = 0

            snr = np.std(prot_values[:, cid]) / self.blank_std[cid]

            pct_range = np.percentile(prot_values[:, cid], [1, 99])
            dyn_range = np.log10(pct_range[1] - pct_range[0])

            self.signal_strengths[prot][chan] = signal
            self.signal_to_noise[prot][chan] = snr
            self.specificities[prot][chan] = np.maximum(signal / (crosstalk + EPSILON), 0)
            self.crosstalks[prot][chan] = crosstalk
            self.dynamic_ranges[prot][chan] = dyn_range

            if (
                snr < self.SNR_WARNING_THRESHOLD
                and signal > 1.0  # Only if we expect signal
                and self.validation_warnings
            ):
                self._log.warning(f"Very low SNR for {prot} in {chan}: {snr:.2f}")

    def _compute_metrics(self):
        """Compute all quality metrics for all channels."""
        for pid, prot in enumerate(self.protein_names):
            # Get protein-specific masks
            prot_mask = np.logical_and(self.single_masks, self.masks[:, pid])
            if not np.any(prot_mask):
                self._log.warning(f"No single-color control events for {prot}")
                continue

            prot_values = self.values[prot_mask]

            # Get other proteins' data
            not_pid_mask = np.logical_not(self.masks[:, pid])
            other_mask = np.logical_and(
                self.single_masks, np.logical_and(not_pid_mask, np.logical_not(self.blank_mask))
            )
            other_values = self.values[other_mask] if np.any(other_mask) else None

            self._compute_single_protein_metrics(pid, prot, prot_values, other_values)

    def get_all_metrics(self):
        """Return all metrics in a format suitable for plotting."""
        metrics = []
        for prot in self.protein_names:
            if prot not in self.signal_strengths:
                continue

            for chan in self.channel_names:
                metrics.append(
                    {
                        'protein': prot,
                        'channel': chan,
                        'signal': self.signal_strengths[prot][chan],
                        'snr': self.signal_to_noise[prot][chan],
                        'dynamic_range': self.dynamic_ranges[prot][chan],
                        'specificity': self.specificities[prot][chan],
                        'crosstalk': self.crosstalks[prot][chan],  # added crosstalk to output
                    }
                )
        return metrics

    def get_quality_summary(self, protein: str, channel: str = None) -> dict:
        """Get quality metrics summary for a protein or protein-channel pair."""
        if protein not in self.protein_names:
            return {}

        if channel is not None:
            if channel not in self.channel_names:
                return {}
            return {
                'signal': self.signal_strengths[protein][channel],
                'snr': self.signal_to_noise[protein][channel],
                'specificity': self.specificities[protein][channel],
                'crosstalk': self.crosstalks[protein][channel],
                'dynamic_range': self.dynamic_ranges[protein][channel],
            }

        return {
            chan: self.get_quality_summary(protein, chan)
            for chan in self.channel_names
            if chan in self.signal_strengths[protein]
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Channel Quality Dashboard Rendering Helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Channels to always exclude from quality metrics display
_EXCLUDED_CHANNEL_PATTERNS = [
    re.compile(r'^TIME$', re.IGNORECASE),
    re.compile(r'^FSC', re.IGNORECASE),
    re.compile(r'^SSC', re.IGNORECASE),
]


def _filter_display_channels(channel_names: list[str]) -> tuple[list[str], list[int]]:
    """Filter out TIME, FSC*, SSC* channels. Returns (filtered_names, kept_indices)."""
    kept = []
    for i, ch in enumerate(channel_names):
        exclude = any(pat.match(ch) for pat in _EXCLUDED_CHANNEL_PATTERNS)
        if not exclude:
            kept.append(i)
    return [channel_names[i] for i in kept], kept


def _compute_quality_grades(
    metrics: ChannelMetrics,
    reference_channels: list[int],
    display_channel_indices: list[int] | None = None,
) -> dict[str, tuple[str, float, str, dict]]:
    """Compute A/B/C/F grade per protein using only displayed (non-excluded) channels.

    Returns protein -> (grade, score, best_display_chan_name, details).
    """
    grades = {}

    # Get list of display channel names
    if display_channel_indices is not None:
        display_chans = [metrics.channel_names[i] for i in display_channel_indices]
    else:
        display_chans = metrics.channel_names

    for prot in metrics.protein_names:
        if prot not in metrics.signal_to_noise:
            grades[prot] = ('F', 0.0, '?', {})
            continue

        # Find best channel among displayed channels (by specificity)
        best_chan, best_spec, best_snr, best_dyn = None, -1, 0, 0
        for chan in display_chans:
            if chan in metrics.specificities[prot]:
                spec = metrics.specificities[prot][chan]
                if spec > best_spec:
                    best_spec = spec
                    best_snr = metrics.signal_to_noise[prot].get(chan, 0)
                    best_dyn = metrics.dynamic_ranges[prot].get(chan, 0)
                    best_chan = chan

        if best_chan is None:
            grades[prot] = ('F', 0.0, '?', {})
            continue

        # Grade based on specificity (primary) and SNR (secondary)
        if best_spec >= 3.0 and best_snr >= 2.0:
            grade, score = 'A', 1.0
        elif best_spec >= 1.5 and best_snr >= 1.0:
            grade, score = 'B', 0.7
        elif best_spec >= 0.8:
            grade, score = 'C', 0.4
        else:
            grade, score = 'F', 0.15

        grades[prot] = (
            grade,
            score,
            best_chan,
            {'snr': best_snr, 'spec': best_spec, 'dyn': best_dyn},
        )

    return grades


def _render_summary_bar(
    ax: Axes,
    grades: dict[str, tuple[str, float, str, dict]],
    spec_vmin: float = 0.1,
    spec_vmax: float = 10.0,
    fontsize: int = 11,
):
    """Render protein quality summary as colored boxes using inferno colormap."""
    n = len(grades)
    if n == 0:
        ax.axis('off')
        return

    proteins = list(grades.keys())
    box_width = 1.0
    ax.set_xlim(0, n * box_width)
    ax.set_ylim(0, 1)

    norm = LogNorm(vmin=spec_vmin, vmax=spec_vmax, clip=True)
    cmap = plt.get_cmap('inferno')

    for i, prot in enumerate(proteins):
        grade, score, ref_chan, details = grades[prot]
        spec_val = details.get('spec', 0.1)
        norm_val = norm(max(spec_val, spec_vmin))
        color = cmap(norm_val)

        # Draw box colored by specificity (same colormap as heatmap)
        ax.add_patch(
            plt.Rectangle(
                (i * box_width, 0.05),
                box_width * 0.95,
                0.90,
                facecolor=color,
                edgecolor='#333',
                linewidth=1.5,
                alpha=0.9,
            )
        )

        # Text color based on brightness
        text_color = 'black' if norm_val > 0.5 else 'white'
        ax.text(
            i * box_width + box_width / 2,
            0.72,
            f'{prot}',
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight='bold',
            color=text_color,
        )
        ax.text(
            i * box_width + box_width / 2,
            0.45,
            f'{grade} │ {ref_chan}',
            ha='center',
            va='center',
            fontsize=fontsize - 2,
            color=text_color,
            alpha=0.85,
        )
        ax.text(
            i * box_width + box_width / 2,
            0.20,
            f'spec={spec_val:.2f}',
            ha='center',
            va='center',
            fontsize=fontsize - 3,
            color=text_color,
            alpha=0.7,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        'Protein Quality Summary', fontsize=fontsize + 1, fontweight='bold', loc='left', pad=4
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_specificity_heatmap(
    ax: Axes,
    spec_data: np.ndarray,
    dynrange_data: np.ndarray,
    sat_data: np.ndarray | None,
    row_labels: list[str],
    col_labels: list[str],
    reference_markers: list[int] | None = None,
    fontsize: int = 10,
):
    """Render specificity heatmap with dynamic range bars and saturation indicators."""
    n_prot, n_chan = spec_data.shape

    # Use log scale for specificity
    vmin, vmax = 0.1, max(10.0, np.nanmax(spec_data))
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    im = ax.imshow(spec_data, cmap='inferno', norm=norm, aspect='auto', interpolation='nearest')

    # Grid
    ax.set_xticks(np.arange(n_chan) + 0.5, minor=True)
    ax.set_yticks(np.arange(n_prot) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5, alpha=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Labels
    ax.set_xticks(np.arange(n_chan))
    ax.set_yticks(np.arange(n_prot))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize + 1)
    ax.set_title(
        'Specificity (Signal / Crosstalk)', fontsize=fontsize + 3, fontweight='bold', pad=10
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.08)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize - 1)

    # Pre-compute best and 2nd best per row
    best_per_row = []
    second_per_row = []
    for y in range(n_prot):
        row_specs = [(x, spec_data[y, x]) for x in range(n_chan) if not np.isnan(spec_data[y, x])]
        row_specs.sort(key=lambda t: t[1], reverse=True)
        best_per_row.append(row_specs[0][0] if len(row_specs) > 0 else None)
        second_per_row.append(row_specs[1][0] if len(row_specs) > 1 else None)

    # Cell annotations: specificity value + dynamic range bar + saturation warning
    dyn_min, dyn_max = np.nanmin(dynrange_data), np.nanmax(dynrange_data)
    dyn_range = max(dyn_max - dyn_min, 1.0)

    for y in range(n_prot):
        for x in range(n_chan):
            spec_val = spec_data[y, x]
            dyn_val = dynrange_data[y, x]
            if np.isnan(spec_val):
                continue

            norm_spec = norm(spec_val)
            base_color = 'white' if norm_spec < 0.5 else 'black'

            # Reference channel marker
            is_ref = reference_markers is not None and x == reference_markers[y]
            if is_ref:
                base_color = '#00ff00' if norm_spec < 0.5 else '#004400'

            # Main text: specificity value (2 decimal places)
            txt = f'{spec_val:.2f}' if spec_val < 100 else f'{spec_val:.1f}'
            if is_ref:
                txt += ' ★'
            ax.text(
                x,
                y - 0.15,
                txt,
                ha='center',
                va='center',
                color=base_color,
                fontsize=fontsize - 1,
                fontweight='bold',
            )

            # Best/2nd best label
            if x == best_per_row[y]:
                ax.text(
                    x,
                    y + 0.38,
                    'best',
                    ha='center',
                    va='center',
                    color=base_color,
                    fontsize=fontsize - 3,
                    fontstyle='italic',
                    alpha=0.8,
                )
            elif x == second_per_row[y]:
                ax.text(
                    x,
                    y + 0.38,
                    '2nd',
                    ha='center',
                    va='center',
                    color=base_color,
                    fontsize=fontsize - 3,
                    fontstyle='italic',
                    alpha=0.6,
                )

            # Dynamic range bar (small horizontal bar below value)
            bar_width = 0.7 * (dyn_val - dyn_min) / dyn_range
            bar_left = x - 0.35
            bar_color = '#aaaaaa' if norm_spec < 0.5 else '#555555'
            ax.plot(
                [bar_left, bar_left + bar_width],
                [y + 0.2, y + 0.2],
                color=bar_color,
                linewidth=3,
                solid_capstyle='round',
                alpha=0.7,
            )

            # Saturation warning (red dot if >1% saturated)
            if sat_data is not None and sat_data[y, x] > 0.01:
                ax.plot(x + 0.35, y - 0.35, 'o', color='red', markersize=5, alpha=0.8)

    # Dynamic range legend (bottom-left)
    legend_y = n_prot - 0.3
    legend_x_start = -0.8
    ax.text(
        legend_x_start,
        legend_y + 0.15,
        'Dyn. range:',
        fontsize=fontsize - 2,
        ha='right',
        va='center',
        color='#666666',
    )
    ax.plot(
        [legend_x_start + 0.05, legend_x_start + 0.25],
        [legend_y + 0.15, legend_y + 0.15],
        color='#888888',
        linewidth=3,
        solid_capstyle='round',
    )
    ax.text(
        legend_x_start + 0.35,
        legend_y + 0.15,
        f'{dyn_min:.1f}–{dyn_max:.1f}',
        fontsize=fontsize - 2,
        ha='left',
        va='center',
        color='#666666',
    )

    return im


def _render_correlation_panel(
    ax: Axes,
    correlations: dict[str, np.ndarray],
    channel_names: list[str],
    channel_indices: list[int] | None = None,
    fontsize: int = 9,
):
    """Render correlation matrices as side-by-side heatmaps with full channel labels."""
    proteins = [
        p for p in correlations.keys() if correlations[p] is not None and correlations[p].size > 0
    ]
    n_prot = len(proteins)

    if n_prot == 0:
        ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', fontsize=fontsize + 2)
        ax.axis('off')
        return

    ax.axis('off')

    # Calculate layout: each protein gets equal horizontal space
    gap = 0.03
    panel_width = (1.0 - gap * (n_prot + 1)) / n_prot

    for pid, prot in enumerate(proteins):
        corr_mat = correlations[prot]

        # Filter to display channels only
        if channel_indices is not None:
            corr_mat = corr_mat[np.ix_(channel_indices, channel_indices)]

        left = gap + pid * (panel_width + gap)
        inset_ax = ax.inset_axes((left, 0.12, panel_width, 0.78))

        im = inset_ax.imshow(corr_mat, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')

        # Channel labels
        display_names = channel_names
        inset_ax.set_xticks(np.arange(len(display_names)))
        inset_ax.set_yticks(np.arange(len(display_names)))
        inset_ax.set_xticklabels(display_names, fontsize=fontsize - 1, rotation=90, ha='center')
        inset_ax.set_yticklabels(display_names, fontsize=fontsize - 1)

        inset_ax.set_title(prot, fontsize=fontsize + 1, fontweight='bold', pad=4)

        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#888888')

        # Add colorbar only for last panel
        if pid == n_prot - 1:
            cbar_ax = ax.inset_axes((left + panel_width + 0.01, 0.12, 0.015, 0.78))
            ax.figure.colorbar(im, cax=cbar_ax)
            cbar_ax.tick_params(labelsize=fontsize - 2)

    ax.set_title(
        'Channel Correlations', fontsize=fontsize + 2, fontweight='bold', y=0.98, loc='left'
    )


class LoadControls(Task):
    """
    Task for loading single, multi and no-protein controls from a list of files.

    ### REQUIRED PIPELINE DATA
    - None

    ### PRODUCED PIPELINE DATA
    - `controls_values`: an array of control values (total events x channels)
    - `controls_masks`: an array of masks telling which proteins were present, for each event (total events x proteins)
    - `channel_names`: the ordered list of channel names
    - `protein_names`: the ordered list of protein names
    - `saturation_thresholds`: {'upper': (channels), 'lower': (channels)} the upper and lower saturation thresholds for each channel
    - `reference_channels`: an array of integer indices of the reference channels for each protein (proteins)
    """

    color_controls: Dict[str | Tuple[str], LoadedData]
    use_channels: List[str] = []
    use_reference_channels: Dict[str, str] = {}  # protein -> channel

    def pick_reference_channels(self, max_sat_proportion=0.001, min_dyn_range=3, **_):
        """
        Pick reference channels for each protein using multiple quality metrics.

        The channel selection uses a weighted scoring system that considers:
        - Dynamic range (log10)
        - Signal-to-noise ratio
        - Specificity (signal vs crosstalk)
        - Saturation proportion (penalizes channels with too many saturated events)
        """
        if self.use_reference_channels is not None:
            user_provided = []
            for p in self._protein_names:
                if p in self.use_reference_channels.keys():
                    ch = self.use_reference_channels[p]
                    assert ch in self._channel_names, f'Unknown channel {ch} in reference_channels'
                    user_provided.append(self._channel_names.index(ch))
                else:
                    user_provided.append(None)
            unknown_provided = [
                p for p in self.use_reference_channels if p not in self._protein_names
            ]
            if len(unknown_provided) > 0:
                self._log.warning(
                    f'Ignoring proteins {unknown_provided}: declared a reference channel but are absent from controls.'
                )
        else:
            user_provided = [None] * len(self._protein_names)

        # compute quality metrics for selection
        metrics = ChannelMetrics(
            self._controls_values, self._controls_masks, self._channel_names, self._protein_names
        )

        single_masks = self._controls_masks.sum(axis=1) == 1
        ref_channels = []

        for pid, prot in enumerate(self._protein_names):
            if user_provided[pid] is not None:
                ref_channels.append(user_provided[pid])
                continue

            # get protein mask and values
            prot_mask = np.logical_and(single_masks, self._controls_masks[:, pid])
            if not np.any(prot_mask):
                ref_channels.append(None)
                continue

            prot_values = self._controls_values[prot_mask]

            # compute saturation proportions
            prot_sat = np.logical_or(
                prot_values <= self._saturarion_thresholds[:, 0],
                prot_values >= self._saturarion_thresholds[:, 1],
            )
            sat_proportions = prot_sat.sum(axis=0) / prot_sat.shape[0]

            # get other metrics for this protein
            channel_scores = []
            for cid, chan in enumerate(self._channel_names):
                # skip channels with too much saturation
                if sat_proportions[cid] > max_sat_proportion:
                    channel_scores.append(-np.inf)
                    continue

                # compile scores from different metrics
                scores = {
                    'dynamic_range': metrics.dynamic_ranges[prot][chan],
                    'snr': metrics.signal_to_noise[prot][chan],
                    'specificity': metrics.specificities[prot][chan],
                    'signal': metrics.signal_strengths[prot][chan],
                }

                # skip channels with insufficient dynamic range
                if scores['dynamic_range'] < min_dyn_range:
                    channel_scores.append(-np.inf)
                    continue

                # normalize each metric to 0-1 range for this protein
                metric_weights = {
                    'dynamic_range': 1.0,
                    'specificity': 0.8,
                    'snr': 0.5,
                    'signal': 0.4,
                }

                scores = normalized_metrics(scores)

                total_score = 0
                for metric, weight in metric_weights.items():
                    score = scores[metric]
                    total_score += weight * score

                channel_scores.append(total_score)

            channel_scores = np.array(channel_scores)

            if np.all(channel_scores == -np.inf):
                ref_channels.append(None)
                raise ValueError(
                    f"""
                    No reference channel found for {prot}!
                    Try increasing max_sat_proportion and/or decreasing min_dyn_range
                    Saturation proportions per channel: {sat_proportions}
                    """
                )
            else:
                best_channel = int(np.argmax(channel_scores))
                ref_channels.append(best_channel)
                self._log.debug(
                    f"Reference channel for {prot} is {self._channel_names[best_channel]} "
                    f"with score {channel_scores[best_channel]:.2f}"
                )

        return ref_channels

    def process(self, ctx):
        loader = ctx.get('cell_data_loader', ut.load_to_df)
        data = loader(ctx.pipeline_input, self.use_channels).values
        assert data.shape[1] == len(self._channel_names)
        return Context(observations_raw=data)

    def initialize(self, ctx):
        loader = ctx.get('cell_data_loader', ut.load_to_df)

        # load both filtered and full data
        self._controls_full = {
            ut.astuple(ut.escape(k)): loader(v)  # no channel filtering
            for k, v in self.color_controls.items()
        }

        self._controls = {
            ut.astuple(ut.escape(k)): loader(v, self.use_channels)
            for k, v in self.color_controls.items()
        }

        self._channel_names = list(list(self._controls.values())[0].columns)
        self._all_channel_names = list(list(self._controls_full.values())[0].columns)

        assert len(self._channel_names) > 0, "No channels found in the provided controls"
        assert all([len(c.columns) == len(self._channel_names) for c in self._controls.values()]), (
            "All controls must have the same number of channels"
        )

        controls_order = sorted(list(self._controls.keys()))
        self._protein_names = sorted(
            list(
                set(
                    [
                        p
                        for c in controls_order
                        for p in c
                        if p not in _VALID_BLANK_NAMES + _VALID_ALL_NAMES
                    ]
                )
            )
        )

        self._controls_values, self._controls_masks = self._process_controls(self._controls)
        self._controls_values_full, self._controls_masks_full = self._process_controls(
            self._controls_full
        )

        assert self._controls_values.shape[0] == self._controls_masks.shape[0], (
            "Mismatch between values and masks"
        )
        assert len(self._protein_names) == self._controls_masks.shape[1], (
            "Mismatch between masks and protein names"
        )
        assert self._controls_values.shape[1] == len(self._channel_names), (
            "Mismatch between values and channel names"
        )

        self._log.debug(f"Loaded {self._controls_values.shape[0]} events")

        self._saturarion_thresholds = utils.estimate_saturation_thresholds(self._controls_values)
        print(f'saturation thresholds: {self._saturarion_thresholds}')

        self._log.debug(f"Saturation thresholds: {self._saturarion_thresholds}")

        self._reference_channels = self.pick_reference_channels()
        self._log.debug(f"Reference channels: {self._reference_channels}")
        self._log.debug(
            f"Using {len(self._channel_names)} channels: {self._channel_names[:8]}{f'+{len(self._channel_names) - 8}' if len(self._channel_names) > 8 else ''}"
        )

        # compute metrics for both filtered and full data
        metrics = ChannelMetrics(
            self._controls_values,
            self._controls_masks,
            self._channel_names,
            self._protein_names,
        )

        # store full metrics for diagnostics
        self._metrics_full = ChannelMetrics(
            self._controls_values_full,
            self._controls_masks_full,
            self._all_channel_names,
            self._protein_names,
        )

        from copy import deepcopy

        self._diag_controls_values = deepcopy(self._controls_values)
        self._diag_controls_masks = deepcopy(self._controls_masks)
        self._diag_channel_names = deepcopy(self._channel_names)
        self._diag_protein_names = deepcopy(self._protein_names)
        self._diag_saturation_thresholds = deepcopy(self._saturarion_thresholds)
        self._diag_reference_channels = deepcopy(self._reference_channels)
        self._diag_metrics = deepcopy(metrics)

        task_metrics = self._compute_task_metrics(metrics)

        return Context(
            channel_correlations=metrics.correlations,
            channel_signal_strengths=metrics.signal_strengths,
            channel_signal_to_noise=metrics.signal_to_noise,
            channel_dynamic_ranges=metrics.dynamic_ranges,
            channel_specificities=metrics.specificities,
            channel_crosstalks=metrics.crosstalks,
            raw_color_controls=self.color_controls,
            controls_values=self._controls_values,
            controls_masks=self._controls_masks,
            protein_names=self._protein_names,
            channel_names=self._channel_names,
            saturation_thresholds=self._saturarion_thresholds,
            reference_channels=self._reference_channels,
            metrics=task_metrics,
        )

    def _compute_task_metrics(self, channel_metrics: ChannelMetrics) -> TaskMetrics:
        metrics = TaskMetrics()
        quality = {}
        for prot in self._protein_names:
            if prot not in channel_metrics.signal_to_noise:
                continue
            ref_idx = self._reference_channels[self._protein_names.index(prot)]
            ref_chan = self._channel_names[ref_idx]
            snr_vals = sorted(channel_metrics.signal_to_noise[prot].values(), reverse=True)
            quality[prot] = {
                'reference_channel': ref_chan,
                'reference_snr': float(channel_metrics.signal_to_noise[prot][ref_chan]),
                'reference_specificity': float(channel_metrics.specificities[prot][ref_chan]),
                'reference_dynamic_range': float(channel_metrics.dynamic_ranges[prot][ref_chan]),
                'reference_dominance': float(snr_vals[0] / snr_vals[1])
                if len(snr_vals) > 1 and snr_vals[1] > 0
                else float('inf'),
            }
        metrics.add(
            'channel_quality_per_protein',
            quality,
            'Per-protein: reference_snr, reference_specificity, reference_dynamic_range, reference_dominance.',
            scope='individual',
        )

        single = self._controls_masks.sum(axis=1) == 1
        events = {
            'blank_events': int(np.sum(~np.any(self._controls_masks, axis=1))),
            'all_color_events': int(np.sum(np.all(self._controls_masks, axis=1))),
            'total_events': int(self._controls_values.shape[0]),
            **{
                f'{p}_single_events': int(np.sum(single & self._controls_masks[:, i].astype(bool)))
                for i, p in enumerate(self._protein_names)
            },
        }
        metrics.add('event_counts', events, 'Events per control type.', scope='individual')

        snr = [d['reference_snr'] for d in quality.values()]
        spec = [d['reference_specificity'] for d in quality.values()]
        dom = [
            d['reference_dominance']
            for d in quality.values()
            if d['reference_dominance'] != float('inf')
        ]

        if snr:
            metrics.add(
                'min_reference_snr', float(np.min(snr)), 'Min SNR. <1 is weak.', scope='global'
            )
            metrics.add('mean_reference_snr', float(np.mean(snr)), 'Mean SNR.', scope='global')
        if dom:
            metrics.add(
                'min_reference_dominance',
                float(np.min(dom)),
                'Min dominance. >2 good, <1.5 uncertain.',
                scope='global',
            )
        if spec:
            metrics.add(
                'min_reference_specificity',
                float(np.min(spec)),
                'Min specificity. <1 is high crosstalk.',
                scope='global',
            )
        metrics.add(
            'num_proteins', len(self._protein_names), 'Proteins in calibration.', scope='global'
        )
        metrics.add('num_channels', len(self._channel_names), 'Channels used.', scope='global')
        metrics.add('protein_names', list(self._protein_names), 'List of proteins.', scope='global')
        metrics.add(
            'channel_names', list(self._channel_names), 'List of channels used.', scope='global'
        )
        return metrics

    def _process_controls(self, controls):
        """Helper to process controls into values and masks arrays."""
        controls_values, controls_masks = [], []
        controls_order = sorted(list(controls.keys()))
        NPROTEINS = len(self._protein_names)
        has_blank = has_all = False

        for m in controls_order:
            vals = controls[m].values
            if m in [ut.astuple(ut.escape(n)) for n in _VALID_BLANK_NAMES]:
                base_mask = np.zeros((NPROTEINS,))
                has_blank = True
            elif m in [ut.astuple(ut.escape(n)) for n in _VALID_ALL_NAMES]:
                base_mask = np.ones((NPROTEINS,))
                has_all = True
            else:
                base_mask = np.array([p in m for p in self._protein_names])

            masks = np.tile(base_mask, (vals.shape[0], 1))
            controls_values.append(vals)
            controls_masks.append(masks)

        assert has_blank, f'BLANK control not found. Should be named any of {_VALID_BLANK_NAMES}'
        assert has_all, f'ALLCOLOR control not found. Should be named any of {_VALID_ALL_NAMES}'

        return np.vstack(controls_values), np.vstack(controls_masks)

    def diagnostics(
        self,
        ctx: Context,
        marker_color: str = 'blue',
        show_all_channels: bool = True,
        show_correlations: bool = True,
        **kw,
    ) -> Optional[List[DiagnosticFigure]]:
        figures = []

        # Raw channels to reference plot
        kw2 = {
            k: v
            for k, v in kw.items()
            if k
            not in [
                'controls_values',
                'controls_masks',
                'reference_channels',
                'channel_names',
                'protein_names',
            ]
        }
        f, _ = plots.plot_channels_to_reference(
            controls_values=self._diag_controls_values,
            controls_masks=self._diag_controls_masks,
            reference_channels=self._diag_reference_channels,
            channel_names=self._diag_channel_names,
            protein_names=self._diag_protein_names,
            **kw2,
        )
        figures.append(DiagnosticFigure(fig=f, name='Raw channels to reference'))

        # Get metrics
        metrics = (
            self._metrics_full
            if show_all_channels
            else ChannelMetrics(
                ctx['controls_values'],
                ctx['controls_masks'],
                ctx['channel_names'],
                ctx['protein_names'],
            )
        )

        all_channel_names = (
            self._all_channel_names if show_all_channels else self._diag_channel_names
        )
        protein_names = self._diag_protein_names
        reference_channels = (
            self._diag_reference_channels
            if show_all_channels
            else ctx.get('reference_channels', self._diag_reference_channels)
        )

        # Filter out TIME, FSC*, SSC* channels for display
        display_channels, display_indices = _filter_display_channels(all_channel_names)
        n_prot, n_disp_chan = len(protein_names), len(display_channels)

        # Map reference channels to display indices
        ref_to_display = []
        for ref_idx in reference_channels:
            if ref_idx in display_indices:
                ref_to_display.append(display_indices.index(ref_idx))
            else:
                ref_to_display.append(None)

        # Build metric arrays (filtered to display channels)
        metrics_data = metrics.get_all_metrics()
        n_all_chan = len(all_channel_names)

        spec_full = np.array([m['specificity'] for m in metrics_data]).reshape(n_prot, n_all_chan)
        dynrange_full = np.array([m['dynamic_range'] for m in metrics_data]).reshape(
            n_prot, n_all_chan
        )

        spec_data = spec_full[:, display_indices]
        dynrange_data = dynrange_full[:, display_indices]

        # Compute saturation proportions (use full data if show_all_channels)
        sat_data = None
        if show_all_channels and hasattr(self, '_controls_values_full'):
            ctrl_vals = self._controls_values_full
            ctrl_masks = self._controls_masks_full
            sat_thresh = utils.estimate_saturation_thresholds(ctrl_vals)
        elif hasattr(self, '_saturarion_thresholds') and self._saturarion_thresholds is not None:
            ctrl_vals = self._controls_values
            ctrl_masks = self._controls_masks
            sat_thresh = self._saturarion_thresholds
        else:
            ctrl_vals = ctrl_masks = sat_thresh = None

        if ctrl_vals is not None:
            single_masks = ctrl_masks.sum(axis=1) == 1
            sat_data = np.zeros((n_prot, n_disp_chan))
            for pid in range(n_prot):
                prot_mask = np.logical_and(single_masks, ctrl_masks[:, pid])
                if np.any(prot_mask):
                    prot_vals = ctrl_vals[prot_mask]
                    for di, ci in enumerate(display_indices):
                        if ci < prot_vals.shape[1] and ci < sat_thresh.shape[0]:
                            saturated = np.logical_or(
                                prot_vals[:, ci] <= sat_thresh[ci, 0],
                                prot_vals[:, ci] >= sat_thresh[ci, 1],
                            )
                            sat_data[pid, di] = saturated.sum() / len(saturated)

        # Compute grades
        grades = _compute_quality_grades(metrics, reference_channels, display_indices)

        # Compute colormap bounds for consistent coloring
        spec_vmin = 0.1
        spec_vmax = max(10.0, np.nanmax(spec_data))

        # Layout: Summary | Specificity | Correlations
        n_rows = 3 if show_correlations else 2
        height_ratios = [0.15, 1.0, 0.8] if show_correlations else [0.15, 1.0]
        fig_height = 12 if show_correlations else 7
        fig_width = max(12, n_disp_chan * 1.5)

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=120)
        gs = GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.40, figure=fig)

        # Summary bar (using same colormap bounds as heatmap)
        ax_summary = fig.add_subplot(gs[0])
        _render_summary_bar(ax_summary, grades, spec_vmin=spec_vmin, spec_vmax=spec_vmax)

        # Specificity heatmap (main panel)
        ax_spec = fig.add_subplot(gs[1])
        _render_specificity_heatmap(
            ax_spec,
            spec_data,
            dynrange_data,
            sat_data,
            protein_names,
            display_channels,
            reference_markers=ref_to_display,
        )

        # Highlight selected channels if showing all
        if show_all_channels and hasattr(self, '_channel_names'):
            selected_in_display = [
                display_channels.index(ch) for ch in self._channel_names if ch in display_channels
            ]
            for idx in selected_in_display:
                if idx < len(ax_spec.get_xticklabels()):
                    ax_spec.get_xticklabels()[idx].set_color(marker_color)

        # Correlation panel (full row)
        if show_correlations:
            ax_corr = fig.add_subplot(gs[2])
            _render_correlation_panel(
                ax_corr, metrics.correlations, display_channels, display_indices
            )

        fig.suptitle('Channel Quality Dashboard', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        figures.append(DiagnosticFigure(fig=fig, name='Channel quality metrics'))
        return figures
