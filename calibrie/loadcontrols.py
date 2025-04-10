"""
This module provides the LoadControls task, which loads single, multi and no-protein controls from a list of files.
"""

from .pipeline import Task, DiagnosticFigure
from typing import List, Dict, Tuple, Optional, Union, Annotated
from . import utils as ut
from calibrie.utils import LoadedData, Context
from calibrie.gui.core import make_ui_field, ListManagerUI, DictEditorUI
import numpy as np
import jax.numpy as jnp
from . import plots, utils
from matplotlib import pyplot as plt
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
        for pid, prot in enumerate(self.protein_names):
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

    color_controls: Annotated[
        Dict[Union[str, Tuple[str]], LoadedData],
        make_ui_field(
            DictEditorUI, help="Map control names (e.g., 'GFP', 'BLANK') or tuples to file paths."
        ),
    ] = {}
    use_channels: Annotated[
        List[str],
        make_ui_field(ListManagerUI, help="List of channel names (exact match) to load and use."),
    ] = []
    use_reference_channels: Annotated[
        Dict[str, str],
        make_ui_field(
            DictEditorUI, help="Map protein names to their primary reference channel name."
        ),
    ] = {}

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
            for pid, p in enumerate(self._protein_names):
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
        assert all(
            [len(c.columns) == len(self._channel_names) for c in self._controls.values()]
        ), "All controls must have the same number of channels"

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

        assert (
            self._controls_values.shape[0] == self._controls_masks.shape[0]
        ), "Mismatch between values and masks"
        assert (
            len(self._protein_names) == self._controls_masks.shape[1]
        ), "Mismatch between masks and protein names"
        assert self._controls_values.shape[1] == len(
            self._channel_names
        ), "Mismatch between values and channel names"

        self._log.debug(f"Loaded {self._controls_values.shape[0]} events")

        self._saturarion_thresholds = utils.estimate_saturation_thresholds(self._controls_values)
        print(f'saturation thresholds: {self._saturarion_thresholds}')

        self._log.debug(f"Saturation thresholds: {self._saturarion_thresholds}")

        self._reference_channels = self.pick_reference_channels()
        self._log.debug(f"Reference channels: {self._reference_channels}")
        self._log.debug(
            f"Using {len(self._channel_names)} channels: {self._channel_names[:8]}{f'+{len(self._channel_names)-8}' if len(self._channel_names)>8 else ''}"
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
        )

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
        marker_size: int = 40,
        show_all_channels: bool = True,
        **kw,
    ) -> Optional[List[DiagnosticFigure]]:
        figures = []

        f, _ = plots.plot_channels_to_reference(
            **ctx,
            **kw,
        )
        figures.append(DiagnosticFigure(fig=f, name='Raw channels to reference'))

        # use appropriate metrics based on whether we're showing all channels
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
        channel_names = self._all_channel_names if show_all_channels else ctx['channel_names']

        # create quality metrics plot
        metrics_data = metrics.get_all_metrics()

        # convert metrics to arrays for plotting
        metrics_array = np.array(
            [[m[key] for m in metrics_data] for key in ['crosstalk', 'snr', 'specificity']]
        )

        axratio = len(channel_names) / len(ctx['protein_names'])
        fig, axes = plt.subplots(3, 1, figsize=(20 * axratio, 15))
        fig.suptitle('Channel Quality Metrics', fontsize=16, y=1)

        titles = [
            'Cross-talk (Z-score): how much signal is picked up from other proteins',
            'Signal-to-Noise Ratio',
            'Specificity (Signal/Cross-talk)',
        ]

        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            data = metrics_array[i].reshape(len(ctx['protein_names']), len(channel_names))

            im = ax.imshow(data, cmap='Blues', interpolation='nearest')

            # add white grid
            ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)

            # set major ticks for labels
            ax.set_xticks(np.arange(len(channel_names)))
            ax.set_yticks(np.arange(len(ctx['protein_names'])))
            ax.set_xticklabels(channel_names, rotation=45, ha='right')
            ax.set_yticklabels(ctx['protein_names'])

            ax.set_title(title)

            # remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # add colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

            aspect = 10
            pad_fraction = 0.5
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1.0 / aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            fig.colorbar(im, cax=cax)

            # add text annotations
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    value = data[y, x]
                    midpoint = ((np.max(data) - np.min(data)) / 2) + np.min(data)
                    color = 'black' if value < midpoint else 'white'
                    text = f'{value:.2f}'
                    ax.text(x, y, text, ha='center', va='center', color=color, weight='bold')

            if show_all_channels:
                selected_indices = [channel_names.index(ch) for ch in self._channel_names]
                for idx in selected_indices:
                    ax.get_xticklabels()[idx].set_color(marker_color)

        plt.tight_layout()
        figures.append(DiagnosticFigure(fig=fig, name='Channel quality metrics'))
        return figures
