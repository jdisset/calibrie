# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

"""
This module provides the LinearCompensation task, which implements linear compensation of fluorescence values using
a spillover matrix computed from single-protein controls.
The module also provides diagnostic plots to visualize the compensation results.

!!! abstract "About linear unmixing"
    The linear unmixing model assumes that the observed fluorescence values $Y$ in each channel are a linear combination of the
    true protein abundances $X$ + some base autofluorescence $A$.
    The proportions of how much a fluo protein contributes to each channel is stored in the rows of the spillover matrix $S$
    (a.k.a bleedthrough matrix or spectral signature).
    $$ Y = XS + A $$

    **The main assumptions are:**

    - each of the studied fluo protein produces fluorescence in all channels (hopefully in negligible amount for many of them)
    - scaling the amount of a given protein in a cell should scale its contribution to the fluorescence in all channels by the same factor
    - the fluo value observed in a channel is the sum of the contributions from all proteins in this channel
"""

from jaxopt import GaussNewton
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .pipeline import Task, DiagnosticFigure
from .utils import Context, Escaped, generate_controls_dict
from .metrics import TaskMetrics
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from . import plots, utils


class LinearCompensation(Task):
    """Task for the linear unmixing of fluorescence values."""

    mode: str = 'weighted'  # 'weighted' or 'simple'
    use_matrix: Optional[np.ndarray] = None
    matrix_corrections: Optional[Escaped[Dict[str, Escaped[Dict[str, float]]]]] = None
    unmix_controls: bool = True
    channel_weight_attenuation_power: float = 2.0

    def initialize(self, ctx: Context):
        self._saturation_thresholds = ctx.saturation_thresholds
        self._controls_values = ctx.controls_values
        self._controls_masks = ctx.controls_masks
        self._protein_names = ctx.protein_names
        self._channel_names = ctx.channel_names
        self._reference_channels = ctx.reference_channels

        # get channel quality metrics from context
        self._signal_to_noise = ctx.channel_signal_to_noise
        self._specificities = ctx.channel_specificities
        self._crosstalks = ctx.channel_crosstalks

        self._autofluorescence = np.mean(
            self._controls_values[np.all(self._controls_masks == 0, axis=1)], axis=0
        )

        if self.use_matrix is not None:
            self._spillover_matrix = self.use_matrix
            self._log.debug(f"Using provided spillover matrix:\n{self._spillover_matrix}")
        else:
            self._log.debug(f"Computing spillover matrix using {self.mode} least squares")
            self._spillover_matrix = self._compute_spillover_matrix()

        # apply any user-provided corrections
        if self.matrix_corrections is not None:
            for p, corrections in self.matrix_corrections.items():
                if p not in self._protein_names:
                    self._log.warning(f"Protein {p} in matrix_corrections not found")
                    continue
                i = self._protein_names.index(p)
                for c, v in corrections.items():
                    j = self._channel_names.index(c)
                    self._spillover_matrix[i, j] *= v
                    self._log.debug(f"Corrected spillover matrix[{i}, {j}] *= {v}")

        # compute channel weights using quality metrics
        self._channel_weights = self._compute_channel_weights()

        if self.unmix_controls:
            self._log.debug("Unmixing controls")
            self._controls_abundance_AU = self.process(
                Context(observations_raw=self._controls_values, channel_names=self._channel_names)
            ).abundances_AU

        # logdebug most things:
        self._log.debug(f"Protein names: {self._protein_names}")
        self._log.debug(f"Channel names: {self._channel_names}")
        self._log.debug(f"Reference channels: {self._reference_channels}")
        self._log.debug(f"Autofluorescence: {self._autofluorescence}")
        self._log.debug(f"Spillover matrix:\n{self._spillover_matrix}")
        self._log.debug(f"Channel weights: {self._channel_weights}")
        self._log.debug(f"Controls abundance AU: {self._controls_abundance_AU}")

        metrics = self._compute_metrics()

        return Context(
            autofluorescence=self._autofluorescence,
            spillover_matrix=self._spillover_matrix,
            channel_weights=self._channel_weights,
            controls_abundances_AU=self._controls_abundance_AU,
            F_mat=self._generate_function_matrix(),
            metrics=metrics,
        )

    def _compute_channel_weights(self) -> np.ndarray:
        """compute channel weights using quality metrics"""
        weights = np.ones(len(self._channel_names))
        self._log.debug("Computing channel weights using quality metrics")

        for i, chan in enumerate(self._channel_names):
            # combine SNR and specificity for each protein that uses this channel
            chan_weights = []
            for prot in self._protein_names:
                self._log.debug(f"Protein {prot}:")
                if chan in self._signal_to_noise[prot]:
                    self._log.debug(f"   Channel {chan}:")
                    snr = self._signal_to_noise[prot][chan]
                    spec = np.maximum(self._specificities[prot][chan], 0)
                    self._log.debug(f"   SNR={snr}, specificity={spec}")
                    # use geometric mean of SNR and specificity
                    chan_weights.append(np.sqrt(snr * spec))
                    self._log.debug(f"    Weight: {chan_weights[-1]}")

            if chan_weights:
                # use max weight across proteins
                weights[i] = np.max(chan_weights)
                self._log.debug(f"Max weight: {weights[i]}")

        w = weights / np.max(weights)  # normalize
        W = w**self.channel_weight_attenuation_power
        return W

    def _compute_spillover_matrix(self) -> np.ndarray:
        """compute spillover matrix using weighted least squares"""
        single_masks = self._controls_masks.sum(axis=1) == 1

        # initialize matrix
        S = np.zeros((len(self._protein_names), len(self._channel_names)))

        # subtract autofluorescence
        Y = self._controls_values - self._autofluorescence

        for i, prot in enumerate(self._protein_names):
            # get protein-specific mask
            prot_mask = np.logical_and(single_masks, self._controls_masks[:, i])
            if not np.any(prot_mask):
                continue

            # get protein values and reference channel
            prot_values = Y[prot_mask]
            ref_chan = self._reference_channels[i]

            if self.mode == 'weighted':
                # compute weights using quality metrics
                weights = np.ones_like(prot_values)
                for j, chan in enumerate(self._channel_names):
                    if chan in self._signal_to_noise[prot]:
                        weights[:, j] *= self._signal_to_noise[prot][chan]
                    if chan in self._specificities[prot]:
                        weights[:, j] *= self._specificities[prot][chan]

                # normalize weights
                weights /= np.max(weights)

                # weighted regression for each channel
                for j in range(len(self._channel_names)):
                    if j == ref_chan:
                        S[i, j] = 1.0
                    else:
                        w = weights[:, j]
                        x = prot_values[:, ref_chan]
                        y = prot_values[:, j]
                        denom = np.sum(w * x * x)
                        if denom > 0:
                            S[i, j] = np.sum(w * x * y) / denom
                        else:
                            S[i, j] = 0.0

            else:  # simple least squares
                ref_values = prot_values[:, ref_chan]
                for j in range(len(self._channel_names)):
                    if j == ref_chan:
                        S[i, j] = 1.0
                    else:
                        denom = np.sum(ref_values**2)
                        if denom > 0:
                            S[i, j] = np.sum(ref_values * prot_values[:, j]) / denom
                        else:
                            S[i, j] = 0.0

        return np.clip(S, 0, None)  # ensure non-negative

    def process(self, ctx: Context):
        observations_raw = ctx.observations_raw
        channel_names = ctx.channel_names

        self._log.debug(f"Linear unmixing of {observations_raw.shape[0]} observations")

        # check for saturation
        not_saturated = (observations_raw > self._saturation_thresholds[:, 0]) & (
            observations_raw < self._saturation_thresholds[:, 1]
        )

        # subtract autofluorescence
        Y = observations_raw - self._autofluorescence

        # solve weighted least squares problem
        @jit
        def solve_abundances(Y, W, S):
            def solve_one(y, w):
                y_weighted = y * w
                S_weighted = S * w[None, :]
                x, _, _, _ = jnp.linalg.lstsq(S_weighted.T, y_weighted, rcond=None)
                return x

            return vmap(solve_one)(Y, W)

        # use channel weights and saturation in solving
        W = not_saturated * self._channel_weights
        X = solve_abundances(Y, W, self._spillover_matrix)

        return Context(abundances_AU=X)

    def _generate_function_matrix(self):
        """generate matrix of linear functions for downstream use"""

        def make_lin_fun(a):
            def lin_fun(x):
                return x * a

            return lin_fun

        return [[make_lin_fun(a) for a in row] for row in self._spillover_matrix]

    def _compute_residual_nonlinearity(self, x_ref, y_off_target, n_bins=20):
        valid = np.isfinite(x_ref) & np.isfinite(y_off_target)
        if np.sum(valid) < 100:
            return np.nan
        x, y = x_ref[valid], y_off_target[valid]
        percentiles = np.linspace(5, 95, n_bins + 1)
        bin_edges = np.percentile(x, percentiles)
        bin_means = []
        for i in range(n_bins):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            if np.sum(mask) > 10:
                bin_means.append(np.median(y[mask]))
        if len(bin_means) < 5:
            return np.nan
        bin_means = np.array(bin_means)
        y_range = np.percentile(y, 99) - np.percentile(y, 1)
        if y_range <= 0:
            return 0.0
        return float(np.mean(np.abs(bin_means)) / (y_range + 1e-10))

    def _compute_zero_centering_bias(self, y_off_target):
        valid = np.isfinite(y_off_target)
        if np.sum(valid) < 100:
            return np.nan
        y = y_off_target[valid]
        median_val = np.median(y)
        mad = np.median(np.abs(y - median_val))
        if mad <= 0:
            return 0.0
        return float(np.abs(median_val) / (mad + 1e-10))

    def _compute_distribution_smoothness(self, y, n_bins=50):
        valid = np.isfinite(y)
        if np.sum(valid) < 100:
            return np.nan
        y = y[valid]
        hist, _ = np.histogram(y, bins=n_bins, density=True)
        smooth_hist = gaussian_filter1d(hist, sigma=1.5)
        second_deriv = np.diff(smooth_hist, n=2)
        return float(np.std(second_deriv) / (np.mean(smooth_hist) + 1e-10))

    def _compute_multimodality_score(self, y, min_prominence_ratio=0.1):
        valid = np.isfinite(y)
        if np.sum(valid) < 100:
            return np.nan
        y = y[valid]
        hist, _ = np.histogram(y, bins=100, density=True)
        smooth_hist = gaussian_filter1d(hist, sigma=2)
        peaks, properties = find_peaks(smooth_hist, prominence=0)
        if len(peaks) == 0:
            return 1
        max_prominence = np.max(properties['prominences'])
        significant_peaks = np.sum(properties['prominences'] >= min_prominence_ratio * max_prominence)
        return int(max(1, significant_peaks))

    def _compute_metrics(self) -> TaskMetrics:
        metrics = TaskMetrics()
        S, n_prot = self._spillover_matrix, len(self._protein_names)

        try:
            cond = float(np.linalg.cond(S))
        except Exception:
            cond = np.nan
        metrics.add('spillover_matrix_condition_number', cond,
            'Condition number. Lower=better. >10 may indicate instability.', scope='global')

        bleedthrough, total_off_diag, max_off_diag = {}, 0.0, 0.0
        for pid, prot in enumerate(self._protein_names):
            ref_idx = self._reference_channels[pid]
            off_diag = np.delete(S[pid, :], ref_idx) if ref_idx is not None else S[pid, :]
            off_diag = off_diag[off_diag > 0]
            bleedthrough[prot] = {
                'max_spillover': float(np.max(off_diag)) if len(off_diag) > 0 else 0.0,
                'sum_spillover': float(np.sum(off_diag)),
                'reference_channel': self._channel_names[ref_idx] if ref_idx is not None else None,
            }
            total_off_diag += bleedthrough[prot]['sum_spillover']
            max_off_diag = max(max_off_diag, bleedthrough[prot]['max_spillover'])

        metrics.add('bleedthrough_per_protein', bleedthrough,
            'Per-protein: max_spillover, sum_spillover, reference_channel.', scope='individual')
        metrics.add('mean_total_bleedthrough', float(total_off_diag / n_prot),
            'Mean off-diagonal spillover. >0.5 is significant.', scope='global')
        metrics.add('max_single_bleedthrough', float(max_off_diag),
            'Max spillover coefficient. >0.3 is problematic.', scope='global')

        if hasattr(self, '_controls_abundance_AU') and self._controls_abundance_AU is not None:
            single_masks = self._controls_masks.sum(axis=1) == 1
            comp_quality = {}
            for ctrl_pid, ctrl_prot in enumerate(self._protein_names):
                prot_mask = single_masks & self._controls_masks[:, ctrl_pid].astype(bool)
                if not np.any(prot_mask):
                    continue
                ctrl_ab = self._controls_abundance_AU[prot_mask]
                x_ref = ctrl_ab[:, ctrl_pid]
                comp_quality[ctrl_prot] = {
                    off_prot: {
                        'nonlinearity': self._compute_residual_nonlinearity(x_ref, ctrl_ab[:, off_pid]),
                        'zero_centering_bias': self._compute_zero_centering_bias(ctrl_ab[:, off_pid]),
                        'distribution_smoothness': self._compute_distribution_smoothness(ctrl_ab[:, off_pid]),
                        'multimodality': self._compute_multimodality_score(ctrl_ab[:, off_pid]),
                    }
                    for off_pid, off_prot in enumerate(self._protein_names) if off_pid != ctrl_pid
                }

            metrics.add('compensation_quality_per_protein', comp_quality,
                'Per-protein-pair: nonlinearity (<0.05 good), zero_centering_bias (<0.5 good), multimodality (1=good).', scope='individual')

            all_vals = [(d['nonlinearity'], d['zero_centering_bias'], d['multimodality'])
                        for pdata in comp_quality.values() for d in pdata.values()]
            all_nonlin = [v[0] for v in all_vals if not np.isnan(v[0])]
            all_bias = [v[1] for v in all_vals if not np.isnan(v[1])]
            all_multi = [v[2] for v in all_vals if not np.isnan(v[2])]

            if all_nonlin:
                metrics.add('mean_nonlinearity', float(np.mean(all_nonlin)),
                    'Mean S-curve metric. <0.05 good, >0.1 problematic.', scope='global')
                metrics.add('max_nonlinearity', float(np.max(all_nonlin)),
                    'Worst-case S-curve.', scope='global')
            if all_bias:
                metrics.add('mean_zero_centering_bias', float(np.mean(all_bias)),
                    'Mean centering bias. <0.5 good.', scope='global')
            if all_multi:
                metrics.add('multimodal_pair_count', sum(1 for m in all_multi if m > 1),
                    'Pairs with multimodal distributions. 0 is ideal.', scope='global')

            q = [1 - min(max_off_diag * 2, 1), 1 - min(total_off_diag / n_prot, 1)]
            if all_nonlin:
                q.append(1 - min(np.mean(all_nonlin) * 10, 1))
            if all_bias:
                q.append(1 - min(np.mean(all_bias) / 2, 1))
            metrics.add('overall_compensation_quality', float(np.mean(q)),
                'Quality score (0-1). >0.8 good, 0.5-0.8 acceptable, <0.5 needs attention.', scope='global')

        return metrics

    def diagnostics(self, ctx: Context, **kw) -> Optional[List[DiagnosticFigure]]:
        if not self.unmix_controls:
            self._log.debug("Unmixing controls")
            self._controls_abundance_AU = self.process(
                Context(observations_raw=ctx.controls_values, channel_names=ctx.channel_names)
            ).abundances_AU

        figs = []

        # Create figure with subplots for spillover and weights analysis
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.3])

        # 1. Spillover matrix plot (main)
        ax_spill = fig.add_subplot(gs[0, 0:2])
        im = ax_spill.imshow(self._spillover_matrix, cmap='Greys', vmin=0, vmax=1)

        # Add matrix annotations
        for i in range(len(self._protein_names)):
            for j in range(len(self._channel_names)):
                color = 'white' if self._spillover_matrix[i, j] > 0.5 else 'black'
                ax_spill.text(
                    j,
                    i,
                    f'{self._spillover_matrix[i, j]:.3f}',
                    ha='center',
                    va='center',
                    color=color,
                )

        ax_spill.set_xticks(np.arange(len(self._channel_names)))
        ax_spill.set_yticks(np.arange(len(self._protein_names)))
        ax_spill.set_xticklabels(self._channel_names, rotation=45, ha='right')
        ax_spill.set_yticklabels(self._protein_names)
        ax_spill.set_title('Spillover Matrix')

        # 2. Quality metrics contribution to weights
        ax_qual = fig.add_subplot(gs[0, 2])

        # Collect quality metrics per channel
        channel_metrics = []
        for chan in self._channel_names:
            chan_data = {
                'channel': chan,
                'snr': [],
                'spec': [],
                'final_weight': self._channel_weights[self._channel_names.index(chan)],
            }
            for prot in self._protein_names:
                if chan in self._signal_to_noise[prot]:
                    chan_data['snr'].append(self._signal_to_noise[prot][chan])
                    chan_data['spec'].append(self._specificities[prot][chan])

            if chan_data['snr']:  # if we have data for this channel
                chan_data['max_snr'] = np.max(chan_data['snr'])
                chan_data['max_spec'] = np.max(chan_data['spec'])
            else:
                chan_data['max_snr'] = 0
                chan_data['max_spec'] = 0

            channel_metrics.append(chan_data)

        # Plot stacked bar of quality metrics
        channels = [d['channel'] for d in channel_metrics]
        snr_vals = [d['max_snr'] for d in channel_metrics]
        spec_vals = [d['max_spec'] for d in channel_metrics]
        weights = [d['final_weight'] for d in channel_metrics]

        # normalize for visualization
        snr_vals = (
            np.array(snr_vals) / np.max(snr_vals)
            if np.max(snr_vals) > 0
            else np.zeros_like(snr_vals)
        )
        spec_vals = (
            np.array(spec_vals) / np.max(spec_vals)
            if np.max(spec_vals) > 0
            else np.zeros_like(spec_vals)
        )

        x = np.arange(len(channels))
        width = 0.35

        ax_qual.bar(x - width / 2, snr_vals, width, label='SNR (normalized)', alpha=0.6)
        ax_qual.bar(x + width / 2, spec_vals, width, label='Specificity (normalized)', alpha=0.6)
        ax_qual.plot(x, weights, 'ro-', label='Final weight', linewidth=2)

        ax_qual.set_xticks(x)
        ax_qual.set_xticklabels(channels, rotation=45, ha='right')
        ax_qual.set_ylabel('Normalized value')
        ax_qual.set_title('Channel Quality Metrics and Weights')
        ax_qual.legend()

        # 3. Weight impact visualization
        ax_impact = fig.add_subplot(gs[1, :])

        # Show how weights affect different channels
        weighted_importance = self._spillover_matrix * self._channel_weights

        im = ax_impact.imshow(weighted_importance, cmap='Greys', vmin=0, vmax=1)
        ax_impact.set_title('Weighted Channel Importance\n(Spillover × Weight)')
        ax_impact.set_xticklabels(self._channel_names, rotation=45, ha='right')

        plt.tight_layout()
        figs.append(DiagnosticFigure(fig=fig, name="Spillover matrix and weighting analysis"))

        # Add compensation results plot
        controls_dict, std_models = utils.generate_controls_dict(
            self._controls_abundance_AU, self._controls_masks, self._protein_names, **kw
        )

        fig_comp, _ = plots.unmixing_plot(
            controls_dict,
            self._protein_names,
            self._channel_names,
            description='linear compensation, ',
            autofluorescence=self._autofluorescence,
            std_models=std_models,
            **kw,
        )
        figs.append(DiagnosticFigure(fig=fig_comp, name="Compensation results"))

        return figs
