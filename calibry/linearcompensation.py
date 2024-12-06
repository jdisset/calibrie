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
from .pipeline import Task, DiagnosticFigure
from .utils import Context, Escaped, generate_controls_dict
import seaborn as sns
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

        return Context(
            autofluorescence=self._autofluorescence,
            spillover_matrix=self._spillover_matrix,
            channel_weights=self._channel_weights,
            controls_abundances_AU=self._controls_abundance_AU,
            F_mat=self._generate_function_matrix(),
        )

    def _compute_channel_weights(self) -> np.ndarray:
        """compute channel weights using quality metrics"""
        weights = np.ones(len(self._channel_names))

        for i, chan in enumerate(self._channel_names):
            # combine SNR and specificity for each protein that uses this channel
            chan_weights = []
            for prot in self._protein_names:
                if chan in self._signal_to_noise[prot]:
                    snr = self._signal_to_noise[prot][chan]
                    spec = self._specificities[prot][chan]
                    # use geometric mean of SNR and specificity
                    chan_weights.append(np.sqrt(snr * spec))

            if chan_weights:
                # use max weight across proteins
                weights[i] = np.max(chan_weights)

        w = weights / np.max(weights)  # normalize
        return w**self.channel_weight_attenuation_power

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
                        S[i, j] = np.sum(w * x * y) / np.sum(w * x * x)

            else:  # simple least squares
                ref_values = prot_values[:, ref_chan]
                for j in range(len(self._channel_names)):
                    if j == ref_chan:
                        S[i, j] = 1.0
                    else:
                        S[i, j] = np.sum(ref_values * prot_values[:, j]) / np.sum(ref_values**2)

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

        # Plot heatmap of weight impact
        sns.heatmap(
            weighted_importance,
            xticklabels=self._channel_names,
            yticklabels=self._protein_names,
            cmap='Greys',
            ax=ax_impact,
        )
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
