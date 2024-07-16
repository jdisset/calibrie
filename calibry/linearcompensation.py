"""
This module provides the LinearCompensation task, which implements linear compensation of fluorescence values using
a spillover matrix computed from single-protein controls.
The module also provides a diagnostic plot to visualize the compensation
results.

!!! abstract "About linear unmixing"
    The linear unmixing model assumes that the observed fluorescence values $Y$ in each channel are a linear combination of the
    true protein abundances $X$ + some base autofluorescence $A$.
    The proportions of how much a fluo protein contributes to each channel is stored in the rows of the spillover matrix $S$
    (a.k.a bleedthrough matrix or spectral signature).
    $$ Y = XS + A $$

    **The main assumptions are:**

    - each of the studied fluo protein produces fluorescence in all channels (hopefully in negligible amount for many of them).
    - scaling the amount of a given protein in a cell should scale its contribution to the fluorescence in all channels by the same factor(i.e. the spectrum stays the same no matter what quantity of said protein is present, and individual contributions are simply added)
    - the fluo value observed in a channel is the sum of the contributions from all proteins in this channel

### Example

```python
import calibry as cal

linpipe = cal.Pipeline(
    [
        cal.LoadControls(color_controls=controls),
        cal.LinearCompensation(),
    ]
)

linpipe.initialize()

# to plot spillover matrix and single-protein compensation:
linpipe.diagnostics("LinearCompensation")

# ... after loading observations:
abundances = linpipe.apply(observations)

```

"""

from jaxopt import GaussNewton
from .pipeline import Task
from .utils import Context
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial
from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional


class LinearCompensation(Task):
    """
    Task for the linear unmixing of fluorescence values.


    ### REQUIRED PIPELINE DATA
    - Basic controls (provided by LoadControls)

    ### PRODUCED PIPELINE DATA
    #### On initialization:
    - `spillover_matrix`: the spillover matrix $S$ (proteins x channels)
    - `autofluorescence`: the autofluorescence vector $A$ (channels)

    #### On application:
    - `abundances_AU`: the estimated abundances $X$ (cells x proteins) in Arbitrary Units (AU)

    """

    mode: str = 'WLR'
    use_matrix: Optional[np.ndarray] = None
    matrix_corrections: Optional[Dict[Tuple[str, str], float]] = None
    use_inverse_variance_estimate_weights: bool = True
    variance_weight_cutoff: float = 100
    channel_weight_attenuation_power: float = 2.0
    unmix_controls: bool = True

    def initialize(self, ctx: Context):

        self._saturation_thresholds = ctx.saturation_thresholds
        self._controls_values = ctx.controls_values
        self._controls_masks = ctx.controls_masks
        self._protein_names = ctx.protein_names
        self._channel_names = ctx.channel_names
        self._reference_channels = ctx.reference_channels

        self._autofluorescence = utils.estimate_autofluorescence(
            ctx.controls_values, ctx.controls_masks
        )

        if self.use_matrix is not None:
            self._spillover_matrix = self.use_matrix
            self._log.debug(f"Using provided spillover matrix:\n{self._spillover_matrix}")

        elif self.mode == 'ALS':
            self._log.debug(f"Computing spillover matrix using Alternating Least Squares")
            self._spillover_matrix, _ = self.compute_spectral_signature_ALS()

        elif self.mode == 'WLR':
            self._log.debug(f"Computing spillover matrix using Weighted Least Squares")
            self._spillover_matrix = self.compute_spectral_signature_WLR()

        else:
            raise ValueError(f"Unknown solver mode {self.mode}")

        self._spillover_matrix = np.array(self._spillover_matrix)

        if self.matrix_corrections is not None:
            for (p, c), v in self.matrix_corrections.items():
                i = self._protein_names.index(p)
                j = self._channel_names.index(c)
                prev_value = self._spillover_matrix[i, j]
                self._spillover_matrix[i, j] *= v
                self._log.debug(
                    f"""Correcting spillover matrix for ({p}, {c}):
                    matrix[{i}, {j}] *= {v} (prev value: {prev_value})"""
                )

        self._channel_weights = self.compute_channel_weights(
            ctx.controls_values, ctx.controls_masks
        )
        self._log.debug(f"Computed channel weights:\n{self._channel_weights}")

        if self.unmix_controls:
            self._log.debug(f"Unmixing controls")
            self._controls_abundance_AU = self.process(
                Context(observations_raw=ctx.controls_values, channel_names=ctx.channel_names)
            ).abundances_AU

        return Context(
            autofluorescence=self._autofluorescence,
            spillover_matrix=self._spillover_matrix,
            channel_weights=self._channel_weights,
            controls_abundances_AU=self._controls_abundance_AU,
            F_mat=generate_function_matrix_from_spillover(self._spillover_matrix),
        )

    def process(self, ctx: Context):

        observations_raw: np.ndarray = ctx.observations_raw
        channel_names: List = ctx.channel_names

        self._log.debug(f"Linear unmixing of {observations_raw.shape[0]} observations")
        self._log.debug(f"spillover_matrix shape: {self._spillover_matrix.shape}")

        assert observations_raw.shape[1] == len(
            channel_names
        ), "Mismatch between observations and channels"

        not_saturated = (observations_raw > self._saturation_thresholds[:, 0]) & (
            observations_raw < self._saturation_thresholds[:, 1]
        )

        Y = observations_raw - self._autofluorescence

        if self.use_inverse_variance_estimate_weights:
            rough_variance_estimate = 1.0 / np.log10(np.clip(Y, self.variance_weight_cutoff, None))
        else:
            rough_variance_estimate = 1.0

        W = not_saturated * rough_variance_estimate * self._channel_weights

        @jit
        def solve_X(Y, W, S):
            def solve_one(y, w):
                y_weighted = y * w
                S_weighted = S * w[None, :]
                x, residuals, rank, s = jnp.linalg.lstsq(S_weighted.T, y_weighted, rcond=None)
                return x

            return vmap(solve_one)(Y, W)

        X = solve_X(Y, W, self._spillover_matrix)

        self._log.debug(f"Linear unmixing done")
        return Context(abundances_AU=X)

    def get_observations_list(self, controls_values, controls_masks):
        # let's isolate all the available single controls
        nproteins = controls_masks.shape[1]
        single_masks = np.eye(nproteins).astype(bool)  # one hot encoding
        null_observations = controls_values[np.all(controls_masks == 0, axis=1)]
        singlectrl_observations = [
            controls_values[np.all(controls_masks == sc, axis=1)] for sc in single_masks
        ]
        return null_observations, singlectrl_observations

    def compute_channel_weights(self, controls_values, controls_masks):
        nproteins, nchannels = (controls_masks.shape[1], controls_values.shape[1])
        corrected_values = controls_values - self._autofluorescence
        null_observations, singleprt_observations = self.get_observations_list(
            corrected_values, controls_masks
        )
        null_std = np.maximum(null_observations.std(axis=0), 1e-6)
        absolute_ranges = np.array(
            [np.subtract(*np.quantile(s, [0.999, 0.001], axis=0)) for s in singleprt_observations]
        )
        assert absolute_ranges.shape == (nproteins, nchannels)

        relative_ranges = absolute_ranges / null_std[None, :]
        norm_relative_ranges = relative_ranges / relative_ranges.max(axis=1)[:, None]

        channel_weights = (
            norm_relative_ranges.max(axis=0)
        ) ** self.channel_weight_attenuation_power

        return channel_weights

    def diagnostics(
        self,
        ctx: Context,
        **kw,
    ):

        assert self._spillover_matrix is not None, "Spillover matrix not computed"
        assert self._autofluorescence is not None, "Autofluorescence not computed"

        if not self.unmix_controls:
            self._log.debug(f"Unmixing controls")

            self._controls_abundance_AU = self.process(
                Context(observations_raw=ctx.controls_values, channel_names=ctx.channel_names)
            ).abundances_AU

        f, a = spillover_matrix_plot(self._spillover_matrix, ctx.channel_names, ctx.protein_names)
        f.suptitle("Spillover matrix", fontsize=16, y=1.05)

        controls_abundances, std_models = utils.generate_controls_dict(
            self._controls_abundance_AU,
            self._controls_masks,
            ctx.protein_names,
            **kw,
        )

        f, a = plots.unmixing_plot(
            controls_abundances,
            ctx.protein_names,
            ctx.channel_names,
            decription='linear model, ',
            std_models=std_models,
            **kw,
        )

    def compute_spectral_signature_ALS(self, max_iterations=50, jax_seed=0, verbose=False, w=None):
        # Spectral signature estimation using Alternating Least Squares (ALS) method.
        # could work with mixed colors (multiple columns of masks being 1) IF the units are the same.

        # Args:
        # Y (array): Observed fluorescence intensities (cells x channels).
        # M (array): Masks matrix from single-color control samples (proteins x channels).
        # max_iterations (int, optional): Maximum number of iterations for ALS. Defaults to 50.
        # jax_seed (int, optional): Seed for JAX random number generator. Defaults to 0.
        # verbose (bool, optional): Display progress information. Defaults to False.

        # Returns:
        # S (array): Spectral signature matrix (proteins x channels).
        # X (array): Estimated protein quantities (proteins x cells).

        # model: Y = XM.S
        # Y: observations
        # M: masks (from controls: single color = (1,0,...))
        # S: spectral signature
        # X: estimated quantity of protein.

        single_masks = self._controls_masks.sum(axis=1) == 1
        M = self._controls_masks[single_masks]
        Y = self._controls_values[single_masks] - self._autofluorescence[single_masks]
        # normalize by row max:
        normalize = lambda x: x / (jnp.maximum(jnp.max(x, axis=1)[:, None], 1e-12))
        # normalize by row norm:
        # normalize = lambda x: x / (jnp.maximum(jnp.linalg.norm(x, axis=1)[:, None], 1e-12))

        X = jax.random.uniform(jax.random.PRNGKey(jax_seed), (M.shape[0],), minval=0.1)
        if w is None:
            w = jnp.ones_like(Y)
        else:
            assert w.shape == Y.shape

        @jit
        def alsq(X):  # one iteration of alternating least squares
            S = jnp.linalg.lstsq(X[:, None] * M, Y, rcond=None)[0]
            S = normalize(S)  # find S from X
            X = Y @ jnp.linalg.pinv(S)  # find X from S
            X = jnp.average(X, axis=1, weights=M)
            X = jnp.maximum(jnp.nan_to_num(X, nan=0), 0)
            return S, X

        Sprev = 0
        for _ in range(max_iterations):
            S, X = alsq(X)
            # error = ((Y - (K[:, None] * M) @ S) ** 2).mean() / jnp.linalg.norm(Y)
            error = jnp.average((Y - (X[:, None] * M) @ S) ** 2, weights=w) / jnp.average(
                Y**2, weights=w
            )
            if verbose:
                print(f'error: {error:.2e}, update: {jnp.mean((Sprev - S)**2):.2e}, S: {S}')
            Sprev = S
        # S = normalize(S)

        return S, X

    def compute_spectral_signature_WLR(
        self,
        min_points=100,
        max_points=100000,
        logger=None,
    ):

        # Spectral signature estimation using Weighted Linear Regression (WLR) method.

        # Returns:
        # S (array): Spectral signature matrix (proteins x channels).

        self._reference_channels = np.array(self._reference_channels)

        Y = self._controls_values - self._autofluorescence
        M = self._controls_masks

        W = (self._controls_values > self._saturation_thresholds[:, 0]) & (
            self._controls_values < self._saturation_thresholds[:, 1]
        )
        rough_variance_estimate = 1.0 / np.log10(np.clip(Y, self.variance_weight_cutoff, None))
        W = W * rough_variance_estimate

        assert Y.shape == (M.shape[0], len(self._channel_names))
        assert M.shape[1] == len(self._protein_names)
        assert W.shape == Y.shape

        # let's reorder Y randomly:
        order = np.random.permutation(Y.shape[0])
        Yp = Y[order]
        Mp = M[order]
        Wp = W[order]

        # first let's isolate all the available single controls
        one_hot = lambda x, n: np.eye(n)[x]
        single_controls = one_hot(
            np.arange(len(self._protein_names)), len(self._protein_names)
        ).astype(bool)
        counts = np.array([np.all(Mp == sc, axis=1).sum() for sc in single_controls])

        if logger is not None:
            logger.debug(f"Single protein controls counts: {counts}")

        for p, c in zip(self._protein_names, counts):
            assert c > min_points, f"Not enough data for single control of {p}"

        largest_count = counts.max()
        pad_to = min(largest_count, max_points)

        all_Y = [Yp[np.all(Mp == sc, axis=1)][:pad_to] for sc in single_controls]
        all_W = [Wp[np.all(Mp == sc, axis=1)][:pad_to] for sc in single_controls]

        # (NPROT, NDATA, NCHAN):
        padded_Y = np.array([np.pad(y, ((0, pad_to - len(y)), (0, 0))) for y in all_Y])
        padded_W = np.array([np.pad(w, ((0, pad_to - len(w)), (0, 0))) for w in all_W])

        assert (
            padded_Y.shape
            == padded_W.shape
            == (len(self._protein_names), pad_to, len(self._channel_names))
        )

        # now we can compute the weighted linear regression for each channel

        def linreg_pair(x, y, w):
            return jnp.sum(w * x * y) / jnp.sum(w * x * x)

        def linreg_one_prot(Y, W, refchan):
            # Y and W are (NDATA, NCHAN)
            return jax.vmap(linreg_pair, in_axes=(None, 1, 1))(Y[:, refchan], Y, W)

        mat = jax.jit(jax.vmap(linreg_one_prot, in_axes=(0, 0, 0)))(
            padded_Y, padded_W, self._reference_channels
        )

        return np.clip(mat, 0, None)


def generate_function_matrix_from_spillover(spillover_matrix):
    def make_lin_fun(a):
        def lin_fun(x):
            return x * a

        return lin_fun

    return [[make_lin_fun(a) for a in row] for row in spillover_matrix]


def spillover_matrix_plot(
    spillover_matrix, channel_names, protein_names, figsize=None, ax=None, **_
):
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
