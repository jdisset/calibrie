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

from .pipeline import Task
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


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

    def __init__(self, mode='WLR', weighted=True, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.weighted = weighted

    def compute_linear_spectral_signature(self, controls_values, controls_masks):
        single_masks = controls_masks.sum(axis=1) == 1
        masks = controls_masks[single_masks]
        Y = controls_values[single_masks] - self.autofluorescence
        self.spillover_matrix, _ = compute_spectral_signature_ALS(Y, masks)
        self.log.debug(f"Computed spillover matrix:\n{self.spillover_matrix}")

    def initialize(
        self,
        controls_values: np.ndarray,
        controls_masks: np.ndarray,
        protein_names,
        channel_names,
        saturation_thresholds,
        reference_channels,
        **_,
    ):
        self.autofluorescence = utils.estimate_autofluorescence(controls_values, controls_masks)
        self.saturation_thresholds = saturation_thresholds
        if self.mode == 'ALS':
            self.compute_linear_spectral_signature(controls_values, controls_masks)

        elif self.mode == 'WLR':
            Y = controls_values
            W = compute_saturation_mask(controls_values, channel_names, saturation_thresholds)
            W = W * 1.0 / (np.maximum(Y, 100))
            self.spillover_matrix = compute_spectral_signature_WLR(
                Y - self.autofluorescence,
                controls_masks,
                W,
                protein_names,
                channel_names,
                reference_channels,
                logger=self.log,
            )
            self.spillover_matrix = np.clip(self.spillover_matrix, 0, None)

        elif self.mode == 'WLR2':

            from sklearn.decomposition import PCA

            # first let's isolate all the available single controls
            one_hot = lambda x, n: np.eye(n)[x]
            single_controls = one_hot(np.arange(len(protein_names)), len(protein_names)).astype(
                bool
            )
            counts = np.array(
                [np.all(controls_masks == sc, axis=1).sum() for sc in single_controls]
            )
            # randomly reorder Y:
            NC = len(channel_names)
            pca = PCA(n_components=NC)
            pca.fit(controls_values)
            K = pca.components_
            Y = np.dot(controls_values, K.T)
            W = np.ones_like(Y)
            self.spillover_matrix = compute_spectral_signature_WLR(
                Y,
                controls_masks,
                W,
                protein_names,
                channel_names,
                reference_channels,
                logger=self.log,
            )
            # self.spillover_matrix = np.clip(self.spillover_matrix, 0, None)

        else:
            raise ValueError(f"Unknown solver mode {mode}")

        return {
            'autofluorescence': self.autofluorescence,
            'spillover_matrix': self.spillover_matrix,
            'controls_values': Y,
            'controls_masks': controls_masks,
            'F_mat': generate_function_matrix_from_spillover(self.spillover_matrix),
        }

    def process(self, observations_raw: np.ndarray, channel_names: List, **_):
        self.log.debug(f"Linear unmixing of {observations_raw.shape[0]} observations")
        self.log.debug(f"spillover_matrix shape: {self.spillover_matrix.shape}")
        assert observations_raw.shape[1] == len(
            channel_names
        ), "Mismatch between observations and channels"

        if self.weighted:
            W = compute_saturation_mask(observations_raw, channel_names, self.saturation_thresholds)
            Y = observations_raw - self.autofluorescence
            W = W * 1.0 / (np.maximum(Y, 100))

            def solve_one(y, w):
                y_weighted = y * w

                # TODO: weight by average error to linreg
                # compute avg error at y for each row of S
                # maybe fit a function that estimates the error and use it as weight
                # WM =

                S_weighted = self.spillover_matrix * w[None, :]

                x, residuals, rank, s = jnp.linalg.lstsq(S_weighted.T, y_weighted, rcond=None)
                return x

            solve = jit(jax.vmap(solve_one))
            X = solve(Y, W)

        else:
            Y = observations_raw - self.autofluorescence
            X = Y @ jnp.linalg.pinv(self.spillover_matrix)  # apply bleedthrough

        self.log.debug(f"Linear unmixing done")
        return {'abundances_AU': X}

    def diagnostics(
        self,
        controls_values: np.ndarray,
        controls_masks: np.ndarray,
        protein_names: List,
        channel_names: List,
        **kw,
    ):

        assert self.spillover_matrix is not None, "Spillover matrix not computed"
        assert self.autofluorescence is not None, "Autofluorescence not computed"

        single_masks = (controls_masks.sum(axis=1) == 1).astype(bool)
        controls_observations = {}
        controls_abundances = {}
        for pid, pname in enumerate(protein_names):
            self.log.debug(f"unmixing {pname}")
            prot_mask = (single_masks) & (controls_masks[:, pid]).astype(bool)
            y = controls_values[prot_mask]
            x = self.process(y, channel_names)['abundances_AU']
            controls_observations[pname] = y
            controls_abundances[pname] = x

        f, a = spillover_matrix_plot(self.spillover_matrix, channel_names, protein_names)
        f.suptitle("Spillover matrix", fontsize=16, y=1.05)

        f, a = plots.unmixing_plot(
            controls_observations,
            controls_abundances,
            protein_names,
            channel_names,
            decription='linear model, ',
            **kw,
        )
        f.suptitle("Linear unmixing results", fontsize=16, y=1.05)


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


def compute_spectral_signature_ALS(Y, M, max_iterations=50, jax_seed=0, verbose=False, w=None):
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

    normalize = lambda x: x / (jnp.maximum(jnp.max(x, axis=1)[:, None], 1e-12))
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

    return S, X


def compute_saturation_mask(Y, channel_names, saturation_thresholds):
    # compute a mask for saturated channels
    assert Y.shape[1] == len(channel_names)
    assert np.all(saturation_thresholds['lower'] <= saturation_thresholds['upper'])
    assert saturation_thresholds['lower'].shape == (len(channel_names),)
    assert saturation_thresholds['upper'].shape == (len(channel_names),)

    return (Y > saturation_thresholds['lower']) & (Y < saturation_thresholds['upper'])


def compute_spectral_signature_WLR(
    Y,
    M,
    W,
    protein_names,
    channel_names,
    reference_channels,
    min_points=100,
    max_points=100000,
    logger=None,
):

    # Spectral signature estimation using Weighted Linear Regression (WLR) method.

    # Main Args:
    # Y (array): Observed fluorescence intensities (cells x channels).
    # M (array): Masks matrix from single-color control samples (proteins x channels).

    # Returns:
    # S (array): Spectral signature matrix (proteins x channels).

    reference_channels = np.array(reference_channels)

    assert Y.shape == (M.shape[0], len(channel_names))
    assert M.shape[1] == len(protein_names)
    assert W.shape == Y.shape

    # let's reorder Y randomly:
    order = np.random.permutation(Y.shape[0])
    Yp = Y[order]
    Mp = M[order]
    Wp = W[order]

    # first let's isolate all the available single controls
    one_hot = lambda x, n: np.eye(n)[x]
    single_controls = one_hot(np.arange(len(protein_names)), len(protein_names)).astype(bool)
    counts = np.array([np.all(Mp == sc, axis=1).sum() for sc in single_controls])

    if logger is not None:
        logger.debug(f"Single protein controls counts: {counts}")

    for p, c in zip(protein_names, counts):
        assert c > min_points, f"Not enough data for single control of {p}"

    largest_count = counts.max()
    pad_to = min(largest_count, max_points)

    all_Y = [Yp[np.all(Mp == sc, axis=1)][:pad_to] for sc in single_controls]
    all_W = [Wp[np.all(Mp == sc, axis=1)][:pad_to] for sc in single_controls]

    # (NPROT, NDATA, NCHAN):
    padded_Y = np.array([np.pad(y, ((0, pad_to - len(y)), (0, 0))) for y in all_Y])
    padded_W = np.array([np.pad(w, ((0, pad_to - len(w)), (0, 0))) for w in all_W])

    assert (
        padded_Y.shape == padded_W.shape == (len(protein_names), largest_count, len(channel_names))
    )

    # now we can compute the weighted linear regression for each channel

    def linreg_pair(x, y, w):
        return jnp.sum(w * x * y) / jnp.sum(w * x * x)

    def linreg_one_prot(Y, W, refchan):
        # Y and W are (NDATA, NCHAN)
        return jax.vmap(linreg_pair, in_axes=(None, 1, 1))(Y[:, refchan], Y, W)

    mat = jax.jit(jax.vmap(linreg_one_prot, in_axes=(0, 0, 0)))(
        padded_Y, padded_W, reference_channels
    )

    return mat
