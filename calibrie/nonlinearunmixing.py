# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

from .pipeline import Task
import jax
from jax import jit, vmap
from jax.tree_util import Partial as partial
from jaxopt import GaussNewton
import jax.numpy as jnp
import numpy as np

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm

def hellonls():
    print('hello')

class NonLinearUnmixing(Task):
    """Task for the non-linear unmixing of fluorescence values."""

    def __init__(self, degree=2, logspace=True, offset=1e4, resolution=8, **kwargs):
        """
        Args:
            degree (int, optional): Degree of the stacked polynomials used for the non-linear transformation. Defaults to 2.
            logspace (bool, optional): Whether to use logspace for the non-linear transformation. Defaults to True.
            offset (float, optional): Offset for the logspace transformation. Defaults to 1e4.
            resolution (int, optional): Resolution of the non-linear transformation. Defaults to 8.

        ### REQUIRED PIPELINE DATA
        - Basic controls (provided by LoadControls)

        ### PRODUCED PIPELINE DATA
        #### On initialization:
        - `autofluorescence`: the autofluorescence vector $A$ (channels)
        - `F`: the generic callable contribution function $F$
        - `arg_mat`: a matrix of arguments (proteins x channels)
            that can be fed to `F` to compute the contribution of each protein to the output of each channel.
        - `F_mat`: a matrix of callable functions $F$ (cells x proteins)
            that compute the contribution of each protein to the output of each channel.
            It's the result of `arg_mat` being fed to `F`.
        """

        super().__init__(**kwargs)
        self.degree = degree
        self.logspace = logspace
        self.offset = offset
        self.resolution = resolution
        if logspace:
            self.transform_data = self.logtransform
            self.inv_transform_data = self.inv_logtransform
        else:
            self.transform_data = lambda x: x
            self.inv_transform_data = lambda x: x

    def logtransform(self, x, scale=1e3):
        return (jnp.log(jnp.clip(x + self.offset, 1, None)) - jnp.log(self.offset)) * scale

    def inv_logtransform(self, x, scale=1e3):
        return jnp.exp(x / scale + jnp.log(self.offset)) - self.offset

    def initialize(
        self,
        controls_values,
        controls_masks,
        protein_names,
        reference_channels,
        saturation_thresholds,
        **_,
    ):
        self.autofluorescence = utils.estimate_autofluorescence(controls_values, controls_masks)

        corrected_values = self.transform_data(controls_values - self.autofluorescence)

        corrected_saturation_thresholds = {
            'lower': saturation_thresholds['lower'] - self.autofluorescence,
            'upper': saturation_thresholds['upper'] - self.autofluorescence - 1,
        }

        low_th, up_th = (
            corrected_saturation_thresholds['lower'],
            corrected_saturation_thresholds['upper'],
        )

        self.F, self.arg_mat = fit_reference_functions(
            corrected_values,
            controls_masks,
            protein_names,
            reference_channels,
            saturation_thresholds=jax.tree_util.tree_map(
                self.transform_data, corrected_saturation_thresholds
            ),
            degree=self.degree,
            resolution=self.resolution,
        )

        pcontrib = vmap(self.F, in_axes=(0, None))  # -> (nChannels,)
        all_pcontribs = vmap(pcontrib)  # -> (nProteins, nChannels)

        @jit
        def lsq_residuals(X, Y):
            Y_pred = all_pcontribs(self.arg_mat, X).sum(axis=0)  # -> (nChannels,)
            not_saturated = (Y < up_th) & (Y > low_th)
            res = (Y_pred - Y) * not_saturated.astype(float)
            variance = jnp.log10(jnp.clip(Y + self.offset, 10, None))  # rough estimate
            res = res / variance
            return res

        @jit
        def lsq_residuals_inner_log(logX, Y):
            logY_contributions = all_pcontribs(self.arg_mat, logX)
            Y_contributions = self.inv_transform_data(logY_contributions)
            Y_pred = Y_contributions.sum(axis=0)  # -> (nChannels,)
            res = Y_pred - Y
            not_saturated = (Y < up_th) & (Y > low_th)
            res = res * not_saturated.astype(float)
            return res

        self.residual_fun = lsq_residuals_inner_log if self.logspace else lsq_residuals
        self._log.debug(f'Using {"logspace" if self.logspace else "linear"} LSQ')

        # for easier plotting purpose, we can create the matrix of functions as partials
        self.F_mat = []
        exploded_arg_mat = utils.tree_unstack(self.arg_mat)
        for arg_row in exploded_arg_mat:
            self.F_mat.append([partial(self.F, arg) for arg in utils.tree_unstack(arg_row)])

        return {
            'autofluorescence': self.autofluorescence,
            'F_mat': self.F_mat,
            'F': self.F,
            'arg_mat': self.arg_mat,
            'jaxlsq_data_transform': self.transform_data,
            'jaxlsq_data_inv_transform': self.inv_transform_data,
            'residual_fun': self.residual_fun,
        }

    def process(
        self,
        observations_raw,
        saturation_thresholds,
        reference_channels,
        max_chunk_size=10000,
        **_,
    ):

        assert self.autofluorescence is not None, "Autofluorescence not computed"
        assert self.F is not None, "Contribution function not computed"

        # remove observations for which any reference channel is saturated:
        self._log.debug(f'Starting solve for {observations_raw.shape[0]} observations')

        selected = (
            observations_raw[:, reference_channels]
            > saturation_thresholds['lower'][reference_channels]
        ) & (
            observations_raw[:, reference_channels]
            < saturation_thresholds['upper'][reference_channels]
        )
        selected = np.all(selected, axis=1)

        # observations = self.transform_data(observations_raw- self.autofluorescence)
        observations = observations_raw - self.autofluorescence

        Xshape = (observations.shape[0], len(reference_channels))
        X0 = self.transform_data(np.zeros(Xshape))

        max_chunk_size = min(max_chunk_size, observations.shape[0])
        NCHUNKS = np.ceil(X0.shape[0] / max_chunk_size).astype(int)
        needs_pad_to = max_chunk_size * NCHUNKS
        xpad = np.pad(X0, ((0, needs_pad_to - X0.shape[0]), (0, 0)))
        obspad = np.pad(observations, ((0, needs_pad_to - observations.shape[0]), (0, 0)))

        gn = GaussNewton(residual_fun=self.residual_fun)

        @jit
        def opt_run(x, obs):
            return vmap(gn.run)(x, obs).params

        trX = [
            opt_run(x, obs)
            for x, obs in tqdm(
                zip(np.split(xpad, NCHUNKS), np.split(obspad, NCHUNKS)),
                desc="JaxLSQ processing",
                total=NCHUNKS,
            )
        ]
        # trX is tuple of (params, residuals)
        trX = np.concatenate(trX, axis=0)[: observations.shape[0]]
        trX_residuals = jit(vmap(self.residual_fun))(trX, observations)

        X = self.inv_transform_data(trX)

        return {'abundances_AU': X, 'abundances_transformed': trX, 'residuals': trX_residuals}

    def diagnostics(
        self,
        controls_values,
        controls_masks,
        protein_names,
        channel_names,
        reference_channels,
        saturation_thresholds,
        **kw,
    ):
        single_masks = (controls_masks.sum(axis=1) == 1).astype(bool)
        controls_observations = {}
        controls_abundances = {}
        ret = {}
        for pid, pname in enumerate(protein_names):
            prot_mask = (single_masks) & (controls_masks[:, pid]).astype(bool)
            y = controls_values[prot_mask]
            ret_ = self.process(
                observations_raw=y,
                channel_names=channel_names,
                protein_names=protein_names,
                reference_channels=reference_channels,
                saturation_thresholds=saturation_thresholds,
            )
            x = ret_['abundances_AU']
            controls_observations[pname] = y
            controls_abundances[pname] = x
            ret_ = {f"{k}_{pname}": v for k, v in ret_.items()}
            ret_[f'controls_observations_{pname}'] = y
            ret.update(ret_)

        f, a = plots.unmixing_plot(
            controls_observations, controls_abundances, protein_names, channel_names, **kw
        )
        f.suptitle("NL LSQ unmixing results", fontsize=16, y=1.05)

        return ret



@partial(jit, static_argnames=('degree', 'max_samples', 'resolution'))
def channel_functions_stacked_poly(
    Y,  # (n_samples, n_channels)
    reference_channel_id,
    saturation_thresholds,
    degree=2,
    max_samples=40000,
    resolution=4,
    key=jax.random.PRNGKey(0),
):
    sample_ids = jax.random.choice(
        key, Y.shape[0], shape=(min(max_samples, Y.shape[0]),), replace=False
    )

    Y_s = Y[sample_ids]

    saturation_weights = (
        (Y_s > saturation_thresholds['lower']) & (Y_s < saturation_thresholds['upper'])
    ).astype(np.float32)


    endpoint = jnp.quantile(Y_s[:, reference_channel_id], 0.999)
    startpoint = Y_s[:, reference_channel_id].min()

    args = vmap(utils.fit_stacked_poly_uniform_spacing, in_axes=(None, 1, 1, None, None, None, None))(
        Y_s[:, reference_channel_id],
        Y_s,
        saturation_weights,
        startpoint,
        endpoint,
        resolution,
        degree,
    )

    return args


def fit_reference_functions(
    controls_values,
    controls_masks,
    protein_names,
    reference_channels,
    saturation_thresholds,
    degree=3,
    resolution=3,
    **_,
):
    arg_mat = []
    single_masks = np.sum(controls_masks, axis=1) == 1

    def f(params, x):
        return utils.evaluate_stacked_poly(x, params)

    for pid in range(len(protein_names)):
        prot_mask = controls_masks[:, pid].astype(bool) & single_masks
        Y = controls_values[prot_mask]
        ref_chan = reference_channels[pid]
        p_args = channel_functions_stacked_poly(
            Y, ref_chan, saturation_thresholds, degree=degree, resolution=resolution
        )
        arg_mat.append(p_args)

    return f, utils.tree_stack(arg_mat)
