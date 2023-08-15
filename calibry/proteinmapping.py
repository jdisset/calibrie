from jaxopt import GaussNewton
from .pipeline import Task
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class ProteinMapping(Task):
    def __init__(
        self,
        reference_protein: str,
        logspace=True,
        model_resolution=10,
        model_degree=2,
        model_quantile_bounds=(0.1, 0.95),
    ):
        self.reference_protein = reference_protein.upper()
        self.logspace = logspace
        self.model_resolution = model_resolution
        self.model_degree = model_degree
        self.model_quantile_bounds = model_quantile_bounds
        if self.logspace:
            self.logt = utils.logtransform
            self.invlogt = utils.inv_logtransform
        else:
            self.logt = lambda x: x
            self.invlogt = lambda x: x

    def initialize(
        self,
        controls_abundances_AU,
        controls_masks,
        protein_names,
        **_,
    ):
        t0 = time.time()
        self.reference_protid = protein_names.index(self.reference_protein)
        self.allprt_abundances = controls_abundances_AU[np.all(controls_masks == 1, axis=1)]

        self.params = []
        ref = self.allprt_abundances[:, self.reference_protid]
        refbounds = np.quantile(ref, np.array(self.model_quantile_bounds))
        self.log.debug(
            f'Mapping {self.allprt_abundances.shape[1]-1} proteins to {self.reference_protein}'
        )

        for i in range(self.allprt_abundances.shape[1]):
            otherp = self.allprt_abundances[:, i]
            xbounds = np.quantile(otherp, np.array(self.model_quantile_bounds))
            w = np.ones_like(otherp)
            # w[otherp < xbounds[0]] = 0
            # w[otherp > xbounds[1]] = 0
            # w[ref < refbounds[0]] = 0
            # w[ref > refbounds[1]] = 0
            self.params.append(
                self.regression(
                    otherp,
                    ref,
                    w=w,
                    xbounds=xbounds,
                )
            )

        self.log.debug(f'Protein mapping took {time.time()-t0:.2f} seconds')

        self.log.debug(f'Mapping controls to {self.reference_protein}')
        self.controls_abundances_mapped = self.apply_prtmap(controls_abundances_AU, self.params)
        self.log.debug(f'Done')

        return {
            'protein_mapping_params': self.params,
            'controls_abundances_mapped': self.controls_abundances_mapped,
            'reference_protein_name': self.reference_protein,
            'reference_protein_id': self.reference_protid,
        }

    def process(self, abundances_AU, **_):
        abundances_mapped = self.apply_prtmap(abundances_AU, self.params)
        return {'abundances_mapped': abundances_mapped}

    def diagnostics(self, controls_masks, protein_names, channel_names, nbins=300, **kw):

        self.plot_mapping(protein_names, nbins=nbins, **kw)

        controls_abundances, std_models = utils.generate_controls_dict(
            self.controls_abundances_mapped,
            controls_masks,
            protein_names,
            **kw,
        )

        f, _ = plots.unmixing_plot(
            controls_abundances,
            protein_names,
            channel_names,
            **kw,
        )
        f.suptitle(f'Unmixing after protein mapping to {self.reference_protein}')

    def plot_mapping(self, protein_names, nbins=300, logspace=True, **kw):
        nprots = self.allprt_abundances.shape[1]
        fig, axes = plt.subplots(1, nprots - 1, figsize=(5 * (nprots - 1), 5))
        # scatter each original abundances and plot regression line
        refx = self.allprt_abundances[:, self.reference_protid]
        target_bounds = np.quantile(refx, np.array([0.0001, 0.9999]))
        # log_target_bounds = self.logt(target_bounds) + np.array([-10, 10])
        j = 0

        for i in range(self.allprt_abundances.shape[1]):
            if i == self.reference_protid:
                continue
            ax = axes[j]
            otherx = self.allprt_abundances[:, i]
            other_bounds = np.quantile(otherx, [0.001, 0.999])
            log_other_bounds = self.logt(other_bounds) + np.array([-10, 10])

            if logspace:
                axtr, invtr, xlims_tr, ylims_tr = plots.make_symlog_ax(ax, other_bounds, target_bounds)
            else:
                axtr = lambda x: x
                xlims_tr = other_bounds
                ylims_tr = target_bounds


            plots.density_histogram2d(ax, axtr(otherx), axtr(refx), xlims_tr, ylims_tr, nbins, noise_smooth=0.3)
            # ax.scatter(tr(self.logt(otherx)), tr(self.logt(refx)), s=1, alpha=0.1)
            ax.set_xlim(*xlims_tr)
            ax.set_ylim(*ylims_tr)

            # transform other into ref
            xfwd = invtr(np.linspace(*axtr(other_bounds), 1000))
            xprimefwd = self.apply_reg(xfwd, self.params[i])

            # # plot first a background line white
            ax.plot(axtr(xfwd), axtr(xprimefwd), lw=6, c='w', dashes=[1, 1])
            ax.plot(axtr(xfwd), axtr(xprimefwd), lw=2, c='k', dashes=[3, 3])
            ax.set_ylabel(f'{self.reference_protein}')
            ax.set_xlabel(f'{protein_names[i]}')

            j += 1
        fig.suptitle(f'Protein mapping to {self.reference_protein}')
        fig.tight_layout()

        return fig, axes

    @partial(jax.jit, static_argnums=(0,))
    def regression(self, x, y, w, xbounds):
        """returns the stacked_poly params to best express y as a function of x"""
        x, y = self.logt(x), self.logt(y)
        xbounds = self.logt(xbounds)
        params = utils.fit_stacked_poly_uniform_spacing(
            x, y, w, *xbounds, self.model_resolution, self.model_degree, endpoint=True
        )
        return params

    @partial(jit, static_argnums=(0,))
    def apply_reg(self, x, p):
        xtr = self.logt(x)
        xprime = utils.evaluate_stacked_poly(xtr, p)
        return self.invlogt(xprime)

    def apply_prtmap(self, X, params):
        return np.array([self.apply_reg(x, p) for x, p in zip(X.T, params)]).T

