from jaxopt import GaussNewton
import jax
from .utils import Context
from .pipeline import Task, DiagnosticFigure
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import time
import lineax as lx
from functools import partial

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable


class ProteinMapping(Task):

    reference_protein: str
    logspace: bool = True
    model_resolution: int = 10
    model_degree: int = 2
    model_quantile_bounds: Tuple[float, float] = (0.05, 0.999)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.reference_protein = self.reference_protein.upper()
        if self.logspace:
            self._logt = utils.logtransform_jax
            self._invlogt = utils.inv_logtransform_jax
        else:
            self._logt = lambda x: x
            self._invlogt = lambda x: x

    def initialize(self, ctx: Context):
        t0 = time.time()
        self._reference_protid = ctx.protein_names.index(self.reference_protein)
        self._allprt_abundances = ctx.controls_abundances_AU[
            np.all(ctx.controls_masks == 1, axis=1)
        ]

        ref = self._allprt_abundances[:, self._reference_protid]
        refbounds = np.quantile(ref, np.array(self.model_quantile_bounds))
        self._log.debug(
            f'Mapping {self._allprt_abundances.shape[1] - 1} proteins to {self.reference_protein}'
        )

        self._regression_results = []
        for i in range(self._allprt_abundances.shape[1]):
            otherp = self._allprt_abundances[:, i]
            xbounds = np.quantile(otherp, np.array(self.model_quantile_bounds))
            w = np.ones_like(otherp)
            res = self._regression(
                otherp,
                ref,
                w=w,
                xbounds=xbounds,
            )
            self._regression_results.append(res)

        self._log.debug(f'Protein mapping took {time.time() - t0:.2f} seconds')

        self._log.debug(f'Mapping controls to {self.reference_protein}')
        self._controls_abundances_mapped = self._apply_prtmap(
            ctx.controls_abundances_AU, self._regression_results
        )
        self._log.debug(f'Done')

        return Context(
            controls_abundances_mapped=self._controls_abundances_mapped,
            reference_protein_name=self.reference_protein,
            reference_protein_id=self._reference_protid,
        )

    def process(self, ctx: Context):
        abundances_mapped = self._apply_prtmap(ctx.abundances_AU, self._regression_results)
        return Context(abundances_mapped=abundances_mapped)

    def diagnostics(self, ctx: Context, **kw):

        # update with self.diagnostic_kwargs

        nbins = kw.pop('nbins', 300)

        fmap, _ = self.plot_mapping(ctx.protein_names, nbins=nbins, **kw)

        controls_abundances, std_models = utils.generate_controls_dict(
            self._controls_abundances_mapped,
            ctx.controls_masks,
            ctx.protein_names,
            **kw,
        )

        f, _ = plots.unmixing_plot(
            controls_abundances,
            ctx.protein_names,
            ctx.channel_names,
            **kw,
        )
        f.suptitle(f'Unmixing after protein mapping to {self.reference_protein}')

        classname = self.__class__.__name__
        return [
            DiagnosticFigure(fmap, f'mapping to ref protein'),
            DiagnosticFigure(f, f'unmixing'),
        ]


    def plot_mapping(
        self, protein_names, nbins=300, logspace=True, target_bounds=None, other_bounds=None, **kw
    ):
        nprots = self._allprt_abundances.shape[1]
        fig, axes = plt.subplots(1, nprots - 1, figsize=(5 * (nprots - 1), 5))
        if nprots == 2:
            axes = [axes]
        # scatter each original abundances and plot regression line
        refx = self._allprt_abundances[:, self._reference_protid]
        if target_bounds is None:
            target_bounds = np.quantile(refx, np.array([0.0001, 1]))
        target_bounds = np.asarray(target_bounds)
        # log_target_bounds = self._logt(target_bounds) + np.array([-10, 10])
        j = 0

        for i in range(self._allprt_abundances.shape[1]):
            if i == self._reference_protid:
                continue
            ax = axes[j]
            otherx = self._allprt_abundances[:, i]
            if other_bounds is None:
                other_bounds = np.quantile(otherx, [0.001, 1])
            other_bounds = np.asarray(other_bounds)
            log_other_bounds = self._logt(other_bounds) + np.array([-10, 10])

            if logspace:
                axtr, invtr, xlims_tr, ylims_tr = plots.make_symlog_ax(
                    ax, other_bounds, target_bounds
                )
            else:
                axtr = lambda x: x
                invtr = lambda x: x
                xlims_tr = other_bounds
                ylims_tr = target_bounds

            plots.density_histogram2d(
                ax, axtr(otherx), axtr(refx), xlims_tr, ylims_tr, nbins, noise_smooth=0.3
            )
            # ax.scatter(tr(self._logt(otherx)), tr(self._logt(refx)), s=1, alpha=0.1)
            ax.set_xlim(*xlims_tr)
            ax.set_ylim(*ylims_tr)

            # transform other into ref
            xfwd = invtr(np.linspace(*axtr(other_bounds), 1000))
            xprimefwd = self._apply_reg(xfwd, self._regression_results[i])

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
    def _regression(self, x, y, w, xbounds):
        """returns the stacked_poly params to best express y as a function of x"""
        x, y = self._logt(x), self._logt(y)
        xbounds = self._logt(xbounds)
        params = utils.fit_stacked_poly_uniform_spacing(
            x, y, w, *xbounds, self.model_resolution, self.model_degree, endpoint=True
        )
        return params

    @partial(jit, static_argnums=(0,))
    def _apply_reg(self, x, p):
        xtr = self._logt(x)
        xprime = utils.evaluate_stacked_poly(xtr, p)
        return self._invlogt(xprime)

    def _apply_prtmap(self, X, params):
        return np.array([self._apply_reg(x, p) for x, p in zip(X.T, params)]).T
