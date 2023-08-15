
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde

from scipy.interpolate import UnivariateSpline
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

from . import plot
import matplotlib.pyplot as plt




def plot_beads_diagnostics(self):

    plot_bead_peaks_diagnostics(
        self.__log_beads_peaks,
        self.__beads_densities,
        self.__beads_vmat,
        self.__log_beads_data,
        self.beads_channel_order,
    )

    plot_beads_after_correction(
        self.beads_transform,
        self.__log_beads_peaks,
        self.__log_beads_mef,
        self.__log_beads_data,
        self.beads_channel_order,
        self.channel_to_unit,
        title="""Beads-based MEF calibration
        right side shows beads intensities in their respective MEF units
        left side shows the real beads data after calibration\n\n""",
    )
    plt.show()


def map_proteins_to_reference(self, unmix_method):
    if self.verbose:
        print(f'Computing all protein mapping to {self.reference_protein}...')
    all_masks = self.controls_masks.sum(axis=1) > 1
    Y = self.controls_values[all_masks] - self.autofluorescence
    selection = self.filter_saturated(Y, only_in_reference_channel=True)
    Y = Y[selection]
    X = unmix_method(Y) + self.offset
    refprotid = self.protein_names.index(self.reference_protein)
    X = X[jnp.all(X > 1, axis=1)]
    logX = utils.logtransform(X, self.log_scale_factor)
    self.cmap_transform, self.cmap_inv_transform = cmap_spline(logX, logX[:, refprotid])

def plot_mapping_diagnostics(
    self, scatter_size=20, scatter_alpha=0.2, max_n_points=15000, figsize=(10, 10)
):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plot transform
    NP = len(self.protein_names)
    all_masks = self.controls_masks.sum(axis=1) > 1
    Y = self.controls_values[all_masks] - self.autofluorescence
    X = Y @ jnp.linalg.pinv(self.spillover_matrix)  # apply bleedthrough
    X += self.offset  # add offset
    logX = logtransform(X, self.log_scale_factor, 0)
    # logX = logX[jnp.all(logX > self.clamp_values[0], axis=1)]
    # logX = logX[jnp.all(logX < self.clamp_values[1], axis=1)]
    refid = self.protein_names.index(self.reference_protein)
    target = logX[:, refid]

    scatter_x = []
    scatter_y = []
    scatter_color = []

    for i in range(NP):
        fpname = self.protein_names[i]
        color = get_bio_color(fpname)
        xrange = (logX[:, i].min(), logX[:, i].max())
        xx = np.linspace(*xrange, 200)
        yy = self.cmap_transform(np.tile(xx, (NP, 1)).T)[:, i]
        scatter_x.append(logX[:, i].flatten())
        scatter_y.append(target.flatten())
        scatter_color.append([color] * len(logX))
        ax.plot(xx, yy, alpha=1, lw=5, color='w')
        if i == refid:
            fpname += ' (reference)'
        ax.plot(xx, yy, label=fpname, alpha=1, lw=1.5, color=color, zorder=10)

    scatter_x = np.concatenate(scatter_x)
    scatter_y = np.concatenate(scatter_y)
    scatter_color = np.concatenate(scatter_color)
    NPOINTS = min(max_n_points, len(scatter_x))
    order = np.random.permutation(len(scatter_x))[:NPOINTS]

    ax.scatter(
        scatter_x[order],
        scatter_y[order],
        c=scatter_color[order],
        s=scatter_size,
        alpha=scatter_alpha,
        lw=0,
        zorder=0,
    )

    yrange = (target.min(), target.max())
    xrange = (scatter_x.min(), scatter_x.max())
    ax.set_ylim(yrange)
    ax.set_xlim(xrange)
    ax.legend()
    ax.set_xlabel('source')
    ax.set_ylabel('target')
    # no grid, grey background
    ax.grid(False)
    ax.set_title(f'Color mapping to {self.reference_protein}')
    plt.show()
