# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import scale as mscaleplo
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.ticker as ticker

# import jax
from . import utils

### {{{                    --     plot styling tools     --

CHAN_WAVELENGTHS = {
    'apc': 650,
    'apc_cy7': 780,
    'fitc': 500,
    'amcyan': 490,
    'pe': 578,
    'pe_texas_red': 615,
    'apc_alexa_700': 700,
    'pacific_blue': 452,
}


def wave2rgb(wave):
    # This is a port of javascript code from  http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 0.8
    wave = np.clip(wave, 385, 775)
    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0
    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0

    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow(c * factor, gamma)

    return f(red), f(green), f(blue)


def get_bio_color(name, default='k'):
    import difflib

    colors = {
        'ebfp': '#529edb',
        'eyfp': '#fbda73',
        'mkate': '#f75a5a',
        'mko2': '#f75a5a',
        'neongreen': '#33f397',
        'irfp720': '#97582a',
    }

    # colors['fitc'] = colors['neongreen']
    # colors['gfp'] = colors['neongreen']
    # colors['pe_texas_red'] = colors['mkate']
    # colors['mcherry'] = colors['mkate']
    # colors['pacific_blue'] = colors['ebfp']
    # colors['tagbfp'] = colors['ebfp']
    # colors['AmCyan'] = '#00a4c7'
    # colors['apc'] = '#f75a5a'
    # colors['percp_cy5_5'] = colors['mkate']

    for k, v in CHAN_WAVELENGTHS.items():
        colors[k] = wave2rgb(v)

    closest = difflib.get_close_matches(name.lower(), colors.keys(), n=1)
    if len(closest) == 0:
        color = default
    else:
        color = colors[closest[0]]
    return color


def mkfig(rows, cols, size=(7, 7), **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]), **kw)
    return fig, ax


def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_topright_spines(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_size(w, h, axes, fig):
    """w, h: width, height in inches"""
    l = min([ax.figure.subplotpars.left for ax in axes])
    r = max([ax.figure.subplotpars.right for ax in axes])
    t = max([ax.figure.subplotpars.top for ax in axes])
    b = min([ax.figure.subplotpars.bottom for ax in axes])
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


def style_violin(parts):
    for pc in parts['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
        pc.set_linewidth(1)
    parts['cbars'].set_linewidth(0)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.5)


import matplotlib.collections as mcoll
import matplotlib.path as mpath


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     spline_exp scale     --


class CustomLocator(ticker.Locator):
    def __init__(self, thresh, base):
        self.thresh = thresh
        self.base = base

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        # Linear ticks
        linear_ticks = np.array([])
        total_decades = np.log10(np.abs(vmax - vmin))
        nlin_ticks = 7
        nlin_ticks = int(max(2, nlin_ticks - total_decades))
        d = int(min(5, 1 + total_decades // 2))
        print(total_decades, nlin_ticks, d)
        if (vmin < self.thresh / d) and (vmax > -self.thresh / d):
            # symmetric around 0
            linmin = max(vmin, -self.thresh) / d
            linmax = min(vmax, self.thresh) / d
            linlim = max(np.abs(linmin), np.abs(linmax))
            linear_ticks = np.linspace(0, linlim, nlin_ticks)
            linear_ticks = np.concatenate((-linear_ticks[1:][::-1], linear_ticks))
            linear_ticks = linear_ticks[(linear_ticks >= vmin) & (linear_ticks <= vmax)]

        # Log ticks for [thresh, vmax] and [-vmin, -thresh]
        log_ticks_positive = np.array([])
        if vmax >= self.thresh:
            log_ticks_positive = ticker.LogLocator(base=self.base).tick_vaues(1, vmax)
            log_ticks_positive = log_ticks_positive[log_ticks_positive >= self.thresh]

        log_ticks_negative = np.array([])
        if vmin <= -self.thresh:
            log_ticks_negative = -ticker.LogLocator(base=self.base).tick_values(1, -vmin)
            log_ticks_negative = log_ticks_negative[log_ticks_negative <= -self.thresh]

        return np.concatenate((log_ticks_negative, linear_ticks, log_ticks_positive))


class CustomMinorLocator(ticker.Locator):
    def __init__(self, thresh, base):
        self.thresh = thresh
        self.base = base

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        log_ticks_positive = np.array([])
        if vmax > self.thresh:
            log_ticks_positive = ticker.LogLocator(
                base=self.base, numticks=10, subs=np.arange(2, 10)
            ).tick_values(
                max(vmin, self.thresh),
                vmax,
            )

        log_ticks_negative = np.array([])
        if -vmin > self.thresh:
            log_ticks_negative = -ticker.LogLocator(
                base=self.base, numticks=10, subs=np.arange(2, 10)
            ).tick_values(-min(vmax, -self.thresh), -vmin)

        return np.concatenate((log_ticks_negative, log_ticks_positive))


def format_powers(x, _):
    abs_x = abs(x)
    if abs_x < 1000:
        if x == int(x):
            return f'{x:.0f}'  # No decimal point
        else:
            return f'{x:.1f}'  # Up to 1 decimal point
    else:
        E = int(np.log10(abs_x))
        if x == int(x):
            return r'${0:.0f}e{1}$'.format(x // 10**E, E)
        else:
            return r'${0:.1f}e{1}$'.format(x / 10**E, E)


class PowerFormatter(ticker.Formatter):
    def __init__(self, values, skip10=False):
        self.values = values
        self.skip10 = skip10

    def __call__(self, x, pos):
        v = self.values[pos]
        if self.skip10 and abs(v) == 10:
            return ''
        return format_powers(v, None)


from matplotlib import scale as mscale


class SplineExponentialScale(mscale.ScaleBase):
    name = 'spline_exp'

    def __init__(self, axis, *, thresh=100, base=10, **kwargs):
        super().__init__(axis)
        if thresh <= 5:
            raise ValueError("thresh must be greater than 5.")
        self.thresh = thresh
        self.base = base

    def get_transform(self):
        return self.SplineExponentialTransform(self.thresh, self.base)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(CustomLocator(self.thresh, self.base))
        axis.set_minor_locator(CustomMinorLocator(self.thresh, self.base))
        axis.set_major_formatter(CustomFormatter(self.thresh, self.base))

    class SplineExponentialTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh, base):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.base = base

        def transform_non_affine(self, a):
            if isinstance(a, np.ma.MaskedArray):
                a = a.filled(np.nan)  # replace masked values with NaNs
            return biexponential_transform(a, self.thresh, self.base)

        def inverted(self):
            return SplineExponentialScale.InvertedSplineExponentialTransform(self.thresh, self.base)

    class InvertedSplineExponentialTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh, base):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.base = base

        def transform_non_affine(self, a):
            if isinstance(a, np.ma.MaskedArray):
                a = a.filled(np.nan)  # replace masked values with NaNs
            return inv_biexponential_transform(a, self.thresh, self.base)

        def inverted(self):
            return SplineExponentialScale.SplineExponentialTransform(self.thresh, self.base)


##────────────────────────────────────────────────────────────────────────────}}}


def fluo_scatter(
    rawx,
    pnames,
    xmin=0,
    xmax=None,
    title=None,
    types=None,
    fname=None,
    logscale=True,
    alpha=None,
    s=5,
    **_,
):
    NPOINTS = rawx.shape[0]
    if alpha is None:
        alpha = np.clip(1 / np.sqrt(NPOINTS / 50), 0.01, 1)
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)
    if len(pnames) == 1:
        axes = [axes]
    if types is None:
        types = [''] * len(pnames)

    xmin = rawx.min() if xmin is None else 10**xmin
    xmax = rawx.max() if xmax is None else 10**xmax

    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        # xcoords = jax.random.normal(jax.random.PRNGKey(0), (rawx.shape[0],)) * 0.1
        xcoords = np.random.normal(size=(rawx.shape[0],)) * 0.1
        ax.scatter(xcoords, rawx[:, xid], color=color, alpha=alpha, s=s, zorder=10, lw=0)
        if logscale:
            ax.set_yscale('symlog')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
        remove_spines(ax)
        ax.set_xticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)

    return fig, axes


def plot_fluo_distribution(ax, data, res=2000):
    from scipy.stats import gaussian_kde

    xmax = np.max(data)
    XX = np.linspace(0.0, 1.1 * xmax, res)
    kde = gaussian_kde(data.T, bw_method=0.01)
    densities = kde(XX.T)
    ldensities = np.log10(1.0 + densities)
    densities = (densities / densities.max()) * 0.4
    ldensities = (ldensities / ldensities.max()) * 0.4
    ax.fill_betweenx(XX, -ldensities, 0, color='k', alpha=0.2, lw=0)
    ax.fill_betweenx(XX, -densities, 0, color='k', alpha=0.2, lw=0)
    ax.plot(-densities, XX, color='k', alpha=0.5, lw=0.7)
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)


def scatter_matrix(
    Y,
    dim_names,
    subsample=50000,
    log=False,
    plot_size=4,
    dpi=200,
    color='k',
    s=0.1,
    alpha=0.2,
    c=None,
    xlims=None,
    ylims=None,
    **kwargs,
):
    n_channels = Y.shape[1]
    idx = np.random.choice(Y.shape[0], min(subsample, Y.shape[0]), replace=False)
    Ysub = Y[idx, :]
    if c is not None:
        c = c[idx]
    # plot each axis against each axis (with scatter plots)
    fig, axes = plt.subplots(
        n_channels, n_channels, figsize=(n_channels * plot_size, n_channels * plot_size), dpi=dpi
    )
    for row in range(n_channels):
        for col in range(n_channels):
            ax = axes[row, col]
            sc = ax.scatter(
                Ysub[:, col], Ysub[:, row], s=s, alpha=alpha, color=color, c=c, **kwargs
            )

            if row == n_channels - 1:
                ax.set_xlabel(dim_names[col])

            if col == 0:
                ax.set_ylabel(dim_names[row])

            if log:
                ax.set_xscale('symlog', linthresh=(50), linscale=0.4)
                ax.set_yscale('symlog', linthresh=(50), linscale=0.2)
            if xlims is not None:
                ax.set_xlim(xlims)
            if ylims is not None:
                ax.set_ylim(ylims)

            # ax.set_xlim(xlims)
            # ax.set_ylim(ylims)

    return fig, axes


def plot_channels_to_reference(
    controls_values,
    controls_masks,
    reference_channels,
    channel_names,
    protein_names,
    F_mat=None,
    coefs=None,
    autofluorescence=None,
    figscale=5,
    logscale=True,
    dpi=150,
    data_transform=None,
    quantiles_limits=None,
    max_n_points=50000,
    disable_reference_plot=True,
    nbins=300,
    xlims=None,
    ylims=None,
    density_min=0.01,
    density_max=None,
    noise_smooth=0.25,
    **_,
):
    NPROT = len(protein_names)
    NCHAN = len(channel_names)
    fig, axes = plt.subplots(NPROT, NCHAN, figsize=(figscale * NCHAN, figscale * NPROT), dpi=dpi)
    single_masks = (controls_masks.sum(axis=1) == 1).astype(bool)
    if autofluorescence is None:
        autofluorescence = np.zeros(NCHAN)
    for ctrl_id, ctrl_name in enumerate(protein_names):
        prot_mask = (single_masks) & (controls_masks[:, ctrl_id]).astype(bool)
        Y = controls_values[prot_mask] - autofluorescence
        if Y.shape[0] > max_n_points:
            idx = np.random.choice(Y.shape[0], max_n_points, replace=False)
            Y = Y[idx, :]
        if data_transform is not None:
            Y = data_transform(Y)
        ref_chan = reference_channels[ctrl_id]
        ref_chan_name = channel_names[ref_chan]
        for cid, chan in enumerate(channel_names):
            ax = axes[ctrl_id, cid]
            if cid == ref_chan and disable_reference_plot:
                ax.text(
                    0.5,
                    0.5,
                    f'{ctrl_name} ctrl\nreference channel {chan}',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=14,
                )
                ax.axis('off')
                continue

            x = Y[:, ref_chan]
            y = Y[:, cid]

            if xlims is None:
                xlims = [np.min(x), np.max(x)]
            if ylims is None:
                ylims = [np.min(y), np.max(y)]

            if quantiles_limits is not None:
                xlims = np.quantile(x, quantiles_limits)
                ylims = np.quantile(y, quantiles_limits)

            if logscale:
                tr, itr, xlims_tr, ylims_tr = make_symlog_ax(ax, xlims, ylims)
                x = tr(x)
                y = tr(y)
            else:
                xlims_tr, ylims_tr = xlims, ylims
                tr = lambda x: x
                itr = lambda x: x

            density_histogram2d(
                ax,
                x,
                y,
                xlims_tr,
                ylims_tr,
                nbins,
                vmin=density_min,
                vmax=density_max,
                noise_smooth=noise_smooth,
            )

            ax.axvline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            ax.axhline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')

            if F_mat is not None:
                # Fmat is a list of list of functions, of shape (NPROT, NCHAN)
                f = F_mat[ctrl_id][cid]
                x = itr(np.linspace(*tr(xlims), 500))
                y = f(x)
                ax.plot(tr(x), tr(y), c='w', lw=3)
                ax.plot(tr(x), tr(y), c='r', lw=1)

            if coefs is not None:
                x = np.linspace(*xlims, 500)
                y = x * coefs[ctrl_id][cid]
                ax.plot(tr(x), tr(y), c='w', lw=3)
                ax.plot(tr(x), tr(y), c='r', lw=1)

            ax.set_ylabel(f'{chan} (a.u.)')
            ax.set_xlabel(f'{ref_chan_name} (a.u.)')
            ax.set_title(f'{ctrl_name} ctrl')

    fig.tight_layout()
    return fig, axes


from functools import partial


def symlog(x, linthresh=50, linscale=0.4):
    # log10 with linear scale around 0
    sign = np.sign(x)
    x = np.abs(x)
    diff = np.log10(linthresh) * (1.0 - linscale)
    x = np.where(x > linthresh, np.log10(x) - diff, linscale * np.log10(linthresh) * x / linthresh)
    x = x * sign
    return x


def inverse_symlog(x, linthresh=50, linscale=0.4):
    # inverse of symlog
    sign = np.sign(x)
    x = np.abs(x)
    diff = np.log10(linthresh) * (1.0 - linscale)
    x = np.where(
        x > linscale * np.log10(linthresh),
        np.power(10, x + diff),
        linthresh * x / (linscale * np.log10(linthresh)),
    )
    x = x * sign
    return x


def powers_of_ten(xmin, xmax, skip10=False):
    # returns a list of powers of 10 that are between xmin and xmax,
    # as well as the list of subpoewrs of 10 that are between xmin and xmax (as a separate array)
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    powers = np.arange(logbounds[0], logbounds[1] + 1)
    p10 = np.power(10, np.abs(powers)) * np.sign(powers)
    # add subticks, that we will return as a separate array
    subticks = np.concatenate([p * np.arange(1, 10) for p in p10 if p != 0])
    if skip10:
        p10 = np.delete(p10, np.where(np.abs(p10) == 10))
    return p10, subticks


def make_symlog_ax(ax, xlims, ylims, linthresh=200, linscale=0.5, skip10=True, margins=0.05):
    """
    Set the axis to a custom symlog-like scale, with the given limits and margins.
    Returns the transformation function, the inverse transformation function, the transformed xlims and the transformed ylims.
    Args:
            ax (matplotlib.axis): the axis to modify
            xlims (tuple): the limits of the x axis
            ylims (tuple): the limits of the y axis
            linthresh (float, optional): the threshold at which the scale switches from linear-like (actually spline) to log. Defaults to 200.
            linscale (float, optional): the compression factor of the linear part. Defaults to 0.4.
            skip10 (bool, optional): whether to skip the 10 tick. Defaults to True.
            margins (float, optional): the margins to add to the limits. Defaults to 0.05.

    Returns:
            tuple: the transformation function, the inverse transformation function, the transformed xlims and the transformed ylims
    """

    tr = partial(utils.spline_biexponential, threshold=linthresh, compression=linscale)
    invtr = partial(utils.inverse_spline_biexponential, threshold=linthresh, compression=linscale)
    return make_tr_ax(ax, tr, invtr, xlims, ylims, skip10=skip10, margins=margins)


def make_tr_ax(ax, tr, invtr, xlims=None, ylims=None, skip10=True, margins=0.05):
    xlims_tr = xlims
    ylims_tr = ylims

    if xlims is not None:
        xlims = np.asarray(xlims)
        # Validate xlims
        if not np.all(np.isfinite(xlims)) or xlims[0] >= xlims[1]:
            xlims = np.array([-100, 1e6])  # fallback to reasonable defaults
        xlims_tr = tr(xlims)
        if np.all(np.isfinite(xlims_tr)) and xlims_tr[0] < xlims_tr[1]:
            xp10, xsub = powers_of_ten(*xlims, skip10=skip10)
            xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
            ax.set_xlim(tuple(xlims_margin))
            ax.xaxis.set_major_formatter(PowerFormatter(xp10, skip10=skip10))
            ax.set_xticks(list(tr(xp10)))
            ax.set_xticks(list(tr(xsub)), minor=True)

    if ylims is not None:
        ylims = np.asarray(ylims)
        # Validate ylims
        if not np.all(np.isfinite(ylims)) or ylims[0] >= ylims[1]:
            ylims = np.array([-100, 1e6])  # fallback to reasonable defaults
        ylims_tr = tr(ylims)
        if np.all(np.isfinite(ylims_tr)) and ylims_tr[0] < ylims_tr[1]:
            yp10, ysub = powers_of_ten(*ylims, skip10=skip10)
            ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
            ax.set_ylim(tuple(ylims_margin))
            ax.yaxis.set_major_formatter(PowerFormatter(yp10, skip10=skip10))
            ax.set_yticks(list(tr(yp10)))
            ax.set_yticks(list(tr(ysub)), minor=True)

    ax.grid(which='major', alpha=0.5, ls='--', lw=0.5)
    return tr, invtr, xlims_tr, ylims_tr


def make_density_cmap(name=None, alpha_start=1.0, alpha_end=1.0, base_cmap='Spectral_r'):
    from matplotlib.colors import LinearSegmentedColormap

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(base_cmap)(range(ncolors))
    # change alpha values
    color_array[:, -1] = np.linspace(alpha_start, alpha_end, ncolors)
    # create a colormap object
    density_cmap = LinearSegmentedColormap.from_list(name=name, colors=color_array)
    density_cmap.set_under('w', alpha=0)
    return density_cmap


CALIBRIE_DEFAULT_DENSITY_CMAP = make_density_cmap(
    'calibrie_density', alpha_start=1.0, alpha_end=1.0
)


def density_histogram2d(
    ax, X, Y, xrange, yrange, nbins, cmap=None, vmin=0.01, vmax=None, noise_smooth=0
):
    if isinstance(nbins, int):
        nbins = (nbins, nbins)

    # Filter out NaN and Inf values
    valid_mask = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_mask]
    Y = Y[valid_mask]

    # Check if we have enough valid data
    if len(X) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return

    # Validate ranges
    xrange = list(xrange)
    yrange = list(yrange)
    if not np.isfinite(xrange[0]) or not np.isfinite(xrange[1]) or xrange[0] >= xrange[1]:
        xrange = [np.nanmin(X), np.nanmax(X)]
        if xrange[0] >= xrange[1]:
            xrange[1] = xrange[0] + 1
    if not np.isfinite(yrange[0]) or not np.isfinite(yrange[1]) or yrange[0] >= yrange[1]:
        yrange = [np.nanmin(Y), np.nanmax(Y)]
        if yrange[0] >= yrange[1]:
            yrange[1] = yrange[0] + 1

    xres = np.abs(np.subtract(*xrange)) / nbins[0]
    yres = np.abs(np.subtract(*yrange)) / nbins[1]

    # w normal noise:
    X = X + np.random.normal(size=X.shape) * noise_smooth * xres
    Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=nbins,
        density=False,
        range=[xrange, yrange],
    )

    h = np.log10(h + 1).T
    if cmap is None:
        cmap = CALIBRIE_DEFAULT_DENSITY_CMAP

    ax.imshow(
        h,
        extent=[*xrange, *yrange],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )


def unmixing_plot(
    controls_abundances,
    protein_names,
    channel_names,
    xlims=[-3e6, 3e6],
    ylims=[-1e2, 3e6],
    description='',
    linthresh=200,
    autofluorescence=None,
    linscale=0.4,
    max_n_points=500000,
    std_models=None,
    std_lims=(1.5, 3),
    nbins=500,
    noise_smooth=0.25,
    density_lims=(1e-6, None),
    draw_acceptable_std_line_at=2,
    **_,
):
    # adding the 'all' to protein names:

    NPROT = len(protein_names)
    FSIZE = 5
    fig, axes = plt.subplots(NPROT, NPROT, figsize=(NPROT * FSIZE, NPROT * FSIZE))
    for ctrl_id, ctrl_name in enumerate(protein_names):
        X = controls_abundances[ctrl_name]
        if X.shape[0] > max_n_points:
            idx = np.random.choice(X.shape[0], max_n_points, replace=False)
            X = X[idx, :]

        for pid, prot_name in enumerate(protein_names):
            ax = axes[ctrl_id, pid]
            if pid == ctrl_id:
                ax.text(
                    0.5,
                    0.5,
                    f'unmixing of\n'
                    f'{prot_name} control\n'
                    f'({description}{len(channel_names)} channels, {X.shape[0]} points)',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis('off')
                continue

            if xlims is None:
                xlims = [np.min(X[:, pid]), np.max(X[:, pid])]
            if ylims is None:
                ylims = [np.min(X[:, ctrl_id]), np.max(X[:, ctrl_id])]

            tr, invtr, xlims_tr, ylims_tr = make_symlog_ax(
                ax, xlims, ylims, linthresh=linthresh, linscale=linscale
            )

            Xtr = tr(X)
            density_histogram2d(
                ax,
                Xtr[:, pid],
                Xtr[:, ctrl_id],
                xlims_tr,
                ylims_tr,
                nbins,
                noise_smooth=noise_smooth,
                vmin=density_lims[0],
                vmax=density_lims[1],
            )

            ax.axvline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            ax.axhline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            if std_models is not None:
                ybounds = np.array(np.quantile(X[:, ctrl_id], [0.01, 0.998]))
                yrange = ybounds[1] - ybounds[0]

                x = np.linspace(*utils.logtransform(ybounds), 200)
                xl = tr(utils.inv_logtransform(x))

                sparams = std_models[ctrl_id][pid]
                std = utils.evaluate_stacked_poly(x, sparams)
                stdl = utils.inv_logtransform(std)

                std_ratio = np.log10(np.maximum(yrange, 1)) / np.log10(np.maximum(stdl, 1.1))

                # we want a background that shows stdl as color
                # xlims_tr = ax.get_xlim()
                # ylims_tr = ax.get_ylim()
                mx, my = np.meshgrid(
                    np.linspace(*xlims_tr, 2),
                    np.concatenate([[ylims_tr[0]], xl, [ylims_tr[1]]]),
                )
                mz = np.ones_like(mx) * np.pad(std_ratio, (1, 1), mode='edge')[:, None]

                # Skip pcolormesh if there are NaN or Inf values
                if np.any(~np.isfinite(mx)) or np.any(~np.isfinite(my)) or np.any(~np.isfinite(mz)):
                    continue

                # put the colormesh in background
                cmap = plt.get_cmap('Reds_r')
                cmap.set_over('w', alpha=0)
                pm = ax.pcolormesh(
                    mx,
                    my,
                    mz,
                    cmap=cmap,
                    alpha=0.5,
                    vmin=std_lims[0],
                    vmax=std_lims[1],
                    shading='auto',
                    zorder=-1,
                )
                if draw_acceptable_std_line_at is not None:
                    cnt = ax.contour(
                        mx,
                        my,
                        mz,
                        levels=[draw_acceptable_std_line_at],
                        colors='k',
                        vmin=std_lims[0],
                        vmax=std_lims[1],
                        linestyles='--',
                        linewidths=1,
                        zorder=0,
                    )
                    ax.clabel(cnt, inline=True, fontsize=8, fmt='%.1f')

            ax.set_xlim(xlims_tr)
            ax.set_ylim(ylims_tr)

            ax.set_ylabel(f'{ctrl_name}')
            ax.set_xlabel(f'{prot_name}')

    fig.tight_layout()
    return fig, axes


def range_plots(
    protein_names, channel_names, absolute_ranges, relative_ranges, unsaturated, dpi=200
):
    from matplotlib.patches import Circle

    base_size = 2.0
    range_size = 0.4
    nchannels = len(channel_names)
    nproteins = len(protein_names)
    coords = np.meshgrid(np.arange(nchannels), np.arange(nproteins))
    coords2d = np.stack(coords, axis=-1).reshape(-1, 2)
    ar = (absolute_ranges / absolute_ranges.max()).flatten()
    rr = (relative_ranges / relative_ranges.max()).flatten()
    unsaturated_proportion = np.mean(unsaturated, axis=1).flatten()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(base_size * nchannels, base_size * nproteins),
        sharex=True,
        sharey=True,
        dpi=dpi,
    )
    colors = [get_bio_color(pname) for pname in protein_names for _ in range(nchannels)]

    # ax.scatter(*coords2d.T, s=(range_size*ar)**2, alpha=0.5, linewidths=1, c=colors, edgecolors='k')
    # ax.scatter(*coords2d.T, s=(range_size*rr)**2, alpha=0.5, facecolors='none', linewidths=1, edgecolors='k', ls='--')

    def draw_clipped_circle(ax, x, y, radius, color, clip_at=0.5, **kwargs):
        # circle = Circle(xy, radius)#, color=color, **kwargs)
        circle = Circle((x, y), radius, color=color, **kwargs, fill=True, edgecolor='none', lw=0)
        contour_circle = Circle((x, y), radius, color='k', fill=False, **kwargs, lw=0.5)
        # clip circle at clip_at * h))eight with a rectangle
        eps = 0.01
        rect = plt.Rectangle(
            (x - radius - eps, y - radius - eps),
            2 * (radius + eps),
            2 * (radius * clip_at + eps),
            facecolor='none',
            edgecolor='none',
        )
        ax.add_patch(rect)
        pc = ax.add_patch(circle)
        pc.set_clip_path(rect)
        ax.add_patch(contour_circle)

    for i, (x, y) in enumerate(coords2d):
        absolute_radius = range_size * np.sqrt(ar[i])
        relative_radius = range_size * np.sqrt(rr[i])
        draw_clipped_circle(ax, x, y, absolute_radius, colors[i], clip_at=unsaturated_proportion[i])
        contour_relative_circle = Circle(
            (x, y), relative_radius, color='k', fill=False, ls='--', alpha=0.5, lw=0.5
        )
        ax.add_patch(contour_relative_circle)
        ax.text(
            x, y - 0.15, f'{100.0 * ar[i]:.1f}%', ha='center', va='center', fontsize=8, color='k'
        )

        if unsaturated_proportion[i] < 0.999:
            ax.text(
                x + 0.25,
                y + 0.5,
                f'{100.0 * (1.0 - unsaturated_proportion[i]):.1f}% o.o.r',
                ha='left',
                va='center',
                fontsize=8,
                color='k',
            )
            ax.plot(
                [x, x + 0.2],
                [y + (absolute_radius * unsaturated_proportion[i]), y + 0.5],
                color='k',
                lw=0.5,
                alpha=0.5,
            )

    # remove frame
    ax.set_frame_on(False)
    # set labels
    ax.set_xticks(np.arange(nchannels))
    ax.set_xticklabels(channel_names, rotation=40)
    ax.set_yticks(np.arange(nproteins))
    ax.set_yticklabels(protein_names)
    # bigger lims
    ax.set_xlim(-0.5, nchannels - 0.5)
    ax.set_ylim(-0.5, nproteins - 0.5)
    ax.set_aspect('equal')

    # grid:
    ax.grid(True, which='both', axis='both', alpha=0.15, color='k', lw=0.5, ls='--')
    ax.set_axisbelow(True)

    # ax.set_title('Absolute ranges (filled) and std normalized ranges (dashed)')
    fig.tight_layout()

    return fig, ax, coords2d


def plot_raw_all_prot_crl(
    controls_values,
    controls_masks,
    channel_names,
    protein_names,
    autofluorescence=None,
    figscale=5,
    logscale=False,
    dpi=150,
    quantiles_limits=None,
    max_n_points=50000,
    nbins=300,
    xlims=None,
    ylims=None,
    density_min=0.01,
    density_max=None,
    noise_smooth=0.25,
    **_,
):
    NPROT = len(protein_names)
    NCHAN = len(channel_names)
    fig, axes = plt.subplots(NCHAN, NCHAN, figsize=(figscale * NCHAN, figscale * NCHAN), dpi=dpi)
    all_masks = (controls_masks.sum(axis=1) == NPROT).astype(bool)
    if autofluorescence is None:
        autofluorescence = np.zeros(NCHAN)
    Y = controls_values[all_masks] - autofluorescence
    if Y.shape[0] > max_n_points:
        idx = np.random.choice(Y.shape[0], max_n_points, replace=False)
        Y = Y[idx, :]
    for c1_id, c1_name in enumerate(channel_names):
        for c2_id, c2_name in enumerate(channel_names):
            ax = axes[c1_id, c2_id]
            x = Y[:, c1_id]
            y = Y[:, c2_id]
            if xlims is None:
                xlims = [np.min(x), np.max(x)]
            if ylims is None:
                ylims = [np.min(y), np.max(y)]
            if quantiles_limits is not None:
                xlims = np.quantile(x, quantiles_limits)
                ylims = np.quantile(y, quantiles_limits)
            if logscale:
                tr, itr, xlims_tr, ylims_tr = make_symlog_ax(ax, xlims, ylims)
                x = tr(x)
                y = tr(y)
            else:
                xlims_tr, ylims_tr = xlims, ylims
                tr = lambda x: x
                itr = lambda x: x

            density_histogram2d(
                ax,
                x,
                y,
                xlims_tr,
                ylims_tr,
                nbins,
                vmin=density_min,
                vmax=density_max,
                noise_smooth=noise_smooth,
            )

            ax.axvline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            ax.axhline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            ax.set_ylabel(f'{c2_name} (a.u.)')
            ax.set_xlabel(f'{c1_name} (a.u.)')

    fig.tight_layout()
    return fig, axes
