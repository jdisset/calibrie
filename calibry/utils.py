from pathlib import Path
import numpy as np
import pandas as pd
import flowio
import jax.numpy as jnp
import jax
from jax.tree_util import Partial as partial
from jax import vmap, jit
from tqdm import tqdm
from scipy.interpolate import LSQUnivariateSpline, interp1d


def generate_controls_dict(
    controls_abundances,
    controls_masks,
    protein_names,
    with_std_model=True,
    std_model_n=1000,
    std_model_resolution=10,
    std_model_degree=2,
    **_,
):
    single_masks = (controls_masks.sum(axis=1) == 1).astype(bool)
    ab_dict = {}

    reg = partial(regression, resolution=std_model_resolution, degree=std_model_degree)

    std_models = [] if with_std_model else None

    for pid, pname in enumerate(protein_names):
        prot_mask = (single_masks) & (controls_masks[:, pid]).astype(bool)
        x = controls_abundances[prot_mask]
        ab_dict[pname] = x
        if with_std_model:
            xbounds = np.array((x.min(), x.max()))
            xs = x[np.random.choice(x.shape[0], std_model_n, replace=True)]
            stdparams = [
                reg(xs[:, pid], np.abs(xs[:, i]), np.ones_like(xs[:, i]), xbounds)
                for i in range(x.shape[1])
            ]
            std_models.append(stdparams)

    return ab_dict, std_models


def escape_name(name):
    return name.replace('-', '_').replace(' ', '_').upper().removesuffix('_A')


def escape(names):
    if isinstance(names, str):
        return escape_name(names)
    if isinstance(names, list):
        return [escape(name) for name in names]
    if isinstance(names, tuple):
        return tuple([escape(name) for name in names])
    if isinstance(names, dict):
        return {escape(k): escape(v) for k, v in names.items()}
    else:
        return names


def astuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def load_fcs_to_df(fcs_file):
    fcs_data = flowio.FlowData(fcs_file.as_posix())
    channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]
    original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    return pd.DataFrame(original_data, columns=escape(channels))


def load_to_df(data, column_order=None):
    # data can be either a pandas dataframe of a path to a file
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = escape(list(df.columns))
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        df = pd.DataFrame(data)
    else:
        data = Path(data)
        assert data.exists(), f'File {data} does not exist'
        if data.suffix == '.fcs':
            df = load_fcs_to_df(data)
        elif data.suffix == '.csv':
            # df = pd.read_csv(data)
            df = pd.read_csv(data, engine="pyarrow")
            df.columns = escape(list(df.columns))
        else:
            raise ValueError(f'File {data} has unknown suffix {data.suffix}')
    if column_order is not None:
        df = df[escape(column_order)]
    # reset index so that it is ordered
    df = df.reset_index(drop=True)
    return df


DEFAULT_LOG_RESCALE = 10
DEFAULT_LOG_OFFSET = 400
def logoffset(x, scale=DEFAULT_LOG_RESCALE, offset=DEFAULT_LOG_OFFSET):
    return (jnp.log(jnp.clip(x + offset, 1, None)) - jnp.log(offset)) * scale
def inv_logoffset(x, scale=DEFAULT_LOG_RESCALE, offset=DEFAULT_LOG_OFFSET):
    return jnp.exp(x / scale + jnp.log(offset)) - offset


### {{{                   --     spline_exp transform     --


def poly_forward_scaled(x, threshold, K, base=10):
    # cubic polynomial that goes through (0,0) and has same first
    # and second derivative as the log function at the threshold
    # it appears monotonically increasing for x in [0, threshold]
    # although I haven't technically proven it
    logthresh = jnp.log(threshold)
    logbase = jnp.log(base)

    a = K * (-3 + 2 * logthresh) / (2 * threshold**3 * logbase)
    b = -(-4 * K + 3 * K * logthresh) / (threshold**2 * logbase)
    c = K * (-5 + 6 * logthresh) / (2 * threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def poly_inverse(y, threshold, base=10):
    # used wolfram to solve the analytical inverse
    lT, lB, cb2 = jnp.log(threshold), jnp.log(base), jnp.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56 + y * lB * (486 - 648 * lT + 216 * lT**2) - 522 * lT + 648 * lT**2 - 216 * lT**3
    )
    B = jnp.sqrt(4 * (-19 * T2 + 12 * T2 * lT) ** 3 + A**2)
    C = jnp.cbrt(A + B)
    D = -9 + 6 * lT
    E = 2 * T * (-4 + 3 * lT) / D
    F = cb2 * (-19 * T2 + 12 * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


def symlog(x, linthresh=50, scale=0.4):
    # log10 with linear scale around 0
    sign = np.sign(x)
    x = np.abs(x)
    x = np.where(x > linthresh, np.log10(x), scale * np.log10(linthresh) * x / linthresh)
    x = x * sign
    return x


def logb(x, base=10):
    return jnp.log(x) / jnp.log(base)


@jit
def biexponential_transform(x, threshold=100, base=10, linscale=0.4):
    def positive_half(x):
        return jnp.where(
            x > threshold, logb(x, base) * linscale, poly_forward_scaled(x, threshold, linscale)
        )

    return jnp.where(x > 0, positive_half(x), -positive_half(-x))


def cubic_exp_fwd(x, threshold, base, scale=1):

    """
    cubic polynomial that goes through (0,0) and has same first
    and second derivative as the log function at the threshold
    it appears monotonically increasing for x in [0, threshold]
    although I haven't technically proven it
    """
    # assert base > 1 and scale > 0, 'Base must be > 1 and scale > 0'
    # assert (
        # 6 * logb(threshold, base) * scale > 5
    # ), 'Threshold too small for given scale (or vice versa)'

    logthresh = jnp.log(threshold)
    logbase = jnp.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def cubic_exp_inv(y, threshold, base, scale):
    """
    inverse of cubic_exp_fwd (on [0,T])
    """
    # used wolfram to solve the analytical inverse
    lT, lB, cb2 = jnp.log(threshold), jnp.log(base), jnp.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56
        + y * lB * (486 - 648 * scale * lT + 216 * scale**2 * lT**2)
        - 522 * scale * lT
        + 648 * scale**2 * lT**2
        - 216 * scale**3 * lT**3
    )
    B = jnp.sqrt(4 * (-19 * T2 + 12 * scale * T2 * lT) ** 3 + A**2)
    C = jnp.cbrt(A + B)
    D = -9 + 6 * scale * lT
    E = 2 * T * (-4 + 3 * scale * lT) / D
    F = cb2 * (-19 * T2 + 12 * scale * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


@jit
def spline_biexponential(x, threshold=100, base=10, compression=1):
    """
    biexponential function with smooth transition to cubic polynomial between [-threshold, threshold]
    """
    x = jnp.asarray(x)
    sign = jnp.sign(x)
    x = jnp.abs(x)
    diff = logb(threshold) * (1.0 - compression)
    x = jnp.where(
        x > threshold,
        logb(x, base) - diff,
        cubic_exp_fwd(x, threshold, base=base, scale=compression),
    )
    return x * sign


@jit
def inverse_spline_biexponential(y, threshold=100, base=10, compression=1):
    """
    inverse of spline_biexponential
    """
    y = jnp.asarray(y)
    sign = jnp.sign(y)
    y = jnp.abs(y)
    diff = logb(threshold) * (1.0 - compression)
    transformed_threshold = cubic_exp_fwd(threshold, threshold, base=base, scale=compression)
    y = jnp.where(
        y > transformed_threshold,
        base ** (y + diff),
        cubic_exp_inv(y, threshold, base=base, scale=compression),
    )
    return y * sign


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                 --     simple optimization functions   --


def affine_opt(Ylog, ref_channel):
    # Ylog is a matrix of log-transformed fluorescence values
    # we need to find the best affine mapping to the reference channel
    # Ylog[:, i] = a_i * Ylog[:, refchanid] + b_i
    # so we solve the linear system
    # Ylog[:, i] = A @ [a_i, b_i] (with A = [Ylog[:, refchanid], 1])

    ref = Ylog[:, ref_channel]

    def solve_column(y):
        A = jnp.stack([y, jnp.ones_like(y)], axis=1)
        return jnp.linalg.lstsq(A, ref, rcond=None)[0]

    res = vmap(solve_column)(Ylog.T)
    a, b = res[:, 0], res[:, 1]
    return a, b


def affine_gd(Y, ref_channel, num_iter=1000, learning_rate=0.1, max_N=500000, verbose=False):
    # gradient descent version of the above affine fit

    NCHAN = Y.shape[1]
    if len(Y) > max_N:
        choice = jax.random.choice(jax.random.PRNGKey(0), len(Y), (min(max_N, len(Y)),))
        Y = Y[choice]

    params = {
        'a': jnp.ones((NCHAN,)),
        'b': jnp.zeros((NCHAN,)),
        'c': jnp.zeros((NCHAN,)),
    }

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    Yref = Y[:, ref_channel]

    @jax.value_and_grad
    def lossf(params):
        yhat = params['a'] * Y + params['b']
        yhat = jnp.maximum(yhat, 0)
        err = (yhat - Yref[:, None]) ** 2
        err = err
        return jnp.mean(err**2)

    @jit
    def update(params, opt_state):
        loss, grad = lossf(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i in range(0, num_iter):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if verbose:
            print(f'iter {i}: loss = {loss:.2e}')

    a, b = params['a'], params['b']
    return a, b


DEFAULT_CMAP_Q = np.array([0.7])  # where to put spline knots
DEFAULT_CLAMP_Q = np.array([0, 1])  # where to clamp values


def cmap_spline(
    X,
    Y,
    spline_order=3,
    CF=100,
    spline_knots_at_quantile=DEFAULT_CMAP_Q,
    switch_to_linear_at_quantile=[0.02, 0.98],
):
    X, Y = X * CF, Y * CF
    x_order = np.argsort(X, axis=0)
    Xx, Yx = np.take_along_axis(X, x_order, axis=0), Y[x_order]
    w = 0.25 * np.ones_like(Y)

    xknots = [np.quantile(X[:, c], spline_knots_at_quantile) for c in range(X.shape[1])]
    print(f'xknots = {xknots}')

    x_linear_transition_lower = np.quantile(
        X, switch_to_linear_at_quantile[0], axis=0
    )  # switch to linear extrapolation at this point
    x_linear_transition_upper = np.quantile(
        X, switch_to_linear_at_quantile[1], axis=0
    )  # switch to linear extrapolation at this point

    splines = []

    for c in range(X.shape[1]):
        try:
            spline = LSQUnivariateSpline(Xx[:, c], Yx[:, c], k=spline_order, t=xknots[c], w=w)
            splines.append(spline)
        except ValueError as e:
            msg = f'Failed to fit spline for channel {c}, with X = {Xx[:, c]}, Y = {Yx[:, c]}, knots = {xknots[c]}: {e}'
            raise ValueError(msg)

    def linear_extrapolation_up(x, spline, x_end, y_end):
        slope = spline.derivative()(x_end)
        return y_end + slope * (x - x_end)

    def linear_extrapolation_down(x, spline, x_start, y_start):
        slope = max(spline.derivative()(x_start), 0)
        return y_start - slope * (x_start - x)

    def combined_spline(x, spline, x_start, y_start, x_end, y_end):
        up = np.where(x > x_end, linear_extrapolation_up(x, spline, x_end, y_end), spline(x))
        return np.where(x < x_start, linear_extrapolation_down(x, spline, x_start, y_start), up)

    y_transition_up = np.array(
        [spline(x_trans) for spline, x_trans in zip(splines, x_linear_transition_upper)]
    )

    y_transition_down = np.array(
        [spline(x_trans) for spline, x_trans in zip(splines, x_linear_transition_lower)]
    )

    transform = lambda X: np.array(
        [
            combined_spline(x * CF, s, xd, yd, xu, yu) / CF
            for s, x, xd, yd, xu, yu in zip(
                splines,
                X.T,
                x_linear_transition_lower,
                y_transition_down,
                x_linear_transition_upper,
                y_transition_up,
            )
        ]
    ).T

    inv_transform = lambda Y: None

    return transform, inv_transform


##────────────────────────────────────────────────────────────────────────────}}}


def tree_stack(trees):
    # stack the leaves of a list of pytrees into a single pytree
    # from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)
    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


### {{{                     --     stacked polynomials fit     --


def gaussian_kernel(x, mean, std, scaler=1e12):
    return scaler * jnp.exp(-((x - mean) ** 2) / (2 * std**2))


def square_kernel(x, mean, std, scaler=1e12):
    return scaler * (jnp.abs(x - mean) < std)


@partial(jit, static_argnames=('degree', 'N'))
def fit_stacked_poly_coeffs_top_n(x, y, w, mticks, stds, degree=1, N=1000):
    weights = vmap(square_kernel, in_axes=(None, 0, 0))(x, mticks, stds)
    weights = weights * w[None, :]

    def pfit(x, y, w):
        return jnp.polyfit(x, y, deg=degree, w=w)

    def get_top_n_idx(w):
        return jnp.argpartition(w, -N)[-N:]

    topn_idx = vmap(get_top_n_idx)(weights)

    def take0(x, idx):
        return x[idx]

    topn_x = vmap(take0, in_axes=(None, 0))(x, topn_idx)
    topn_y = vmap(take0, in_axes=(None, 0))(y, topn_idx)
    topn_weights = vmap(take0, in_axes=(0, 0))(weights, topn_idx)

    coeffs = vmap(pfit)(topn_x, topn_y, topn_weights)

    return coeffs


@partial(jit, static_argnames=('degree'))
def fit_stacked_poly_coeffs(x, y, w, mticks, stds, degree=1):
    # weights = vmap(square_kernel, in_axes=(None, 0, 0))(x, mticks, stds)
    weights = vmap(gaussian_kernel, in_axes=(None, 0, 0))(x, mticks, stds)
    weights = weights * w[None, :]

    def pfit(x, y, w):
        ww = jnp.where(w.sum() > 0, w, jnp.ones_like(w) * 1e-12)
        return jnp.polyfit(x, y, deg=degree, w=ww)

    coeffs = vmap(pfit, in_axes=(None, None, 0))(x, y, weights)

    return coeffs


@jit
def evaluate_stacked_poly(x, params):
    coeffs, mticks, stds = params
    evals = vmap(jnp.polyval, in_axes=(0, None))(coeffs, x)
    eval_weights = vmap(gaussian_kernel, in_axes=(None, 0, 0))(x, mticks, stds)

    # when to the left of the first tick or to the right of the last tick,
    # extrapolate the weights to the nearest tick:
    EPS = 1e-9

    eval_weights = eval_weights.at[0].set(
        jnp.where(x < mticks[0], jnp.clip(eval_weights[0], EPS, None), eval_weights[0])
    )
    eval_weights = eval_weights.at[-1].set(
        jnp.where(x > mticks[-1], jnp.clip(eval_weights[-1], EPS, None), eval_weights[-1])
    )

    return jnp.average(evals, weights=eval_weights, axis=0)


# STACKED_POLY_HEURISTIC_FIT_STD = lambda spacing: spacing**0.75
# STACKED_POLY_HEURISTIC_EVAL_STD = lambda spacing: 1.4 * spacing**0.80

STACKED_POLY_SCALING = 1e2

STACKED_POLY_HEURISTIC_FIT_STD = lambda spacing: spacing
STACKED_POLY_HEURISTIC_EVAL_STD = lambda spacing: spacing


@partial(jit, static_argnames=('resolution', 'degree', 'endpoint'))
def fit_stacked_poly_uniform_spacing(x, y, w, xmin, xmax, resolution, degree=1, endpoint=True):
    scale_ratio = STACKED_POLY_SCALING / (xmax - xmin)
    spacing = (xmax - xmin) / (resolution)
    scaled_spacing = spacing * scale_ratio
    mticks = jnp.linspace(xmin, xmax, resolution, endpoint=endpoint)
    if not endpoint:
        mticks = mticks + spacing / 2
    fit_stds = jnp.ones_like(mticks) * (
        STACKED_POLY_HEURISTIC_FIT_STD(scaled_spacing) / scale_ratio
    )
    eval_stds = jnp.ones_like(mticks) * (
        STACKED_POLY_HEURISTIC_EVAL_STD(scaled_spacing) / scale_ratio
    )
    coeffs = fit_stacked_poly_coeffs(x, y, w, mticks, fit_stds, degree=degree)
    return (coeffs, mticks, eval_stds)


@partial(jit, static_argnames=('resolution', 'degree'))
def fit_stacked_poly(x, y, w, resolution, degree=1, qmin=0.0001, qmax=0.9999):
    xmin, xmax = jnp.quantile(x, jnp.array([qmin, qmax]))
    return fit_stacked_poly_uniform_spacing(x, y, w, xmin, xmax, resolution, degree=degree)


@partial(jit, static_argnames=('degree',))
def fit_stacked_poly_at_quantiles(x, y, w, quantiles, degree=1):
    mticks = jnp.quantile(x, quantiles)
    # std is the average of the previous and next quantile
    diff = jnp.pad(jnp.diff(mticks), (1, 1), mode='edge')
    stds = (diff[:-1] + diff[1:]) / 2
    coeffs = fit_stacked_poly_coeffs(x, y, w, mticks, stds, degree=degree)
    return (coeffs, mticks, stds)


@partial(jax.jit, static_argnames=("resolution", "degree", "logspace", "endpoint"))
def regression(x, y, w, xbounds, resolution, degree, logspace=True, endpoint=True):
    """returns the stacked_poly params to best express y as a function of x"""
    if logspace:
        x, y = logtransform(x), logtransform(y)
        xbounds = logtransform(xbounds)
    # params = fit_stacked_poly_uniform_spacing(x, y, w, *xbounds, resolution, degree, endpoint)
    quantiles = jnp.linspace(0.01, 0.99, resolution, endpoint=endpoint)
    params = fit_stacked_poly_at_quantiles(x, y, w, quantiles, degree)
    return params


##────────────────────────────────────────────────────────────────────────────}}}

logtransform = partial(spline_biexponential, threshold=100, compression=0.8)
inv_logtransform = partial(inverse_spline_biexponential, threshold=100, compression=0.8)

### {{{           --     simple functions shared accross tasks     --


def estimate_autofluorescence(controls_values, controls_masks):
    zero_masks = controls_masks.sum(axis=1) == 0
    return np.median(controls_values[zero_masks], axis=0)


def estimate_saturation_thresholds(controls_values, epsilon=1e-6):
    return np.quantile(controls_values, [0.0001, 0.9999], axis=0).T + np.array([epsilon, -epsilon])


##────────────────────────────────────────────────────────────────────────────}}}
