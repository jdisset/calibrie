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

from jax.scipy.stats import gaussian_kde

from scipy.interpolate import UnivariateSpline
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

from . import BeadsBase

class MEFBeadsCalibration(BeadsBase):
    """ Calibrate a reference protein to MEF units using beads """

    def __init__(
        self,

        model_resolution: int = 7,
        model_degree: int = 1,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.model_resolution = model_resolution
        self.model_degree = model_degree

    def initialize(
        self,
        channel_names,
        reference_channels,
        reference_protein_name,
        reference_protein_id,
        controls_abundances_mapped,
        **kwargs,
    ):

        super().initialize(channel_names=channel_names, **kwargs)

        self.calibration_channel_name = channel_names[reference_channels[reference_protein_id]]
        if self.use_channels is None:
            self.use_channels = channel_names

        if self.channel_units[self.calibration_channel_name] is None:
            raise ValueError(f'{self.calibration_channel_name} has no associated units')

        self.log.debug(
            f'Calibrating with channel {self.calibration_channel_name} and protein {reference_protein_name}'
        )

        if self.calibration_channel_name not in channel_names:
            raise ValueError('Calibration channel not found in the main channels')
        if self.calibration_channel_name not in self.use_channels:
            raise ValueError(
                'Calibration channel not found in the channels used for beads (use_channels)'
            )

        self.calibration_units_name = self.channel_units[self.calibration_channel_name]
        if self.calibration_units_name not in self.beads_mef_values:
            raise ValueError(f'No beads values for {self.calibration_units_name}')

        self.controls_abundances_MEF = self.process(controls_abundances_mapped)['abundances_MEF']

        self.calibration_channel_id = self.use_channels.index(self.calibration_channel_name)
        self.log.debug(f'Calibrated abundances will be expressed in {self.calibration_units_name}')

        return {
            'controls_abundances_MEF': self.controls_abundances_MEF,
            'calibration_units_name': self.calibration_units_name,
        }

    def process(self, abundances_mapped, **_):
        self.log.debug(f'Calibrating values to {self.calibration_units_name}')
        params = self.params[self.calibration_channel_id]
        tr_abundances = self.tr(abundances_mapped)
        tr_calibrated = vmap(utils.evaluate_stacked_poly, in_axes=(1, None))(tr_abundances, params)
        abundances_MEF = self.itr(tr_calibrated).T
        return {'abundances_MEF': abundances_MEF}


    def diagnostics(self, controls_masks, protein_names, channel_names, **kw):
        controls_abundances, std_models = utils.generate_controls_dict(
            self.controls_abundances_MEF,
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
        f.suptitle(f'Unmixing after calibration to {self.calibration_units_name}', y=1.05)
        peakfig = self.plot_bead_peaks_diagnostics()
        regfig = self.plot_regressions()
        classname = self.__class__.__name__
        return {
            f'diagnostics_{classname}': {
                'mefcontrols': f,
                'peak_id': peakfig,
                'channel_regression': regfig,
            }
        }



