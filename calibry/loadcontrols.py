"""
This module provides the LoadControls task, which loads single, multi and no-protein controls from a list of files.
"""

from .pipeline import Task
from typing import List, Dict, Tuple
from . import utils as ut
from calibry.utils import LoadedData
import numpy as np
import jax.numpy as jnp
from . import plots, utils

_VALID_BLANK_NAMES = ['BLANK', 'EMPTY', 'INERT', 'CNTL']
_VALID_ALL_NAMES = ['ALL', 'ALLCOLOR', 'ALLCOLORS', 'FULL']

EPSILON = 1e-6


class LoadControls(Task):
    """
    Task for loading single, multi and no-protein controls from a list of files.

    ### REQUIRED PIPELINE DATA
    - None

    ### PRODUCED PIPELINE DATA
    - `controls_values`: an array of control values (total events x channels)
    - `controls_masks`: an array of masks telling which proteins were present, for each event (total events x proteins)
    - `channel_names`: the ordered list of channel names
    - `protein_names`: the ordered list of protein names
    - `saturation_thresholds`: {'upper': (channels), 'lower': (channels)} the upper and lower saturation thresholds for each channel
    - `reference_channels`: an array of integer indices of the reference channels for each protein (proteins)

    Load controls from a list of files.

    Args:
        color_controls: a dictionary mapping the names of the color controls to their filenames.
            Required names are:
            - `$pname$`: a single protein control (e.g. `mKate`)
            - `[BLANK, EMPTY, INERT, CNTL]`: a blank control (e.g. `BLANK`)
            - `[ALL, ALLCOLOR, ALLCOLORS, FULL]`: a multi-protein control (e.g. `ALL`)

        use_channels: a list of channels to keep when the data is processed by other tasks such as compensation.
            If None, all channels are kept.
        reference_channels: a dictionary mapping the names of the proteins to the names of the channels to use as reference
            for compensation. If None, the reference channel is chosen as the one with the highest dynamic range (and no saturation).
    """

    color_controls: Dict[str | Tuple[str], LoadedData]
    use_channels: List[str] = []
    use_reference_channels: Dict[str, str] = {} # protein -> channel

    def pick_reference_channels(self, max_sat_proportion=0.001, min_dyn_range=3, **_):
        # for each protein ctrl, we'll pick a reference channel as the one with the highest dynamic range
        if self.use_reference_channels is not None:
            user_provided = []
            for pid, p in enumerate(self._protein_names):
                if p in self.use_reference_channels.keys():
                    ch = self.use_reference_channels[p]
                    assert ch in self._channel_names, f'Unknown channel {ch} in reference_channels'
                    user_provided.append(self._channel_names.index(ch))
                else:
                    user_provided.append(None)
            unknown_provided = [
                p for p in self.use_reference_channels if p not in self._protein_names
            ]
            if len(unknown_provided) > 0:
                self._log.warning(
                    f'Unknown proteins {unknown_provided} in user_provided_reference_channels. Ignoring them.'
                )
        else:
            user_provided = [None] * len(self._protein_names)

        single_masks = (self._controls_masks.sum(axis=1) == 1).astype(bool)
        ref_channels = []
        for pid, prot in enumerate(self._protein_names):
            if user_provided[pid] is not None:
                ref_channels.append(user_provided[pid])
                continue
            prot_mask = (single_masks) & (self._controls_masks[:, pid]).astype(bool)
            prot_values = self._controls_values[prot_mask]

            prot_sat = (prot_values <= self._saturarion_thresholds[:, 0]) | (
                prot_values >= self._saturarion_thresholds[:, 1]
            )

            sat_proportion = prot_sat.sum(axis=0) / prot_sat.shape[0]
            dyn_range = np.log10(np.quantile(prot_values, [0.001, 0.999], axis=0).ptp(axis=0))
            effective_range = dyn_range.copy()
            effective_range[sat_proportion > max_sat_proportion] = 0
            effective_range[effective_range < min_dyn_range] = 0
            if effective_range.sum() == 0:
                ref_channels.append(None)
                raise ValueError(
                    f"""
                    No reference channel found for {prot}!
                    Try increasing max_sat_proportion and/or decreasing min_dyn_range'
                    Saturation proportions per channel: {sat_proportion}
                    Log10 dynamic range per channel: {dyn_range}
                    """
                )
            else:
                ref_channels.append(effective_range.argmax())
            self._log.debug(
                f"Reference channel for {prot} is {self._channel_names[ref_channels[-1]]}"
            )
        return ref_channels


    def process(self, ctx):
        loader = ctx.get('cell_data_loader', ut.load_to_df)

        data = loader(ctx.pipeline_input, self.use_channels).values
        assert data.shape[1] == len(self._channel_names)
        return Context(observations_raw=data)


    def initialize(self, ctx):
        loader = ctx.get('cell_data_loader', ut.load_to_df)

        self._controls = {
            ut.astuple(ut.escape(k)): loader(v, self.use_channels)
            for k, v in self.color_controls.items()
        }

        self._channel_names = list(list(self._controls.values())[0].columns)

        assert len(self._channel_names) > 0, "No channels found in the provided controls"
        assert all(
            [len(c.columns) == len(self._channel_names) for c in self._controls.values()]
        ), "All controls must have the same number of channels"

        self._log.debug(f"Loaded {len(self._controls)} controls: {list(self._controls.keys())}")
        self._log.debug(
            f"Using {len(self._channel_names)} channels: {self._channel_names[:8]}{f'+{len(self._channel_names)-8}' if len(self._channel_names)>8 else ''}"
        )

        controls_order = sorted(list(self._controls.keys()))

        self._protein_names = sorted(
            list(
                set(
                    [
                        p
                        for c in controls_order
                        for p in c
                        if p not in _VALID_BLANK_NAMES + _VALID_ALL_NAMES
                    ]
                )
            )
        )
        self._log.debug(f"For {len(self._protein_names)} proteins: {self._protein_names}")

        controls_values, controls_masks = [], []
        NPROTEINS = len(self._protein_names)
        has_blank, has_all = False, False
        for m in controls_order:  # m is a tuple of protein names
            vals = self._controls[m].values
            if m in [ut.astuple(ut.escape(n)) for n in _VALID_BLANK_NAMES]:
                base_mask = np.zeros((NPROTEINS,))
                has_blank = True
            elif m in [ut.astuple(ut.escape(n)) for n in _VALID_ALL_NAMES]:
                base_mask = np.ones((NPROTEINS,))
                has_all = True
            else:
                for p in m:
                    assert p in self._protein_names, f'Unknown protein {p}'
                base_mask = np.array([p in m for p in self._protein_names])

            masks = np.tile(base_mask, (vals.shape[0], 1))
            controls_values.append(vals)
            controls_masks.append(masks)

        assert has_blank, f'BLANK control not found. Should be named any of {_VALID_BLANK_NAMES}'
        assert has_all, f'ALLCOLOR control not found. Should be named any of {_VALID_ALL_NAMES}'

        self._controls_values = np.vstack(controls_values)
        self._controls_masks = np.vstack(controls_masks)

        assert (
            self._controls_values.shape[0] == self._controls_masks.shape[0]
        ), "Mismatch between values and masks"
        assert (
            len(self._protein_names) == self._controls_masks.shape[1]
        ), "Mismatch between masks and protein names"
        assert self._controls_values.shape[1] == len(
            self._channel_names
        ), "Mismatch between values and channel names"

        self._log.debug(f"Loaded {self._controls_values.shape[0]} events")

        self._saturarion_thresholds = utils.estimate_saturation_thresholds(self._controls_values)

        self._log.debug(f"Saturation thresholds: {self._saturarion_thresholds}")

        self._reference_channels = self.pick_reference_channels()
        self._log.debug(f"Reference channels: {self._reference_channels}")

        self._log.debug(
            f"Using {len(self._channel_names)} channels: {self._channel_names[:8]}{f'+{len(self._channel_names)-8}' if len(self._channel_names)>8 else ''}"
        )

        return {
            'controls_values': self._controls_values,
            'controls_masks': self._controls_masks,
            'protein_names': self._protein_names,
            'channel_names': self._channel_names,
            'saturation_thresholds': self._saturarion_thresholds,
            'reference_channels': self._reference_channels,
        }
