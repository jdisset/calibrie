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


class AbundanceCutoff(Task):
    def __init__(
        self,
        cutoffs: Dict[str, float],
        **_,
    ):
        self.cutoffs = cutoffs

    def initialize(
        self,
        protein_names,
        controls_abundances_AU,
        controls_masks,
        controls_values,
        **_,
    ):
        r = self.process(protein_names, controls_abundances_AU)
        controls_abundances_AU = r['abundances_AU']
        self.deleted = r['deleted']
        new_controls_masks = controls_masks[~self.deleted]
        new_controls_values = controls_values[~self.deleted]
        return {'controls_abundances_AU': controls_abundances_AU, 'controls_masks': new_controls_masks, 'controls_values': new_controls_values}


    def process(self, protein_names, abundances_AU):
        to_delete = np.zeros(len(abundances_AU), dtype=bool)
        for pname, cutoff in self.cutoffs.items():
            if pname not in protein_names:
                self.log.warning(f'Protein {pname} not found in dataset')
                continue
            protid = protein_names.index(pname)
            new_to_delete = abundances_AU[:, protid] > cutoff
            self.log.info(f'Deleting {new_to_delete.sum()/len(to_delete)*100:.2f}% of events ({pname} > {cutoff})')
            to_delete = to_delete | new_to_delete
        new_abundances_AU = abundances_AU[~to_delete]
        return {'abundances_AU': new_abundances_AU, 'deleted': to_delete}


    def diagnostics(self):
        pass

