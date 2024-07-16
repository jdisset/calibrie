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
from .utils import Context


class AbundanceCutoff(Task):

    cutoffs: Dict[str, float] = {}
    affects_controls: bool = False

    def initialize(self, ctx: Context):
        if self.affects_controls:
            r = self.process(
                Context(
                    protein_names=ctx.protein_names,
                    abundances_AU=ctx.controls_abundances_AU
                )
            )
            controls_abundances_AU = r.abundances_AU
            self.deleted = r.deleted
            new_controls_masks = ctx.controls_masks[~self.deleted]
            new_controls_values = ctx.controls_values[~self.deleted]
            return Context(
                controls_abundances_AU=controls_abundances_AU,
                controls_masks=new_controls_masks,
                controls_values=new_controls_values
            )
        else:
            return Context()

    def process(self, ctx: Context):
        to_delete = np.zeros(len(ctx.abundances_AU), dtype=bool)
        for pname, cutoff in self.cutoffs.items():
            if pname not in ctx.protein_names:
                self._log.warning(f'Protein {pname} not found in dataset')
                continue
            protid = ctx.protein_names.index(pname)
            new_to_delete = ctx.abundances_AU[:, protid] > cutoff
            self._log.info(
                f'Deleting {new_to_delete.sum()/len(to_delete)*100:.2f}% of events ({pname} > {cutoff})'
            )
            to_delete = to_delete | new_to_delete
        new_abundances_AU = ctx.abundances_AU[~to_delete]
        return Context(abundances_AU=new_abundances_AU, deleted=to_delete)
