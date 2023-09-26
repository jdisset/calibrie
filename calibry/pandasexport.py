import pandas as pd
from .pipeline import Task
import numpy as np
import time
from functools import partial
from . import plots, utils
from typing import List, Dict, Tuple


class PandasExport(Task):
    def __init__(
        self,
        what,
        **_,
    ):
        self.what = what

    def initialize( self, **_,):
        return {}


    def process(self, protein_names, **kw):
        assert self.what in kw
        output = kw[self.what]
        assert output.shape[1] == len(protein_names)
        return {'output_df':pd.DataFrame(output, columns=protein_names)}


    def diagnostics(self):
        pass

