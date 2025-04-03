import pandas as pd
from .pipeline import Task
import numpy as np
import time
from functools import partial
from . import plots, utils
from typing import List, Dict, Tuple

from .utils import Context


class PandasExport(Task):

    context_field_to_export: str = 'abundances_MEF'

    def process(self, ctx: Context):
        output = ctx[self.context_field_to_export]
        assert output.shape[1] == len(
            ctx.protein_names
        ), f"Output shape {output.shape} does not match number of proteins {len(ctx.protein_names)}"
        output_df = pd.DataFrame(output, columns=ctx.protein_names)
        return Context(output_df=output_df)


    # processed.to_csv(output_path / f'{f.stem}.csv', index=False)
