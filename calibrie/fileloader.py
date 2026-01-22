# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

import pandas as pd
from .pipeline import Task
import numpy as np
import time
from functools import partial
from . import plots, utils
from typing import List, Dict, Tuple

from .utils import Context


class FileLoader(Task):

    data_folder: str
    calibration_files: Dict[str, str]
    bead_file: str


    def process(self):

        # glob all files in the data folder
        import glob

        self.data_folder = Path(self.data_folder).expanduser().resolve()

        all_fcs_files = glob.glob(
            str(self.data_folder / '*.fcs')




        samples = self._xpdata['samples']
        control_samples = [s for s in samples if s['control']]

        # load beads
        beads_file = self._datadir / self._xpdata['beads_file']
        assert beads_file.exists(), f'File {beads_file} not found for beads'

        # load color controls
        self._control_files = {
            s['name']: (self._datadir / s['file']).as_posix() for s in control_samples
        }
        for name, path in self._control_files.items():
            assert Path(path).exists(), f'File {path} not found for control {name}'

        ctx = {}
        ctx['$BEAD_FILE'] = beads_file.as_posix()
        ctx['$CONTROL_FILES'] = self._control_files
        ctx['$XP_FILE'] = self.xpfile
        ctx['$XP_DIRNAME'] = Path(self.xpfile).parent.name
        ctx['$XP_DATADIR'] = self._datadir.as_posix()

        self._resolved_pipeline = self.pipeline.construct(context=ctx)
        resolve_all_lazy(self._resolved_pipeline)
        if not isinstance(self._resolved_pipeline, cal.Pipeline):
            self._resolved_pipeline = cal.Pipeline(**self._resolved_pipeline)

        ctx['$PIPELINE_NAME'] = self._resolved_pipeline.get_namehash()

        self._outputdir = self.outputdir.construct(context=ctx).resolve()
        self._outputdir = Path(self._outputdir).expanduser().resolve()

        self._outputdir.mkdir(parents=True, exist_ok=True)

        self._resolved_pipeline.loglevel = self.loglevel
        self._resolved_pipeline.setup_logging()

        return self._resolved_pipeline


    def process(self, ctx: Context):
        output = ctx[self.file]
        assert output.shape[1] == len(
            ctx.protein_names
        ), f"Output shape {output.shape} does not match number of proteins {len(ctx.protein_names)}"
        output_df = pd.DataFrame(output, columns=ctx.protein_names)
        return Context(output_df=output_df)


    # processed.to_csv(output_path / f'{f.stem}.csv', index=False)
