import dracon as dr
import json
from dracon.utils import with_indent
from dracon.deferred import DeferredNode
from dracon.lazy import LazyDraconModel, resolve_all_lazy
import pandas as pd
import time
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import dracon
import calibrie as cal
from pathlib import Path
from typing import List, Tuple, Union, Annotated, Dict, Any, Optional, Literal
from pydantic import BaseModel
import sys
import json5
from calibrie.pipeline import DiagnosticFigure
from calibrie import (
    GatingTask,
    PolygonGate,
    LoadControls,
    LinearCompensation,
    ProteinMapping,
    MEFBeadsCalibration,
    MEFBeadsTransform,
    PandasExport,
    Colinearization,
    AbundanceCutoff,
    Pipeline,
)
import dracon as dr
import matplotlib
import logging

log = logging.getLogger(__name__)

matplotlib.use('Agg')


def parse_xpfile(xpfile: str):
    filepath = Path(xpfile).expanduser().resolve()

    with open(filepath, 'r') as f:
        data = json5.load(f)

    return data


def run_and_save_diagnostics(pipeline, outputdir):
    figs: List[DiagnosticFigure] = pipeline.all_diagnostics()
    outputdir = Path(outputdir).expanduser().resolve()
    outputdir.mkdir(parents=True, exist_ok=True)

    for fig in figs:
        try:
            fname = f'{fig.source_task}_{fig.name}'.lower().replace(' ', '_')
            fig.fig.savefig(outputdir / f'{fname}.pdf', dpi=300, bbox_inches='tight')
        except Exception as e:
            log.error(f'Error saving figure {fig.name}: {e}. Skipping.')
            import traceback

            traceback.print_exc()


class CalibrationProgram(LazyDraconModel):
    pipeline: Annotated[
        DeferredNode[cal.Pipeline], Arg(help='The pipeline to execute', is_file=True)
    ]

    xpfile: Annotated[str, Arg(help='Input experiment file to load')] = './experiment.json5'

    outputdir: Annotated[DeferredNode[str], Arg(help='Output directory')] = (
        '${"$XP_DATADIR/../calibrated/$PIPELINE_NAME"}'
    )

    datapath: Annotated[
        str, Arg(help='The path to the directory containing the data files, relative to the xpfile')
    ] = './data/raw_data/'

    export_format: Annotated[Literal['csv', 'parquet'], Arg(help='Output file format')] = 'parquet'

    diagnostics: Annotated[bool, Arg(help='Whether to generate diagnostic figures')] = True
    diagnostics_output_dir: Annotated[
        Optional[str], Arg(help='The directory to save diagnostic figures to')
    ] = None
    diagnostics_only: Annotated[bool, Arg(help='Just run diagnostics')] = False

    def build_pipeline(self):
        self._datadir = (Path(self.xpfile).parent / self.datapath).expanduser().resolve()
        self._xpdata = parse_xpfile(self.xpfile)

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

        self._context = {}
        self._context['$BEAD_FILE'] = beads_file.as_posix()
        self._context['$CONTROL_FILES'] = self._control_files
        self._context['$XP_FILE'] = self.xpfile
        self._context['$XP_DIRNAME'] = Path(self.xpfile).parent.name
        self._context['$XP_DATADIR'] = self._datadir.as_posix()

        self._resolved_pipeline = self.pipeline.construct(context=self._context)
        resolve_all_lazy(self._resolved_pipeline)
        if not isinstance(self._resolved_pipeline, cal.Pipeline):
            self._resolved_pipeline = cal.Pipeline(**self._resolved_pipeline)

        self._context['$PIPELINE_NAME'] = self._resolved_pipeline.get_namehash()

        self._outputdir = self.outputdir.construct(context=self._context).resolve()
        self._outputdir = Path(self._outputdir).expanduser().resolve()

        self._outputdir.mkdir(parents=True, exist_ok=True)

        return self._resolved_pipeline

    def calibrate_file(self, sample, file_format='parquet'):
        data_path = (self._datadir / sample['file']).as_posix()
        ctx_out = self._resolved_pipeline.apply_all(data_path)
        data: pd.DataFrame = ctx_out.output_df
        assert isinstance(data, pd.DataFrame), f'Expected DataFrame, got {type(data)}'

        pipeline_dict = self._resolved_pipeline.model_dump()
        xp_dict = {k: v for k, v in self._xpdata.items() if k != 'samples'}
        sample_dict = sample

        data.attrs['calibration'] = {
            'pipeline': pipeline_dict,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_args': sys.argv[1:],
            'namehash': self._resolved_pipeline.get_namehash(),
        }
        data.attrs['xp'] = xp_dict
        data.attrs['sample'] = sample_dict
        self._outputdir.mkdir(parents=True, exist_ok=True)
        fname = f'{sample["name"]}.{file_format}'
        output_path = self._outputdir / fname
        print(f'Saving calibrated data to {output_path}')

        if file_format == 'csv':
            data.to_csv(output_path, index=False)
        elif file_format == 'parquet':
            data.to_parquet(output_path, compression='zstd')
        else:
            raise ValueError(f'Unsupported file format: {file_format}')

    def run(self):
        self.build_pipeline()
        self._resolved_pipeline.initialize()
        if self.diagnostics:
            if self.diagnostics_output_dir:
                diag_output_dir = Path(self.diagnostics_output_dir).expanduser().resolve()
            else:
                diag_output_dir = (
                    (self._outputdir / f'diagnostics_{self._resolved_pipeline.name}')
                    .expanduser()
                    .resolve()
                )
            run_and_save_diagnostics(self._resolved_pipeline, diag_output_dir.as_posix())

        # save the full pipeline
        pipeline_dump = dr.dump(self._resolved_pipeline)
        with open(self._outputdir / 'calibration.yaml', 'w') as f:
            f.write(pipeline_dump)

        if not self.diagnostics_only:
            samples = self._xpdata['samples']
            for s in samples:
                if not s['control']:
                    print(f"Processing sample {s['name']}")
                    self.calibrate_file(s, self.export_format)


def main():
    prog = make_program(
        CalibrationProgram,
        name='calibrie-run',
        description='Calibration of data files and experiments.',
    )
    calib, args = prog.parse_args(
        sys.argv[1:],
        context={
            'GatingTask': GatingTask,
            'Colinearization': Colinearization,
            'PolygonGate': PolygonGate,
            'LoadControls': LoadControls,
            'LinearCompensation': LinearCompensation,
            'ProteinMapping': ProteinMapping,
            'MEFBeadsCalibration': MEFBeadsCalibration,
            'MEFBeadsTransform': MEFBeadsTransform,
            'PandasExport': PandasExport,
            'AbundanceCutoff': AbundanceCutoff,
            'Pipeline': Pipeline,
        },
        deferred_paths=[
            '/pipeline',
            '/outputdir',
        ],
    )
    calib.run()


if __name__ == '__main__':
    main()
