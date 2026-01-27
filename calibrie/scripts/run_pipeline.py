# File: calibrie/scripts/run_pipeline.py
# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

import dracon as dr
import json
from dracon.trace import with_indent
from dracon.deferred import DeferredNode, make_deferred
from dracon.lazy import LazyDraconModel, resolve_all_lazy
import pandas as pd
import time
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import dracon
import calibrie as cal
from pathlib import Path
from typing import List, Tuple, Union, Annotated, Dict, Any, Optional, Literal
from pydantic import BaseModel, BeforeValidator
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


class Timer:
    _enabled = False

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if Timer._enabled:
            print(f"[TIMING] {self.name}...", flush=True)
            self.start = time.time()
        return self

    def __exit__(self, *args):
        if Timer._enabled and hasattr(self, 'start'):
            elapsed = time.time() - self.start
            print(f"[TIMING] {self.name} done in {elapsed:.2f}s", flush=True)


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
        DeferredNode[cal.Pipeline],
        Arg(help='Path to the YAML pipeline configuration file.', is_file=True),
    ]

    loglevel: Annotated[
        Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        Arg(help='Set the logging level.'),
    ] = 'INFO'

    xpfile: Annotated[
        str,
        Arg(help='Path to the experiment metadata file (JSON5 format).'),
    ] = './experiment.json5'

    outputdir: Annotated[
        DeferredNode[str],
        BeforeValidator(make_deferred),
        Arg(help='Directory to save calibrated output files.'),
    ] = '${"$XP_DATADIR/../calibrated/$PIPELINE_NAME"}'

    datapath: Annotated[
        str,
        Arg(help='Base directory containing the raw data files, relative to the xpfile.'),
    ] = './data/raw_data/'

    export_format: Annotated[
        Literal['csv', 'parquet', 'json', 'hdf5', 'feather'],
        Arg(help='Format for the exported calibrated data files.'),
    ] = 'parquet'

    skip_diagnostics: Annotated[
        bool,
        Arg(help='If set, skip generating diagnostic figures.'),
    ] = False

    diagnostics_output_dir: Annotated[
        Optional[str],
        Arg(
            help='Directory to save diagnostic figures (if generated).',
            default_str='<outputdir>/diagnostics-<namehash>',
        ),
    ] = None

    diagnostics_only: Annotated[
        bool,
        Arg(help='If set, only run diagnostics, do not process samples.'),
    ] = False

    no_figures: Annotated[
        bool,
        Arg(help='If set, skip generating diagnostic figures but still output metrics.'),
    ] = False

    timings: Annotated[
        bool,
        Arg(help='If set, print timing information for each step.'),
    ] = False

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

        ctx = {}
        ctx['$BEAD_FILE'] = beads_file.as_posix()
        ctx['$CONTROL_FILES'] = self._control_files
        ctx['$XP_FILE'] = self.xpfile
        ctx['$XP_DIRNAME'] = Path(self.xpfile).parent.name
        ctx['$XP_DATADIR'] = self._datadir.as_posix()

        with Timer("pipeline.construct"):
            self._resolved_pipeline = self.pipeline.construct(context=ctx)
        with Timer("resolve_all_lazy"):
            resolve_all_lazy(self._resolved_pipeline)
        if not isinstance(self._resolved_pipeline, cal.Pipeline):
            self._resolved_pipeline = cal.Pipeline(**self._resolved_pipeline)

        with Timer("get_namehash"):
            ctx['$PIPELINE_NAME'] = self._resolved_pipeline.get_namehash()

        self._outputdir = self.outputdir.construct(context=ctx)
        self._outputdir = Path(self._outputdir).expanduser().resolve()

        self._outputdir.mkdir(parents=True, exist_ok=True)

        self._resolved_pipeline.loglevel = self.loglevel
        self._resolved_pipeline.setup_logging()

        return self._resolved_pipeline

    def calibrate_file(self, sample, file_format='parquet'):
        data_path = (self._datadir / sample['file']).as_posix()
        ctx_out = self._resolved_pipeline.apply_all(data_path)
        data: pd.DataFrame = ctx_out.output_df
        assert isinstance(data, pd.DataFrame), f'Expected DataFrame, got {type(data)}'

        pipeline_json = json.loads(self._resolved_pipeline.model_dump_json_safe())
        xp_dict = {k: v for k, v in self._xpdata.items() if k != 'samples'}
        sample_dict = sample

        data.attrs['calibration'] = {
            'pipeline': pipeline_json,
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
        elif file_format == 'json':
            data.to_json(output_path, orient='records', indent=2)
        elif file_format == 'hdf5':
            try:
                import tables  # noqa: F401
            except ImportError:
                log.error('HDF5 export requires tables. pip install tables')
                return
            data.to_hdf(output_path, key='data', mode='w', complevel=5, complib='zlib')
        elif file_format == 'feather':
            try:
                import pyarrow  # noqa F401
            except ImportError:
                log.error('Feather export requires pyarrow. pip install pyarrow')
                return
            data.to_feather(output_path)
        else:
            raise ValueError(f'Unsupported file format: {file_format}')

    def run(self):
        Timer._enabled = self.timings
        with Timer("build_pipeline"):
            self.build_pipeline()
        with Timer("pipeline.initialize"):
            self._resolved_pipeline.initialize()
        if not self.skip_diagnostics:
            if self.diagnostics_output_dir:
                diag_output_dir = Path(self.diagnostics_output_dir).expanduser().resolve()
            else:
                diag_output_dir = (
                    (self._outputdir / f'diagnostics_{self._resolved_pipeline.name}')
                    .expanduser()
                    .resolve()
                )
            if not self.no_figures:
                with Timer("run_and_save_diagnostics"):
                    run_and_save_diagnostics(self._resolved_pipeline, diag_output_dir.as_posix())
            with Timer("save_metrics"):
                self._resolved_pipeline.save_metrics(diag_output_dir)

        with Timer("dr.dump pipeline"):
            pipeline_dump = dr.dump(self._resolved_pipeline)
        with open(self._outputdir / 'calibration.yaml', 'w') as f:
            f.write(pipeline_dump)

        if not self.diagnostics_only:
            samples = self._xpdata['samples']
            for s in samples:
                if not s['control']:
                    with Timer(f"calibrate {s['name']}"):
                        print(f"Processing sample {s['name']}")
                        self.calibrate_file(s, self.export_format)


def main():
    prog = make_program(
        CalibrationProgram,
        name='calibrie-run',
        description=('Execute a Calibrie fluorescence calibration pipeline.'),
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
    )

    print(dr.dump(calib))

    calib.run()


if __name__ == '__main__':
    main()
