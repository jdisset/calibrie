import dracon as dr
import json
from dracon.utils import with_indent
from dracon.interpolation import LazyDraconModel
import pandas as pd
import time
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import dracon
import calibry as cal
from pathlib import Path
from typing import List, Tuple, Union, Annotated, Dict, Any
from pydantic import BaseModel
import sys
from pathlib import Path
import json5
from calibry.pipeline import DiagnosticFigure
from calibry import (
    GatingTask,
    PolygonGate,
    LoadControls,
    LinearCompensation,
    ProteinMapping,
    MEFBeadsCalibration,
    PandasExport,
    AbundanceCutoff,
    Pipeline,
)
import matplotlib

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
        fname = f'{fig.source_task}_{fig.name}'.lower().replace(' ', '_')
        fig.fig.savefig(outputdir / f'{fname}.pdf', dpi=300)


class CalibrationProgram(LazyDraconModel):
    pipeline: Annotated[
        Resolvable[cal.Pipeline], Arg(help='The pipeline to execute', resolvable=True, is_file=True)
    ]
    xpfile: Annotated[str, Arg(help='Input experiment file to load')] = './experiment.json5'
    outputdir: Annotated[str, Arg(help='The output directory')] = '.'
    datapath: Annotated[
        str, Arg(help='The path to the directory containing the data files, relative to the xpfile')
    ] = './data/raw_data/'
    diagnostics: Annotated[bool, Arg(help='Whether to generate diagnostic figures')] = True

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

        self._outputdir = Path(self.outputdir).expanduser().resolve()

        self._context = {}
        self._context['$BEAD_FILE'] = beads_file.as_posix()
        self._context['$CONTROL_FILES'] = self._control_files
        self._context['$XP_FILE'] = self.xpfile
        self._context['$XP_DIRNAME'] = Path(self.xpfile).parent.name
        self._context['$XP_DATADIR'] = self._datadir.as_posix()
        self._context['$CALIBRATION_OUTPUT_DIR'] = self._outputdir.as_posix()
        print(f'Context: {self._context}')

        self._resolved_pipeline = self.pipeline.resolve(context=self._context, interpolate_all=True)

        return self._resolved_pipeline

    def calibrate_file(self, sample):
        data_path = (self._datadir / sample['file']).as_posix()
        ctx_out = self._resolved_pipeline.apply_all(data_path)
        data: pd.DataFrame = ctx_out.output_df
        assert isinstance(data, pd.DataFrame), f'Expected DataFrame, got {type(data)}'

        # pipeline_json = pipeline.model_dump_json(indent=2)
        # xp_json = json.dumps({k: v for k, v in calib._xpdata.items() if k != 'samples'}, indent=2)
        # sample_json = json.dumps(sample, indent=2)

        pipeline_dict = self._resolved_pipeline.model_dump()
        xp_dict = {k: v for k, v in self._xpdata.items() if k != 'samples'}
        sample_dict = sample

        data.attrs['calibration'] = {
            'pipeline': pipeline_dict,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_args': sys.argv[1:],
        }
        data.attrs['xp'] = xp_dict
        data.attrs['sample'] = sample_dict
        self._outputdir.mkdir(parents=True, exist_ok=True)
        output_path = self._outputdir / f'{sample["name"]}.parquet'
        print(f'Saving calibrated data to {output_path}')
        # data.to_parquet(output_path, compression='snappy')
        data.to_parquet(output_path, compression='zstd')

    def run(self):
        self.build_pipeline()
        self._resolved_pipeline.initialize()
        if self.diagnostics:
            run_and_save_diagnostics(self._resolved_pipeline, self._outputdir)

        samples = self._xpdata['samples']
        for s in samples:
            if not s['control']:
                print(f"Processing sample {s['name']}")
                self.calibrate_file(s)


# xpfile = '~/Dropbox (MIT)/Biocomp_v2/Experiments/2024-08-09_BPBRTL/experiment.json5'
# calib, args = prog.parse_args(['--pipeline', './example_pipeline.yaml', '--xpfile', xpfile, '--outputdir', './output'])

def main():
    prog = make_program(
        CalibrationProgram,
        name='calibry-run',
        description='Calibration of data files and experiments.',
    )

    calib, args = prog.parse_args(sys.argv[1:])
    calib.run()

if __name__ == '__main__':
    main()
