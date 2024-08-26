import dracon as dr
from dracon.utils import with_indent
from dracon.interpolation import LazyDraconModel
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import calibry as cal
from pathlib import Path
from typing import List, Tuple, Union, Annotated, Dict, Any
from pydantic import BaseModel
import sys
from pathlib import Path
import json5
from calibry import (
    GatingTask,
    PolygonGate,
    LoadControls,
    LinearCompensation,
    ProteinMapping,
    MEFBeadsCalibration,
    PandasExport,
    AbundanceCutoff,
    Pipeline
)

def parse_xpfile(xpfile: str):
    filepath = Path(xpfile).expanduser().resolve()

    with open(filepath, 'r') as f:
        data = json5.load(f)

    return data


class CalibrationProgram(LazyDraconModel):
    pipeline: Annotated[Resolvable[cal.Pipeline], Arg(help='The pipeline to execute', resolvable=True, is_file=True)]
    xpfile: Annotated[str, Arg(help='Input experiment file to load')] = './experiment.json5'
    outputdir: Annotated[str, Arg(help='The output directory')] = '.'
    datapath: Annotated[
        str, Arg(help='The path to the directory containing the data files, relative to the xpfile')
    ] = './data/raw_data/'

    def get_pipeline(self):

        self._datadir = (Path(self.xpfile).parent / self.datapath).expanduser().resolve()
        self._xpdata = parse_xpfile(self.xpfile)

        samples = self._xpdata['samples']
        control_samples = [s for s in samples if s['control']]

        # load beads
        beads_file = self._datadir / self._xpdata['beads_file']
        assert beads_file.exists(), f'File {beads_file} not found for beads'

        # load color controls
        self._control_files = {s['name']: (self._datadir / s['file']).as_posix() for s in control_samples}
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

        pipeline = self.pipeline.resolve(context=self._context, interpolate_all=True)

        return pipeline



prog = make_program(
    CalibrationProgram,
    name='calibry-run',
    description='Calibration of data files and experiments.',
)

xpfile = '~/Dropbox (MIT)/Biocomp_v2/Experiments/2024-08-09_BPBRTL/experiment.json5'
calib = prog.parse_args(['--pipeline', './example_pipeline.yaml', '--xpfile', xpfile])
pipeline = calib.get_pipeline()

pipeline.tasks[0]

##

pipeline.initialize()

##

all_figs = pipeline.all_diagnostics()


## {{{                     --     run + diagnostics     --

import matplotlib.pyplot as plt

# diagnostic_dir = Path(calibration_dir / 'unmixing_diagnostics')
# diagnostic_dir.mkdir(exist_ok=True)


# try:
# some diagnostic plots:
cal.plots.plot_channels_to_reference(
    **pipeline._context,
    logscale=True,
    ylims=[-1e2, 2e5],
    dpi=150,
)
fig = plt.gcf()
fig.tight_layout()
# fig.savefig(diagnostic_dir / 'channels_to_reference.png', dpi=150)
# except Exception as e:
# print(f'Error while plotting: {e}')

pipeline.diagnostics("GatingTask", use_files=calib._control_files.values())
fig = plt.gcf()
fig.tight_layout()
# fig.savefig(diagnostic_dir / 'linear_compensation.png', dpi=150)

try:
    pipeline.diagnostics(
        "LinearCompensation",
        xlims=[-1e5, 1e5],
        ylims=[-1e3, 5e5],
    )
    fig = plt.gcf()
    fig.tight_layout()
    # fig.savefig(diagnostic_dir / 'linear_compensation.png', dpi=150)
except Exception as e:
    print(f'Error while plotting linear compensation: {e}')

pipeline.diagnostics(
    "ProteinMapping",
    xlims=[-1e5, 1e5],
    ylims=[-1e3, 2e5],
    target_bounds=[-100, 1e6],
    other_bounds=[-100, 1e6],
    # logspace=False,
)
# for n, f in pipeline._context['diagnostics_ProteinMapping'].items():
    # f.savefig(diagnostic_dir / f'{n}_pmap.png', dpi=150)
# except Exception as e:
# print(f'Error while plotting protein mapping: {e}')


try:
    pipeline.diagnostics("Colinearization")
    fig = plt.gcf()
    # fig.savefig(diagnostic_dir / 'colinearization.png', dpi=150)
except Exception as e:
    print(f'Error while plotting colinearization: {e}')

try:
    pipeline.diagnostics(
        'MEFBeadsCalibration',
        xlims=[-1e5, 1e5],
        ylims=[-1e3, 5e5],
    )
    # for n, f in pipeline._context['diagnostics_MEFBeadsCalibration'].items():
        # f.savefig(diagnostic_dir / f'{n}_beads.png', dpi=150)
except Exception as e:
    print(f'Error while plotting beads calibration: {e}')

# cal.plots.plot_raw_all_prot_crl(**linpipe.context, logscale=True, dpi=150)

##────────────────────────────────────────────────────────────────────────────}}}


##
samples = calib._xpdata['samples']
for s in samples:
    if not s['control']:
        print(f"Processing sample {s['name']}")
        fullpath = (calib._datadir / s['file']).as_posix()

## TODO:
# remove the _A remover
