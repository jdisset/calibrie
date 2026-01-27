# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

import pandas as pd
from dracon.representer import DraconRepresenter

def _represent_dataframe(representer, df):
    if hasattr(df, 'attrs') and 'from_file' in df.attrs:
        return representer.represent_str(str(df.attrs['from_file']))
    raise ValueError(f"Cannot represent DataFrame without 'from_file' attr: {df.shape}")

DraconRepresenter.add_representer(pd.DataFrame, _represent_dataframe)

from .pipeline import *

# from .gating_v2.gating import GatingTask, PolygonGate
from .gating import GatingTask, PolygonGate
from .linearcompensation import LinearCompensation
from .nonlinearunmixing import hellonls, fit_reference_functions, NonLinearUnmixing
from .loadcontrols import LoadControls
from .mefbeadstransform import MEFBeadsTransform
from .proteinmapping import ProteinMapping
from .colinearization import Colinearization
from .cutoff import AbundanceCutoff
from .mefcalibration import MEFBeadsCalibration
from .pandasexport import PandasExport
from . import plots, utils, scripts
