<img src="docs/logo_sd.png" alt="calibry logo" width="64">

# Calibry

Calibry is a Python package for calibrating fluorescence values in flow cytometry data. 
It uses color controls, blank controls, and beads with known fluorescence values for calibration.

## Installation

To install the Calibry package, follow these steps:

### For windows users
If you don't have a python development environment, I'd suggest the following procedure:

1. I recommend installing miniconda for windows: [Latest miniconda for windows 64 bits][https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe]
2. Open the Anaconda Prompt (from the start menu)
3. Create a new environment with python 3.8, as it's a version  that works for both cytoflow and calibry (calibry will work with any version >= 3.8) :
   `conda create -n <name> python=3.8` where <name> is any name you want to give your environment (such as "flow")
5. Activate the environment: `conda activate <name>`
6. Install jupyterlab and git with `conda install jupyterlab git`
7. Start jupyterlab with `jupyter lab` and open a terminal from the jupyterlab interface
8. Clone and install calibry in the directory of your choice, from the jupyterlab terminal:
	```bash
	git clone <calibry_repository_url>`
	cd calibry
	pip install jax[cpu] # this is needed to install all the jax dependencies
	pip install -e .
	```
9. Clone and install the Calibry repository:
	```bash
	git clone <calibry_repository_url>`
	cd calibry
	pip install -e .
	```
10. Windows require specific versions of certain packages so we'll install the dependencies manually:
	```bash
	pip install "jax[cpu]===0.3.25" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
	pip install flax==0.6.3 ott-jax==0.3.1 chex==0.1.5 optax==0.1.4 flax==0.6.3 orbax==0.0.15 --no-deps
	pip install pandas numpy Pillow matplotlib PyYAML scipy flowio tqdm
	```
	

### General procedure
	```bash
	git clone <calibry_repository_url>`
	cd calibry
	pip install -e .
	```

## Update

To update the Calibry package to the latest version, first pull the latest changes from the GitHub repository:

```
git pull
```

Then, reinstall the package:

```
pip install --upgrade .
```

## Usage

Here's an example of how to use the Calibration class from the Calibry package:

```python

from calibry import Calibration
import pandas as pd

# Your control and beads files
beads = "path/to/beads/file.fcs"

# create the control dictionnary:
# key = name of the control, value = path to the csv or fcs file
# key follows the following convention (case insensitive):
# single color control : the name of the protein. 'EBFP2', 'EYFP', ...
# all color control: 'ALL'
# blank: 'CNTL' or 'EMPTY'
control_files = list(raw_path.glob('color_controls/*.csv'))
# in this example I'm assuming that the control files are named like "KEY.blablabla.csv"
# e.g. "EBFP2.2023-03-10.csv", "all.whatever.csv", "eMpTy.my_empty_ctrl.csv"
# the relevant is the first part of the file name, before the first dot
# so it's easy to extract it with the following line
controls = {c.stem.split('.')[0]: c for c in control_files}

# Initialize the Calibration instance
cal = Calibration(controls, beads, reference_protein='MKATE', 
				use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700'])

# Fit the calibration model
cal.fit()

# OPTIONAL: plot diagnostics, very useful to check that everything looks right
cal.plot_bleedthrough_diagnostics() # plots the spectral signature matrix
cal.plot_beads_diagnostics() # plots the results of the beads detection, assignment and calibration steps
cal.plot_color_mapping_diagnostics() # plots the mapping from every protein to the reference one

# Apply the calibration to your data
data = pd.read_csv("path/to/data/file.csv")
calibrated = cal.apply(data)
```

## Calibration Class

The Calibration class is the main class of the Calibry package. It handles the calibration of the fluorescence values in a given experiment and can export to a calibrated pandas dataframe.

### Constructor Arguments
- `color_controls_files`: A dictionary with keys as control protein names (e.g., 'eYFP') and values as paths to their corresponding FCS or CSV files.
- `beads_file`: A string representing the path to the beads control FCS file.
- `reference_protein`: (Optional) A string representing the name of the reference protein used to align other protein quantities. The default value is 'mkate'.
- `reference_channel`: (Optional) A string representing the name of the reference channel used to convert to standard MEF units. The default value is 'PE_TEXAS_RED'.
- `beads_mef_values`: (Optional) A dictionary associating the unit name with the list of reference values for the beads in this unit. The default value is SPHEROTECH_RCP_30_5a.
- `channel_to_unit`: (Optional) A dictionary associating the channel name with the unit name. The default value is FORTESSA_CHANNELS.
- `random_seed`: (Optional) An integer representing the random seed for any resampling that might be done. The default value is 42.
- `use_channels`: (Optional) A list of channel names to use when computing protein quantity from channel values (i.e., bleedthrough correction). The default value is None, which means all available channels will be used.
- `max_value`: (Optional) An integer representing the maximum fluorescence value used to detect saturation. The default value is FORTESSA_MAX_V.
- `offset`: (Optional) An integer representing the offset to add to the fluorescence values to avoid negative values. The default value is FORTESSA_OFFSET.


Please refer to the comments and documentation in the calibry.py file for more details.
