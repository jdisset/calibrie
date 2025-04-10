#<img src="docs/assets/calibrie_horizontal.png" alt="Calibrie Logo" width="400" style="vertical-align: bottom;">

**Calibrie** is a modular Python library for the **analysis**, **unmixing**, and **calibration** of fluorescence flow cytometry data, particularly tailored for synthetic biology applications.

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://jdisset.github.io/calibrie/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Add other badges as relevant (e.g., PyPI version, build status) -->

It implements multiple algorithms to turn raw fluorescence measurements into standardized, quantitative protein abundance units (like Molecules of Equivalent Fluorophore - MEF), along with numerous diagnostics and quality assessment tools. Calibrie is designed to promote **reproducible**, **modular**, and **sharable** cytometry analysis workflows.

!!! warning "Alpha Stage"
Calibrie is currently in **alpha**. While functional, the API might undergo changes, and documentation is actively being developed. Feedback and contributions are welcome!

## Installation

Calibrie is not yet on PyPI. To install it, clone the repository and install it locally using pip:

```bash
git clone https://github.com/jdisset/calibrie.git # Replace with your repo URL if different
cd calibrie
pip install .
# For development (editable install):
# pip install -e .
```

Please ensure you have the necessary dependencies, especially JAX (refer to the [JAX Installation Guide](https://github.com/google/jax#installation) for hardware-specific instructions).

## Documentation

**Full documentation, including tutorials, concept explanations, and API references, is available at:**

**https://jdisset.github.io/calibrie**

The documentation provides detailed guides on:

- Getting started with calibration pipelines.
- Using the Python API vs. YAML configuration.
- Understanding the theory behind compensation, mapping, and MEF calibration.
- Detailed API reference for all tasks and utilities.

## Quick Example (Python API)

```python
import calibrie as cal
import pandas as pd

# Define controls and bead file paths
controls_dict = {'EBFP2': 'path/to/ebfp2.fcs', ... , 'BLANK': 'path/to/blank.fcs'}
beads_file = 'path/to/beads.fcs'

# Define pipeline tasks (example)
pipeline = cal.Pipeline(tasks={
    "controls": cal.LoadControls(color_controls=controls_dict, ...),
    "lincomp": cal.LinearCompensation(...),
    "protmap": cal.ProteinMapping(reference_protein='EBFP2', ...),
    "beads": cal.MEFBeadsTransform(beads_data=beads_file, ...), # Provides context for protmap
    "export": cal.PandasExport(context_field_to_export='abundances_MEF')
})

# Initialize and run
pipeline.initialize()
sample_data = cal.load_to_df("path/to/sample.fcs")
results = pipeline.apply_all(sample_data)
calibrated_df = results.output_df

print(calibrated_df.head())
```

_(See the full documentation for details on task parameters and YAML usage)_

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

Calibrie is distributed under the MIT License. See `LICENSE` file for details.
