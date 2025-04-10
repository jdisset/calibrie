
<style>
  .md-typeset h1,
  .md-content__button {
    /* display: none; Hide default H1 title if using logo */
  }
  .md-typeset figure {
    text-align: center;
  }
</style>

<figure markdown>
  ![Calibrie Logo](./assets/calibrie_full.png){ width="400" }
</figure>

**Calibrie** is a modular Python library for the **analysis**, **unmixing**, and **calibration** of fluorescence flow cytometry data, particularly tailored for synthetic biology applications.

It implements multiple algorithms to turn raw fluorescence measurements into standardized, quantitative protein abundance units (like Molecules of Equivalent Fluorophore - MEF), along with numerous diagnostics and quality assessment tools. Calibrie is designed to promote **reproducible**, **modular**, and **sharable** cytometry analysis workflows.

!!! warning "Alpha Stage"
    Calibrie is currently in **alpha**. While functional, the API might undergo changes, and documentation is actively being developed. Feedback and contributions are welcome!

## Core Philosophy

Calibrie is built on a few key principles inspired by the need for robust characterization in synthetic biology:

*   **Modularity:** Analysis steps are broken down into independent `Task` classes (like `LoadControls`, `LinearCompensation`, `MEFCalibration`). This allows flexible construction of analysis `Pipeline`s tailored to specific experimental needs.
*   **Reproducibility:** By standardizing units (MEF-equivalents based on bead standards and protein mapping) and using explicit configuration files (YAML), Calibrie aims to make analyses repeatable across different datasets and even different labs (with caveats about instrument differences).
*   **Transparency:** The `Context` object explicitly tracks the flow of data between `Tasks`. Extensive `diagnostics` methods within tasks allow visualization of intermediate steps and final results, aiding in quality control and debugging.
*   **Flexibility:** Pipelines can be defined either directly in Python code for maximum control or through declarative YAML configuration files (leveraging the `dracon` library) for easier sharing and modification without code changes.

## Installation

Calibrie is not yet on PyPI. To install it, clone the repository and install it locally:

```bash
git clone https://github.com/jdisset/calibrie.git
cd calibrie
pip install .
# Or for development:
# pip install -e .
```
Make sure you have the necessary dependencies installed (see `pyproject.toml`). You might need to install JAX separately depending on your hardware (CPU/GPU/TPU): [JAX Installation Guide](https://github.com/google/jax#installation).

## Quick Start

Here's a minimal example of defining and running a pipeline in Python:

```python
import calibrie as cal
import pandas as pd

# Define controls and bead file paths (replace with actual paths)
controls_dict = {
    'EBFP2': 'path/to/ebfp2_control.fcs',
    'MNEONGREEN': 'path/to/mneongreen_control.fcs',
    'ALL': 'path/to/all_control.fcs',
    'BLANK': 'path/to/blank_control.fcs'
}
beads_file = 'path/to/beads.fcs'

# Example minimal pipeline (adjust parameters as needed)
pipeline = cal.Pipeline(
    tasks={
        "controls": cal.LoadControls(
            color_controls=controls_dict,
            use_channels=['PACIFIC_BLUE_A', 'FITC_A'] # Example
        ),
        "lincomp": cal.LinearCompensation(),
        # Add ProteinMapping and MEFCalibration for full calibration
    }
)

# Initialize (e.g., load controls, compute spillover)
# In a real scenario, LoadControls needs file paths.
# Here we assume they are set correctly inside the task definition.
pipeline.initialize()

# Load sample data
sample_data = cal.load_to_df("path/to/sample.fcs") # Use Calibrie's loader

# Apply pipeline
result_context = pipeline.apply_all(sample_data)
# Get unmixed arbitrary units
unmixed_abundances = result_context.abundances_AU

print(unmixed_abundances.head())

# Generate diagnostics (optional)
# figs = pipeline.all_diagnostics()
# if figs:
#   figs[0].fig.savefig("lincomp_diagnostics.png") # Save diagnostic figures
```

**Explore the documentation to learn more about:**

*   [Getting Started](getting_started.md): A detailed tutorial using YAML configuration.
*   [Usage Patterns](usage/index.md): How to define and run pipelines in Python and YAML.
*   [Core Concepts](concepts/index.md): The theory behind the calibration steps.
*   [API Reference](reference/index.md): Details on each task and class.
