
# Usage: Command Line Tools

Calibrie provides command-line scripts to facilitate running pipelines and performing specific actions like interactive gating.

## `calibrie-run`

This is the primary script for executing calibration pipelines defined in YAML files.

**Synopsis:**

```bash
calibrie-run --pipeline <pipeline.yaml> --xpfile <experiment.json5> [options]
```

**Core Arguments:**

*   `--pipeline <path/to/pipeline.yaml>`: **Required.** Specifies the main YAML file defining the pipeline structure and tasks.
*   `--xpfile <path/to/experiment.json5>`: **Required.** Specifies the JSON5 file containing experiment metadata, control definitions, and paths to raw data files (relative to `--datapath`).

**Common Options:**

*   `--outputdir <path/to/output/directory>`: Specifies the directory where calibrated data files (e.g., Parquet/CSV) will be saved.
    *   *Default:* Creates a directory based on the experiment name, pipeline name, and a hash of the pipeline configuration (e.g., `output/<xp_name>/<pipeline_name>-<hash>`).
*   `--datapath <path/to/data/directory>`: Specifies the base directory relative to which file paths within `experiment.json5` (`file`, `beads_file`) are resolved.
    *   *Default:* `./data/raw_data/` (relative to where `calibrie-run` is executed).
*   `--diagnostics <true|false>`: Whether to generate diagnostic plots for each applicable task.
    *   *Default:* `true`. Use `--no-diagnostics` or `--diagnostics false` to disable.
*   `--diagnostics_output_dir <path/to/diagnostics/directory>`: Specifies where to save diagnostic plots (typically as a combined PDF or individual image files).
    *   *Default:* A subdirectory named `diagnostics_<pipeline_name>` within the main `--outputdir`.
*   `--diagnostics_only <true|false>`: If set, only generates diagnostics without processing and saving sample data. Useful for quickly reviewing results after an initial run.
    *   *Default:* `false`.
*   `--export_format <csv|parquet>`: Format for the exported calibrated data files. Parquet is generally recommended for efficiency.
    *   *Default:* `parquet`.

**Execution Process:**

1.  Parses the `--xpfile` to identify control files, sample files, and the beads file.
2.  Creates context variables like `$CONTROL_FILES` (a dictionary mapping control names to full paths) and `$BEAD_FILE` (full path) based on the `--xpfile` and `--datapath`.
3.  Loads the `--pipeline` YAML file, substituting variables (`${...}`) using the context derived from the experiment file and potentially environment variables or other configurations (handled by `dracon`).
4.  Instantiates the `calibrie.Pipeline` object based on the loaded YAML definition.
5.  Calls `pipeline.initialize()` to set up all tasks (load controls, calculate spillover, etc.).
6.  If `diagnostics_only` is false:
    *   Iterates through each *sample* file defined in the `--xpfile`.
    *   Calls `pipeline.apply_all()` for the sample data.
    *   Saves the output specified by the `PandasExport` task (or equivalent) to the `--outputdir`.
7.  If `diagnostics` is true:
    *   Calls `pipeline.all_diagnostics()`.
    *   Saves the resulting figures to the `--diagnostics_output_dir`.

**Example:**

```bash
# Run the example pipeline from the project root
calibrie-run \
    --pipeline experiments/myxp/calibration/regular.yaml \
    --xpfile experiments/myxp/experiment.json5 \
    --datapath experiments/myxp/data/raw_data/ \
    --outputdir output/cascades_calibrated \
    --diagnostics_output_dir output/cascades_diagnostics
```

## `calibrie-gating`

This script launches a Graphical User Interface (GUI) for interactively defining polygon gates on flow cytometry data.

**Synopsis:**

```bash
calibrie-gating
```

**Functionality:**

1.  Opens a GUI window based on `dearpygui`.
2.  Allows loading FCS files using the "Load File(s)" button.
3.  Displays scatter plots of the loaded data.
4.  Lets users select X and Y channels for plotting.
5.  Provides tools ("Add Points", "Del Points") to draw polygon gates directly on the plots.
6.  Allows saving the defined gates (channels, vertices, gate name) to a YAML file (e.g., `gating_task.yaml`).
7.  Allows loading previously saved gating YAML files.

**Output:**

The primary output is a YAML file containing a `!GatingTask` definition, including a list of `!PolygonGate` objects with their vertices and channel assignments. This file can then be included (`*file:`) in a main calibration pipeline YAML, as shown in the `regular.yaml` example.

**Usage:**

Typically, you would run `calibrie-gating` first to define your cell population gates (e.g., based on FSC/SSC), save the results to a `gating_task.yaml`, and then reference that file in your main pipeline run with `calibrie-run`.
