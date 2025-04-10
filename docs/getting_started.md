
# Getting Started: A Calibration Example

This tutorial walks through a typical Calibrie workflow using YAML configuration files and the command-line runner. We'll calibrate a sample experiment involving multiple fluorescent proteins.

## Prerequisites

1.  **Install Calibrie:** Follow the instructions on the [Overview](index.md) page.
2.  **Example Files:** You'll need experiment data (FCS or CSV files for samples and controls), bead data (FCS), and configuration files. We'll refer to the example structure provided:
    ```
    calibrie/
    ├── CalibrationTemplates/
    │   └── WeissFortessa/
    │       ├── 4color_controls.yaml
    │       ├── beads_urcp-100-2H.yaml
    │       ├── ebfp2_mapping.yaml
    │       └── simple_linear.yaml
    ├── experiments/
    │   └── myxp/
    │       ├── calibration/
    │       │   ├── gating/
    │       │   │   └── gating_task.yaml
    │       │   └── regular.yaml
    │       ├── data/ # <= IMPORTANT: Assumed location for data files
    │       │   ├── raw_data/
    │       │   │   ├── 2023-10-01_Cascades_CCv4_1.fcs # Sample
    │       │   │   ├── ... (other samples) ...
    │       │   │   ├── 2023-10-01_Cascades_CCv4_21.fcs # 1xirfp720 Control
    │       │   │   ├── ... (other controls) ...
    │       │   │   ├── 2023-10-01_Cascades_CCv4_27.fcs # CNTL Control
    │       │   │   └── 2023-10-01_Cascades_CCv4_beads.fcs # Beads
    │       └── experiment.json5
    ├── calibrie/
    ├── ... (other project files) ...
    ```
    *Note: You'll need to create the `data/raw_data/` directory and place your FCS files there, matching the names in `experiment.json5`.*

## 1. Experiment Metadata (`experiment.json5`)

Calibrie uses a JSON5 file (JSON with comments and trailing commas) to store metadata about the experiment and links to the raw data files.

File: `experiments/myxp/experiment.json5`
```json5
{!experiments/myxp/experiment.json5!}
```

**Key Sections:**

*   `name`: A descriptive name for the experiment.
*   `samples`: A list of all runs in the experiment. Each entry has:
    *   `name`: Unique name for the sample/run.
    *   `recipe`: (Optional) Experimental condition identifier.
    *   `control`: `1` if this is a control file (single color, blank, all), `0` otherwise.
    *   `file`: The **relative path** (from the `datapath` specified in the run command, default is `./data/raw_data/`) to the FCS or CSV file.
    *   `notes`: (Optional) Any relevant notes.
*   `beads_file`: The **relative path** to the bead calibration file.
*   Other fields: Metadata like dates, operators, machine, cell line, etc. (Currently not used directly by the core Calibrie tasks, but good practice for record-keeping).

This file tells the `calibrie-run` script which files are samples and which are controls, essential for tasks like `LoadControls`.

## 2. Pipeline Definition (`regular.yaml`)

The pipeline defines the sequence of analysis steps (`Tasks`). Calibrie uses YAML files, often leveraging the `dracon` library for features like includes (`*file:`) and variable substitution (`${...}`).

File: `experiments/myxp/calibration/regular.yaml`
```yaml
{!experiments/myxp/calibration/regular.yaml!}
```

**Explanation:**

*   `name: FINAL`: Sets a name used in the output directory structure.
*   `tasks:`: Defines the dictionary of tasks to be executed. The keys (`gating`, `controls`, `lincomp`, etc.) are arbitrary names for the tasks within this pipeline.
*   `*file:$DIR/gating/gating_task.yaml`: This includes the gating definition from another file located in the same directory (`$DIR`). The `*file:` syntax is specific to `dracon`.
*   `*file:$CALIBRATION_TEMPLATE_DIR/...`: Includes predefined task configurations from the `CalibrationTemplates` directory. `$CALIBRATION_TEMPLATE_DIR` is likely a variable set by the environment or `dracon` configuration. This promotes reuse of standard task settings.
*   **Task Order:** The `priority` field within each included task file (e.g., in `4color_controls.yaml`) determines the execution order (lower priority runs first). The sequence here is roughly:
    1.  `gating` (Priority 0.0): Filters cells based on scatter properties.
    2.  `beads` (Priority 5): Pre-calculates bead information needed for MEF conversion, but doesn't apply it yet (`apply_on_load: false`).
    3.  `controls` (Priority 10): Loads control files specified in `experiment.json5`, calculates metrics.
    4.  `lincomp` (Priority 20): Performs linear compensation based on controls.
    5.  `protmap` (Priority 40): Maps protein abundances to the reference protein's scale (EBFP2 in this case).
    6.  `export` (Priority 1000): Exports the final calibrated data (`abundances_MEF` produced implicitly by `ProteinMapping` when `apply_mef_transform: true` and the `beads` task has run).
*   **Overriding:** You can override parameters from included files directly in this main file (demonstrated by the commented-out `lincomp` override example).

## 3. Running the Pipeline

Execute the pipeline from the **project root directory** using the `calibrie-run` script:

```bash
calibrie-run --pipeline experiments/myxp/calibration/regular.yaml --xpfile experiments/myxp/experiment.json5
```

**Command Breakdown:**

*   `calibrie-run`: The command-line script installed with the package.
*   `--pipeline experiments/myxp/calibration/regular.yaml`: Specifies the main pipeline definition file.
*   `--xpfile experiments/myxp/experiment.json5`: Specifies the experiment metadata file. `calibrie-run` will parse this and make `$CONTROL_FILES` (a dictionary mapping control names like 'EBFP2' to their full paths) and `$BEAD_FILE` (the full path to the beads file) available for substitution within the YAML pipeline definition.
*   `--datapath experiments/myxp/data/raw_data/` (Optional): Specifies the base directory for relative file paths in `experiment.json5`. If omitted, it defaults to `./data/raw_data/`.
*   `--outputdir output/myxp_calibrated` (Optional): Specifies where to save results. Defaults to a directory based on the experiment and pipeline names.
*   `--diagnostics` (Optional, default: True): Generate diagnostic plots. Use `--no-diagnostics` to disable.
*   `--diagnostics_output_dir output/myxp_diagnostics` (Optional): Specify where to save diagnostic plots. Defaults to a subdirectory within the main output directory.

## 4. Expected Output

After running, Calibrie will:

1.  **Initialize Tasks:** Load controls, calculate spillover matrix, fit mapping functions, process beads, etc.
2.  **Process Samples:** For each *non-control* file listed in `experiment.json5`:
    *   Load the data.
    *   Apply gating.
    *   Apply linear compensation.
    *   Apply protein mapping.
    *   Apply MEF calibration (implicitly done by `ProteinMapping` using the info from the `beads` task).
    *   Export the final `abundances_MEF` data to a file (e.g., Parquet or CSV) in the specified output directory.
3.  **Generate Diagnostics:** If enabled, create plots for each task (spillover matrix, mapping functions, bead calibration plots, etc.) and save them, typically as a combined PDF or individual files in the diagnostics output directory.

You now have calibrated fluorescence data in MEF units (specifically, "MEF units relative to EBFP2 on the Pacific Blue channel" in this example) ready for downstream analysis!
