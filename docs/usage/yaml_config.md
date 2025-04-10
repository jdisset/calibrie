
# Usage: YAML Configuration

Defining pipelines using YAML files is often convenient for reproducibility, sharing, and modifying analyses without altering Python code. Calibrie leverages the `dracon` library (or a similar mechanism) to parse these YAML files, allowing for features like includes and variable substitution.

## Structure of a Pipeline YAML

A pipeline YAML file typically defines a top-level `Pipeline` object and specifies its `tasks`.

```yaml
# Example: experiments/myxp/calibration/regular.yaml
name: FINAL_YAML_EXAMPLE # Name for organization/output folder naming

tasks:
  # Each key is a unique name for the task within this pipeline
  gating: !GatingTask # The '!TaskName' syntax likely maps to the Python class
    # Parameters specific to this task instance can be defined here
    # OR included from another file:
    <<: *file:$DIR/gating/gating_task.yaml # '<<:' is merge key, '*file:' includes another YAML

  controls: !LoadControls
    # Include common settings from a template
    <<: *file:$CALIBRATION_TEMPLATE_DIR/WeissFortessa/4color_controls.yaml
    # Variables like $CONTROL_FILES are substituted at runtime
    color_controls: ${$CONTROL_FILES}

  lincomp: !LinearCompensation
    <<: *file:$CALIBRATION_TEMPLATE_DIR/WeissFortessa/simple_linear.yaml
    # Example of overriding a parameter from the template:
    # channel_weight_attenuation_power: 1.5

  protmap: !ProteinMapping
    <<: *file:$CALIBRATION_TEMPLATE_DIR/WeissFortessa/ebfp2_mapping.yaml
    reference_protein: EBFP2 # Ensure this matches controls

  beads: !MEFBeadsTransform
    <<: *file:$CALIBRATION_TEMPLATE_DIR/WeissFortessa/beads_urcp-100-2H.yaml
    beads_data: ${"$BEAD_FILE"} # Variable substitution

  export: !PandasExport
    priority: 1000
    context_field_to_export: abundances_MEF
```

**Key Concepts:**

*   **`!TaskName`:** This YAML tag tells the parser (e.g., `dracon`) which Python class (`calibrie.Task` subclass) to instantiate for this task.
*   **`<<: *file:` (Merge Include):** This is a standard YAML merge key combined with a likely `dracon` extension (`*file:`). It means "load the YAML content from the specified file and merge its key-value pairs into the current dictionary." This is used heavily to pull settings from templates (like those in `CalibrationTemplates/`).
*   **Variable Substitution (`${...}`):** Placeholders like `${$CONTROL_FILES}` or `${"$BEAD_FILE"}` are replaced at runtime, typically by the command-line runner (`calibrie-run`) based on the provided `experiment.json5` file or environment variables. `$DIR` usually refers to the directory of the current YAML file, and `$CALIBRATION_TEMPLATE_DIR` would be another configured path.
*   **Overrides:** You can specify parameters directly within a task definition to override values inherited from an included template (see commented-out `lincomp` example).
*   **Task Naming:** The keys in the `tasks` dictionary (`gating`, `controls`, etc.) are names used internally by the pipeline runner and for organizing diagnostics.
*   **Priorities:** Execution order is determined by the `priority` parameter defined within the task's configuration (often in the included template file). Lower numbers run earlier.

## Experiment Metadata (`experiment.json5`)

As shown in the [Getting Started](getting_started.md) guide, the `experiment.json5` file complements the pipeline YAML by providing the specific file inputs and metadata for a particular experimental run. The command-line runner uses this file to resolve variables like `${$CONTROL_FILES}` and `${"$BEAD_FILE"}` in the pipeline YAML.

## Running YAML Pipelines

YAML-defined pipelines are typically executed using the `calibrie-run` command-line tool. See the [Command Line Tools](command_line.md) page for details.

Using YAML configuration allows easy modification of parameters (e.g., changing the `reference_protein` in `protmap`) or swapping entire task implementations (e.g., replacing `simple_linear.yaml` with a different compensation method) without needing to edit Python code.
