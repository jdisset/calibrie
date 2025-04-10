
# Usage: Python API

You can define and run Calibrie pipelines directly within Python scripts. This offers the most flexibility for complex workflows or integration into larger applications.

## Defining a Pipeline

A pipeline is created by instantiating the `calibrie.Pipeline` class, passing it a dictionary where keys are custom task names and values are instances of `calibrie.Task` subclasses.

```python
import calibrie as cal
import pandas as pd

# 1. Define paths to controls and beads
# (Replace with your actual file paths or logic to find them)
controls_dict = {
    'EBFP2': 'path/to/data/raw_data/ebfp2_control.fcs',
    'MNEONGREEN': 'path/to/data/raw_data/mneongreen_control.fcs',
    'MKO2': 'path/to/data/raw_data/mko2_control.fcs',
    '1XIRFP720': 'path/to/data/raw_data/1xirfp720_control.fcs', # Example control names
    'ALL': 'path/to/data/raw_data/all_control.fcs',
    'BLANK': 'path/to/data/raw_data/blank_control.fcs' # Or 'CNTL', 'EMPTY' etc.
}
beads_file = 'path/to/data/raw_data/beads.fcs'

# 2. Instantiate Tasks with Parameters
# Note: Parameter values here mirror the example YAML configuration.

# Optional Gating (can be loaded from a saved YAML if preferred)
# gating_task = cal.GatingTask(...)

load_controls_task = cal.LoadControls(
    color_controls=controls_dict,
    use_channels=[
        'PACIFIC_BLUE_A', # EBFP2
        'FITC_A',         # MNEONGREEN
        'PE_TEXAS_RED_A', # MKO2
        'APC_ALEXA_700_A' # 1XIRFP720
        # Add other channels used for unmixing if different from reference channels
    ],
    use_reference_channels={ # Explicitly define, or let LoadControls try to guess
        'EBFP2': 'PACIFIC_BLUE_A',
        'MNEONGREEN': 'FITC_A',
        'MKO2': 'PE_TEXAS_RED_A',
        '1XIRFP720': 'APC_ALEXA_700_A',
    },
    priority=10
)

lincomp_task = cal.LinearCompensation(
    mode='weighted', # Or 'simple'
    channel_weight_attenuation_power=1.75,
    unmix_controls=True,
    priority=20
)

protmap_task = cal.ProteinMapping(
    reference_protein='EBFP2', # Must match a name in controls_dict
    logspace=True,
    model_resolution=10,
    model_degree=2,
    apply_mef_transform=True, # Set to True to get MEF output
    priority=40
)

# MEF Calibration setup - needed by ProteinMapping if apply_mef_transform=True
# Loads bead info and makes the transform function available in the context
mef_setup_task = cal.MEFBeadsTransform(
    beads_data=beads_file,
    # These often point to package data or user-defined YAMLs
    channel_units='pkg:calibrie:config/beads/fortessa/spherotech_urcp-100-2H@channel_units',
    beads_mef_values='pkg:calibrie:config/beads/fortessa/spherotech_urcp-100-2H@mef_values',
    ignore_channels_with_missing_units=True,
    apply_on_load=False, # IMPORTANT: Keep False unless you have a specific reason
    priority=5 # Run early to provide context for ProteinMapping
)

export_task = cal.PandasExport(
    context_field_to_export='abundances_MEF', # Or 'abundances_mapped', 'abundances_AU'
    priority=1000 # Run last
)

# 3. Create the Pipeline Instance
pipeline = cal.Pipeline(
    tasks={
        # "gating": gating_task, # Optional
        "mef_setup": mef_setup_task,
        "controls": load_controls_task,
        "lincomp": lincomp_task,
        "protmap": protmap_task,
        "export": export_task,
    },
    name="MyPythonPipeline",
    loglevel="INFO" # Set logging level (DEBUG, INFO, WARNING, ERROR)
)

## Running the Pipeline

# 1. Initialization (Load controls, calculate spillover, fit models, etc.)
print("Initializing pipeline...")
pipeline.initialize()
print("Initialization complete.")
# The pipeline._context now holds initialized data (spillover matrix, models, etc.)

# 2. Process Sample Data
print("Processing sample data...")
# Use calibrie's loader, which handles FCS/CSV and column selection/escaping
# Specify only the channels needed for the *first* processing step (usually LoadControls)
sample_data_raw = cal.load_to_df(
    "path/to/data/raw_data/my_sample.fcs",
    column_order=load_controls_task.use_channels
)

# Apply the full pipeline to the sample data
result_context = pipeline.apply_all(sample_data_raw)
print("Processing complete.")

# 3. Access Results
# The result_context contains the final state after all tasks.
# The export task specifically places the desired DataFrame under 'output_df'
final_dataframe = result_context.output_df # This is abundances_MEF in this case

print("\nFinal Calibrated Data (head):")
print(final_dataframe.head())

# You can also access intermediate results stored in the pipeline's context
# print("Spillover Matrix:\n", pipeline._context.spillover_matrix)

# 4. Generate Diagnostics (Optional)
print("Generating diagnostics...")
diagnostic_figures = pipeline.all_diagnostics() # List[DiagnosticFigure]

output_dir = pathlib.Path("output/python_pipeline_diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)
for diag_fig in diagnostic_figures:
    savename = output_dir / f"{diag_fig.source_task}_{diag_fig.name.replace(' ', '_')}.png"
    diag_fig.fig.savefig(savename, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic: {savename}")
print("Diagnostics complete.")

```

This approach gives you fine-grained control over task instantiation and allows complex logic around pipeline execution. Remember to ensure the `priority` attributes are set correctly to control the execution order if it matters (which it usually does!).
