
# Architecture: Pipelines, Tasks, and Context

Calibrie is designed around a modular architecture to promote flexibility and reusability. The core components are Pipelines, Tasks, and the Context object.

## Tasks (`calibrie.pipeline.Task`)

A `Task` represents a distinct step in the data processing or analysis workflow. Examples include loading control data (`LoadControls`), performing linear compensation (`LinearCompensation`), or calibrating to MEF units (`MEFCalibration`).

Key characteristics of a Task:

*   **Inheritance:** Tasks inherit from the `calibrie.pipeline.Task` base class (which itself inherits from Pydantic's `BaseModel` for configuration handling).
*   **Methods:**
    *   `initialize(self, ctx: Context) -> Context`: Called once *before* processing any sample data. Used for setup that relies on controls or initial configuration (e.g., calculating a spillover matrix, loading bead data, fitting calibration models). It takes the current `Context` and returns an updated `Context` with any data it produces (e.g., the spillover matrix).
    *   `process(self, ctx: Context) -> Context`: Called for *each* sample dataset being processed. Performs the main work of the task on the sample data (e.g., applying compensation, applying calibration). It uses data from the `Context` (both initialization data and results from previous `process` steps) and returns an updated `Context` with its results (e.g., calibrated abundances).
    *   `diagnostics(self, ctx: Context, **kw) -> Optional[List[DiagnosticFigure]]`: Generates plots or other diagnostic information to visualize the task's inputs, outputs, or internal state. Returns a list of `DiagnosticFigure` objects or `None`.
*   **Configuration:** Tasks are configured using parameters defined as Pydantic fields (e.g., `reference_protein` in `ProteinMapping`). These can be set when creating the task instance in Python or defined in YAML configuration files.
*   **Priority:** An optional `priority` field (float, lower runs first) helps control the execution order within a pipeline if not explicitly ordered otherwise.

## Pipeline (`calibrie.pipeline.Pipeline`)

A `Pipeline` manages a collection of named `Task` instances and orchestrates their execution.

Key responsibilities:

*   **Task Collection:** Holds a dictionary mapping user-defined names to `Task` instances.
*   **Ordering:** Sorts tasks based on their `priority` before execution.
*   **Context Management:** Maintains a central `Context` object. It passes the context to each task during initialization and processing, updating it with the results from each step.
*   **Execution Flow:**
    1.  Calls `initialize()` on each task sequentially, updating the context.
    2.  For each input sample dataset, calls `process()` on each task sequentially, again updating the context.
*   **Diagnostics:** Provides methods (`diagnostics`, `all_diagnostics`) to trigger the generation of diagnostic figures from individual tasks or all tasks in the pipeline.
*   **Configuration:** Can be configured with parameters like `name` and `loglevel`.

## Context (`calibrie.utils.Context`)

The `Context` acts as the central data bus for the pipeline. It's essentially a specialized dictionary that carries information between tasks.

Key features:

*   **Data Storage:** Holds key-value pairs where keys are strings (e.g., `'controls_values'`, `'spillover_matrix'`, `'abundances_MEF'`) and values can be any Python object (often NumPy arrays, Pandas DataFrames, fitted models, etc.).
*   **Attribute Access:** Allows accessing items using dot notation (e.g., `ctx.spillover_matrix`) in addition to dictionary-style access (`ctx['spillover_matrix']`).
*   **Provenance Tracking:** (As implemented in `calibrie.utils`) It can optionally store the history of which task and function generated or modified each key. This is extremely valuable for debugging complex pipelines, allowing you to trace where a specific piece of data came from.
*   **Passing:** The *same* `Context` object instance is passed through the `initialize` methods of all tasks. A *copy* or updated version of the context is typically passed through the `process` methods for each sample, potentially isolating sample-specific results.

## Workflow Summary

1.  Define a set of `Task` instances, configuring their parameters.
2.  Create a `Pipeline` instance, providing the tasks.
3.  Call `pipeline.initialize()`. This runs the `initialize` method of each task in order, building up the initial context (e.g., control data analysis, model fitting).
4.  For each sample data file:
    *   Load the sample data.
    *   Call `pipeline.apply_all(sample_data)`. This runs the `process` method of each task in order on the sample data, using the initialized context and passing results forward via the context.
    *   The final context returned by `apply_all` contains the end results for that sample.
5.  Optionally, call `pipeline.all_diagnostics()` to generate visualizations.

This architecture promotes modularity by encapsulating logic within Tasks and decoupling them via the Context, making pipelines easier to build, modify, and understand.
