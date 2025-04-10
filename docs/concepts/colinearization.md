
# Concepts: Colinearization (Advanced)

!!! warning "Advanced Topic"
    Colinearization is an advanced pre-processing step that addresses specific non-ideal behaviors in fluorescence measurements. It may not be necessary for all experiments but can improve accuracy when linear compensation assumptions are violated.

## The Problem: Non-Linear Channel Relationships

Standard linear compensation assumes that if a single fluorophore (e.g., EYFP) emits light detected in two channels (e.g., primarily FITC-A, secondarily PE-A), the signal intensity in PE-A is *directly proportional* to the signal intensity in FITC-A. That is, `Signal_PE-A = constant * Signal_FITC-A`.

However, sometimes this isn't true due to:

*   **Instrument Non-linearities:** Detectors (PMTs) or amplifiers might have slightly non-linear responses, especially at the extremes of their range.
*   **Complex Reporter Physics:** Förster Resonance Energy Transfer (FRET) or complex photophysics within a reporter could lead to non-proportional emission in different channels.
*   **Subtle Spectral Shifts:** Cellular environment might slightly alter emission spectra in ways that affect different channels non-uniformly.

If the relationship is non-linear (e.g., `Signal_PE-A = f(Signal_FITC-A)` where `f` is not a simple multiplication), the standard spillover matrix calculation (which assumes linearity) and subsequent unmixing will be inaccurate.

## The Colinearization Strategy

The `calibrie.Colinearization` task attempts to correct this *before* linear compensation. Its goal is to find mathematical transformations (`T_channel`) for each channel such that the *transformed* signals *do* exhibit a linear relationship.

`T_PE-A(Signal_PE-A) ≈ constant * T_FITC-A(Signal_FITC-A)`

How it works:

1.  **Requires Context:** Needs single-color control data (`controls_values`, `controls_masks`, etc.) from `LoadControls`.
2.  **Analyze Control Data:** It examines the relationships between different channels *within each single-color control*. For example, in the EYFP control, it looks at how PE-A changes as FITC-A changes.
3.  **Pathfinding:** It determines an optimal sequence or "path" for applying transformations. Because transforming one channel might affect its relationship with others, it tries to find a sensible order (e.g., transform PE-A relative to FITC-A, then transform PE-TexasRed-A relative to the *already transformed* PE-A). This involves analyzing channel "information content" and signal ranges (`compute_possible_linearization_steps`, `find_best_linearization_path`).
4.  **Fit Transformations:** For each step in the path (e.g., relating PE-A to FITC-A using EYFP control data), it fits a transformation function (typically a log-poly spline similar to those in `MEFCalibration`) for the "source" channel (PE-A) that makes it best match the (potentially already transformed) "destination" channel (FITC-A).
5.  **Produces Context:** The primary output is a modified `cell_data_loader` function.
6.  **Processing/Loading:** When data is loaded *after* colinearization has been initialized, the modified `cell_data_loader` intercepts the raw data, applies the fitted transformations (`T_channel`) to the appropriate columns, and then passes the *transformed* data to subsequent tasks like `LinearCompensation`.

## Impact

By applying colinearization, the data fed into `LinearCompensation` better satisfies the linearity assumption. This can lead to:

*   More accurate spillover matrix calculation.
*   Cleaner separation of signals during unmixing (`abundances_AU`).
*   Potentially more accurate downstream mapping and calibration.

However, it adds significant complexity and computational cost during initialization. It's typically used when standard linear compensation diagnostics reveal significant non-linear "hooks" or curves in plots that should ideally be linear (like off-diagonal plots in single-color control diagnostics).
