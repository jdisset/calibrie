
# Concepts: MEF Calibration

After linear compensation and protein mapping, we have protein abundances expressed in arbitrary units relative to a chosen reference protein (e.g., "AU relative to EBFP2"). The final step is often to convert these relative units into **absolute, standardized units** that can be compared across different experiments, instruments (with caveats), and labs. The most common standard unit in flow cytometry is **Molecules of Equivalent Fluorophore (MEF)**.

## Why MEF?

MEF units quantify the fluorescence intensity of a cell or particle in terms of "how many molecules of a specific, standardized fluorophore (like fluorescein or PE) would be needed to produce the same intensity on this specific channel under these specific instrument settings?"

## Calibration Beads

Achieving MEF calibration requires **calibration beads** (like the Spherotech Rainbow or Ultra Rainbow particles). These beads have key properties:

*   **Multiple Populations:** They contain several distinct populations (e.g., 8 or 9 peaks) with precisely controlled, increasing amounts of embedded stable fluorophores.
*   **Known MEF Values:** The manufacturer (e.g., Spherotech) has characterized each bead population (for a specific product type and lot number) and assigned it MEF values for various standard channels (e.g., MEFL for FITC channel, MEPE for PE channel, MEAPC for APC channel). These values are provided in datasheets or configuration files.

## How `MEFCalibration` / `MEFBeadsTransform` Work

The `calibrie.MEFCalibration` and `calibrie.MEFBeadsTransform` tasks use these beads to establish the link between your instrument's arbitrary units and the standard MEF scale. While they have slightly different roles in the example pipeline, the core calibration logic is similar:

1.  **Requires Context/Input:** Needs bead data (`beads_data`), mappings between your instrument channels and the manufacturer's units (`channel_units`), and the manufacturer's MEF values for those units (`beads_mef_values`). `MEFCalibration` also typically requires the mapped abundances (`abundances_mapped`) and information about the reference protein/channel from previous steps. `MEFBeadsTransform` can operate more independently or earlier.
2.  **Process Bead Data:**
    *   Load the bead data file (FCS).
    *   **Identify Bead Peaks:** Locate the distinct bead populations in the measured data. Calibrie uses sophisticated methods involving Optimal Transport (`ott-jax`) and Kernel Density Estimation (KDE) to assign individual bead events to the most likely population peak and find the precise center (in AU) of each peak across relevant channels. This is more robust than simple gating or 1D peak finding.
    *   **Fit Calibration Function:** For a specific channel chosen for calibration (in `MEFCalibration`, this is the reference channel for the reference protein; `MEFBeadsTransform` can do this for multiple specified channels), fit a function that maps the measured AU of the bead peak centers to the known MEF values provided by the manufacturer for that channel/unit. Calibrie uses a flexible stacked polynomial spline, often fitted in log-space (`logspace=True`), to capture potential non-linearities in the instrument's response. Factors like peak resolution (`relative_density_threshold`) and overlap (`_peak_weights`) can influence the fit quality.
3.  **Produces Context / Function:**
    *   `MEFBeadsTransform` (when `apply_on_load=False`): Primarily stores the fitted calibration functions and makes a `transform_to_MEF` function available in the context. This function can later be used (e.g., by `ProteinMapping`) to convert specific channel data to MEF.
    *   `MEFCalibration`: Uses the context (including `abundances_mapped` and the reference channel info) and the fitted calibration function for the reference channel/unit to convert the *mapped* abundances into MEF units, producing `controls_abundances_MEF` (during initialize) and `abundances_MEF` (during process). It also stores the `calibration_units_name`.
4.  **Processing Samples (`MEFCalibration`):** Takes the `abundances_mapped` (which are in AU relative to the reference protein) for a sample, applies the *single* calibration function derived for the reference channel/unit, and outputs `abundances_MEF`.

## Result: Calibrated Abundances (MEF)

The final output (`abundances_MEF`) represents the estimated protein abundances in MEF units. Crucially, because the input was already mapped relative to the reference protein, the output MEF units are all effectively **"MEF units equivalent to the reference fluorophore (e.g., MEFL or MEPE corresponding to the reference channel), scaled as if produced by the reference protein (e.g., EBFP2)."**

This provides a standardized, quantitative measure that accounts for spectral overlap, relative protein scale differences, and instrument response, anchored to a physical bead standard. While spectral differences between instruments can slightly alter the absolute meaning of "MEF" for a given channel (see notes in previous discussion), this calibrated unit is highly reproducible on a given setup and comparable across similarly configured instruments, achieving the primary goal of standardization for synthetic biology characterization.
