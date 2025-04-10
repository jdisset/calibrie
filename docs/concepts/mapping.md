
# Concepts: Protein Mapping

After linear compensation, we have estimates of the signal originating from each fluorescent protein in **Arbitrary Units (AU)** (stored in `abundances_AU`). However, these AU scales are generally *not directly comparable* between different proteins.

## Why Mapping is Needed

1000 AU measured for EBFP2 (on the Pacific Blue channel) doesn't necessarily correspond to the same number of molecules or the same biological "strength" as 1000 AU measured for mNeonGreen (on the FITC channel). This is because:

*   **Intrinsic Brightness:** Proteins have different quantum yields and extinction coefficients.
*   **Maturation:** Fluorescent proteins take time to fold correctly and become fluorescent, and maturation rates can differ.
*   **Expression/Translation/Degradation:** Even if driven by identical promoters, cellular context might affect the final steady-state concentration differently for different protein sequences.
*   **Instrument Sensitivity:** The detectors (PMTs) and filters might have different efficiencies for the wavelengths emitted by each protein, even after linear compensation corrects for spectral *overlap*.

To compare the functional effect of different proteins (e.g., if EBFP2 reports on Input A and mNeonGreen reports on Output B, how does A relate to B?), we need to bring their signals onto a common, relative scale.

## The Mapping Strategy

Protein mapping aims to standardize these arbitrary scales by relating them all to a single, chosen **reference protein**. The `calibrie.ProteinMapping` task does this by:

1.  **Requires Context:** It needs the unmixed arbitrary units (`abundances_AU`), protein names (`protein_names`), and control masks (`controls_masks`) from previous steps. Crucially, it relies on having an **"all-color" control** sample where *all* the relevant fluorescent proteins were intended to be co-expressed (ideally from linked promoters or a setup where relative expression levels are somewhat consistent).
2.  **Select Reference:** You specify a `reference_protein` (e.g., 'EBFP2').
3.  **Analyze "All-Color" Control:** The task looks at the data from the "all-color" control cells. For these cells, it examines the relationship between the measured AU of the `reference_protein` and the measured AU of *each other protein*.
4.  **Fit Mapping Functions:** For each non-reference protein `P`, it fits a function `F_P` such that:
    `AU_reference ≈ F_P(AU_P)`
    This function essentially learns "how many AU of the reference protein typically correspond to a given AU measurement of protein P *when they are co-expressed*". The function is often a flexible one, like the stacked polynomial spline used in Calibrie (potentially fit in log-space, controlled by the `logspace` parameter), to capture potentially non-linear relationships.
5.  **Produces Context:**
    *   `controls_abundances_mapped`: The *control* data abundances after mapping functions have been applied.
    *   `reference_protein_name`, `reference_protein_id`: Identifiers for the chosen reference.
    *   (Implicitly stores the fitted mapping functions internally).
6.  **Processing Samples:** In its `process` method, it takes the `abundances_AU` for a sample and applies the corresponding fitted function `F_P` to each protein `P`'s column, producing `abundances_mapped`.

## Result: Mapped Abundances

The output `abundances_mapped` is still in arbitrary units, but now they are all expressed on the scale *relative to the reference protein*. For example, if EBFP2 is the reference, a value of 1000 in the mNeonGreen column of `abundances_mapped` means "a signal equivalent to what 1000 AU of EBFP2 typically represents under co-expression conditions".

This step standardizes the *relative biological signal strengths* before the final absolute calibration to physical units (MEF). It's a crucial intermediate step for making meaningful comparisons between the levels of different reporters within the same cell or experiment. The accuracy depends on how well the relative expression in the "all-color" control reflects the typical behavior.
