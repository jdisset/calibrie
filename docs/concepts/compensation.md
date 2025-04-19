# Concepts: Linear Compensation (Unmixing)

Fluorescent proteins used in synthetic biology often have broad emission spectra. This means the light emitted by one protein (e.g., EYFP) might be detected not only in its primary channel (e.g., FITC-A) but also spill over into channels intended for other proteins (e.g., PE-A). This phenomenon is called **spectral overlap** or **bleed-through**.

Linear compensation (often called unmixing) is the process of computationally correcting for this overlap to estimate the true signal originating from each individual fluorophore.

## The Linear Model

The core assumption of linear compensation (aka "unmixing") is that the total signal measured in a particular detector channel (`Yi`) is a linear combination of:

1.  **Autofluorescence (`Ai`):** Background signal from the cell itself in that channel, independent of the expressed fluorescent proteins.
2.  **Primary Signal:** The signal from the fluorophore primarily intended for that channel (`fi`).
3.  **Bleed-through:** Contributions from all _other_ fluorophores (`fj`) spilling into channel `i`.

This is typically expressed mathematically as:

`Yi = Ai + S_ii * fi + Σ(S_ji * fj)` for `j ≠ i`

Where:

- `Yi`: Measured fluorescence intensity in channel `i`.
- `Ai`: Autofluorescence component in channel `i`.
- `fi`: True, unobserved fluorescence intensity originating _only_ from the protein primarily associated with channel `i`.
- `fj`: True, unobserved fluorescence intensity originating _only_ from the protein primarily associated with channel `j`.
- `S_ji`: The **spillover coefficient**, representing the fraction of signal from protein `j` that is detected in channel `i`. `S_ii` is typically normalized to 1 (or represents the primary signal efficiency).

We usually use a matrix representation for this model, where:

- `Y` is the vector of observed fluorescence intensities across all channels.
- `A` is the vector of autofluorescence intensities across all channels.
- `S` is the **spillover matrix**, where `S_ij` is the contribution of protein `j`'s signal to channel `i`'s measurement.
- `X` is the vector of true underlying fluorescence signals (the quantities of each protein).

Which gives us the equation:
`Y = S * X + A`

The goal of unmixing is to estimate `X` (the quantities of each protein), from `Y` (the observations in each channel) and estimates of `A` and `S`:

`X = S⁻¹ * (Y - A)`

Where `S⁻¹` is the inverse of the spillover matrix. This equation allows us to estimate the true abundances of each protein in the sample.

## How `LinearCompensation` Works

The `calibrie.LinearCompensation` task implements this process:

1.  **Requires Context:** It needs `controls_values`, `controls_masks`, `channel_names`, `protein_names`, `reference_channels` (usually from `LoadControls`). It also benefits from channel quality metrics (`channel_signal_to_noise`, `channel_specificities`) if available.

2.  **Estimate Autofluorescence (`A`):** It calculates the median signal in each channel from the 'blank' control samples

3.  **Estimate Spillover Matrix (`S`):** This is the crucial step. It uses the _single-color controls_. For each control (e.g., cells expressing only EYFP):

    - It subtracts the estimated autofluorescence (`A`) from the measurements.
    - It assumes that the signal in the _reference channel_ for that protein (e.g., FITC-A for EYFP, defined in `LoadControls`) is proportional to the true amount of that protein (`X_EYFP`).
    - It calculates how much signal appears in _other_ channels (e.g., PE-A) relative to the signal in the reference channel. This ratio gives the spillover coefficient (e.g., `S_EYFP_to_PE-A`).
    - This is done using least-squares regression. The default `mode='weighted'` option uses channel quality metrics (SNR, specificity) provided by `LoadControls` to give more importance to reliable channels when calculating these coefficients, making the spillover estimation more robust.

4.  **Produces Context:**

    - `autofluorescence`: The estimated autofluorescence vector `A`.
    - `spillover_matrix`: The calculated spillover matrix `S`.
    - `channel_weights`: The weights used if `mode='weighted'`.
    - `controls_abundances_AU`: If `unmix_controls=True`, the control data itself, unmixed.
    - `F_mat`: A matrix representation suitable for potential downstream non-linear steps (though not typically used if only linear compensation is performed).

5.  **Processing Samples:** In its `process` method, it takes raw sample observations (`observations_raw`), subtracts the stored `autofluorescence`, and solves the linear system (`X = S⁻¹ * Y'`) using the stored `spillover_matrix` to produce `abundances_AU` (abundances in Arbitrary Units).

Linear compensation is a fundamental step to disentangle mixed fluorescence signals, providing a more accurate estimate of the contribution of each individual fluorescent reporter _before_ attempting any further normalization or calibration. The main end result is a set of abundances in Arbitrary Units (AU) for each protein in the sample, which can be used for downstream analysis or visualization.
