
def pick_calibration_units(self):
    reference_protein_id = self.protein_names.index(self.reference_protein)
    reference_channel_id = self.reference_channels[reference_protein_id]
    reference_channel_name = self.channel_names[reference_channel_id]
    # check that the reference protein's reference channel is available
    if self.channel_to_unit[reference_channel_name] is None:
        raise ValueError(
            f'Cannot calibrate with {reference_channel_name} because its unit is unknown'
        )
    if reference_channel_name not in self.beads_channel_order:
        raise ValueError(
            f'Cannot calibrate with {reference_channel_name} because it is not selected for beads calibration'
        )
    self.calibration_units = self.channel_to_unit[self.channel_names[reference_channel_id]]
    self.calibration_channel_name = reference_channel_name
    self.calibration_channel_id = self.beads_channel_order.index(reference_channel_name)

    if self.verbose:
        print(f'Calibrated abundances will be expressed in {self.calibration_units}')


def map_proteins_to_reference(self, unmix_method):
    if self.verbose:
        print(f'Computing all protein mapping to {self.reference_protein}...')
    all_masks = self.controls_masks.sum(axis=1) > 1
    Y = self.controls_values[all_masks] - self.autofluorescence
    selection = self.filter_saturated(Y, only_in_reference_channel=True)
    Y = Y[selection]
    X = unmix_method(Y) + self.offset
    refprotid = self.protein_names.index(self.reference_protein)
    X = X[jnp.all(X > 1, axis=1)]
    logX = logtransform(X, self.log_scale_factor)
    self.cmap_transform, self.cmap_inv_transform = cmap_spline(logX, logX[:, refprotid])

def plot_mapping_diagnostics(
    self, scatter_size=20, scatter_alpha=0.2, max_n_points=15000, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plot transform
    NP = len(self.protein_names)
    all_masks = self.controls_masks.sum(axis=1) > 1
    Y = self.controls_values[all_masks] - self.autofluorescence
    X = Y @ jnp.linalg.pinv(self.spillover_matrix)  # apply bleedthrough
    X += self.offset  # add offset
    logX = logtransform(X, self.log_scale_factor, 0)
    # logX = logX[jnp.all(logX > self.clamp_values[0], axis=1)]
    # logX = logX[jnp.all(logX < self.clamp_values[1], axis=1)]
    refid = self.protein_names.index(self.reference_protein)
    target = logX[:, refid]

    scatter_x = []
    scatter_y = []
    scatter_color = []

    for i in range(NP):
        fpname = self.protein_names[i]
        color = get_bio_color(fpname)
        xrange = (logX[:, i].min(), logX[:, i].max())
        xx = np.linspace(*xrange, 200)
        yy = self.cmap_transform(np.tile(xx, (NP, 1)).T)[:, i]
        scatter_x.append(logX[:, i].flatten())
        scatter_y.append(target.flatten())
        scatter_color.append([color] * len(logX))
        ax.plot(xx, yy, alpha=1, lw=5, color='w')
        if i == refid:
            fpname += ' (reference)'
        ax.plot(xx, yy, label=fpname, alpha=1, lw=1.5, color=color, zorder=10)

    scatter_x = np.concatenate(scatter_x)
    scatter_y = np.concatenate(scatter_y)
    scatter_color = np.concatenate(scatter_color)
    NPOINTS = min(max_n_points, len(scatter_x))
    order = np.random.permutation(len(scatter_x))[:NPOINTS]

    ax.scatter(
        scatter_x[order],
        scatter_y[order],
        c=scatter_color[order],
        s=scatter_size,
        alpha=scatter_alpha,
        lw=0,
        zorder=0,
    )

    yrange = (target.min(), target.max())
    xrange = (scatter_x.min(), scatter_x.max())
    ax.set_ylim(yrange)
    ax.set_xlim(xrange)
    ax.legend()
    ax.set_xlabel('source')
    ax.set_ylabel('target')
    # no grid, grey background
    ax.grid(False)
    ax.set_title(f'Color mapping to {self.reference_protein}')
    plt.show()

