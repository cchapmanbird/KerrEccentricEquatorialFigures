import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load data from HDF5 file
h5_file = "trajectory_data.h5"
with h5py.File(h5_file, "r") as f:
    groups = list(f.keys())
    results = []
    labels = []
    for group_name in groups:
        group = f[group_name]
        delta_phi_values = group["delta_phi_values"][:]
        plunge_mask = group["plunge_mask"][:]
        p0_vec = group["p0_vec"][:]
        e0_vec = group["e0_vec"][:]
        M = group["M"][()]
        mu = group["mu"][()]
        a = group["a"][()]
        results.append((delta_phi_values, plunge_mask))
        labels.append((M, mu, a))

results = results[:3]
# ===================== GENERATE PLOT =====================
fig, ax = plt.subplots(1, len(results), figsize=(16, 7), sharey=True)

for k, (delta_phi_values, plunge_mask) in enumerate(results):
    P0, E0 = np.meshgrid(p0_vec, e0_vec)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("grey")

    pcm = ax[k].pcolormesh(P0, E0, delta_phi_values, shading='auto', cmap=cmap, norm=LogNorm(vmin=1e-5, vmax=1e6))
    plunge_contour = ax[k].contour(P0, E0, plunge_mask, levels=[0.5], colors='red', linewidths=1, linestyles='dashed')
    ax[k].clabel(plunge_contour, fmt={0.5: "\u00A0Plunge\u00A0"}, colors='red', fontsize=15)

    contour_levels = [0.1, 1]
    contours = ax[k].contour(P0, E0, delta_phi_values, levels=contour_levels, colors='white', linewidths=1, linestyles='dashed')
    ax[k].clabel(contours, fmt={0.1: r"$0.1$", 1: r"$1$"}, colors='white', fontsize=15)

    ax[k].set_xlabel(r'$p_0$', fontsize=20)
    if k == 0:
        ax[k].set_ylabel(r'$e_0$', fontsize=15)
    ax[k].set_title(rf'$(M/M_\odot, \mu/M_\odot, a) = ({labels[k][0]}, {labels[k][1]}, {labels[k][2]})$', fontsize=14)

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.15)
cbar.set_label(r'$|\Phi^{(Kerr)}_{\phi,-1} - \Phi^{(5PN)}_{\phi,-1}|$', fontsize=25)
cbar.ax.tick_params(labelsize=20)

# Save and show the plot
plot_dir = "./"
plot_filename = "dephasing_PN_vs_Kerr_masses_w_contours_w_plunge_contours.png"
plt.savefig(plot_dir + plot_filename, bbox_inches="tight")
plt.show()