import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.colors import LogNorm
from seaborn import color_palette
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

results = h5py.File("Kerr_PN_trajectory_data.h5", "r")
p0_vec = results["p0_vec"][:]
e0_vec = results["e0_vec"][:]

# labels = [["1e5", "1"], ["1e6", "10"], ["1e7", "100"]]
labels=["10^5\,M_\odot", "10^6\,M_\odot", "10^7\,M_\odot"]
fig, ax = plt.subplots(1, 3, figsize=(9, 4.9), sharey=True)

for k in range(3):
    delta_phi_values = results[f"delta_phi_values_{k}"][:]
    plunge_mask = results[f"plunge_mask_{k}"][:]
    
    P0, E0 = np.meshgrid(p0_vec, e0_vec)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("grey")

    pcm = ax[k].pcolormesh(P0, E0, delta_phi_values, shading='auto', cmap=cmap, norm=LogNorm(vmin=1e-5, vmax=1e5), rasterized=True)
    plunge_contour = ax[k].contour(P0, E0, plunge_mask, levels=[0.5], colors='blue', linewidths=1, linestyles='dashed')
    ax[k].clabel(plunge_contour, fmt={0.5: "\u00A0Plunging\u00A0"}, colors='blue', fontsize=14, inline_spacing=40)

    contour_levels = [0.01, 0.1, 1]
    contours = ax[k].contour(P0, E0, delta_phi_values, levels=contour_levels, colors='white', linewidths=1, linestyles='dashed')
    ax[k].clabel(contours, fmt={0.01: r"$0.01$", 0.1: r"$0.1$", 1: r"$1$"}, colors='white', fontsize=14, inline_spacing=20)

    ax[k].set_xlabel(r'$p_0$', fontsize=14)
    if k == 0:
        ax[k].set_ylabel(r'$e_0$', fontsize=14)
    ax[k].set_title(rf'$m_1 = {labels[k]}$', fontsize=14)

plt.tight_layout()

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.2)
cbar.set_label(r'$\left|\Phi^{\mathrm{(Kerr)}}_\phi - \Phi^{\mathrm{(PN5)}}_\phi\right|$', fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.savefig("./dephasing_PN_vs_Kerr_masses_w_contours_w_plunge_contours.pdf", bbox_inches="tight")
plt.show()
