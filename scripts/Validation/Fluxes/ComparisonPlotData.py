import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]



level_list = [ -11,-10,-9,-8,-7,-6, -5,-4, -3, -2]
#level_list = [ 1e-11,1e-9,1e-7, 1e-5, 1e-3]
#tick_list = [ '$10^{-11}$', '$10^{-9}$', '$10^{-7}$',  '$10^{-5}$',  '$10^{-3}$']
vmin = min(level_list)
vmax = max(level_list)
levels = np.array(level_list)

label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16
for name in ["SchwarzEccFlux", "PN5"]:
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))  # Create 1x2 subplots
    ax1 = axs[0]
    ax2 = axs[1]

    pdotsRelDiffLoaded = np.loadtxt(f"{name}_ComparisonPdot.txt")
    edotsRelDiffLoaded = np.loadtxt(f"{name}_ComparisonEdot.txt")
    ps = np.loadtxt(f"{name}_ComparisonPs.txt")
    es = np.loadtxt(f"{name}_ComparisonEs.txt")

    contourf1 = ax1.contourf(ps, es, pdotsRelDiffLoaded, cmap='plasma', levels=levels, vmin=vmin, vmax=vmax)
    ax1.set_xlabel(rf'$p$', fontsize=label_fontsize)
    ax1.set_ylabel(rf'$e$', fontsize=label_fontsize)

    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    contourf2 = ax2.contourf(ps, es, edotsRelDiffLoaded, cmap='plasma', levels=levels, vmin=vmin, vmax=vmax)
    ax2.set_xlabel(rf'$p$', fontsize=label_fontsize)
    ax2.set_ylabel(rf'$e$', fontsize=label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    cbar = fig.colorbar(contourf1, ax=ax1, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(rf'$ \log_{{10}} \left| 1 - \hat{{f}}_p^{{\mathrm{{FEW}}}}/ \hat{{f}}_p^{{\mathrm{{{name}}}}} \right| $', fontsize=14)
    cbar = fig.colorbar(contourf2, ax=ax2, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(rf'$ \log_{{10}} \left| 1 - \hat{{f}}_e^{{\mathrm{{FEW}}}}/ \hat{{f}}_e^{{\mathrm{{{name}}}}} \right|$', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{name}_Comparison.pdf")
