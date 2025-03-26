import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]



level_list = [ -11,-10,-9,-8,-7,-6, -5,-4, -3]
#level_list = [ 1e-11,1e-9,1e-7, 1e-5, 1e-3]
#tick_list = [ '$10^{-11}$', '$10^{-9}$', '$10^{-7}$',  '$10^{-5}$',  '$10^{-3}$']
vmin = min(level_list)
vmax = max(level_list)
levels = np.array(level_list)

label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16

fig, axs = plt.subplots(2, 1, figsize=(6,10))  # Create 1x2 subplots
ax1 = axs[0]
ax2 = axs[1]

pdotsRelDiffLoaded = np.loadtxt("PNComparisonPdot.txt")
edotsRelDiffLoaded = np.loadtxt("PNComparisonEdot.txt")
ps = np.loadtxt("PNComparisonPs.txt")
es = np.loadtxt("PNComparisonEs.txt")

contourf1 = ax1.contourf(ps, es, pdotsRelDiffLoaded, cmap='plasma',levels=levels,vmin=vmin, vmax=vmax)
ax1.set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
ax1.set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
ax1.set_title(r'$ \log_{10} \left(| 1 - f_p^{0PA}/ f_p^{5PN} | \right)$', fontsize=title_fontsize)

ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

contourf2 = ax2.contourf(ps, es, edotsRelDiffLoaded, cmap='plasma',levels=levels,vmin=vmin, vmax=vmax)
ax2.set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
ax2.set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
ax2.set_title(r'$ \log_{10} \left(| 1 - f_e^{0PA}/ f_e^{5PN} | \right)$', fontsize=title_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)


cbar = fig.colorbar(contourf1, ax=ax1, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
cbar.ax.tick_params(labelsize=tick_fontsize)
cbar = fig.colorbar(contourf2, ax=ax2, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
cbar.ax.tick_params(labelsize=tick_fontsize)

plt.tight_layout()
plt.savefig("ComparisonWith5PN.pdf")