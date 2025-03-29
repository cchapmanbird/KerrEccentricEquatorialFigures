import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

level_list = np.linspace(-10, 0, 11)
#level_list = [ 1e-11,1e-9,1e-7, 1e-5, 1e-3]
#tick_list = [ '$10^{-11}$', '$10^{-9}$', '$10^{-7}$',  '$10^{-5}$',  '$10^{-3}$']
vmin = min(level_list)
vmax = max(level_list)
levels = np.array(level_list)

label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16

# Load amplitude data from the HDF5 file
with h5py.File('amplitude_diff.h5', 'r') as f:
    ps = f.attrs['pvals']
    es = f.attrs['evals']
    ellvec = f.attrs['ellvec']
    mvec = f.attrs['mvec']
    nvec = f.attrs['nvec']
    for ell in ellvec:
        for m in range(-ell, ell + 1):
            for n in nvec:
                amps_diff = f[f'amp_diff_{ell}{m}{n}'][...]
                name = '(' + str(ell) + ',' + str(m) + ',' + str(n) + ')'
                real_diff = np.log10(np.abs(amps_diff.real)).reshape(len(ps), len(es))
                imag_diff = np.log10(np.abs(amps_diff.imag)).reshape(len(ps), len(es))

                fig, axs = plt.subplots(2, 1, figsize=(6, 10))  # Create 1x2 subplots
                ax1 = axs[0]
                ax2 = axs[1]

                contourf1 = ax1.contourf(ps, es, real_diff.T, cmap='plasma', levels=levels, vmin=vmin, vmax=vmax)
                ax1.set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
                ax1.set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
                # ax1.set_title(rf'$| 1 - f_p^{{0PA}}/ f_p^{{{name}}} |$', fontsize=title_fontsize)
                # ax1.set_title(f'Real of $A_{{{ell,m,n}}}$ l={ell}, m={m}, n={n}', fontsize=title_fontsize)
                ax1.set_title(rf'$\log_{{10}}| \Im ( A_{{{name}}} -  A_{{{name}}} ) |$', fontsize=title_fontsize)

                ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

                contourf2 = ax2.contourf(ps, es, imag_diff.T, cmap='plasma', levels=levels, vmin=vmin, vmax=vmax)
                ax2.set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
                ax2.set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
                # ax2.set_title(rf'$ \log_{{10}} \left(| 1 - f_e^{{0PA}}/ f_e^{{{name}}} | \right)$', fontsize=title_fontsize)
                ax2.set_title(rf'$\log_{{10}}| \Re (A_{{{name}}} -  A_{{{name}}} ) |$', fontsize=title_fontsize)
                ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

                cbar = fig.colorbar(contourf1, ax=ax1, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
                cbar.ax.tick_params(labelsize=tick_fontsize)
                cbar = fig.colorbar(contourf2, ax=ax2, orientation='vertical', fraction=0.2, pad=0.01, ticks=level_list)
                cbar.ax.tick_params(labelsize=tick_fontsize)

                plt.tight_layout()
                plt.savefig(f"{name}_Comparison.pdf")
