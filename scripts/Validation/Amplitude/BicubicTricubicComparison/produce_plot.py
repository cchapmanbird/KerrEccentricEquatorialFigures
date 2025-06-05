import numpy as np
import matplotlib.pyplot as plt
from multispline.spline import TricubicSpline
import h5py
from few.utils.mappings.kerrecceq import kerrecceq_backward_map, z_of_a, a_of_z, p_of_u, e_of_uwz
from few.utils.geodesic import get_separatrix

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

few_mode_amp, tric_mode_amp = np.load('A220_ecc_slice.npy')
apex = np.load('A220_ecc_pars.npy')
sep = apex[-1]

uplot = np.linspace(0,1,501)
wplot = np.ones(501)*0.5
zplot = z_of_a(0.998)
eplot = e_of_uwz(uplot, wplot, np.ones(501)*zplot, 3)
pseps = get_separatrix(0.998, eplot, np.ones_like(eplot))
pplots = p_of_u(uplot, pseps, 1/3)

plt.figure(figsize=(3.,4.5), dpi=150)
plt.subplot(211)
plt.scatter(apex[1] - sep, apex[2], c=np.log10(np.abs(1 - (few_mode_amp) / (tric_mode_amp))), vmin=-3, vmax=0, s=3, rasterized=True, cmap='plasma')
# plt.text(0.05, 0.9, r'$a=0.998$', transform=plt.gca().transAxes, fontsize=12)
plt.plot(pplots - pseps, eplot, c='turquoise', ls='--', lw=1, label=r'$w = 0.5$')
plt.xscale('log')
plt.xlim(1e-3, 10)
plt.ylim(0, 0.9)
plt.ylabel(r'$e$')
plt.tick_params(axis='x', labelbottom=False)
plt.legend(frameon=False, loc='upper left')

few_mode_amp, tric_mode_amp = np.load('A220_spin_slice.npy')
apex = np.load('A220_spin_pars.npy')
sep = apex[-1]

plt.subplot(212)
sc = plt.scatter(apex[1] - sep, 1 - apex[0], c=np.log10(np.abs(1 - (few_mode_amp) / (tric_mode_amp))), vmin=-3, vmax=0, s=3, rasterized=True, cmap='plasma')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$1-a$')
plt.xlabel(r'$p - p_{\mathrm{sep}}$')
plt.xlim(1e-3, 10)
plt.ylim(1e-3, 2)

plt.tight_layout()
plt.subplots_adjust(hspace=0.08)

# add colorbar on right-hand side of figure, with its own axes
cbar_ax = plt.gcf().add_axes([0.95, 0.17, 0.03, 0.75])
cbar = plt.colorbar(sc, cax=cbar_ax, label=r'$\log_{10} \left| 1 - \frac{\mathcal{A}_{220}^{\mathrm{BIC}}}{\mathcal{A}_{220}^{\mathrm{TRI}}}\right|$')
plt.savefig('bic_tric_comp_220.pdf', bbox_inches='tight')
plt.show()
