import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

f = h5py.File('mismatches_a0.895200.h5', 'r')

p = f['pv'][()]
e = f['ev'][()]
mism = f['mismatch'][()]
seps = f['separatrix'][()]

filt = ~np.isnan(mism)
pplot = (p - seps)[filt]
eplot = np.asarray(e)[filt]
mplot = np.log10(mism)[filt]
mplot[mplot< -9] = -9

plt.figure(dpi=150,figsize=(3,2.7))
plt.tricontourf(pplot, eplot, mplot, cmap='plasma', levels=[-9, -8, -7, -6, -5, -4, -3, -2])

cbar = plt.colorbar()
cbar.set_label(r'$\log_{10} \mathcal{M}_\mathrm{amp}$')
plt.ylabel(r'$e$')
plt.xlabel(r'$p - p_{\mathrm{sep}}$')
plt.text(0.05, 0.95, r'$a = 0.8952$', transform=plt.gca().transAxes, fontsize=12, va='top', ha='left')
plt.xscale('log')
plt.savefig('sah_amp_few_comp_contourf_lin.pdf', bbox_inches='tight')
plt.close()
