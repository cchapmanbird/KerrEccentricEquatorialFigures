import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Need to convert seaborn colors to matplotlib colors
colorblind_palette = sns.color_palette('colorblind')
colorblind_hex = [mcolors.to_hex(color) for color in colorblind_palette]

# General Plot Settings
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams['axes.labelsize'] = 14

misms_all  = np.load('downsample_mismatch_arr.npy')[:9501]

misms = np.log10(misms_all[:,-3:])

retain = (misms_all[:,4] < 1) & (misms_all[:,2] < 1)
misms = misms[retain]
misms_all = misms_all[retain]

extrapolated = np.array([np.polynomial.Polynomial.fit([1, 2], misms[i,:2], deg=1)(0) for i in range(misms.shape[0])])

bins = np.arange(-10, np.log10(1), 0.25)

plt.figure(figsize=(5,3))
plt.hist(misms[:,0], bins=bins, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="1/2 grid points")
plt.hist(misms[:,1], bins=bins, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="1/4 grid points")
plt.hist(misms[:,2], bins=bins, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="1/8 grid points")
plt.hist(extrapolated, bins=bins, histtype='stepfilled', facecolor='none', linestyle='--', edgecolor=colorblind_hex[3], linewidth=2, label="Extrapolated")

plt.xlabel(r'$\log_{10}(\mathcal{M})$')
plt.ylabel('Count')
plt.legend(frameon=False)
plt.savefig("mismatch_histogram.pdf", bbox_inches='tight')

vmin = -8
vmax = 0

# mism_plot = extrapolated
mism_plot = misms[:,0]

plt.figure(figsize=(3.5,6), dpi=200)
plt.subplot(2, 1, 1)
plt.scatter(misms_all[:,4], misms_all[:,3], s=8, c = mism_plot, cmap='plasma', vmin=vmin, vmax=vmax, rasterized=True)
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel(r'$p_0$')

plt.subplot(2, 1, 2)
sc = plt.scatter(misms_all[:,4], misms_all[:,2], s=8, c = mism_plot, cmap='plasma', vmin=vmin, vmax=vmax, rasterized=True)
plt.ylabel(r'$a$')
plt.xlabel(r'$e_0$')

plt.tight_layout()

# one colorbar for both plots, on its own axes
cbar_ax = plt.gcf().add_axes([1.02, 0.13, 0.03, 0.82])
cbar = plt.colorbar(sc, cax=cbar_ax)
cbar.set_label(r'$\mathcal{M}(h^\mathrm{1DS}, h^\mathrm{2DS})$')
cbar.ax.set_yticklabels([f'$10^{{{int(i):d}}}$' for i in cbar.ax.get_yticks()])

plt.savefig("mismatch_scatter.pdf", bbox_inches='tight')
# plt.show()

# plt.figure()
# sc = plt.scatter(misms_all[:,0], misms_all[:,1], s=30, c = mism_plot, cmap='plasma', vmin=vmin, vmax=vmax)
# plt.ylabel('$m_1$')
# plt.xlabel('$m_2$')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()