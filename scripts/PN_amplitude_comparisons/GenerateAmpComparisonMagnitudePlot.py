import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

p1l2m2n0diffs=np.loadtxt('p1l2m2n0diffs.txt')
p1l2m2n1diffs=np.loadtxt('p1l2m2n1diffs.txt')
# plot_diffs(p1l2m2n0diffs,0)
# plot_diffs(p1l2m2n1diffs,1)

p2l2m2n0diffs=np.loadtxt('p2l2m2n0diffs.txt')
p2l2m2n1diffs=np.loadtxt('p2l2m2n1diffs.txt')
p2l6m4n2diffs=np.loadtxt('p2l6m4n2diffs.txt')
# plot_diffs2(p2l2m2n0diffs,2,2,0)
# plot_diffs2(p2l2m2n1diffs,2,2,1)
# plot_diffs2(p2l6m4n2diffs,6,4,2)

p3l2m2n0diffs=np.loadtxt('p3l2m2n0diffs.txt')
p3l2m2n1diffs=np.loadtxt('p3l2m2n1diffs.txt')
# plot_diffs3(p3l2m2n0diffs,0)
# plot_diffs3(p3l2m2n1diffs,1)


plt.figure(figsize=(4.5,5))
plt.subplot(221)
plt.scatter(p2l2m2n0diffs[:,0], p2l2m2n0diffs[:,1], c=p2l2m2n0diffs[:,4], vmin=-5, vmax=0, s=7, rasterized=True, cmap='plasma')
plt.ylabel('e')
plt.title(r'$(\ell, m, n) = (2, 2, 0)$')

plt.subplot(222)
plt.scatter(p2l2m2n1diffs[:,0], p2l2m2n1diffs[:,1], c=p2l2m2n1diffs[:,4], vmin=-5, vmax=0, s=7, rasterized=True, cmap='plasma')
plt.tick_params(axis='y', labelleft=False)
plt.title(r'$(\ell, m, n) = (2, 2, 1)$')

plt.subplot(223)
plt.scatter(p2l6m4n2diffs[:,0], p2l6m4n2diffs[:,1], c=p2l6m4n2diffs[:,4], vmin=-5, vmax=0, s=7, rasterized=True, cmap='plasma')
plt.ylabel('e')
plt.xlabel('p')
plt.title(r'$(\ell, m, n) = (6, 4, 2)$')

plt.subplot(224)
s = plt.scatter(p3l2m2n0diffs[:,0], p3l2m2n0diffs[:,1], c=p3l2m2n0diffs[:,4], vmin=-5, vmax=0, s=7, rasterized=True, cmap='plasma')
plt.plot(p3l2m2n0diffs[:,9], p3l2m2n0diffs[:,1], c='gray', ls='-', lw=1)
plt.tick_params(axis='y', labelleft=False)
plt.xlabel('p')
plt.title(r'$(\ell, m, n) = (2, 2, 0)$')

plt.tight_layout()

# add colorbar on its own axes
cbar_ax = plt.gcf().add_axes([1.015, 0.15, 0.03, 0.75])
cbar = plt.colorbar(s,cax=cbar_ax)
cbar.set_label(r'$\log_{10} \left| 1 - \frac{\mathcal{A}_{\ell mn}^{\mathrm{FEW}}}{\mathcal{A}_{\ell mn}^{\mathrm{PN5}}} \right|$', fontsize=14)

plt.savefig('AmpMagComparison.pdf', bbox_inches='tight')
# plt.show()