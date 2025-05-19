import numpy as np
import matplotlib.pyplot as plt
from few.utils.geodesic import get_separatrix
from seaborn import color_palette
import h5py
from functools import reduce

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
cpal = color_palette("colorblind", 6)


f = h5py.File('data.h5', 'r')
a = f.attrs['a']
tout = f.attrs['trajectory']
pvals = f.attrs['pvals']
evals = f.attrs['evals']
ellvec = f.attrs['ellvec']
mvec = f.attrs['mvec']
nvec = f.attrs['nvec']
pow_max = f.attrs['pow_max']
amps_out = [f[f'amps_{i}'] for i in range(4)]
fig = plt.figure(figsize=(7, 8), dpi=150)

plt.subplot(7, 9, (1, 18))
plt.plot(tout[1], tout[2], c=cpal[0])
plt.scatter(pvals, evals, c=cpal[1], s=24, marker='d', zorder=100, label='Mode spectrum points')
egrid = np.linspace(1e-5, 0.99, 101)
seps = get_separatrix(a, egrid, np.ones_like(egrid))
plt.plot(seps, egrid, lw=1, ls=':', c='k', label=r'$p_\mathrm{LSO}$')
plt.xlabel(r'$p$')
plt.ylabel(r'$e$')
plt.ylim(0.01, 0.83)

vmax = 0
vmin = -10

modes_onepercent_power = []
modes_fiducial_power = []

for i in range(4):
    power_arr_plot = amps_out[i]
    power_arr_flat = power_arr_plot[()].flatten()
    sort_order = np.flip(np.argsort(power_arr_flat))
    cumsum = np.cumsum(power_arr_flat[sort_order])
    cutind = np.where(cumsum > 0.99*cumsum[-1])[0][0]
    modes_onepercent_power.append(sort_order[:cutind])    
    cutind = np.where(cumsum > 0.99999*cumsum[-1])[0][0]
    modes_fiducial_power.append(sort_order[:cutind])    

    for k in range(9):
        plt.subplot(7, 9, 27 + i*9 + k+1)
        # plt.pcolormesh(mvec, nvec, np.log10(power_arr_plot).sum(0).T, cmap='cividis')
        plt.pcolormesh(mvec[:k+3], nvec, np.log10(power_arr_plot)[k][:k+3].T, cmap='cividis', vmin=vmin, vmax=vmax)
        # plt.xlim(0,2)
        if k > 0:
            plt.tick_params(axis='y', which='both', labelleft=False, left=False)
        if i < 3:
            plt.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

        if i == 0:
            plt.title(fr'$\ell = {k+2}$')
        
        if i == 3:
            tick_temp = np.arange(k+3)
            if tick_temp.size == 11:
                tick_temp = tick_temp[::5]
            elif tick_temp.size > 5:
                tick_temp = tick_temp[::2]

            plt.xticks(tick_temp, tick_temp)

        if k == 0:
            plt.yticks([-50, -25, 0, 25, 50], [-50, -25, 0, 25, 50])
            if i < 3:
                plt.ylabel(f"({pvals[i]:.0f}, {evals[i]:.2f})")
            elif i == 3:
                ax = plt.gca()
                lb = ax.set_ylabel(fr"$(p, e) = $({pvals[i]:.0f}, {evals[i]:.2f})")
                lb.set_position((0., 0.36))

# fig.supylabel(r'$n$ mode index', x=0.06, y=0.35)
fig.supylabel(r'$n$ mode index', y=0.35)
fig.supxlabel(r'$m$ mode index', x=0.55, y=0.025)

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.15)

# one colorbar for all subplots, sits in a big axis at the side
cax = plt.axes([1.01, 0.1, 0.02, 0.47])
cb = plt.colorbar(cax=cax)
cb.set_label(label=r"$P_{\ell m n} / P_{\ell m n, \mathrm{tot}}^{(2, 0.09)}$")
plt.savefig('amp_heatmap.pdf', bbox_inches='tight')
plt.close()

for arr in modes_onepercent_power:
    print('Sizes (1%)')
    print(arr.size)

print('Union', reduce(np.union1d, modes_onepercent_power).size)

for arr in modes_fiducial_power:
    print('Sizes (1E-5)')
    print(arr.size)

print('Union', reduce(np.union1d, modes_fiducial_power).size)