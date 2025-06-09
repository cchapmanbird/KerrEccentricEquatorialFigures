from few.trajectory.ode.flux import KerrEccEqFlux
import matplotlib.pyplot as plt
import numpy as np
from few.utils.mappings.kerrecceq import kerrecceq_forward_map

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]


base_str = 'sah_flux/eind_{}.flux'
nums = [0, 32, 64, 96, 112, 128]

loadeds_shaped = np.concatenate([np.loadtxt(base_str.format(num)).reshape(129, 129, -1)[None,...] for num in nums])

ode = KerrEccEqFlux()

interps = np.zeros((6, 129, 129, 2))
for i in range(6):
    for j in range(129):
        for k in range(129):
            try:
                interps[i,j,k,:] = ode.interpolate_flux_grids(*loadeds_shaped[i,j,k,1:4], a=loadeds_shaped[i,j,k,0])
            except ValueError:
                interps[i,j,k,:] = [np.nan,np.nan]

pdot_int = interps[:,:,:,0]
edot_int = interps[:,:,:,1]

edot_grid = loadeds_shaped[:,:,:,-4] + loadeds_shaped[:,:,:,-3]
pdot_grid = loadeds_shaped[:,:,:,-6] + loadeds_shaped[:,:,:,-5]

a_plot = loadeds_shaped[:,:,:,0]*loadeds_shaped[:,:,:,3]
p_plot = loadeds_shaped[:,:,:,1] - loadeds_shaped[:,:,:,7]
e_plot = loadeds_shaped[:,:,:,2]

fig = plt.figure(figsize=(4.,4.), dpi=200)
for i, k in enumerate([0, 2, 4, 5]):
    plt.subplot(2, 2, i+1)

    toplot =np.log10(np.abs(1 - pdot_int / pdot_grid))[k,:,:].flatten()

    plt.scatter(1 - a_plot[i,:,:].flatten(), p_plot[i,:,:].flatten(), s=3, c=toplot, vmin=-8, vmax=-2, cmap='plasma', rasterized=True)

    plt.text(0.52, 0.875, rf'$e_\mathrm{{max}}$={e_plot[k, p_plot[k] <= 5].max():.2f}', fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))

    if i == 0 or i == 2:
        plt.ylabel(r'$p - p_{\mathrm{sep}}$')
    else:
        plt.tick_params(axis='y', labelleft=False)
    
    if i == 0 or i == 1:
        plt.tick_params(axis='x', labelbottom=False)
    else:
        plt.xlabel(r'$1 - a$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 2)
    plt.ylim(1e-3, 5)

# one colorbar for all subplots, given its own axes
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(-8, -2)), cax=cbar_ax)
cbar.set_label(r'$\log_{10} \left| 1 - f_p^{\mathrm{FEW}} / f_p^{\mathrm{SAH}} \right|$')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
plt.savefig('SAH_nearISO_Comparison.pdf', bbox_inches='tight')