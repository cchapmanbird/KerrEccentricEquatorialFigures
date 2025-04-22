import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.mappings.kerrecceq import apex_of_uwyz, apex_of_UWYZ
from few.utils.geodesic import get_separatrix

ode1 = KerrEccEqFlux()
ode2 = KerrEccEqFlux(downsample=[[2,2,2],[2,2,2]])

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# The excluded grid nodes from the down-sampled grid.
uv = np.linspace(0,1,129)[1::2]
wv = np.linspace(0,1,65)[1::2]
zv = np.linspace(0,1,65)[1]

ugrid, wgrid = np.asarray(np.meshgrid(uv, wv, indexing='ij')).reshape(2, -1)
zgrid = np.ones_like(ugrid) * zv

#near grid
agrid, pgrid, egrid, xgrid = apex_of_uwyz(ugrid, wgrid, np.ones_like(zgrid), zgrid)
seps = get_separatrix(agrid, egrid, xgrid)

int_full = np.asarray([ode1.interpolate_flux_grids(pgrid[i]+1e-10, egrid[i], xgrid[i], a=agrid[i]) for i in range(len(agrid))])
int_half = np.asarray([ode2.interpolate_flux_grids(pgrid[i]+1e-10, egrid[i], xgrid[i], a=agrid[i]) for i in range(len(agrid))])

#far grid
Agrid, Pgrid, Egrid, Xgrid = apex_of_UWYZ(ugrid, wgrid, np.ones_like(zgrid), zgrid, True)
Seps = get_separatrix(Agrid, Egrid, Xgrid)

Int_full = np.asarray([ode1.interpolate_flux_grids(Pgrid[i]+1e-10, Egrid[i], Xgrid[i], a=Agrid[i]) for i in range(len(Agrid))])
Int_half = np.asarray([ode2.interpolate_flux_grids(Pgrid[i]+1e-10, Egrid[i], Xgrid[i], a=Agrid[i]) for i in range(len(Agrid))])

p_all = np.r_[pgrid, Pgrid]
e_all = np.r_[egrid, Egrid]
seps_all = np.r_[seps, Seps]
int_full_all = np.r_[int_full, Int_full]
int_half_all = np.r_[int_half, Int_half]

plt.figure(figsize=(4,3), dpi=150)
plt.scatter(p_all - seps_all, e_all, s=5, c= np.log10(np.abs(1 - int_full_all/int_half_all))[:,0], cmap='plasma',rasterized=True, marker='o', vmin=-8, vmax=-2)
plt.xlim(1e-3, 25)
plt.ylim(0, 0.9)
plt.colorbar(label=r'$\log_{10} \left| 1 - \hat{f}_p^{\mathrm{FEW}} / \hat{f}_p^{\mathrm{FEW/2}} \right|$')
plt.xscale('log')
plt.xlabel(r'$p - p_{\mathrm{sep}}$')
plt.ylabel(r'$e$')
plt.savefig('downsample_flux_interpolation_error.pdf', bbox_inches='tight')
plt.show()