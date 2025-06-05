import numpy as np
from multispline.spline import TricubicSpline
import h5py
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.mappings.kerrecceq import kerrecceq_backward_map, z_of_a, a_of_z
from few.utils.geodesic import get_separatrix

bicubic = AmpInterpKerrEccEq()


# ecc first
with h5py.File('A220_slice.h5') as f:
    tricubic_A_Re = TricubicSpline(np.linspace(0,1,33), np.linspace(0,1,33), np.linspace(0,1,33), f['modeA'][()][:,:,:,0])
    tricubic_A_Im = TricubicSpline(np.linspace(0,1,33), np.linspace(0,1,33), np.linspace(0,1,33), f['modeA'][()][:,:,:,1])

    tricubic_B_Re = TricubicSpline(np.linspace(0,1,33), np.linspace(0,1,33), np.linspace(0,1,33), f['modeB'][()][:,:,:,0])
    tricubic_B_Im = TricubicSpline(np.linspace(0,1,33), np.linspace(0,1,33), np.linspace(0,1,33), f['modeB'][()][:,:,:,1])

zvec = np.linspace(0, 1, 33)
z_betw = (zvec[0] + zvec[1]) / 2
a = a_of_z(z_betw)
z = z_of_a(a)

u = np.linspace(1e-5, 1-1e-5, 301)
w = np.linspace(1e-5, 1-1e-5, 301)

ugrid, wgrid = np.asarray(np.meshgrid(u, w, indexing='ij')).reshape(2, -1)

apex = kerrecceq_backward_map(ugrid, wgrid, np.ones_like(ugrid), np.ones_like(ugrid)*z, regionA=True, kind="amplitude")
sep = get_separatrix(apex[0], apex[2], apex[3], )

few_mode_amp = bicubic(a, apex[1], apex[2], apex[3], specific_modes=[(2,2,0)])[(2,2,0)].reshape(u.size, w.size)

tric_mode_amp = (
    tricubic_A_Re(ugrid, wgrid, np.ones_like(ugrid)*z) + 1j*tricubic_A_Im(ugrid, wgrid,  np.ones_like(ugrid)*z)
).reshape(u.size, w.size)

np.save('A220_ecc_pars.npy', np.array([*apex, sep]))
np.save('A220_ecc_slice.npy', np.array([few_mode_amp, tric_mode_amp]))

zvec = np.linspace(0, 1, 301)

u = np.linspace(1e-5, 1-1e-5, 301)
w = 0.5

ugrid, zgrid = np.asarray(np.meshgrid(u, zvec, indexing='ij')).reshape(2, -1)

apex = kerrecceq_backward_map(ugrid, np.ones_like(ugrid)*0.5, np.ones_like(ugrid), zgrid, regionA=True, kind="amplitude")
sep = get_separatrix(apex[0], apex[2], apex[3], )

few_mode_amp = np.asarray([bicubic(apex[0][i], apex[1][i], apex[2][i], apex[3][i], specific_modes=[(2,2,0)])[(2,2,0)] for i in range(len(apex[0]))]).reshape(u.size, zvec.size)

tric_mode_amp = (
    tricubic_A_Re(ugrid, np.ones_like(ugrid)*w, zgrid) + 1j*tricubic_A_Im(ugrid,  np.ones_like(ugrid)*0.5, zgrid)
).reshape(u.size, zvec.size)

np.save('A220_spin_pars.npy', np.array([*apex, sep]))
np.save('A220_spin_slice.npy', np.array([few_mode_amp, tric_mode_amp]))

