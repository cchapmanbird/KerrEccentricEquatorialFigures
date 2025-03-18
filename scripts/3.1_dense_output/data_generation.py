import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t, get_fundamental_frequencies
from few.utils.constants import MTSUN_SI
import h5py


tr = EMRIInspiral(func="KerrEccEqFlux")

pars = [
    1e6,
    1e1,
    0.998,
    5.,
    0.5,
    1.
]

T = 1.
dt = 50.  # The smaller the better, but strongly impacts the runtime. This should be fine.

pars[3] = get_p_at_t(tr, T, [pars[0], pars[1], pars[2], pars[4], pars[5]])

T = 0.999  # Cut off the very last part of the inspiral for plot dynamic range purposes

outp_dense = tr(*pars, T=T, dt=dt, DENSE_STEPPING=True)
outp = tr(*pars, T=T, dt=dt)

t_dense = outp_dense[0]
phase_dense = np.asarray([outp_dense[4], outp_dense[5], outp_dense[6]]).T
fr_dense = np.asarray(get_fundamental_frequencies(pars[2], outp_dense[1], outp_dense[2], outp_dense[3])).T / (pars[0] * MTSUN_SI)
fdot_dense = np.gradient(fr_dense, t_dense, axis=0, edge_order=2)

phase_ups = tr.inspiral_generator.eval_integrator_spline(t_dense)[:,[3,4,5]]
fr_ups = tr.inspiral_generator.eval_integrator_derivative_spline(t_dense)[:,[3,4,5]]
fdot_ups = tr.inspiral_generator.eval_integrator_derivative_spline(t_dense, order=2)[:,[3,4,5]]

with h5py.File('data.h5', 'w') as f:
    f.attrs['T'] = T
    f.attrs['dt'] = dt
    f.create_dataset('t_dense', data=t_dense)
    f.create_dataset('phase_dense', data=phase_dense)
    f.create_dataset('fr_dense', data=fr_dense)
    f.create_dataset('fdot_dense', data=fdot_dense)
    f.create_dataset('phase_ups', data=phase_ups)
    f.create_dataset('fr_ups', data=fr_ups)
    f.create_dataset('fdot_ups', data=fdot_ups)
