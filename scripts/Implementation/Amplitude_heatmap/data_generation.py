from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.trajectory.inspiral import EMRIInspiral
import numpy as np
import h5py

traj = EMRIInspiral(func="KerrEccEqFlux")
amp_module = AmpInterpKerrEccEq()

tout = traj(1e6, 1e2, 0.998, 8, 0.8, 1., T=10.)

pvals = np.array([8., 6., 4., 2.,])
evals = np.array([0.8, 0.533, 0.283, 0.091,])

# pvals = np.array([2.7, 2.35, 2., 1.65])
# evals = np.array([0.5, 0.4, 0.3, 0.22,])

mode_selection = []
for ell in range(2,11):
    for m in range(-ell, ell+1):
        for n in range(-55, 56):
            mode_selection.append((ell, m, n))

amps_out = []
for p, e in zip(pvals, evals):
    # e = 0.3
    a = 0.998
    x = 1.

    amps_here = amp_module(a, p, e, x, specific_modes=mode_selection)

    amp_arr = np.zeros((9, 21, 111), dtype=np.complex128)
    for key, val in amps_here.items():
        ell, m, n = key
        amp_arr[ell - 2, m + 10, n + 55] = val[0]

    amps_out.append(amp_arr)

ellvec = np.arange(2,11)
mvec = np.arange(11)
nvec = np.arange(-55,56)

pow_max = (np.abs(amps_out[-1][:,10:,:])**2).sum()

with h5py.File('data.h5', 'w') as f:
    f.attrs['trajectory'] = np.asarray(tout)
    f.attrs['a'] = a
    f.attrs['pvals'] = pvals
    f.attrs['evals'] = evals
    f.attrs['ellvec'] = ellvec
    f.attrs['mvec'] = mvec
    f.attrs['nvec'] = nvec
    f.attrs['pow_max'] = pow_max
    for i, amp_arr in enumerate(amps_out):
        amp_arr = amps_out[i]
        amp_arr_plot = amp_arr[:,10:,:]
        power_arr_plot = np.abs(amp_arr_plot)**2
        # power_arr_plot /= np.max(power_arr_plot)
        power_arr_plot /= pow_max
        power_arr_plot[power_arr_plot < 1e-10] = 0.
        f.create_dataset(f'amps_{i}', data=power_arr_plot)