from few.amplitude.ampinterp2d import AmpInterpKerrEccEq, AmpInterpSchwarzEcc
from few.trajectory.inspiral import EMRIInspiral
import numpy as np
import h5py

amps_out = []

traj = EMRIInspiral(func="KerrEccEqFlux")
amp_module = AmpInterpKerrEccEq()
ampS = AmpInterpSchwarzEcc()

ps_vec = np.linspace(8., 40, 10)
es_vec = np.linspace(0.01, 0.7, 10)
# create grid of parameters
a = 0.0
ps, es = np.meshgrid(ps_vec, es_vec)
ps = ps.flatten()
es = es.flatten()
xs = np.ones(len(ps))

nmin = -3
nmax = 3
mode_selection = []
lmin = 2
lmax = 3
for ell in range(lmin, lmax+1):
    for m in range(-ell, ell + 1):
        for n in range(nmin, nmax + 1):
            mode_selection.append((ell, m, n))

amps_here = amp_module(a, ps, es, xs, specific_modes=mode_selection)
ampS_here = ampS(a, ps, es, xs, specific_modes=mode_selection)

# amps_out.append(amp_arr)

ellvec = np.arange(lmin, lmax + 1)
mvec = np.arange(-lmax, lmax + 1)
nvec = np.arange(nmin, nmax + 1)


with h5py.File('amplitude_diff.h5', 'w') as f:
    f.attrs['a'] = 0.0
    f.attrs['pvals'] = ps_vec
    f.attrs['evals'] = es_vec
    f.attrs['ellvec'] = ellvec
    f.attrs['mvec'] = mvec
    f.attrs['nvec'] = nvec
    
    for key, val in amps_here.items():
        ell, m, n = key
        diff_amp = val - ampS_here[key]
        f.create_dataset(f'amp_Kerr_{ell}{m}{n}', data=val.get())
        f.create_dataset(f'amp_Schw_{ell}{m}{n}', data=ampS_here[key].get())
        f.create_dataset(f'amp_diff_{ell}{m}{n}', data=diff_amp.get())
