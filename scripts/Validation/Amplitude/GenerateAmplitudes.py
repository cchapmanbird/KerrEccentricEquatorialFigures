from few.amplitude.ampinterp2d import AmpInterpKerrEccEq, AmpInterpSchwarzEcc
from few.trajectory.inspiral import EMRIInspiral
import numpy as np
import h5py
import json


amps_out = {}
# save dictionary to json
with open('amplitude_diff.json', 'w') as f:
    json.dump(amps_out, f, indent=4)

traj = EMRIInspiral(func="KerrEccEqFlux")
amp_module = AmpInterpKerrEccEq()
ampS = AmpInterpSchwarzEcc()

# pvals = np.array([8., 10., 20.0])
# evals = np.array([0.1, 0.2, 0.6])
ps = np.linspace(10, 40, 50)
es = np.linspace(0.01, 0.7, 50)
nmin = -10
nmax = 10
mode_selection = []
for ell in range(2,3):
    for m in range(-ell, ell+1):
        for n in range(nmin, nmax+1):
            mode_selection.append((ell, m, n))
for p, e in zip(ps, es):
    a = 0.0
    x = 1.

    amps_here = amp_module(a, p, e, x, specific_modes=mode_selection)
    ampS_here = ampS(a, p, e, x, specific_modes=mode_selection)
    for key, val in amps_here.items():
        ell, m, n = key
        amps_out[str(key)] = float(np.abs(val[0]-ampS_here[key][0]))
        print(key, p, e)
        print(f"diff: {amps_out[str(key)]}")

# save dictionary to json
with open('amplitude_diff.json', 'w') as f:
    json.dump(amps_out, f)