"""

The formatted GREMLIN data this script uses to compute mismatches is too large for Github, but is available on request.

"""

import h5py
import numpy as np
import os
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms
from tqdm import tqdm
from few.utils.geodesic import get_separatrix

few_amp = AmpInterpKerrEccEq()

spin = 0.8952

# We consider only modes with m >= 0 for simplicity, as without the spherical harmonics 
# including m < 0 amounts to scaling inner products by 2, which drops out of the mismatch.
few_mode_list = []
for l in range(2,11):
    for m in range(0, l+1):
        for n in range(-55, 56):
            few_mode_list.append((l, m, n))

f = h5py.File('./a0.895200_data.h5', 'r')

pvals = f['p'][()]
evals = f['e'][()]
ampvals = f['amps'][()]

mismatches = []
for j in tqdm(range(len(pvals)), total=len(pvals)):
    p = pvals[j]
    e = evals[j]
    try:
        few_amplitudes = few_amp(spin, p, e, 1., specific_modes=few_mode_list)
    except:
        mismatches.append(np.nan)
        continue
    
    amp_pairs = []
    for k, (ell, emm, enn) in enumerate(few_mode_list):
        amp_here = ampvals[j,k]

        try:
            few_amp_here = few_amplitudes[(ell, emm, enn)].item()
        except:
            few_amp_here = 0
        
        amp_pairs.append((few_amp_here, amp_here))

    amp_pairs = np.asarray(amp_pairs)
    f_f = (amp_pairs[:,0].conj() * amp_pairs[:,0]).real.sum()
    s_s = (amp_pairs[:,1].conj() * amp_pairs[:,1]).real.sum()
    f_s = (amp_pairs[:,0].conj() * amp_pairs[:,1]).real.sum()

    mismatches.append(1 - f_s / (np.sqrt(f_f * s_s)))

mismatches = np.asarray(mismatches)

outp = h5py.File('mismatches_a0.895200.h5', 'w')
outp.create_dataset('mismatch', data=mismatches)
outp.create_dataset('pv', data=pvals)
outp.create_dataset('ev', data=evals)

seps = get_separatrix(spin, np.asarray(evals), np.ones_like(evals))
outp.create_dataset('separatrix', data=seps)
outp.close()