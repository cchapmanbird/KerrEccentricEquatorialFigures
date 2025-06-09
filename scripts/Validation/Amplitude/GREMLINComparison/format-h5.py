"""

The GREMLIN data this script formats is too large for Github, but is available on request.

"""

import h5py
import numpy as np
import os
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms
from tqdm import tqdm

few_amp = AmpInterpKerrEccEq()
ylm = GetYlms()

all_files = os.listdir('./a0.895200')
all_files = [f for f in all_files if f.endswith('.h5')]

def get_pe_from_fn(fn):
    p = float(fn.split('_')[1].split('p')[1])
    e = float(fn.split('_')[2].split('e')[1])
    
    return p, e

spin = 0.8952


few_mode_list = []
for l in range(2,11):
    for m in range(0, l+1):
        for n in range(-55, 56):
            few_mode_list.append((l, m, n))

formstr = "l{:d}m{:d}k0n{:d}"
f = h5py.File(os.path.join('a0.895200', all_files[0]), 'r')

outp = h5py.File('a0.895200_data.h5', 'w')
pv = []
ev = []
ampv = []

for fn in tqdm(all_files):
    p, e = get_pe_from_fn(fn)
    f = h5py.File(os.path.join('a0.895200', fn), 'r')['modes']

    amp_h = []
    for ell, emm, enn in few_mode_list:
        try:
            amp_here =  f[formstr.format(ell, emm, enn)][()]
            amp_here = amp_here[0] + 1j* amp_here[1]
        except:
            amp_here = 0
        amp_h.append(amp_here)

    ampv.append(amp_h)
    pv.append(p)
    ev.append(e)

outp.create_dataset('p', data=pv)
outp.create_dataset('e', data=ev)
outp.create_dataset('amps', data=ampv)