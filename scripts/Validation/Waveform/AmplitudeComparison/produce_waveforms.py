import numpy as np
import matplotlib.pyplot as plt
import h5py

from few.waveform import FastKerrEccentricEquatorialFlux
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms
from few.utils.constants import YRSID_SI
import os
import h5py
from few.utils.utility import get_mismatch

wf = FastKerrEccentricEquatorialFlux()
ylm_gen = GetYlms(include_minus_m=True)
# ampmod = AmpInterpKerrEccEq()

# load the amplitudes
all_files = list(reversed((sorted(os.listdir('./a0.998_trajdat')))))

modes = []

# for l in range(2,11):
#     for m in range(0, l+1):
#         for n in range(-55, 56):
#             modes.append((l,m, n))

for i in range(len(wf.l_arr_no_mask)):
    l = wf.l_arr_no_mask[i]
    m = wf.m_arr_no_mask[i]
    n = wf.n_arr_no_mask[i]
    modes.append((l, m, n))

teuk_modes_arr = np.load('teuk_modes_a0.998.npy')
teuk_modes_arr = teuk_modes_arr[:, :, 0] + 1j * teuk_modes_arr[:, :, 1]

teuk_modes_arr_tric = np.load('a0.998-tric-amps.npy')
teuk_modes_arr_tric = teuk_modes_arr_tric[:, :, 0] + 1j * teuk_modes_arr_tric[:, :, 1]

tr = EMRIInspiral(func="KerrEccEqFlux")

m1 = 1e6
m2 = 1e1
a = 0.998
e0 = 0.7
x0 = 1.

theta = np.pi/3
phi = np.pi/4

T = 2

p0 = get_p_at_t(tr, T, [m1, m2, a, e0, x0])

traj_out = tr(m1, m2, a, p0, e0, x0, T=T)
# amps_check = ampmod(a, traj_out[1], traj_out[2], traj_out[3])

print(traj_out[0][-1] / YRSID_SI)
wave1 = wf(
    m1,
    m2,
    a,
    p0,
    e0,
    x0,
    theta,
    phi,
    dt=5,
    T=traj_out[0][-1] / YRSID_SI,
    mode_selection_threshold=1e-7
)
print('Wave made')
sum = InterpolatedModeSum()

retain_inds = wf.special_index_map_arr[wf.ls, wf.ms, wf.ns]
teuk_modes_arr = teuk_modes_arr[:, retain_inds]

ylms = ylm_gen(wf.ls, wf.ms, theta, phi)

scott_wave = sum(
    traj_out[0],
    teuk_modes_arr,
    ylms,
    tr.integrator_spline_t,
    tr.integrator_spline_phase_coeff[:,[0,2]],
    # wf.l_arr_no_mask,
    # wf.m_arr_no_mask,
    # wf.n_arr_no_mask,
    wf.ls,
    wf.ms,
    wf.ns,
    T=2,
    dt=5
)
print('Wave made')

teuk_modes_arr_tric = teuk_modes_arr_tric[:, retain_inds]

tric_wave = sum(
    traj_out[0],
    teuk_modes_arr_tric,
    ylms,
    tr.integrator_spline_t,
    tr.integrator_spline_phase_coeff[:,[0,2]],
    # wf.l_arr_no_mask,
    # wf.m_arr_no_mask,
    # wf.n_arr_no_mask,
    wf.ls,
    wf.ms,
    wf.ns,
    T=2,
    dt=5
)
print('Wave made')

scott_wave_arr = np.array([scott_wave.real, scott_wave.imag]).T
wave1_arr = np.array([wave1.real, wave1.imag]).T
tric_wave_arr = np.array([tric_wave.real, tric_wave.imag]).T
np.save('a0.998waves', np.vstack((scott_wave_arr.T, wave1_arr.T, tric_wave_arr.T)))

breakpoint()

plt.plot(scott_wave.real)
plt.plot(wave1.real, ls='--')
plt.plot(tric_wave.real, ls='--')
plt.show()
