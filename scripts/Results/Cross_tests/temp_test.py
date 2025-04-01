


import numpy as np
from matplotlib import pyplot as plt



import multiprocessing
num_threads =  multiprocessing.cpu_count()

from few.utils.constants import YRSID_SI, MTSUN_SI, Pi

from bhpwave.waveform import KerrWaveform
from  bhpwave.trajectory.inspiral import InspiralGenerator
from bhpwave.waveform import scaled_amplitude
import multiprocessing
Zach_gen_Kerr = KerrWaveform()
num_threads_zach = multiprocessing.cpu_count()
traj_Zach = InspiralGenerator(trajectory_data=None)
print("num_threads_zach:", num_threads_zach, "num_threads:", num_threads)












M = 1e6
mu = 1e1
a0 = 0.99
# p0 = 12.0
dt = 10
Tobs = 1.0
e0 = 0.0
x0 = 1.0
theta = np.pi/5.
phi = np.pi/3.
p0 = 7.0
dist = 1.0
Phi_phi0 = 0.0

lmax = 10
specific_modes = []
for l in range(2,lmax+1):
    for m in range(0,l+1):
        specific_modes += [(l,m,0)]


# specific_modes = [(10,10,0)]
zach_scaled_amp = scaled_amplitude(mu, dist)
Zach_source = zach_scaled_amp * Zach_gen_Kerr.source_frame(M, mu, a0, p0, theta, phi, Phi_phi0 , dt=dt, T = Tobs,select_modes=specific_modes,  num_threads=num_threads_zach)
t_arr = np.arange(len(Zach_source.real))*dt/YRSID_SI
# waveform_KerrEcc = Kerr_ecc_wave(M, mu, a0, p0, e0, x0, theta, phi,dt=dt, T=Tobs, dist = dist, mode_selection=specific_modes)
# waveform_KerrEcc = waveform_KerrEcc.get()
# t_arr = np.arange(len(waveform_KerrEcc.real))*dt/YRSID_SI

plt.plot(t_arr[-100:], Zach_source.real[-100:], label = f"Zach's code")

specific_modes = [(2,2,0)]
zach_scaled_amp = scaled_amplitude(mu, dist)
Zach_source = zach_scaled_amp * Zach_gen_Kerr.source_frame(M, mu, a0, p0, theta, phi, Phi_phi0 , dt=dt, T = Tobs,mode_selection=specific_modes,  num_threads=num_threads_zach)
plt.plot(t_arr[-100:], Zach_source.real[-100:], label = f"Zach's code")
