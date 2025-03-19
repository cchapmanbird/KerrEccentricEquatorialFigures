import numpy as np
import matplotlib.pyplot as plt
from few.utils.constants import YRSID_SI
from scipy.signal.windows import tukey
import h5py

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

cmap = 'cividis'

f = h5py.File("waveform.h5", "r")
wave = f["prograde_waveform"][:]
T = f.attrs["T"]
dt = f.attrs["dt"]

tseg = 12 * 3600
samples_per_seg = int(tseg / dt)
nseg = int(len(wave) * dt / tseg)
wave_plot = wave[:int(nseg*samples_per_seg)]

wave_tf = 2 * np.fft.rfft(dt * tukey(samples_per_seg, alpha=1.)[None,:] * wave_plot.real.reshape(nseg, samples_per_seg), axis=1) / tseg
t_seg = np.arange(nseg) * tseg / YRSID_SI
f_seg = np.fft.rfftfreq(samples_per_seg, d=dt)

# vmax = 3e-24
vmax = 5e-24

plt.figure(figsize=(10, 9), dpi=200)
plt.subplot(211)
im = plt.pcolormesh(t_seg, f_seg, np.abs(wave_tf).T, shading='auto', cmap=cmap, vmax=vmax, rasterized=True)
plt.yscale('log')
plt.ylim(3e-4, 1e-1)
plt.xlim(0,1)
plt.text(0.015, 0.96, '(a)', color='white', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
# bring in the colorbar a bit
# cb = plt.colorbar(im, pad=0.02)
# cb.ax.tick_params(labelsize=13)
# cb.set_label(label='Strain (source-frame)', size=14)

plt.ylabel('Frequency [Hz]', fontsize=16)
# turn off the xtick labels but not the ticks themselves
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.tick_params(axis='both', which='major', labelsize=14)

wave = f["retrograde_waveform"][:]
T = f.attrs["T"]
dt = f.attrs["dt"]

tseg = 12 * 3600
samples_per_seg = int(tseg / dt)
nseg = int(len(wave) * dt / tseg)
wave_plot = wave[:int(nseg*samples_per_seg)]

wave_tf = 2 * np.fft.rfft(dt *tukey(samples_per_seg, alpha=1.)[None,:] * wave_plot.real.reshape(nseg, samples_per_seg), axis=1) / tseg

t_seg = np.arange(nseg) * tseg / YRSID_SI
f_seg = np.fft.rfftfreq(samples_per_seg, d=dt)

plt.subplot(212)
im2 = plt.pcolormesh(t_seg, f_seg, np.abs(wave_tf).T, shading='auto', cmap=cmap, vmax=vmax, rasterized=True)
plt.ylabel('Frequency [Hz]', fontsize=16)
# cb = plt.colorbar(im2, pad=0.02)
# # make the colorbar tick labels bigger
# cb.set_label(label='Strain (source-frame)', size=14)
# cb.ax.tick_params(labelsize=13)
plt.ylim(3e-4, 1e-1)
plt.xlim(0,1)
plt.yscale('log')
plt.xlabel('Time [y]', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(0.015, 0.96, '(b)', color='white', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')

plt.tight_layout()

# One colorbar for both plots
fig = plt.gcf()
cbar_ax = fig.add_axes([1.01, 0.1, 0.02, 0.85])
cb = fig.colorbar(im, cax=cbar_ax)
# cb = plt.colorbar(im, pad=0.02)
cb.ax.tick_params(labelsize=13)
cb.set_label(label='Gravitational-wave strain', size=14)


plt.savefig("waveform_tf.pdf", bbox_inches='tight')
plt.close()
