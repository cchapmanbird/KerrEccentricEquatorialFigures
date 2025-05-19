import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
import h5py

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams['font.size'] = 13
cpal = color_palette("colorblind", 4)

f = h5py.File("data.h5", "r")
t_dense = f["t_dense"][:]
phase_dense = f["phase_dense"][:]
fr_dense = f["fr_dense"][:]
fdot_dense = f["fdot_dense"][:]
phase_ups = f["phase_ups"][:]
fr_ups = f["fr_ups"][:]
fdot_ups = f["fdot_ups"][:]


plt.figure(figsize=(5, 6.7), dpi=300)

# Phase and Phase Difference
plt.subplot(3, 2, 1, rasterized=True)
form = plt.ScalarFormatter()
form.set_powerlimits((2, 5))
form.set_scientific(True)
plt.plot(t_dense, phase_ups[:,0], c=cpal[0], label=r"$\alpha=\phi$")
plt.plot(t_dense, phase_ups[:,1], c=cpal[1], label=r"$\alpha=\theta$")
plt.plot(t_dense, phase_ups[:,2], c=cpal[2], label=r"$\alpha=r$")
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel(r"$\Phi_\alpha^{{\rm Adaptive}}$")
plt.gca().yaxis.set_major_formatter(form)
plt.legend(frameon=False)

plt.subplot(3, 2, 2, rasterized=True)
plt.plot(t_dense, np.abs(phase_dense[:,0] - phase_ups[:,0]), c=cpal[0])
plt.plot(t_dense, np.abs(phase_dense[:,1] - phase_ups[:,1]), c=cpal[1])
plt.plot(t_dense, np.abs(phase_dense[:,2] - phase_ups[:,2]), c=cpal[2])
plt.ylim(1e-7, 5e-5)
plt.yscale('log')
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel(r"$\Phi_\alpha^{{\rm Adaptive}} - \Phi_\alpha^{{\rm Dense}}$")

# Frequency and Frequency Difference
plt.subplot(3, 2, 3, rasterized=True)
plt.plot(t_dense, fr_ups[:,0], c=cpal[0])
plt.plot(t_dense, fr_ups[:,1], c=cpal[1])
plt.plot(t_dense, fr_ups[:,2], c=cpal[2])
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel(r"$\Omega_\alpha^{{\rm Adaptive}}$")

plt.subplot(3, 2, 4, rasterized=True)
plt.plot(t_dense, np.abs(fr_dense[:,0] - fr_ups[:,0]), c=cpal[0])
plt.plot(t_dense, np.abs(fr_dense[:,1] - fr_ups[:,1]), c=cpal[1])
plt.plot(t_dense, np.abs(fr_dense[:,2] - fr_ups[:,2]), c=cpal[2])
plt.yscale('log')
plt.tick_params(axis='x', labelbottom=False)
plt.ylim(1e-13, 2e-9)
plt.ylabel(r"$\Omega_\alpha^{{\rm Adaptive}} - \Omega_\alpha^{{\rm Dense}}$")

# Frequency Derivative and Frequency Derivative Difference
plt.subplot(3, 2, 5, rasterized=True)
plt.plot(t_dense, fdot_ups[:,0], c=cpal[0])
plt.plot(t_dense, fdot_ups[:,1], c=cpal[1])
plt.plot(t_dense, fdot_ups[:,2], c=cpal[2])
plt.xlabel("Time [s]")
plt.ylabel(r"$\dot{\Omega}_\alpha^{{\rm Adaptive}}$")

plt.subplot(3, 2, 6, rasterized=True)
plt.plot(t_dense, np.abs(fdot_dense[:,0] - fdot_ups[:,0]), c=cpal[0])
plt.plot(t_dense, np.abs(fdot_dense[:,1] - fdot_ups[:,1]), c=cpal[1])
plt.plot(t_dense, np.abs(fdot_dense[:,2] - fdot_ups[:,2]), c=cpal[2])
plt.yscale('log')
plt.ylim(1e-15, 5e-12)
plt.xlabel("Time [s]")
plt.ylabel(r"$\dot{\Omega}_\alpha^{{\rm Adaptive}} - \dot{\Omega}_\alpha^{{\rm Dense}}$")

plt.tight_layout()
plt.savefig("adaptive_dense_phase_comparison_transposed.pdf", bbox_inches='tight')
plt.close()