import unittest
import warnings
import time
from few.waveform import GenerateEMRIWaveform
import numpy as np

# Try to import CuPy for GPU acceleration, fall back to NumPy if not available
try:
    import cupy as xp
    use_gpu = True
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    warnings.warn("CuPy is not installed or a GPU is not available. Using NumPy instead.")
    use_gpu = False

# Keyword arguments for inspiral generator (Kerr Waveform)
inspiral_kwargs_Kerr = {
    "DENSE_STEPPING": 0,  # We want a sparsely sampled trajectory
    "max_init_len": int(1e3)  # All of the trajectories will be well under len = 1000
}

# Print whether we are using a GPU
print(f"Using GPU: {use_gpu}")

ntests = 1  # Number of tests to run for each parameter set

# Test the speed of the Kerr and Schwarzschild waveform models
inspiral_kwargs_Schwarz = inspiral_kwargs_Kerr.copy()
inspiral_kwargs_Schwarz["func"] = "SchwarzEccFlux"

# Initialize waveform generators
wave_generator_Kerr = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", use_gpu=use_gpu)
wave_generator_Schwarz = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu=use_gpu)

# Fixed parameters
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0
dist = 1.0
dt = 10.0
T = 2.0
eps = 1e-5  # Example epsilon value, adjust as needed

# Measure time for Kerr and Schwarzschild waveforms
M = 10**np.random.uniform(5, 7)
mu = np.random.uniform(1, 100)
p0 = np.random.uniform(10.0, 16.0)
e0 = np.random.uniform(0.0, 0.5)
wave_generator_Kerr(M, mu, 0.0, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
wave_generator_Schwarz(M, mu, 0.0, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)

# Lists to store the speed results and parameters
speed_kerr = []
speed_schwarz = []
params = []
size = 10000  # Number of tests to run

# Loop over the number of tests
for ii in range(size):
    print(f"Test {ii+1}/{size}")
    M = 10**np.random.uniform(5, 7)
    a = np.random.uniform(0.0, 0.99)
    mu = np.random.uniform(1, 100)
    p0 = np.random.uniform(10.0, 16.0)
    e0 = np.random.uniform(0.0, 0.5)

    # Measure time for Kerr waveform
    start_time = time.perf_counter()
    for _ in range(ntests):
        wave_generator_Kerr(M, mu, a, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt, eps=eps)
    kerr_duration = time.perf_counter() - start_time

    # Measure time for Schwarzschild waveform
    start_time = time.perf_counter()
    for _ in range(ntests):
        wave_generator_Schwarz(M, mu, 0.0, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt, eps=eps)
    schwarz_duration = time.perf_counter() - start_time

    # Print and store the results
    print(f"Kerr waveform generation time: {kerr_duration/ntests} seconds")
    print(f"Schwarzschild waveform generation time: {schwarz_duration/ntests} seconds")
    speed_kerr.append(kerr_duration / ntests)
    speed_schwarz.append(schwarz_duration / ntests)
    params.append([M, mu, a, p0, e0])

speed_kerr = np.array(speed_kerr)
speed_schwarz = np.array(speed_schwarz)
params = np.array(params)

# Save the results to files
np.save("speed_kerr.npy", speed_kerr)
np.save("speed_schwarz.npy", speed_schwarz)
np.save("params.npy", params)

# Save the configuration used
with open("config.txt", "w") as f:
    f.write(f"eps: {eps}\n")
    f.write(f"dt: {dt}\n")
    f.write(f"T: {T}\n")


# create corner plot of speed results
import corner
import matplotlib.pyplot as plt

# create corner plot and save to file
to_plot = np.hstack((params, speed_kerr[:,None]))
fig = corner.corner(to_plot, labels=[r"$M$", r"$\mu$", r"$a$", r"$p_0$", r"$e_0$", "speed"], show_titles=False)
plt.savefig("speed_corner_kerr.png")

to_plot = np.hstack((params, speed_schwarz[:,None]))
fig = corner.corner(to_plot, labels=[r"$M$", r"$\mu$", r"$a$", r"$p_0$", r"$e_0$", "speed"], show_titles=False)
plt.savefig("speed_corner_schw.png")
