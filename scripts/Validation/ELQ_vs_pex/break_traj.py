import os 
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

# from lisatools.sensitivity import noisepsd_AE2,noisepsd_T # Power spectral densities

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux, KerrEccEqFluxAPEX

from few.utils.utility import get_separatrix, get_p_at_t

# Import features from eryn

MAKE_PLOT = True
# Import parameters
sys.path.append("/home/ad/burkeol/work/Parameter_Estimation_EMRIs/Kerr_FEW_PE/mcmc_code")
from EMRI_settings import (M, mu, a, p0, e0, x_I0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T, delta_t)
use_gpu = True

YRSID_SI = 31558149.763545603
N_channels = 2
xp = cp

M = 1e7
mu = 1e5
p0 = 23.6015
a = 0.95
e0 = 0.85
x_I0 = 1.0

T = 2.0
delta_t = 10.0

## ===================== CHECK TRAJECTORY ====================
# 
breakpoint()

inspiral_kwargs_ELQ = {'flux_output_convention':'ELQ'}
inspiral_kwargs_pex = {'flux_output_convention':'pex'}

traj_ELQ = EMRIInspiral(func=KerrEccEqFlux, inspiral_kwargs = inspiral_kwargs_ELQ) # Inspiral with ELQ
traj_pex = EMRIInspiral(func=KerrEccEqFlux, inspiral_kwargs = inspiral_kwargs_pex) # Inspiral with pex

t_traj_ELQ, p_traj_ELQ, e_traj_ELQ, Y_traj_ELQ, Phi_phi_traj_ELQ, Phi_theta_traj_ELQ, Phi_r_traj_ELQ = traj_ELQ(M, mu, a, p0, e0, x_I0, T=T)
t_traj_pex, p_traj_pex, e_traj_pex, Y_traj_pex, Phi_phi_traj_pex, Phi_theta_traj_pex, Phi_r_traj_pex = traj_pex(M, mu, a, p0, e0, x_I0, T=T)

breakpoint()
# ================================ STOP ===================

quit()
# Compute trajectory 
import time
time_vec = []

for i in range(0,10):
    print(f"Working on iteration {i}")
    start = time.time()
    t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_theta_traj, Phi_r_traj = traj(M, mu, a, p0, e0, x_I0, T=T)
    time_vec.append(time.time() - start)

print("Minimum time to compute trajectories is ", np.min(time_vec))

breakpoint()