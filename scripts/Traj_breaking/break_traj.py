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
traj = EMRIInspiral(func=KerrEccEqFluxAPEX)  # Set up trajectory module, pn5 AAK


traj_args = [M, mu, a, e_traj[0], Y_traj[0]]
index_of_p = 3
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    # bounds=[6, 15]
    bounds=None
)


print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))

print("Now going to load in class")

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