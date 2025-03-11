import os 
import sys
from tqdm import tqdm as tqdm

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

# Import relevant EMRI packages

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.utils.utility import get_separatrix, get_p_at_t

MAKE_PLOT = True


YRSID_SI = 31558149.763545603


# Set flux conventions. Integrate ELQ or pex
inspiral_kwargs_ELQ = {'flux_output_convention':'ELQ',
                        'err':1e-12}
inspiral_kwargs_pex = {'flux_output_convention':'pex',
                        'err':1e-12}

# initialise classes 
traj_ELQ = EMRIInspiral(func=KerrEccEqFlux, **inspiral_kwargs_ELQ)
traj_pex = EMRIInspiral(func=KerrEccEqFlux, **inspiral_kwargs_pex)

def compute_traj(M,mu,a,e0,T = 4):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: two separate trajectories from ELQ and pex module 
    """
    
    traj_args = [M, mu, a, e0, 1.0]
    # Compute value of p to give T year inspiral
    p_new = get_p_at_t(
        traj_pex,
        T,
        traj_args,
        bounds=None
    )
    # Compute trajectories for ELQ and pex
    out_ELQ = traj_ELQ(M, mu, a, p_new, e0, 1.0, T=T)  # ELQ
    out_pex = traj_pex(M, mu, a, p_new, e0, 1.0, T=T,  
                       new_t=out_ELQ[0], upsample=True) # pex, NOTE: using ELQ time array. 


    return out_ELQ, out_pex

def check_phasing(M,mu,a,e0,T = 2):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: phasing information from the ELQ and pex traj modules

    Very useful for debugging and checking!  
    """

    out_ELQ, out_pex = compute_traj(M,mu,a,e0,T = 4)

    print("Difference in p, final point", abs(out_ELQ[1][-1] - out_pex[1][-1])) # semi-latus rectum
    print("Difference in e, final point", abs(out_ELQ[2][-1] - out_pex[2][-1])) # Eccentricity
    print("Difference in Phi_phi, final point", abs(out_ELQ[-3][-1] - out_pex[-3][-1])) # orbital phase in phi

    print("Number of data points in pex: ", out_ELQ[0].shape)
    print("Number of data points in ELQ: ", out_pex[0].shape)

    print("")
    return out_ELQ, out_pex

print("Feel free to debug with check_phasing(M,mu,a,e0,T = 2) module")
breakpoint()

# Fix secondary mass mu and spin parameter.
# Four year observation. Spin chosen to be worst case scenario (super strong field)
mu = 10; a = 0.999; T = 4.0

j = 0
Phi_phi_dephasing_vec = [[],[],[]]

M_vec = [1e5, 1e6, 1e7] # Chosen primary masses
e_vec = np.arange(0.01,0.8, 0.01) # Eccentricities
for M_val in M_vec:
    print(f"Now focusing on M = {M_val}")
    for e_val in tqdm(e_vec):
        try:
            out_ELQ, out_pex = compute_traj(M_val,mu,a,e_val,T = T)  # Compute two trajectory modules 
        except ValueError:
            break
        dephasing_phi = abs(out_pex[-3][-1] - out_ELQ[-3][-1])  # Compute dephasing between each
        Phi_phi_dephasing_vec[j].append(dephasing_phi)          # Add dephasing to list
    j+=1

# PLOT THE RESULTS 

label_M_vec = [r"$10^5$",r"$10^6$",r"$10^7$"]
for j in range(0,3):
    plt.semilogy(e_vec[0:len(Phi_phi_dephasing_vec[j])], Phi_phi_dephasing_vec[j], 'o', label = rf'$(M,\mu,a)$ = ({label_M_vec[j]},{mu},{a})')
plt.xlabel(r'eccentricity')
plt.ylabel(r'$|\Phi^{ELQ}_{\phi} - \Phi^{pex}_{\phi}|_{-1}$', fontsize = 18)
plt.axhline(y = 1, linestyle = 'dashed', c = 'black', label = r'Criterion')
plt.grid(True)
plt.legend()
plt.savefig(f"plots/comparison_ELQ_pex_phasing_subtract_PN_norms.png",bbox_inches = "tight")
breakpoint()
quit()
