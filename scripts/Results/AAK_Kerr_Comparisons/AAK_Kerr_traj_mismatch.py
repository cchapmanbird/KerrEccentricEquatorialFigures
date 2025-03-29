import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
sys.path.append("../")

# from scipy.signal import tukey       # I'm always pro windowing.  

from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, KerrEccEqFlux

from few.utils.utility import get_separatrix, get_p_at_t

def compute_traj(M,mu,a,p0,e0,T = 4):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: two separate trajectories from ELQ and pex module 
    """
    
    # Compute value of p to give T year inspiral
    # Compute trajectories for ELQ and pex
    out_pn5 = traj_pn5(M, mu, a, p0, e0, 1.0, T=T)  # ELQ
    out_Kerr = traj_Kerr(M, mu, a, p0, e0, 1.0, T=T,  
                       new_t=out_pn5[0], upsample=True) # pex, NOTE: using ELQ time array. 


    return out_pn5, out_Kerr

def check_phasing(M,mu,a,p0,e0, report_results = False, T = 4):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: phasing information from the ELQ and pex traj modules

    Very useful for debugging and checking!  
    """

    out_pn5, out_Kerr = compute_traj(M,mu,a,p0,e0,T = T)
    Delta_phi = abs(out_pn5[-3][-1] - out_Kerr[-3][-1]) 

    if report_results == True:
        print("Difference in p, final point", abs(out_pn5[1][-1] - out_Kerr[1][-1])) # semi-latus rectum
        print("Difference in e, final point", abs(out_pn5[2][-1] - out_Kerr[2][-1])) # Eccentricity
        print("Difference in Phi_phi, final point", Delta_phi)

    return Delta_phi, out_pn5, out_Kerr

##======================Likelihood and Posterior (change this)=====================

M = 1e6; mu = 10.0; a = 0.998

T = 4.0     # Evolution time [years]

## ===================== BUILD TRAJECTORIES ====================
# 
traj_Kerr = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
traj_pn5 = EMRIInspiral(func=PN5)  # Set up trajectory module, pn5 AAK

fig, ax = plt.subplots(1,3, figsize = (16,7), sharey = True)
from tqdm import tqdm as tqdm
from matplotlib.colors import LogNorm

# Production runs 
p0_vec = np.arange(7,60.25,0.25) 
e0_vec = np.arange(0.01,0.8,0.01)

# Test runs
# p0_vec = np.arange(5,60.25,5)
# e0_vec = np.arange(0.01,0.8,0.1)

M_mu_vec = [[1e5,1.0], [1e6, 1e1], [1e7, 1e2]]
labels = [["1e5", "1"], ["1e6", "10"], ["1e7", "100"]]
k = 0
for M_mu in tqdm(M_mu_vec):
    # Initialize heat map storage
    plunge_mask = np.zeros((len(e0_vec),len(p0_vec)))
    delta_phi_values = np.zeros((len(e0_vec), len(p0_vec)))
    for i, e0 in (enumerate(e0_vec)):
        for j, p0 in enumerate(p0_vec):
            try:
                # Compute Delta_phi for each (e0, p0) pair
                delta_phi_values[i, j], out_pn5, out_kerr = check_phasing(M_mu[0], M_mu[1], a, p0, e0, T=T)

                # Check if plunging
                p_sep = get_separatrix(a, e0, 1.0)
                if out_pn5[1][-1] < (p_sep + 0.1):  
                    plunge_mask[i, j] = 1  # Mark as plunging

            except AssertionError or ValueError:
                delta_phi_values[i, j] = np.nan  # Assign NaN to invalid values
                plunge_mask[i, j] = np.nan  # Mask invalid regions


    # Create meshgrid
    P0, E0 = np.meshgrid(p0_vec, e0_vec)

    # Use pcolormesh for smooth gradients
    cmap = plt.get_cmap("viridis").copy()  # Copy the viridis colormap
    cmap.set_bad("grey")  # Set NaN values to appear as grey

    pcm = ax[k].pcolormesh(P0, E0, delta_phi_values, shading='auto', cmap=cmap, norm=LogNorm(vmin=1e-5, vmax=1e6))
    
    plunge_contour = ax[k].contour(P0, E0, plunge_mask, levels=[0.5], colors='red', linewidths=1, linetyles = 'dashed')
    ax[k].clabel(plunge_contour, fmt={0.5: "\u00A0Plunge\u00A0"}, colors='red', fontsize=15)

    # Define contour levels at 0.1 and 1 radians
    contour_levels = [0.1, 1]

    # Add white contour lines on top of the heatmap
    contours = ax[k].contour(P0, E0, delta_phi_values, levels=contour_levels, colors='white', linewidths=1, linestyles='dashed')

    # Add labels to contours
    ax[k].clabel(contours, fmt={0.1: r"$0.1$", 1: r"$1$"}, colors='white', fontsize=15)
    
    # Set labels and title
    ax[k].set_xlabel(r'$p_0$', fontsize=20)
    if k == 0:  # Only leftmost plot gets the Y-label
        ax[k].set_ylabel(r'$e_0$', fontsize=15)
    ax[k].set_title(rf'$(M/M_\odot, \mu/M_\odot, a) = ({labels[k][0]}, {labels[k][1]}, 0.998)$',fontsize = 14)

    k+=1
# Add a single colorbar for all subplots
cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.15)
cbar.set_label(r'$|\Phi^{(Kerr)}_{\phi,-1} - \Phi^{(5PN)}_{\phi,-1}|$', fontsize=25)
cbar.ax.tick_params(labelsize=20)  # Set tick label font size

plot_dir = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/plots/traj_plots/"
plt.savefig(plot_dir + "dephasing_PN_vs_Kerr_masses_w_contours_w_plunge_contours.png",bbox_inches="tight")

plt.show()