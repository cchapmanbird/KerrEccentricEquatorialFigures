import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
sys.path.append("../")

from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, KerrEccEqFlux

from few.utils.utility import get_separatrix, get_p_at_t
from tqdm import tqdm as tqdm
from matplotlib.colors import LogNorm

# Define functions
def compute_traj(M, mu, a, p0, e0, T=4):
    out_pn5 = traj_pn5(M, mu, a, p0, e0, 1.0, T=T)  # ELQ
    out_Kerr = traj_Kerr(M, mu, a, p0, e0, 1.0, T=T, new_t=out_pn5[0], upsample=True)  # pex
    return out_pn5, out_Kerr

def check_phasing(M, mu, a, p0, e0, report_results=False, T=4):
    out_pn5, out_Kerr = compute_traj(M, mu, a, p0, e0, T=T)
    Delta_phi = abs(out_pn5[-3][-1] - out_Kerr[-3][-1]) 

    if report_results:
        print("Difference in p, final point", abs(out_pn5[1][-1] - out_Kerr[1][-1]))  # semi-latus rectum
        print("Difference in e, final point", abs(out_pn5[2][-1] - out_Kerr[2][-1]))  # Eccentricity
        print("Difference in Phi_phi, final point", Delta_phi)

    return Delta_phi, out_pn5, out_Kerr

# ===================== COMPUTE QUANTITIES =====================
M = 1e6; mu = 10.0; a = 0.998
T = 4.0  # Evolution time [years]

traj_Kerr = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
traj_pn5 = EMRIInspiral(func=PN5)  # Set up trajectory module, pn5 AAK

p0_vec = np.arange(7, 60.25, 0.25) 
e0_vec = np.arange(0.01, 0.8, 0.01)

M_mu_vec = [[1e5, 1.0], [1e6, 10.0], [1e7, 100.0]]
labels = [["1e5", "1"], ["1e6", "10"], ["1e7", "100"]]

results = []
for M_mu in tqdm(M_mu_vec):
    plunge_mask = np.zeros((len(e0_vec), len(p0_vec)))
    delta_phi_values = np.zeros((len(e0_vec), len(p0_vec)))
    for i, e0 in enumerate(e0_vec):
        for j, p0 in enumerate(p0_vec):
            try:
                delta_phi_values[i, j], out_pn5, out_kerr = check_phasing(M_mu[0], M_mu[1], a, p0, e0, T=T)
                p_sep = get_separatrix(a, e0, 1.0)
                if out_pn5[1][-1] < (p_sep + 0.1):  
                    plunge_mask[i, j] = 1  # Mark as plunging
            except (AssertionError, ValueError):
                delta_phi_values[i, j] = np.nan  # Assign NaN to invalid values
                plunge_mask[i, j] = np.nan  # Mask invalid regions
    results.append((delta_phi_values, plunge_mask))

# ===================== GENERATE PLOT =====================
fig, ax = plt.subplots(1, 3, figsize=(16, 7), sharey=True)

for k, (delta_phi_values, plunge_mask) in enumerate(results):
    P0, E0 = np.meshgrid(p0_vec, e0_vec)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("grey")

    pcm = ax[k].pcolormesh(P0, E0, delta_phi_values, shading='auto', cmap=cmap, norm=LogNorm(vmin=1e-5, vmax=1e6))
    plunge_contour = ax[k].contour(P0, E0, plunge_mask, levels=[0.5], colors='red', linewidths=1, linestyles='dashed')
    ax[k].clabel(plunge_contour, fmt={0.5: "\u00A0Plunge\u00A0"}, colors='red', fontsize=15)

    contour_levels = [0.1, 1]
    contours = ax[k].contour(P0, E0, delta_phi_values, levels=contour_levels, colors='white', linewidths=1, linestyles='dashed')
    ax[k].clabel(contours, fmt={0.1: r"$0.1$", 1: r"$1$"}, colors='white', fontsize=15)

    ax[k].set_xlabel(r'$p_0$', fontsize=20)
    if k == 0:
        ax[k].set_ylabel(r'$e_0$', fontsize=15)
    ax[k].set_title(rf'$(M/M_\odot, \mu/M_\odot, a) = ({labels[k][0]}, {labels[k][1]}, 0.998)$', fontsize=14)

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.15)
cbar.set_label(r'$|\Phi^{(Kerr)}_{\phi,-1} - \Phi^{(5PN)}_{\phi,-1}|$', fontsize=25)
cbar.ax.tick_params(labelsize=20)

plot_dir = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/plots/traj_plots/"
plt.savefig(plot_dir + "dephasing_PN_vs_Kerr_masses_w_contours_w_plunge_contours.png", bbox_inches="tight")
plt.show()
