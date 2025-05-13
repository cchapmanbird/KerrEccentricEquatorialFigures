import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, KerrEccEqFlux

from few.utils.geodesic import get_separatrix
from tqdm import tqdm as tqdm
import h5py

# Define functions
def compute_traj(M, mu, a, p0, e0, T=4):
    out_pn5 = traj_pn5(M, mu, a, p0, e0, 1.0, T=T, err=1e-10)  # ELQ
    out_Kerr = traj_Kerr(M, mu, a, p0, e0, 1.0, T=T, err=1e-10)  # pex

    if out_Kerr[0][-1] > out_pn5[0][-1]:
        out_Kerr = traj_Kerr(M, mu, a, p0, e0, 1.0, T=T, new_t=out_pn5[0], upsample=True, err=1e-10)  # pex
    else:
        out_pn5 = traj_pn5(M, mu, a, p0, e0, 1.0, T=T, new_t=out_Kerr[0], upsample=True, err=1e-10)  # ELQ
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

traj_Kerr = EMRIInspiral(func=KerrEccEqFlux)
traj_pn5 = EMRIInspiral(func=PN5)

p0_vec = np.arange(7, 60.25, 0.25) 
e0_vec = np.arange(0.01, 0.81, 0.01)

M_mu_vec = [[1e5, 1.0], [1e6, 10.0], [1e7, 100.0]]
labels = [["1e5", "1"], ["1e6", "10"], ["1e7", "100"]]

results = []
for M_mu in M_mu_vec:
    plunge_mask = np.zeros((len(e0_vec), len(p0_vec)))
    delta_phi_values = np.zeros((len(e0_vec), len(p0_vec)))
    for i, e0 in tqdm(enumerate(e0_vec), total=len(e0_vec)):
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

# Save the results to a file
with h5py.File("Kerr_PN_trajectory_data.h5", "w") as f:
    for k, (delta_phi_values, plunge_mask) in enumerate(results):
        f.create_dataset(f"delta_phi_values_{k}", data=delta_phi_values)
        f.create_dataset(f"plunge_mask_{k}", data=plunge_mask)
    
    f.create_dataset("p0_vec", data=p0_vec)
    f.create_dataset("e0_vec", data=e0_vec)

