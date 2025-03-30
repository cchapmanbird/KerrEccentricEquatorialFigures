import numpy as np
import sys

# Import relevant EMRI packages
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, KerrEccEqFlux

from few.utils.utility import get_separatrix
from tqdm import tqdm as tqdm

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
num = 10
p0_vec = np.linspace(7, 60.25, num) 
e0_vec = np.linspace(0.01, 0.8, num)

M_mu_vec = [[M_,mu_,a_] for mu_ in [1.0, 10., 100., 1000.] for M_ in [1e5, 1e6, 1e7] for a_ in [-0.99, -0.5, 0.0, 0.5, 0.99]]

results = []
max_dephasing = 0  # Initialize variable to track the largest dephasing
max_dephasing_params = None  # To store the parameters corresponding to the largest dephasing

for M_mu in tqdm(M_mu_vec):
    plunge_mask = np.zeros((len(e0_vec), len(p0_vec)))
    delta_phi_values = np.zeros((len(e0_vec), len(p0_vec)))
    for i, e0 in enumerate(e0_vec):
        for j, p0 in enumerate(p0_vec):
            print(f"Computing for M={M_mu[0]}, mu={M_mu[1]}, a={a}, p0={p0}, e0={e0}")
            try:
                delta_phi, out_pn5, out_kerr = check_phasing(M_mu[0], M_mu[1], M_mu[2], p0, e0, T=T)
                delta_phi_values[i, j] = delta_phi
                p_sep = get_separatrix(a, e0, 1.0)
                if out_pn5[1][-1] < (p_sep + 0.1):  
                    plunge_mask[i, j] = 1  # Mark as plunging

                # Update max dephasing if current delta_phi is larger
                if delta_phi > max_dephasing:
                    max_dephasing = delta_phi
                    max_dephasing_params = (M_mu[0], M_mu[1], M_mu[2], p0, e0)

            except (AssertionError, ValueError):
                delta_phi_values[i, j] = np.nan  # Assign NaN to invalid values
                plunge_mask[i, j] = np.nan  # Mask invalid regions
    results.append((delta_phi_values, plunge_mask))

# Print the largest dephasing and its parameters
print(f"Largest dephasing: {max_dephasing}")
if max_dephasing_params:
    print(f"Parameters for largest dephasing: M={max_dephasing_params[0]}, mu={max_dephasing_params[1]}, "
          f"a={max_dephasing_params[2]}, p0={max_dephasing_params[3]}, e0={max_dephasing_params[4]}")

# save to hdf5
import h5py
with h5py.File("trajectory_data.h5", "w") as f:
    for i, (delta_phi_values, plunge_mask) in enumerate(results):
        group = f.create_group(f"M{M_mu_vec[i][0]}_mu{M_mu_vec[i][1]}_a{M_mu_vec[i][2]}")
        group.create_dataset("delta_phi_values", data=delta_phi_values)
        group.create_dataset("plunge_mask", data=plunge_mask)
        group.create_dataset("p0_vec", data=p0_vec)
        group.create_dataset("e0_vec", data=e0_vec)
        group.create_dataset("M", data=M_mu_vec[i][0])
        group.create_dataset("mu", data=M_mu_vec[i][1])
        group.create_dataset("a", data=M_mu_vec[i][2])
        group.create_dataset("T", data=T)
        group.create_dataset("num", data=num)
        group.attrs['description'] = "Trajectory data for different M and mu values"

