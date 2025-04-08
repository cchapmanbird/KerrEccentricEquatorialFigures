import matplotlib.pyplot as plt
import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from tqdm import tqdm
from numpy.random import seed, uniform, randint
import time

from few.utils.constants import YRSID_SI

#Importing Flux Trajectory
from few.trajectory.ode import KerrEccEqFlux
from few.utils.utility import get_p_at_t

# Defining the duration of the trajectory
T = 4.0 # years

# Defining the number of samples to use

N = 1000

# Defining the different trajectory objects

full_flux = EMRIInspiral(func=KerrEccEqFlux)

skip_points = 2
downsampled_2 = EMRIInspiral(func=KerrEccEqFlux, downsample=[(skip_points, skip_points, skip_points), (skip_points, skip_points, skip_points)])

skip_points = 4
downsampled_4 = EMRIInspiral(func=KerrEccEqFlux,downsample=[(skip_points, skip_points, skip_points), (skip_points, skip_points, skip_points)])

skip_points = 8
downsampled_8 = EMRIInspiral(func=KerrEccEqFlux,downsample=[(skip_points, skip_points, skip_points), (skip_points, skip_points, skip_points)])

kip_points = 2
downsampled_u = EMRIInspiral(func=KerrEccEqFlux, downsample=[(skip_points, 1, 1), (skip_points, 1, 1)])

skip_points = 2
downsampled_w = EMRIInspiral(func=KerrEccEqFlux,downsample=[(1, skip_points, 1), (1, skip_points, 1)])

skip_points = 2
downsampled_z = EMRIInspiral(func=KerrEccEqFlux,downsample=[(1, 1, skip_points), (1, 1, skip_points)])

# Jonathan's randomised parameters code

def gen_parameters(NEVAL, duration, seed_in=314159):


    M_range = [1E5, 1E7]
    mu_range = [1,1E2]
    a_range = [-0.999, 0.999]
    e_range = [0.0, 0.9]

    x0 = 1.0  # will be ignored in Schwarzschild waveform

    _base_params = [
        1E5, # M
        10,  # mu
        0.0, # a
        0.0, # p0
        0.0, # e0
        x0,  # x0
    ]

    seed(seed_in)
    M_seed, mu_seed, a_seed, e_seed = randint(1E3, 1E5, size=4)

    seed(M_seed)
    M_list = uniform(low=M_range[0], high=M_range[1], size=NEVAL)
    seed(mu_seed)
    mu_list = uniform(low=mu_range[0], high=mu_range[1], size=NEVAL)
    seed(a_seed)
    a_list = uniform(low=a_range[0], high=a_range[1], size=NEVAL)
    seed(e_seed)
    e_list = uniform(low=e_range[0], high=e_range[1], size=NEVAL)

    output_params_list = []
    failed_params_list = []

    for i, (M, mu, a, e) in enumerate(zip(
        M_list,
        mu_list,
        a_list,
        e_list,
    )):

        try:
            # print(f"{i+1}:\t{M}, {mu}, {a}, {e}")
            updated_params = _base_params.copy()

            updated_params[0] = M
            updated_params[1] = mu
            updated_params[2] = a
            updated_params[4] = e
            updated_params[3] = get_p_at_t(
                full_flux,
                duration * 1.01,
                [updated_params[0], updated_params[1], updated_params[2], updated_params[4], 1.0],
                index_of_p=3,
                index_of_a=2,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-6,
                rtol=8.881784197001252e-6,
            )

            output_params_list.append(
                updated_params.copy()
            )
        except ValueError:
            failed_params_list.append([M, mu, a, e, duration])

    return output_params_list, failed_params_list

# Get randomised parameters
traj_pars = gen_parameters(N, T, seed_in=314159)[0]

# Calculate the dephasings
dephasings = np.zeros([len(traj_pars),6])

for i in range(len(traj_pars)):

    # print(f"Trajectory {i+1} of {len(traj_pars)}")
    t, p, e, xI, Phi_phi, Phi_theta, Phi_r  = full_flux(*traj_pars[i], T=T,err=1e-12)
    t2, p2, e2, xI2, Phi_phi2, Phi_theta2, Phi_r2  = downsampled_2(*traj_pars[i], T=T,err=1e-12)
    t4, p4, e4, xI4, Phi_phi4, Phi_theta4, Phi_r4  = downsampled_4(*traj_pars[i], T=T,err=1e-12)
    t8, p8, e8, xI8, Phi_phi8, Phi_theta8, Phi_r8  = downsampled_8(*traj_pars[i], T=T,err=1e-12)
    tu, pu, eu, xIu, Phi_phiu, Phi_thetau, Phi_ru  = downsampled_u(*traj_pars[i], T=T,err=1e-12)
    tw, pw, ew, xIw, Phi_phiw, Phi_thetaw, Phi_rw  = downsampled_w(*traj_pars[i], T=T,err=1e-12)
    tz, pz, ez, xIz, Phi_phiz, Phi_thetaz, Phi_rz  = downsampled_z(*traj_pars[i], T=T,err=1e-12)


    dephasings[i,0] = np.log10(np.abs(Phi_phi[-1] - Phi_phi2[-1]))
    dephasings[i,1] = np.log10(np.abs(Phi_phi[-1] - Phi_phi4[-1]))
    dephasings[i,2] = np.log10(np.abs(Phi_phi[-1] - Phi_phi8[-1]))
    dephasings[i,3] = np.log10(np.abs(Phi_phi[-1] - Phi_phiu[-1]))
    dephasings[i,4] = np.log10(np.abs(Phi_phi[-1] - Phi_phiw[-1]))
    dephasings[i,5] = np.log10(np.abs(Phi_phi[-1] - Phi_phiz[-1]))

np.savetxt('DownsampledFluxesData.txt', dephasings)



