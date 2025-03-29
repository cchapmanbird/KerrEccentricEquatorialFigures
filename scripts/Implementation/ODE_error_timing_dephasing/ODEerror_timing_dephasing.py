import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux
from few.utils.globals import get_logger
from few.utils.utility import get_p_at_t
from scipy.interpolate import CubicSpline
def initialize_parameters(traj, Ntest, massratio, Tobs=1.0, seed=42):
    np.random.seed(seed)
    evec = np.random.uniform(0.0, 0.7, Ntest)
    avec = np.random.uniform(0.0, 0.999, Ntest)
    Mvec = 10**np.random.uniform(5, 7, Ntest)
    # get
    
    pvec = np.asarray([get_p_at_t(traj,Tobs, 
                [M, massratio*M, a, ecc, 1.0],
                index_of_p=3,index_of_a=2,
                index_of_e=4,index_of_x=5,
                traj_kwargs={},xtol=2e-6,rtol=8.881784197001252e-6,bounds=None,) for M, ecc, a in zip(Mvec, evec, avec)])
    return pvec, evec, avec, Mvec

def get_N_Phif_evalT(traj, M, mu, a, p0, e0, err, insp_kw):
    insp_kw['err'] = err
    tic = time.perf_counter()
    for _ in range(5):
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, **insp_kw)
    toc = time.perf_counter()
    return t, Phi_phi, (toc - tic)/5

def compute_results(traj, M, mass_ratio, a, p0, e0, err_vec, insp_kw):
    # create dictionary to store results
    results = {err: [] for err in err_vec}
    # interpolate with cubic splines the time and phi
    
    t_true, Phi_phi_true, evalT_true = get_N_Phif_evalT(traj, M, mass_ratio*M, a, p0, e0, 1e-11, insp_kw)
    cs_true = CubicSpline(t_true, Phi_phi_true)
    # loop over errors
    for err in err_vec:
        # loop over initial conditions
        t, Phi_phi, evalT = get_N_Phif_evalT(traj, M, mass_ratio*M, a, p0, e0, err, insp_kw)
        cs = CubicSpline(t, Phi_phi)
        t_final = np.min([t[-1], t_true[-1]])
        results[err] = [len(t), np.abs(cs(t_final) - cs_true(t_final)), evalT]
    return results

def plot_results(err_vec, phase_difference, timing, N_points, Ntest, mass_ratio):

    plt.figure(figsize=(10, 8))

    # Plot 1: Mean phase difference
    plt.subplot(3, 1, 1)
    plt.title(f'Average over {Ntest} random initial conditions')
    median_phase_diff = np.median(phase_difference, axis=0)
    sigma_phase_diff = np.std(phase_difference, axis=0)
    
    plt.loglog(err_vec, median_phase_diff, '-o', label=f'mass ratio={mass_ratio}')
    plt.fill_between(err_vec, median_phase_diff - sigma_phase_diff, median_phase_diff + sigma_phase_diff, alpha=0.3)
    # plt.loglog(err_vec, phase_difference, '-o', label=f'mass ratio={mass_ratio}')
    plt.axhline(1.0, color='k', linestyle='--', label='1.0 rad')
    plt.axhline(0.1, color='k', linestyle='-', label='0.1 rad')
    plt.ylabel('Final Phase Difference')
    plt.legend()

    # Plot 2: Mean timing
    plt.subplot(3, 1, 2)
    median_timing = np.median(timing, axis=0)
    sigma_timing = np.std(timing, axis=0)
    
    plt.semilogx(err_vec, median_timing, '-o', label=f'mass ratio={mass_ratio}')
    plt.fill_between(err_vec, median_timing - sigma_timing, median_timing + sigma_timing, alpha=0.3)
    # plt.semilogx(err_vec, timing, '-o', label=f'mass ratio={mass_ratio}')
    plt.ylabel('Timing [seconds]')

    # Plot 3: Mean number of points
    plt.subplot(3, 1, 3)
    median_N_points = np.median(N_points, axis=0)
    sigma_N_points = np.std(N_points, axis=0)
    plt.semilogx(err_vec, median_N_points, '-o', label=f'mass ratio={mass_ratio}')
    plt.fill_between(err_vec, median_N_points - sigma_N_points, median_N_points + sigma_N_points, alpha=0.3)
    # plt.semilogx(err_vec, N_points, '-o', label=f'mass ratio={mass_ratio}')
    plt.xlabel('Error ODE')
    plt.ylabel('Number of Points')

    plt.savefig(f'Trajectory_timing_ODEerror_dephasing.png')

if __name__ == "__main__":
    insp_kw = {
        "T": 10.0,
        "dt": 1.0,
        "err": 1e-10,
        "DENSE_STEPPING": 0,
        "buffer_length": int(1e4),
        "upsample": False,
    }

    mass_ratio = 1e-4
    err_vec = np.logspace(-10, -5, 5)
    
    traj = EMRIInspiral(func=KerrEccEqFlux)
    
    Ntest = 100
    pvec, evec, avec, Mvec = initialize_parameters(traj, Ntest, mass_ratio, Tobs=4.0, seed=42)
    print("Initial parameters generated")
    total_array = []
    for ii in range(Ntest):
        
        results = compute_results(traj, Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii], err_vec, insp_kw)

        N_points = [results[err][0] for err in err_vec]
        phase_difference = [results[err][1] for err in err_vec]
        timing = [results[err][2] for err in err_vec]
        par = [Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii]]
        total_array.append(np.asarray(par + N_points + phase_difference + timing))
    total_array = np.asarray(total_array)
    Np = np.asarray(total_array[:, len(par):len(par) + len(N_points)])
    dphi = np.asarray(total_array[:, len(par) + len(N_points):len(par) + len(N_points) + len(phase_difference)])
    eval_timing = np.asarray(total_array[:, len(par) + len(N_points) + len(phase_difference):])
    np.savez('results_ode_error_timing_phase.npz', M_a_p_e_N_dphi_timing=total_array, error_array=err_vec, header='M, massratio, a, p0, e0, N_points, phase_difference, timing')
    plot_results(err_vec, dphi, eval_timing, Np, Ntest, mass_ratio)
