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
from seaborn import color_palette
cpal = color_palette("colorblind", 4)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16

def initialize_parameters(traj, Ntest, massratio, Tobs=1.0, seed=42):
    np.random.seed(seed)
    evec = np.random.uniform(0.0, 0.7, Ntest)
    avec = np.random.uniform(0.0, 0.999, Ntest)
    Mvec = 10**np.random.uniform(5, 6.5, Ntest)
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

def plot_results(err_vec, results_dict, Ntest):
    plt.figure(figsize=(7, 10))

    # Plot 1: Mean phase difference
    plt.subplot(3, 1, 1)
    plt.title(f'Average over {Ntest} random initial conditions', fontsize=title_fontsize)
    for idx, (mass_ratio, data) in enumerate(results_dict.items()):
        phase_difference = data['phase_difference']
        median_phase_diff = np.median(phase_difference, axis=0)
        sigma_phase_diff = np.std(phase_difference, axis=0)
        plt.loglog(err_vec, median_phase_diff, '-o', color=cpal[idx])#, label=f'mass ratio={mass_ratio}')
        plt.fill_between(err_vec, median_phase_diff - sigma_phase_diff, median_phase_diff + sigma_phase_diff, color=cpal[idx], alpha=0.3)
    plt.axhline(1.0, color='k', linestyle='--', label='1.0 rad')
    plt.axhline(0.1, color='k', linestyle='-', label='0.1 rad')
    plt.ylabel('Final Phase Difference', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize, loc='lower right')

    # Plot 2: Mean timing
    plt.subplot(3, 1, 2)
    for idx, (mass_ratio, data) in enumerate(results_dict.items()):
        timing = data['timing']
        median_timing = np.median(timing, axis=0)
        sigma_timing = np.std(timing, axis=0)
        plt.semilogx(err_vec, median_timing, '-o', color=cpal[idx], label=f'mass ratio={mass_ratio}')
        plt.fill_between(err_vec, median_timing - sigma_timing, median_timing + sigma_timing, color=cpal[idx], alpha=0.3)
    plt.ylabel('Timing [s]', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize)

    # Plot 3: Mean number of points
    plt.subplot(3, 1, 3)
    for idx, (mass_ratio, data) in enumerate(results_dict.items()):
        N_points = data['N_points']
        median_N_points = np.median(N_points, axis=0)
        sigma_N_points = np.std(N_points, axis=0)
        plt.semilogx(err_vec, median_N_points, '-o', color=cpal[idx], label=f'mass ratio={mass_ratio}')
        plt.fill_between(err_vec, median_N_points - sigma_N_points, median_N_points + sigma_N_points, color=cpal[idx], alpha=0.3)
    plt.xlabel('ODE Error tolerance', fontsize=label_fontsize)
    plt.ylabel('Number of Points', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.savefig(f'Trajectory_timing_ODEerror_dephasing_combined.png')

if __name__ == "__main__":
    insp_kw = {
        "T": 10.0,
        "dt": 1.0,
        "err": 1e-10,
        "DENSE_STEPPING": 0,
        "buffer_length": int(1e4),
        "upsample": False,
    }

    mass_ratios = [1e-6, 1e-5, 1e-4]
    err_vec = np.logspace(-10, -4, 6)
    traj = EMRIInspiral(func=KerrEccEqFlux)
    Ntest = 20

    results_dict = {}
    # for mass_ratio in mass_ratios:
    #     pvec, evec, avec, Mvec = initialize_parameters(traj, Ntest, mass_ratio, Tobs=4.0, seed=42)
    #     print(f"Initial parameters generated for mass ratio {mass_ratio}")
    #     total_array = []
    #     for ii in range(Ntest):
    #         print(f"Computing for M={Mvec[ii]}, mu={mass_ratio*Mvec[ii]}, a={avec[ii]}, p0={pvec[ii]}, e0={evec[ii]}")
    #         results = compute_results(traj, Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii], err_vec, insp_kw)
    #         N_points = [results[err][0] for err in err_vec]
    #         phase_difference = [results[err][1] for err in err_vec]
    #         timing = [results[err][2] for err in err_vec]
    #         par = [Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii]]
    #         total_array.append(np.asarray(par + N_points + phase_difference + timing))
    #     total_array = np.asarray(total_array)
    #     results_dict[mass_ratio] = {
    #         'N_points': np.asarray(total_array[:, len(par):len(par) + len(N_points)]),
    #         'phase_difference': np.asarray(total_array[:, len(par) + len(N_points):len(par) + len(N_points) + len(phase_difference)]),
    #         'timing': np.asarray(total_array[:, len(par) + len(N_points) + len(phase_difference):])
    #     }
    #     np.savez(f'results_ode_error_timing_phase_massratio_{mass_ratio:.0e}.npz', M_a_p_e_N_dphi_timing=total_array, error_array=err_vec, header='M, massratio, a, p0, e0, N_points, phase_difference, timing')

    # plot_results(err_vec, results_dict, Ntest)

    # Read the data and plot results
    for mass_ratio in mass_ratios:
        data = np.load(f'results_ode_error_timing_phase_massratio_{mass_ratio:.0e}.npz')
        results_dict[mass_ratio] = {
            'N_points': data['M_a_p_e_N_dphi_timing'][:, 5:5 + len(err_vec)],
            'phase_difference': data['M_a_p_e_N_dphi_timing'][:, 5 + len(err_vec):5 + 2 * len(err_vec)],
            'timing': data['M_a_p_e_N_dphi_timing'][:, 5 + 2 * len(err_vec):]
        }

    plot_results(err_vec, results_dict, Ntest)
