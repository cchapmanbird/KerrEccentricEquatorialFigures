import numpy as np
import time
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.utility import get_p_at_t
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import h5py

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


if __name__ == "__main__":
    insp_kw = {
        "T": 10.0,
        "dt": 1.0,
        "err": 1e-11,
        "DENSE_STEPPING": 0,
        "buffer_length": int(1e4),
        "upsample": False,
    }

    with h5py.File('ODEerror_timing_dephasing_data.h5', 'w') as f:
        Ntest = 100
        err_vec = np.logspace(-11, -6, 6)
        mass_ratio_vec = np.array([1e-4, 1e-5, 1e-6])
        f.attrs['Ntest'] = Ntest
        f.attrs['err_vec'] = err_vec
        f.attrs['mass_ratio'] = mass_ratio_vec

        for mass_ratio in mass_ratio_vec:
            print(f"Mass ratio: {mass_ratio}")
            grp = f.create_group(f'mass_ratio_{mass_ratio}')            
            traj = EMRIInspiral(func=KerrEccEqFlux)
            pvec, evec, avec, Mvec = initialize_parameters(traj, Ntest, mass_ratio, Tobs=4.0, seed=42)
            print("Initial parameters generated")
            
            total_array = []
            for ii in tqdm(range(Ntest)):
                try:
                    results = compute_results(traj, Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii], err_vec, insp_kw)
                except ValueError:
                    breakpoint()
                N_points = [results[err][0] for err in err_vec]
                phase_difference = [results[err][1] for err in err_vec]
                timing = [results[err][2] for err in err_vec]
                par = [Mvec[ii], mass_ratio, avec[ii], pvec[ii], evec[ii]]
                total_array.append(np.asarray(par + N_points + phase_difference + timing))
            total_array = np.asarray(total_array)
            Np = np.asarray(total_array[:, len(par):len(par) + len(N_points)])
            dphi = np.asarray(total_array[:, len(par) + len(N_points):len(par) + len(N_points) + len(phase_difference)])
            eval_timing = np.asarray(total_array[:, len(par) + len(N_points) + len(phase_difference):])

            grp.create_dataset('N_points', data=Np)
            grp.create_dataset('phase_difference', data=dphi)
            grp.create_dataset('timing', data=eval_timing)
            # np.savez('results_ode_error_timing_phase.npz', M_a_p_e_N_dphi_timing=total_array, error_array=err_vec, header='M, massratio, a, p0, e0, N_points, phase_difference, timing')
    