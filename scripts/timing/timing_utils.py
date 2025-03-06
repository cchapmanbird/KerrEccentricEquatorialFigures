import numpy as np
import time
from tqdm import tqdm
from numpy.random import seed, uniform

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.utility import get_p_at_t

def get_parameter_to_index_mapping():

    return {
            "mass_1": 0,
            "mass_2": 1,
            "spin": 2,
            "p0": 3,
            "e0": 4,
            "x0": 5,
            "dist": 6,
            "qS": 7,
            "phiS": 8,
            "qK": 9,
            "phiK": 10,
            "Phi_phi0": 11,
            "Phi_theta0": 12,
            "Phi_r0": 13
        }


def transform_masses(log_mass1, log_mass_ratio):
    # here mass_ratio is defined mass_2 / mass_1
    mass_1 = 10**log_mass1
    mass_ratio = 10**log_mass_ratio
    mass_2 = mass_ratio * mass_1
    return mass_1, mass_2


def gen_parameters(N_SAMPLES, duration, seed_in=314159, verbose=False):

    traj_module = EMRIInspiral(func=KerrEccEqFlux)

    log10_mass1_range = [5, 7]
    log10_massratio_range = [-6, -4]
    spin_range = [0.0, 0.999]
    ecc_range = [0.0, 0.9]

    prior_ranges = np.array(
        [log10_mass1_range, log10_massratio_range, spin_range, ecc_range]
    ).T

    seed(seed_in)
    samples = uniform(low=prior_ranges[0], high=prior_ranges[1], size=(N_SAMPLES, 4))

    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = np.pi / 3  # polar spin angle
    phiK = np.pi / 3  # azimuthal viewing angle
    qS = np.pi / 3  # polar sky angle
    phiS = np.pi / 3  # azimuthal viewing angle
    distance = 1.0  # distance
    # initial phases
    Phi_phi0 = np.pi / 3
    Phi_theta0 = 0.0
    Phi_r0 = np.pi / 3

    _base_params = [
        1e5,  # mass_1
        10,  # mass_1
        0.0,  # spin
        0.0,  # p0
        0.0,  # e0
        x0,
        distance,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
    ]

    output_params_list = []
    failed_params_list = []

    for i, (log_10_m1, log_10_eta, spin, ecc) in enumerate(samples):

        mass_1, mass_2 = transform_masses(log_10_m1, log_10_eta)

        try:
            if verbose:
                print(f"{i+1}:\t{mass_1}, {mass_2}, {spin}, {ecc}")

            updated_params = _base_params.copy()

            updated_params[0] = mass_1
            updated_params[1] = mass_2
            updated_params[2] = spin
            updated_params[4] = ecc


            print(updated_params)

            updated_params[3] = get_p_at_t(
                traj_module,
                duration * 0.99,
                [
                    updated_params[0],
                    updated_params[1],
                    updated_params[2],
                    updated_params[4],
                    1.0,
                ],
                index_of_a=2,
                index_of_p=3,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-6,
                rtol=8.881784197001252e-6,
            )

            output_params_list.append(updated_params.copy())
        except ValueError:
            failed_params_list.append([mass_1, mass_2, spin, ecc, duration])

    return output_params_list, failed_params_list


# create a function that times the FD and TD waveform generation for different input parameters
def time_full_waveform_generation(
    fd_waveform_func,
    td_waveform_func,
    input_params,
    waveform_kwargs,
    iterations=10,
    duration=1.0,
    delta_t=5.0,
    verbose=False,
):
    """
    Times the FD and TD waveform generation for different input parameters.

    Parameters:
    fd_waveform_func (function): Function to generate FD waveform.
    td_waveform_func (function): Function to generate TD waveform.
    input_params (list): List of dictionaries containing input parameters for the waveform functions.
    waveform_kwargs (dict): Dictionary of waveform kwargs
    iterations (int, default 10): number of waveform generation iterations to average timing over
    verbose (bool, default True): verbose runtime information using tqdm

    Returns:
    list: List of dictionaries containing input parameters and their corresponding FD and TD generation times.
    """
    results = []

    if verbose:
        iterator = tqdm(input_params, total=len(input_params))
    else:
        iterator = input_params

    for params in iterator:
        # Time FD waveform generation
        start_time = time.time()
        for _ in range(iterations):
            fd_waveform_func(*params, **waveform_kwargs)

        fd_time = (time.time() - start_time) / iterations

        # Time TD waveform generation
        start_time = time.time()
        for _ in range(iterations):
            td_waveform_func(*params, **waveform_kwargs)

        td_time = (time.time() - start_time) / iterations

        # Store the results
        key_map = get_parameter_to_index_mapping()
        result = {k:params[i] for k,i in key_map.items()}
        result.update({
            "duration": duration,
            "delta_t": delta_t,
            "iterations": iterations,
            "fd_time": fd_time,
            "td_time": td_time,
        })
        results.append(result.copy())

    return results
