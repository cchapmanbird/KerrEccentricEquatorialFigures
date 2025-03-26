import numpy as np
import timeit
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
        "Phi_r0": 13,
    }


def transform_masses(log_10_mass1, log_10_mass_ratio):
    # here mass_ratio is defined mass_2 / mass_1
    mass_1 = 10**log_10_mass1
    mass_ratio = 10**log_10_mass_ratio
    mass_2 = mass_ratio * mass_1
    return mass_1, mass_2


def gen_parameters(
    N_SAMPLES, duration, seed_in=314159, verbose=False, exact_length=True
):
    # get_p_at_t can fail for uniform priors, so if exact_length is specified then we
    # overdraw samples and only generate a number of paraeters until we are at the end

    if exact_length:
        # should be enough to guarantee the full N_SAMPLES is generated
        _n_samples = 2 * N_SAMPLES
    else:
        _n_samples = N_SAMPLES

    traj_module = EMRIInspiral(func=KerrEccEqFlux)

    log10_mass1_range = [5, 7]
    log10_massratio_range = [-6, -4]
    spin_range = [0.0, 0.999]
    ecc_range = [0.0, 0.9]

    prior_ranges = np.array(
        [log10_mass1_range, log10_massratio_range, spin_range, ecc_range]
    ).T

    seed(seed_in)
    samples = uniform(low=prior_ranges[0], high=prior_ranges[1], size=(_n_samples, 4))

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

    successful_params = 0

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
            successful_params += 1
        except ValueError:
            failed_params_list.append([mass_1, mass_2, spin, ecc, duration])

        # this will break the loop if we have oversampled and request a specific
        # number of samples
        if successful_params == N_SAMPLES:
            break

    return output_params_list, failed_params_list


# create a function that times the FD and TD waveform generation for different input parameters
def time_full_waveform_generation(
    fd_waveform_func,
    td_waveform_func,
    input_params,
    waveform_kwargs_base,
    iterations=10,
    duration=1.0,
    verbose=False,
    vary_delta_t=False,
    vary_epsilon=False,
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

    dt_list = [5.0, 10.0, 15.0]
    eps_list = [1e-2, 1e-3, 1e-4]

    # run things once to cache, this will sometimes take a few seconds
    fd_waveform_func(*input_params[0], **waveform_kwargs_base)

    if verbose:
        iterator = tqdm(input_params, total=len(input_params))
    else:
        iterator = input_params

    if vary_delta_t:
        dt_list_to_use = dt_list
    else:
        dt_list_to_use = [waveform_kwargs_base["dt"]]

    if vary_epsilon:
        eps_list_to_use = eps_list
    else:
        eps_list_to_use = [waveform_kwargs_base["eps"]]

    for params in iterator:
        internal_param_list = []
        for dt in dt_list_to_use:
            for eps in eps_list_to_use:
                wvf_kwargs = waveform_kwargs_base.copy()
                wvf_kwargs.update({"dt": dt, "eps": eps})
                # Time FD waveform generation
                fd_start_time = timeit.default_timer()
                for _ in range(iterations):
                    fd_waveform_func(*params, **wvf_kwargs)

                fd_end_time = timeit.default_timer()

                fd_time = (fd_end_time - fd_start_time) / iterations

                # Time TD waveform generation
                td_start_time = timeit.default_timer()
                for _ in range(iterations):
                    td_waveform_func(*params, **wvf_kwargs)
                    
                td_end_time = timeit.default_timer()

                td_time = (td_end_time - td_start_time) / iterations

                internal_results_dict = {
                    "dt": wvf_kwargs["dt"],
                    "eps": wvf_kwargs["eps"],
                    "fd_timing": fd_time,
                    "td_timing": td_time,
                }

                internal_param_list.append(internal_results_dict.copy())

        # Store the results
        key_map = get_parameter_to_index_mapping()
        result = dict(parameters={k: params[i] for k, i in key_map.items()})
        result.update(
            {
                "duration": duration,
                "iterations": iterations,
                "timing_results": internal_param_list.copy(),
            }
        )
        results.append(result.copy())

    return results
