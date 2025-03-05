import numpy as np
import time
from tqdm import tqdm
from numpy.random import seed, uniform, randint

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.utility import get_p_at_t

# 9700484.894655073, 25.806654874357793, 0.7735415084736439, 0.6834440767965327 stuck parameters
# 352224.8117850187, 63.9495298217371, 0.424046325083201, 0.7687664552808469

def gen_parameters(NEVAL, duration, seed_in=314159):

    traj_module = EMRIInspiral(func=KerrEccEqFlux)

    M_range = [1E5, 1E7]
    mu_range = [1,1E2]
    a_range = [0.0, 0.999]
    e_range = [0.0, 0.9]

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
        1E5, # M
        10,  # mu
        0.0, # a
        0.0, # p0
        0.0, # e0
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
            print(f"{i+1}:\t{M}, {mu}, {a}, {e}")
            updated_params = _base_params.copy()

            updated_params[0] = M
            updated_params[1] = mu
            updated_params[2] = a
            updated_params[4] = e
            updated_params[3] = get_p_at_t(
                traj_module,
                duration * 0.99,
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


def time_amp_generation(
    fd_waveform_func,
    td_waveform_func,
    input_params,
    waveform_kwargs,
    iterations=10,
    verbose=True,
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
        result = {
            "mass_1": params[0],
            "mass_2": params[1],
            "spin": params[2],
            "p0": params[3],
            "e0": params[4],
            "x0": params[5],
            "dist": params[6],
            "qS": params[7],
            "phiS": params[8],
            "qK": params[9],
            "phiK": params[10],
            "Phi_phi0": params[11],
            "Phi_theta0": params[12],
            "Phi_r0": params[13],
            "fd_time": fd_time,
            "td_time": td_time,
        }
        results.append(result)

    return results



# create a function that times the FD and TD waveform generation for different input parameters
def time_full_waveform_generation(
    fd_waveform_func,
    td_waveform_func,
    input_params,
    waveform_kwargs,
    iterations=10,
    verbose=True,
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
        result = {
            "mass_1": params[0],
            "mass_2": params[1],
            "spin": params[2],
            "p0": params[3],
            "e0": params[4],
            "x0": params[5],
            "dist": params[6],
            "qS": params[7],
            "phiS": params[8],
            "qK": params[9],
            "phiK": params[10],
            "Phi_phi0": params[11],
            "Phi_theta0": params[12],
            "Phi_r0": params[13],
            "fd_time": fd_time,
            "td_time": td_time,
        }
        results.append(result)

    return results
