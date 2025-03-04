import time
from tqdm import tqdm


# create a function that times the FD and TD waveform generation for different input parameters
def time_waveform_generation(
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
            "e0": params[3],
            "p0": params[4],
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
