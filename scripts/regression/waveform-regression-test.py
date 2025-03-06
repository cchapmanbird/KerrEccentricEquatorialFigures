import argparse
import json
import logging
import os

import numpy as np

from numpy.random import uniform, seed, randint

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_p_at_t

parser = argparse.ArgumentParser(
    description="Regression testing script to ensure that our waveform models are unchanged after code changes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-p", "--parameters", help="path to input parameter file", type=str, default=""
)
parser.add_argument("-o", "--output", help="path to output file", type=str, default="")
parser.add_argument(
    "-s",
    "--seed",
    help="seed used for parameter file generation",
    type=int,
    default=123456,
)
parser.add_argument(
    "-g",
    "--generate",
    help="used to generate parameter list for testing",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-l",
    "--length",
    help="number of random parameters generated when creating parameter list",
    type=int,
    default=100,
)
parser.add_argument(
    "-d",
    "--duration",
    help="length of signals to be tested (in years)",
    type=float,
    default=1.0,
)


# -- utility functions ---------------------
def transform_mass_ratio(logM, logeta):
    # takes in log10 of values
    return [10**logM, 10**logM * 10**logeta]


def get_amp_phase(h):
    amp = np.abs(h)
    phase = np.unwrap(np.angle(h))
    return amp, phase


def sum_sqr_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def generate_params(
    traj_module,
    duration,
    output_filename,
    generate_seed,
    generate_length,
):

    log10_mass1_range = [5, 7]
    log10_massratio_range = [-6, -4]
    spin_range = [0.0, 0.999]
    ecc_range = [0.0, 0.9]

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
        1e5,  # M
        10,  # mu
        0.0,  # a
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

    seed(generate_seed)
    M_seed, mu_seed, a_seed, e_seed = randint(1e3, 1e5, size=4)

    # sample uniformly in log10_M and log10_massratio
    seed(M_seed)
    log10_mass1_list = uniform(
        low=log10_mass1_range[0], high=log10_mass1_range[1], size=generate_length
    )
    seed(mu_seed)
    log10_massratio_list = uniform(
        low=log10_massratio_range[0],
        high=log10_massratio_range[1],
        size=generate_length,
    )
    mass_1_list, mass_2_list = transform_mass_ratio(
        log10_mass1_list, log10_massratio_list
    )
    seed(a_seed)
    spin_list = uniform(low=spin_range[0], high=spin_range[1], size=generate_length)
    seed(e_seed)
    ecc_list = uniform(low=ecc_range[0], high=ecc_range[1], size=generate_length)

    output_params_list = []

    for mass_1, mass_2, spin, ecc in zip(mass_1_list, mass_2_list, spin_list, ecc_list):
        updated_params = _base_params.copy()

        updated_params[0] = mass_1
        updated_params[1] = mass_2
        updated_params[2] = spin
        updated_params[4] = ecc
        try:
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
                index_of_p=3,
                index_of_a=2,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-6,
                rtol=8.881784197001252e-6,
            )

            output_params_list.append(updated_params.copy())
        except ValueError:
            continue

    json.dump(output_params_list, open(output_filename, "w"), indent=4)


def gen_test_data(input_params, few_gen, waveform_kwargs):
    """
    compute the difference between two waveforms
    and compare to expected value
    """

    pars1 = input_params.copy()

    # perturb mass 2
    pars2 = input_params.copy()
    pars2[1] += 10.0

    try:
        hp1, hc1 = few_gen(*pars1, **waveform_kwargs)
        hp2, hc2 = few_gen(*pars2, **waveform_kwargs)
    except RuntimeError:
        return [0.0, 0.0, 0.0, 0.0]

    # compute amp and phase
    h1_amp, h1_phase = get_amp_phase(hp1 + 1j * hc1)

    h2_amp, h2_phase = get_amp_phase(hp2 + 1j * hc2)

    h_amp_diff = sum_sqr_diff(h1_amp, h2_amp)
    h_phase_diff = sum_sqr_diff(h1_phase, h2_phase)

    return h_amp_diff, h_phase_diff


# -- run the tests ------------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    params_file = args.parameters
    output_file = args.output
    generate_bool = args.generate
    generate_seed = args.seed
    generate_length = args.length
    duration = args.duration

    # produce sensitivity function
    traj_module = EMRIInspiral(func=KerrEccEqFlux)

    waveform_kwargs = {
        "T": duration,
        "dt": 15.0,
        "eps": 1e-2,
    }

    # Initialize waveform generators
    # frequency domain
    few_gen = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
        return_list=True,
    )

    output_filename = output_file

    if generate_bool:
        logging_output_filename = f"waveform-regression-parameter-generation.log"

        logging.basicConfig(
            filename=logging_output_filename,
            encoding="utf-8",
            level=logging.INFO,
            filemode="a",
        )
        logging.info("waveform-regression-test.py called to generate parameters.")
        if output_filename == "":
            logging.info(
                f"Generating {generate_length} precessing parameters with seed {generate_seed}."
            )
            output_filename = "testing_parameters.json"

        # generating regression test parameters
        generate_params(
            traj_module,
            duration,
            output_filename=output_filename,
            generate_seed=generate_seed,
            generate_length=generate_length,
        )
    else:
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)

        try:
            conda_env = os.environ["CONDA_DEFAULT_ENV"]
        except Exception:
            conda_env = "unknown"

        logging_output_filename = f"waveform-regression-{conda_env}.log"

        logging.basicConfig(
            filename=logging_output_filename,
            encoding="utf-8",
            level=logging.INFO,
            filemode="a",
        )
        logging.info(f"waveform-regression-test.py called to generate waveform data.")
        logging.info(f"Running in conda env -- {conda_env}")
        if params_file == "":
            raise ValueError(
                f"Please pass in valid command-line arguments: {params_file}"
            )

        try:
            with open(params_file, "r") as f:
                params = json.load(f)
        except Exception:
            raise ValueError("Can't open input parameter file.")

        if output_filename == "":
            output_filename = f"test_results_{conda_env}"

        test_results = []
        logging.info(f"Generating waveform data in {conda_env}.")

        for input_pars in params:

            test_results.append(
                np.array(gen_test_data(input_pars, few_gen, waveform_kwargs))
            )

        logging.info(f"Saving data to {output_filename}.")
        np.save(output_filename, test_results)
