import numpy as np
import sys
import os
import argparse

sys.path.append(os.getcwd())

from timing_utils import time_full_waveform_generation
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.utility import get_p_at_t
from few.waveform import GenerateEMRIWaveform

import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Timing tests for FastEMRIWaveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--delta-t", help="time step", type=float, default=10.0)
    parser.add_argument(
        "-f",
        "--filename",
        help="output json filename (without path)",
        type=str,
        default="initial_timing_results",
    )
    parser.add_argument(
        "--duration", help="signal duration in years", type=float, default=1.0
    )
    parser.add_argument(
        "--epsilon", help="mode threshold cutoff", type=float, default=1e-2
    )
    parser.add_argument(
        "--iterations",
        help="number of waveforms generated when averaging timing",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Turn verbosity on/off",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-g",
        "--generate-parameters",
        help="flag to generate random seed parameters for test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--nsamples",
        help="specifies number of samples to generate if random parameters requested",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        help="specifies seed for random parameters requested",
        type=int,
        default=314159,
    )

    args = parser.parse_args()

    iterations = args.iterations
    dt = args.delta_t  # time interval
    duration = args.duration
    output_filename = args.filename + ".json"

    waveform_kwargs = {
        "T": duration,
        "dt": dt,
        "eps": args.epsilon,
    }

    if args.verbose:
        print(
            (
                "Running FEW timing test"
                + f"\noutput filename: {output_filename},"
                + f"\niterations: {iterations},"
                + f"\ndelta_t: {dt},"
                + f"\nduration: {duration}"
            )
        )

    # produce sensitivity function
    traj_module = EMRIInspiral(func=KerrEccEqFlux)

    # Initialize waveform generators
    # frequency domain
    few_gen = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
        return_list=True,
    )

    # time domain
    td_gen = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True, odd_len=True),
        return_list=True,
    )

    # define the injection parameters
    mass_1 = 0.5e6  # central object mass
    mass_2 = 10.0  # secondary object mass
    spin = 0.9  # dimensionful Kerr parameter at 1MSun
    p0 = 12.0  # initial semi-latus rectum
    e0 = 0.1  # eccentricity

    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = np.pi / 3  # polar spin angle
    phiK = np.pi / 3  # azimuthal viewing angle
    qS = np.pi / 3  # polar sky angle
    phiS = np.pi / 3  # azimuthal viewing angle
    dist = 1.0  # distance
    # initial phases
    Phi_phi0 = np.pi / 3
    Phi_theta0 = 0.0
    Phi_r0 = np.pi / 3

    emri_injection_params = [
        mass_1,
        mass_2,
        spin,
        0.0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
    ]
    emri_injection_params[3] = get_p_at_t(
        traj_module,
        duration * 0.99,  # buffer for... reasons?
        [
            emri_injection_params[0],
            emri_injection_params[1],
            emri_injection_params[2],
            emri_injection_params[4],
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

    # run things once to cache, this will sometimes take a few seconds
    few_gen(*emri_injection_params, **waveform_kwargs)

    timing_results = []

    # check to generate random parameters
    if args.generate_parameters:
        from timing_utils import gen_parameters

        parameter_list, flist = gen_parameters(
            args.nsamples, duration, seed_in=args.seed, verbose=args.verbose
        )
        print(parameter_list)
        print(flist)
    else:
        parameter_list = []
        # create a list of input parameters
        # ranging over mass_1 and eccentricity values
        for mass in [1e5, 1e6, 1e7]:
            for ecc in np.linspace(0.1, 0.6, 3, endpoint=False):
                temp = emri_injection_params.copy()
                temp[0] = mass
                temp[4] = ecc
                temp[3] = get_p_at_t(
                    traj_module,
                    duration * 0.99,
                    [temp[0], temp[1], temp[2], temp[4], 1.0],
                    index_of_a=2,
                    index_of_p=3,
                    index_of_e=4,
                    index_of_x=5,
                    traj_kwargs={},
                    xtol=2e-6,
                    rtol=8.881784197001252e-6,
                )
                parameter_list.append(temp.copy())

    timing_results = time_full_waveform_generation(
        few_gen,
        td_gen,
        parameter_list,
        waveform_kwargs,
        iterations=iterations,
        verbose=args.verbose,
    )

    output = {
        "metadata": {
            "duration": duration,
            "delta_t": dt,
            "iterations": iterations,
            "epsilon": args.epsilon,
        },
        "results": timing_results.copy(),
    }

    json.dump(output, open(output_filename, "w"), indent=4)
