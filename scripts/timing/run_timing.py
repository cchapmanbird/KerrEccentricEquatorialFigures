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
    parser.add_argument("--delta-t", help="time step", type=float, default=5.0)
    parser.add_argument(
        "-f",
        "--filename",
        help="output json filename (without path)",
        type=str,
        default="initial_timing_results.json",
    )
    parser.add_argument(
        "--duration", help="signal duration in years", type=float, default=1.0
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

    args = parser.parse_args()

    iterations = args.iterations
    dt = args.delta_t  # time interval
    duration = args.duration
    output_filename = (
        args.filename + f"dur_{duration}_dt_{dt}_iters_{iterations}" + ".json"
    )

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
    M = 0.5e6  # central object mass
    a = 0.9  # will be ignored in Schwarzschild waveform
    mu = 10.0  # secondary object mass
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
        M,
        mu,
        a,
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

    timing_results = []
    vec_par = []

    # create a list of input parameters for M, mu, a, p0, e0, x0
    for mass in [1e5, 1e6, 1e7]:
        for el in np.linspace(0.1, 0.6, num=3):
            temp = emri_injection_params.copy()
            temp[0] = mass
            temp[4] = el
            temp[3] = get_p_at_t(
                traj_module,
                duration * 0.99,
                [temp[0], temp[1], temp[2], temp[4], 1.0],
                index_of_p=3,
                index_of_a=2,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-6,
                rtol=8.881784197001252e-6,
            )
            vec_par.append(temp.copy())

    waveform_kwargs = {
        "T": duration,
        "dt": dt,
        "eps": 1e-2,
    }
    timing_results = time_full_waveform_generation(
        few_gen,
        td_gen,
        vec_par,
        waveform_kwargs,
        iterations=iterations,
        verbose=args.verbose,
    )

    json.dump(timing_results, open(output_filename, "w"), indent=4)
