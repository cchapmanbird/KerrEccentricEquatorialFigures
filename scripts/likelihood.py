import numpy as np
from eryn.backends import HDFBackend
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner
from lisatools.utils.utility import AET
from lisatools.detector import scirdv1

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

from lisatools.sensitivity import get_sensitivity

from few.waveform import GenerateEMRIWaveform
from eryn.utils import TransformContainer

from fastlisaresponse import ResponseWrapper

from few.utils.constants import *
import os
np.random.seed(1112)

try:
    import cupy as xp

    # set GPU device
    xp.cuda.runtime.setDevice(7)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

import warnings

warnings.filterwarnings("ignore")

# whether you are using
use_gpu = True

if use_gpu is True:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")


# function call
def run_emri_pe(emri_injection_params, Tobs, dt, fp, ntemps, nwalkers, emri_kwargs={}):

    # sets the proper number of points and what not

    N_obs = int(
        Tobs * YRSID_SI / dt
    )  # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)

    few_gen = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True),
        use_gpu=use_gpu,
        return_list=False,
    )

    from lisatools.detector import EqualArmlengthOrbits

    tdi_gen = "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        order=order,
        tdi=tdi_gen,
        tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird

    wave_gen = ResponseWrapper(
        few_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    # wave_gen = few_gen

    # for transforms
    # this is an example of how you would fill parameters
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
        "ndim_full": 14,
        "fill_values": np.array([+1.0, 0.0]),  # spin and inclination and Phi_theta
        "fill_inds": np.array([5, 12]),
    }

    (M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0) = (
        emri_injection_params
    )

    # get the right parameters
    # log of large mass
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[7] = np.cos(emri_injection_params[7])
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9])
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)

    # phases
    emri_injection_params[-1] = emri_injection_params[-1] % (2 * np.pi)
    emri_injection_params[-2] = emri_injection_params[-2] % (2 * np.pi)
    emri_injection_params[-3] = emri_injection_params[-3] % (2 * np.pi)

    # remove inc, phi_theta we are not sampling those
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(1e6)),  # M
                1: uniform_dist(1.0, 100.0),  # mu
                2: uniform_dist(0.0, 0.9),
                3: uniform_dist(12.0, 16.0),  # p0
                4: uniform_dist(0.001, 0.4),  # e0
                5: uniform_dist(0.01, 100.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        )
    }

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        0: np.exp,  # M
        7: np.arccos,  # qS
        9: np.arccos,  # qK
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )

    # sampler treats periodic variables by wrapping them properly
    periodic = {"emri": {7: 2 * np.pi, 9: 2 * np.pi, 10: 2 * np.pi, 11: 2 * np.pi}}

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # get XYZ
    data_channels = wave_gen(*injection_in, **emri_kwargs)

    from lisatools.sensitivity import AE1SensitivityMatrix

    print("NEED TO FIX TDI SENSITIVITY")

    sens_mat = AE1SensitivityMatrix(
        np.logspace(-5, 0, 1000), stochastic_params=(YRSID_SI,)
    )

    check_snr = snr(
        [data_channels[0], data_channels[1]],
        dt=dt,
        psd="A1TDISens",
        # psd_kwargs={"stochastic_params": (YRSID_SI,)},
    )

    print(check_snr)
    # this is a parent likelihood class that manages the parameter transforms

    # form DataResidualArray
    data_res_container = DataResidualArray([data_channels[0], data_channels[1]], dt=dt)

    sens_mat = AE1SensitivityMatrix(data_res_container.f_arr, stochastic_params=(Tobs,))
    from lisatools.analysiscontainer import AnalysisContainer

    analysis = AnalysisContainer(data_res_container, sens_mat, signal_gen=wave_gen)

    like_kwargs = dict(
        source_only=True,
        waveform_kwargs=emri_kwargs,
        data_res_arr_kwargs=dict(dt=dt),
        transform_fn=transform_fn,
    )

    test_params_inj = emri_injection_params_in

    ll_injection = analysis.eryn_likelihood_function(test_params_inj, **like_kwargs)

    test_params_adjust = test_params_inj.copy()
    check_tmp = []
    for i in np.arange(20):
        val = i * 0.0000000025
        for sign in [+1, -1]:
            test_params_adjust[2] = test_params_inj[2] + sign * val
            tmptmptmp = analysis.eryn_likelihood_function(test_params_adjust, **like_kwargs)
            check_tmp.append([test_params_adjust[2], tmptmptmp])
        print(val)
    check_tmp = np.asarray(check_tmp)
    return


if __name__ == "__main__":
    # set parameters
    M = 7e5
    a = 0.8  # 1324211123  
    mu = 20.0
    p0 = 13.0
    e0 = 0.2
    x0 = +1.0  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 3.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    Tobs = 2.05
    dt = 15.0
    fp = "test_run_emri_pe_5.h5"

    emri_injection_params = np.array(
        [M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    )

    ntemps = 4
    nwalkers = 30

    waveform_kwargs = {"T": 1.0, "dt": 15.0, "eps": 1e-2}

    run_emri_pe(
        emri_injection_params,
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        emri_kwargs=waveform_kwargs,
    )
    # frequencies to interpolate to
