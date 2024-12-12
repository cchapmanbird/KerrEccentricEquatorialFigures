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

from utils_mcmc import get_loglike_kwargs_prior

# set parameters
M = 5.4e5
mu = 50.0
p0 = 10.35
e0 = 0.3
sinqK = 0.6471975512
phiK = 2.5471975512
dist = 8.75
Phi_phi0 = 0.0
Phi_r0 = 0.0
Phi_theta0 = 0.0
sinqS = 0.471975512
phiS = 0.9071975512

a = 0.0
x0 = +1.0  # will be ignored in Schwarzschild waveform
qK = np.arcsin(sinqK)  # polar spin angle
phiK = 0.2  # azimuthal viewing angle
qS = np.arcsin(sinqS)  # polar sky angle
phiS = 0.3  # azimuthal viewing angle
dist = 8.75 * 1e-2

Tobs = 3150000 / YRSID_SI
dt = 10.0
fp = "test_run_emri_pe_5.h5"

emri_injection_params = np.array([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])

ntemps = 4
nwalkers = 16

waveform_kwargs = {"T": 1.0, "dt": 10.0, "eps": 1e-5}

ll, like_kwargs, priors, periodic, emri_injection_params_in =  get_loglike_kwargs_prior(emri_injection_params, Tobs, dt, emri_kwargs=waveform_kwargs)

ndim = len(priors['emri'].priors)

start_params = priors["emri"].rvs(size=ntemps * nwalkers)
start_params = np.random.multivariate_normal(emri_injection_params_in, np.eye(ndim)*1e-16, ntemps * nwalkers)
start_prior = priors["emri"].logpdf(start_params)
start_like = np.asarray([ll(el, **like_kwargs) for el in start_params])
# prtint start 
print(start_like)
# check true params likelihood
print(ll(emri_injection_params_in, **like_kwargs))
start_state = State(
            {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)},
            log_like=start_like.reshape(ntemps, nwalkers),
            log_prior=start_prior.reshape(ntemps, nwalkers),
        )

# MCMC moves (move, percentage of draws)
moves = [StretchMove()]

# prepare sampler
sampler = EnsembleSampler(
    nwalkers,
    [ndim],  # assumes ndim_max
    ll,
    priors,
    tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
    moves=moves,
    kwargs=like_kwargs,
    backend=fp,
    vectorize=False,
    periodic=periodic,  # TODO: add periodic to proposals
    # update_fn=None,
    # update_iterations=-1,
    branch_names=["emri"],
)

# TODO: check about using injection as reference when the glitch is added
# may need to add the heterodyning updater

nsteps = 1000
out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

# get samples
samples = sampler.get_chain(discard=0, thin=1)["emri"][:, 0].reshape(-1, ndim)

# plot
fig = corner.corner(samples, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2))
fig.savefig(fp[:-3] + "_corner.png", dpi=150)