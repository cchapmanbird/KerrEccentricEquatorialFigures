import os 
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

# from lisatools.sensitivity import noisepsd_AE2,noisepsd_T # Power spectral densities
from fastlisaresponse import ResponseWrapper             # Response

from lisatools.sensitivity import get_sensitivity
from lisatools.utils.utility import AET
from lisatools.detector import scirdv1
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import AE1SensitivityMatrix


# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, get_p_at_t

from few.summation.directmodesum import DirectModeSum 
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.modeselector import ModeSelector, NeuralModeSelector

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

MAKE_PLOT = True
# Import parameters
sys.path.append("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/PE_studies")
from EMRI_settings import (M, mu, a, p0, e0, x_I0, SNR_choice, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T, delta_t)
use_gpu = True

dist = 1.0 # Choose new distance, normalise 
YRSID_SI = 31558149.763545603

np.random.seed(1234)

tdi_gen = "2nd generation"

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
t0 = 20000.0  # throw away on both ends when our orbital information is weird

TDI_channels = ['TDIA','TDIE']
N_channels = len(TDI_channels)


def noise_PSD_AE(f, TDI = 'TDI2'):
    """
    Inputs: Frequency f [Hz]
    Outputs: Power spectral density of noise process for TDI1 or TDI2.

    TODO: Incorporate the background!! 
    """
    # Define constants
    L = 2.5e9
    c = 299_792_458 # CORRECT
    # c = 299758492
    x = 2*np.pi*(L/c)*f
    
    # Test mass acceleration
    Spm = (3e-15)**2 * (1 + ((4e-4)/f)**2)*(1 + (f/(8e-3))**4) * (1/(2*np.pi*f))**4 * (2 * np.pi * f/ c)**2
    # Optical metrology subsystem noise 
    Sop = (15e-12)**2 * (1 + ((2e-3)/f)**4 )*((2*np.pi*f)/c)**2
    
    S_val = (2 * Spm *(3 + 2*np.cos(x) + np.cos(2*x)) + Sop*(2 + np.cos(x))) 
    
    if TDI == 'TDI1':
        S = 8*(np.sin(x)**2) * S_val
    elif TDI == 'TDI2':
        S = 32*np.sin(x)**2 * np.sin(2*x)**2 * S_val

    S[S < S[0]] = S[0]

    return cp.asarray(S)

def zero_pad(data):
    """
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain 
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product 
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))

##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    """
    # Intrinsic Parameters
    M_val = float(params[0])
    mu_val =  float(params[1])
    a_val =  float(params[2])            
    p0_val = float(params[3])
    e0_val = float(params[4])
    log_e0_val = float(params[4])
    e0_val = 10**(log_e0_val)
    x_I0_val = 1.0 
    
    # Luminosity distance 
    D_val = float(params[5])

    # Angular Parameters
    qS_val = float(params[6])
    phiS_val = float(params[7])
    qK_val = float(params[8])
    phiK_val = float(params[9])

    # Angular parameters
    Phi_phi0_val = float(params[10])
    Phi_theta0_val = Phi_theta0
    Phi_r0_val = float(params[11])

    # Deal with retrograde orbits:

    # print("Original value: (",a_val,Y0_val,")")
    # if a_val < 0:
    #     a_val *= -1.0
    #     Y0_val *= -1.0
    # print("New value: (",a_val,Y0_val,")")

    # Propose new waveform model
    # Recover with relativistic model 
    
    waveform_prop = EMRI_TDI_Model(M_val, mu_val, a_val, p0_val, e0_val, 
                                  x_I0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0_val, Phi_theta0_val, Phi_r0_val)  # EMRI waveform across A, E and T.

    # Taper and then zero pad. 
    EMRI_AET_w_pad_prop = [zero_pad(waveform_prop[i]) for i in range(N_channels)]

    # Compute in frequency domain
    EMRI_AET_fft_prop = [xp.fft.rfft(item) for item in EMRI_AET_w_pad_prop]

    # Compute (d - h| d- h)
    diff_f_AET = [data_f_AET[k] - EMRI_AET_fft_prop[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],N_t,delta_t,PSD_AET[k]) for k in range(N_channels)])
    
    # Return log-likelihood value as numpy val. 
    llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod))) 
    return (llike_val_np)

## =================== SET UP PARAMETERS =====================

N_channels = 2
xp = cp

## ===================== CHECK TRAJECTORY ====================
# 
traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK

# Compute trajectory 
if a < 0:
    a *= -1.0 
    Y0 *= -1.0

t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a, p0, e0, x_I0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

traj_args = [M, mu, a, e_traj[0], Y_traj[0]]
index_of_p = 3
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=None
)

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))

print("Separation between separatrix and final p = ",abs(get_separatrix(a,e_traj[-1],1.0) - p_traj[-1]))
print("Now going to load in class")

# inspiral_kwargs = {'flux_output_convention':'ELQ'} 

# inspiral_kwargs = {‘flux_output_convention’:‘ELQ’}
# inspiral_kwargs = {‘flux_output_convention’:‘pex’}


Kerr_waveform = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True),
        # inspiral_kwargs = inspiral_kwargs,
        use_gpu=use_gpu,
        return_list=False,
    )


# Build the response wrapper
print("Building the responses!")
EMRI_TDI_Model = ResponseWrapper(
        Kerr_waveform,
        T,
        delta_t,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

####=======================True Responsed waveform==========================

params = [M, mu, a, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

print("Running the truth waveform")
Kerr_TDI_waveform = EMRI_TDI_Model(*params)

# Taper and then zero_pad signal
Kerr_FEW_TDI_pad = [zero_pad(Kerr_TDI_waveform[i]) for i in range(N_channels)]

N_t = len(Kerr_FEW_TDI_pad[0])

# Compute signal in frequency domain
Kerr_TDI_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in Kerr_FEW_TDI_pad])

freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)

PSD_AET = 2*[noise_PSD_AE(freq_np)]

# Compute optimal matched filtering SNR
SNR2_Kerr_FEW = xp.asarray([inner_prod(Kerr_TDI_fft[i],Kerr_TDI_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])

SNR_Kerr_FEW = xp.asnumpy(xp.sum(SNR2_Kerr_FEW)**(1/2))

print("SNR for Kerr_FEW is",SNR_Kerr_FEW)

dist = (SNR_Kerr_FEW/SNR_choice)

# Recompute SNR -- Gross code -- TODO: FIX!! 

params = [M, mu, a, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
Kerr_TDI_waveform = EMRI_TDI_Model(*params)
Kerr_FEW_TDI_pad = [zero_pad(Kerr_TDI_waveform[i]) for i in range(N_channels)]
Kerr_TDI_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in Kerr_FEW_TDI_pad])
SNR2_Kerr_FEW = xp.asarray([inner_prod(Kerr_TDI_fft[i],Kerr_TDI_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])
SNR_Kerr_FEW = xp.asnumpy(xp.sum(SNR2_Kerr_FEW)**(1/2))
print("New SNR for Kerr_FEW is",SNR_Kerr_FEW)


# ================== PLOT THE A CHANNEL ===================

if MAKE_PLOT == True:
    plt.loglog(freq_np[1:], freq_np[1:]*abs(cp.asnumpy(Kerr_TDI_fft[0][1:])), label = "Waveform frequency domain")
    plt.loglog(freq_np[1:], np.sqrt(freq_np[1:] * cp.asnumpy(PSD_AET[0][1:])), label = "TDI2 PSD")
    plt.xlabel(r'Frequency [Hz]', fontsize = 30)
    plt.ylabel(r'Magnitude',fontsize = 30)
    plt.title(fr'$(M, \mu, a, p_0, e_0, D_L)$ = {M,mu,p0,a, e0,dist}')
    plt.grid()
    plt.xlim([1e-5,freq_np[-1]])
    plt.savefig(f"/home/ad/burkeol/work/Parameter_Estimation_EMRIs/Kerr_FEW_PE/mcmc_code/plots/charac_strain_{M,mu,p0,e0,dist}.png", bbox_inches = "tight")


##=====================Noise Setting: Currently 0=====================

# Compute Variance and build noise realisation
variance_noise_AET = [N_t * PSD_AET[k] / (4*delta_t) for k in range(N_channels)]

variance_noise_AET[0] = 2*variance_noise_AET[0]
variance_noise_AET[-1] = 2*variance_noise_AET[-1]

noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]

# Compute noise in frequency domain
noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

for i in range(N_channels):
    noise_f_AET[i][0] = noise_f_AET[i][0].real
    noise_f_AET[i][-1] = noise_f_AET[i][-1].real

data_f_AET = Kerr_TDI_fft + 0*noise_f_AET   # define the data

##===========================MCMC Settings============================

iterations = 20000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains

# USING ntemps = 5 for the Kerr inj AAK rec runs
ntemps = 1             # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

d = 1.0 # A parameter that can be used to dictate how close we want to start to the true parameters
# Useful check: If d = 0 and noise_f = 0, llike(*params)!!

# We start the sampler exceptionally close to the true parameters and let it run. This is reasonable 
# if and only if we are quantifying how well we can measure parameters. We are not performing a search. 

if x_I0 < 0:
    a *= -1.0 
    x_I0 *= -1.0

# Intrinsic Parameters

start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))   
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_a = a*(1. + d * 1e-7 * np.random.randn(nwalkers,1))

start_p0 = p0*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))
start_e0 = np.log10(e0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1)))

# Luminosity distance
start_D = dist*(1 + d * 1e-6 * np.random.randn(nwalkers,1))

# Angular parameters
start_qS = qS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = phiS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = qK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = phiK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

# Initial phases 
start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_r0 = Phi_r0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_mu, start_a, start_p0, start_e0, start_D, 
start_qS, start_phiS, start_qK, start_phiK,start_Phi_Phi0, start_Phi_r0))

if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other. 
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
# ================= SET UP PRIORS ========================

n = 2500 # size of prior

# Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta_theta_D = dist/np.sqrt(np.sum(SNR_Kerr_FEW))

priors_in = {
    # Intrinsic parameters
    0: uniform_dist(M - n*Delta_theta_intrinsic[0], M + n*Delta_theta_intrinsic[0]), # Primary Mass M
    # 1: uniform_dist(mu - n*Delta_theta_intrinsic[1], mu + n*Delta_theta_intrinsic[1]), # Secondary Mass mu
    1: uniform_dist(mu - 1000, mu + 1000), # Secondary Mass mu for very heavy IMRI
    2: uniform_dist(a - n*Delta_theta_intrinsic[2], 0.999), # Spin parameter a
    3: uniform_dist(p0 - n*Delta_theta_intrinsic[3], p0 + n*Delta_theta_intrinsic[3]), # semi-latus rectum p0
    4: uniform_dist(-100 , 0), # eccentricity e0 # Crazy wide uniform prior. Disgusting. 
    5: uniform_dist(dist - n*Delta_theta_D, dist + n* Delta_theta_D), # distance D
    # Extrinsic parameters -- Angular parameters
    6: uniform_dist(0, np.pi), # Polar angle (sky position)
    7: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    8: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    9: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    10: uniform_dist(0, 2*np.pi), # Phi_phi0
    11: uniform_dist(0, 2*np.pi) # Phi_r0
}  

priors = ProbDistContainer(priors_in, use_cupy = False)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=2.0, use_gpu=True)

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0])) 
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0])) 

os.chdir('/work/scratch/data/burkeol/kerr_few_paper/low_e0_simulations/')
# Low eccentricity result 
fp = "low_e0_Kerr_M_1e6_mu_25_a_0p998_p0_10p628_e0_0p00001_SNR_50_dt_5_T_2.h5"
backend = HDFBackend(fp)


ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
Reset_Backend = False # NOTE: CAREFUL HERE. ONLY TO USE IF WE RESTART RUNS!!!!
if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    start = backend.get_last_sample() # Start from last sample
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler

##===========================MCMC Settings (change this)============================

