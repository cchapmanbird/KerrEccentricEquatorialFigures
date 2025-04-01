import os 
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

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

xp = cp
N_channels = 2
MAKE_PLOT = True
# Import parameters
use_gpu = True

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

def SNR_function(sig1_t, PSD, dt, N_channels = 2):

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1] 

    PSD = 2 * [xp.asarray(noise_PSD_AE(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR

def mismatch_function(sig1_t,sig2_t, PSD, dt,N_channels=2):

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    sig2_f = [xp.fft.rfft(zero_pad(sig2_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    

    aa = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])
    bb = xp.asarray([inner_prod(sig2_f[i], sig2_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])
    ab = xp.asarray([inner_prod(sig1_f[i], sig2_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    overlap_channels = 0.5 * xp.sum(xp.asarray([ab[j]/np.sqrt(aa[j]*bb[j]) for j in range(N_channels)]))
    mismatch_channels = 1 - overlap_channels
    return mismatch_channels


## ===================== Set up parameters ========================================

M = 1e6; mu = 25; a = 0.998; p0 = 10.628; e0 = 0.1; x_I0 = 1.0
dist = 5.0; 
qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

delta_t = 5.0; T = 2.0

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

inspiral_kwargs = {"err":1e-12}
Kerr_waveform = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True), 
        inspiral_kwargs = inspiral_kwargs,
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

params = [M, mu, a, p0, 0.0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

print("Running the truth waveform")
Kerr_TDI_waveform_circ = EMRI_TDI_Model(*params, eps = 1e-5)

# Taper and then zero_pad signal
Kerr_FEW_TDI_pad = [zero_pad(Kerr_TDI_waveform_circ[i]) for i in range(N_channels)]

N_t = len(Kerr_FEW_TDI_pad[0])

# Compute signal in frequency domain
Kerr_TDI_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in Kerr_FEW_TDI_pad])

freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)

PSD_AET = 2*[noise_PSD_AE(freq_np)]

# Compute optimal matched filtering SNR

SNR_Kerr_FEW = SNR_function(Kerr_TDI_waveform_circ, PSD_AET, delta_t, N_channels = 2)

print("SNR for Kerr_FEW is",SNR_Kerr_FEW)

mismatch_EMRI = mismatch_function(Kerr_TDI_waveform_circ,Kerr_TDI_waveform_circ, PSD_AET, delta_t,N_channels=2)
# ================== PLOT THE A CHANNEL ===================

if MAKE_PLOT == True:
    plt.loglog(freq_np[1:], freq_np[1:]*abs(cp.asnumpy(Kerr_TDI_fft[0][1:])), label = "Waveform frequency domain")
    plt.loglog(freq_np[1:], np.sqrt(freq_np[1:] * cp.asnumpy(PSD_AET[0][1:])), label = "TDI2 PSD")
    plt.xlabel(r'Frequency [Hz]', fontsize = 30)
    plt.ylabel(r'Magnitude',fontsize = 30)
    plt.title(fr'$(M, \mu, a, p_0, e_0, D_L)$ = {M,mu,p0,a, e0,dist}')
    plt.grid(True)
    plt.xlim([1e-5,freq_np[-1]])
    plt.savefig(f"plots/waveform_plots/charac_strain_{M,mu,p0,e0,dist}.png", bbox_inches = "tight")
    plt.clf()

# Now try to compute mismatches as we change e0. 

err = [1e-11, 1e-12, 1e-13]
k = 0
for err_val in err:
    inspiral_kwargs = {"err":err_val}
    waveform_kwargs = {"inspiral_kwargs":inspiral_kwargs}
    little_e0 = np.logspace(-8, -3, 50) 

    mismatch_vec = []
    for ecc in tqdm(little_e0):
        params = [M, mu, a, p0, ecc, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 
        EMRI_TDI_perturb_e0 = EMRI_TDI_Model(*params, eps = 1e-5, **waveform_kwargs)

        mismatch_val = mismatch_function(EMRI_TDI_perturb_e0,Kerr_TDI_waveform_circ, PSD_AET, delta_t,N_channels=2)
        mismatch_vec.append(xp.asnumpy(mismatch_val))

    mismatch_vec_np = cp.asnumpy(mismatch_vec)

    if err_val == 1e-11:
        # Example data (replace with your actual values)
        index_e0_1eneg5 = np.argwhere(little_e0 > 1e-5)[0][0]
        index_e0_1eneg4 = np.argwhere(little_e0 > 5e-4)[0][0]

        little_e0_point = np.array([little_e0[index_e0_1eneg5], little_e0[index_e0_1eneg4]])  # x-values (eccentricities)
        mismatch_vec_np_point = np.array([mismatch_vec_np[index_e0_1eneg5], mismatch_vec_np[index_e0_1eneg4]])  # y-values (mismatch)

        # Convert to log10 space
        log_e0_points = np.log10(little_e0_point)
        log_mismatch_points = np.log10(mismatch_vec_np_point)

        # Fix slope m = 4.0 and solve for intercept
        b = np.mean(log_mismatch_points - 4.0 * log_e0_points)  # Average to fit all points

        # # Perform linear fit in log-space
        # m, b = np.polyfit(log_e0_points, log_mismatch_points, 1)

        m = 4.0
        # Define fitted function (power-law relationship)
        def fitted_curve(e0_val,m,b):
            return 10**(m * np.log10(e0_val) + b)

        print(f"Gradient = {m} and intercept {b}")
        # Generate smooth points for plotting the fitted curve
        e0_smooth = np.logspace(-6, -3, 150)  # Log-spaced x-values
        mismatch_fit = fitted_curve(e0_smooth,m,b)


    
    plt.loglog(little_e0, mismatch_vec_np, 'o', ms = 5, label = f"Error = {err_val}")
    if k == 2:
        plt.loglog(e0_smooth, mismatch_fit, linestyle = 'dashed', c = 'red', label = r"$\propto (e_{0})^{4}$")
        plt.axhline(y = (1/(2*cp.asnumpy(SNR_Kerr_FEW)**2)), c = 'black', linestyle = 'dashed', label = r'$1/(2\rho^{2})$')
    plt.xlabel(r'Eccentricity', fontsize = 16)
    plt.ylabel(r'$\mathcal{M}$', fontsize = 16)
    plt.title(r'Low eccentricity limit', fontsize = 16)
    plt.legend()
    plt.grid(True)
    k+=1
plt.savefig("plots/low_e0_plots/low_e0_mismatch_err_tol.png",bbox_inches = "tight")
