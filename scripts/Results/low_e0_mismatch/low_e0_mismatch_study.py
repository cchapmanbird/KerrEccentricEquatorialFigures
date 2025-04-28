import os 
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

# Cosmology stuff
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value; cosmo = Planck18

def get_redshift(distance):
    return (z_at_value(cosmo.luminosity_distance, distance * u.Gpc )).value

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

from fastlisaresponse import ResponseWrapper             # Response

# from lisatools.sensitivity import get_sensitivity
# from lisatools.utils.utility import AET
from lisatools.detector import EqualArmlengthOrbits


from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux
run_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/low_e0_mismatch/"
sys.path.append("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/config_files/")
from psd_utils import (write_psd_file, load_psd_from_file, load_psd)

xp = cp
N_channels = 2
MAKE_PLOT = False
# Import parameters
use_gpu = True
if use_gpu:
    xp = cp 
else:
    xp = np

YRSID_SI = 31558149.763545603

np.random.seed(1234)

tdi_gen = "2nd generation"

order = 25  # interpolation order (should not change the result too much)
tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
    order=order,
    tdi=tdi_gen,
    tdi_chan="AE",
)  

index_lambda = 8
index_beta = 7

# with longer signals we care less about this
t0 = 20000.0  # throw away on both ends when our orbital information is weird

TDI_channels = ['TDIA','TDIE']
N_channels = len(TDI_channels)

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

def SNR_function(sig1_t, PSD_interp, dt, N_channels = 2):

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq = xp.fft.rfftfreq(N_t, dt)

    freq[0] = freq[1] 

    PSD_AE_array = PSD_interp(freq)

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD_AE_array[i]) for i in range(N_channels)])

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
SNR_choice = 50.0;

qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

delta_t = 10.0; T = 2.0

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

inspiral_kwargs = {"err":1e-11}

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

params_unnormed = [M, mu, a, p0, 0.0, 1.0, 1.0, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

print("Running the truth waveform")
Kerr_TDI_waveform_circ_unnormed = EMRI_TDI_Model(*params_unnormed)

# Taper and then zero_pad signal
Kerr_FEW_TDI_pad_unnormed = [zero_pad(Kerr_TDI_waveform_circ_unnormed[i]) for i in range(N_channels)]

N_t = len(Kerr_FEW_TDI_pad_unnormed[0])

# Compute signal in frequency domain
Kerr_TDI_fft_unnormed = xp.asarray([xp.fft.rfft(waveform) for waveform in Kerr_FEW_TDI_pad_unnormed])

freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
# First, write PSD to a file.

PSD_filename = "tdi2_AE_w_background.npy"
kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE', 
                           tdi2=True, include_foreground=True, 
                           filename = run_direc + PSD_filename, **kwargs_PSD)

PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=xp)

PSD_AE = PSD_AE_interp(freq)

# Compute optimal matched filtering SNR

SNR_Kerr_FEW_dL_1 = SNR_function(Kerr_TDI_waveform_circ_unnormed, PSD_AE_interp, delta_t, N_channels = 2)

print("SNR for Kerr_FEW with dL_1 ",SNR_Kerr_FEW_dL_1)

dist = (SNR_Kerr_FEW_dL_1/SNR_choice)

check_redshift = get_redshift(xp.asnumpy(dist))

Kerr_TDI_fft = (1/dist) * Kerr_TDI_fft_unnormed
Kerr_TDI_waveform_circ = (1/dist) * xp.asarray(Kerr_TDI_waveform_circ_unnormed)

SNR_Kerr_FEW = SNR_function(Kerr_TDI_waveform_circ, PSD_AE_interp, delta_t, N_channels = 2)

print(f"SNR for Kerr_FEW is {SNR_Kerr_FEW} at distance {dist}, with redshift z = {check_redshift}")

mismatch_EMRI = mismatch_function(Kerr_TDI_waveform_circ,Kerr_TDI_waveform_circ, PSD_AE, delta_t, N_channels=2)
# ================== PLOT THE A CHANNEL ===================

if MAKE_PLOT == True:
    plt.loglog(freq_np[1:], freq_np[1:]*abs(xp.asnumpy(Kerr_TDI_fft[0][1:])), label = "Waveform frequency domain")
    plt.loglog(freq_np[1:], np.sqrt(freq_np[1:] * xp.asnumpy(PSD_AET[0][1:])), label = "TDI2 PSD")
    plt.xlabel(r'Frequency [Hz]', fontsize = 30)
    plt.ylabel(r'Magnitude',fontsize = 30)
    plt.title(fr'$(M, \mu, a, p_0, e_0, D_L)$ = {M,mu,p0,a, e0,dist}')
    plt.grid(True)
    plt.xlim([1e-5,freq_np[-1]])
    plt.savefig(f"plots/waveform_plots/charac_strain_{M,mu,p0,e0,dist}.png", bbox_inches = "tight")
    plt.clf()

# Now try to compute mismatches as we change e0. 

err = [1e-11]
k = 0
for err_val in err:
    inspiral_kwargs = {"err":err_val}
    waveform_kwargs = {"inspiral_kwargs":inspiral_kwargs}
    little_e0 = np.logspace(-8, -3, 30) 

    mismatch_vec = []
    for ecc in tqdm(little_e0):
        params = [M, mu, a, p0, ecc, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 
        EMRI_TDI_perturb_e0 = EMRI_TDI_Model(*params, eps = 1e-5, **waveform_kwargs)

        mismatch_val = mismatch_function(EMRI_TDI_perturb_e0,Kerr_TDI_waveform_circ, PSD_AE, delta_t,N_channels=2)
        mismatch_vec.append(mismatch_val)

    if use_gpu:
        mismatch_vec_np = np.array([cp.asnumpy(x) for x in mismatch_vec])
    else:
        mismatch_vec_np = np.array(mismatch_vec)

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

# Save Data

breakpoint()

np.save(run_direc + "/data_for_plots/" + "SNR_choice.npy", np.array(SNR_choice))
np.save(run_direc + "/data_for_plots/" + "e0_vec.npy", little_e0)
np.save(run_direc + "/data_for_plots/" + "mismatch_vec.npy", mismatch_vec_np)

np.save(run_direc + "/data_for_plots/" + "e0_for_fit.npy", e0_smooth)
np.save(run_direc + "/data_for_plots/" + "mismatch_fit.npy", mismatch_fit)
    
print("End :) ")