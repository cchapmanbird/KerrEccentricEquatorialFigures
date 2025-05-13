import cupy as cp
import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
sys.path.append("../")
run_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/AAK_Kerr_Comparisons/"
from fastlisaresponse import ResponseWrapper             # Response

# Cosmology stuff
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value; cosmo = Planck18

def get_redshift(distance):
    return (z_at_value(cosmo.luminosity_distance, distance * u.Gpc )).value

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.trajectory.ode.flux import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

# Import relevant EMRI packages
from lisatools.detector import EqualArmlengthOrbits

# Import useful PSD files 
sys.path.append("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/config_files/")
from psd_utils import (write_psd_file, load_psd_from_file, load_psd)

YRSID_SI = 31558149.763545603

np.random.seed(1234)
use_gpu = True

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
def SNR_function(sig1_t, dt, N_channels = 2):
    N_t = len(sig1_t[0])

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq = xp.fft.rfftfreq(N_t, dt)

    freq[0] = freq[1] 

    PSD_AE_list = PSD_AE_interp(freq) 

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD_AE_list[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR


if sys.argv[1] == '0':
    # SNR comparisons for M = 1e6 and mu = 1e1
    data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/AAK_Kerr_Comparisons/SNR_data/M1e7_mu1e2/"
    # Set masses, and focus on prograde orbits
    M = 1e7; mu = 100; x_I0 = 1.0; dist = 1.0

    # Set angular parameters 
    qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
    # Set initial phases
    Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

    # Set sampling properties
    delta_t = 10.0;  # Sampling interval [seconds]
    T = 2.0     # Evolution time [years]
elif sys.argv[1] == '1':
    # SNR comparisons for M = 1e6 and mu = 1e1
    data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/AAK_Kerr_Comparisons/SNR_data/M1e6_mu1e1/"
    # Set masses, and focus on prograde orbits
    M = 1e6; mu = 10; x_I0 = 1.0; dist = 1.0 

    # Fix angular parameters 
    qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 

    # Set initial phases
    Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

    # Set sampling properties
    delta_t = 5.0
    T = 2.0
elif sys.argv[1] == '2':
    # SNR comparisons for M = 1e5 and mu = 1e1
    data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/AAK_Kerr_Comparisons/SNR_data/M1e5_mu1/"
    M = 1e5; mu = 1; x_I0 = 1.0; dist = 1.0
    
    # Fix angular parameters 
    qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
    
    # Set initial phases
    Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

    delta_t = 2.0;  # Sampling interval [seconds]
    T = 2.0     # Evolution time [years]
else:
    print("You need to enter job array, quitting")
    quit()

## =================== SET UP PARAMETERS =====================

N_channels = 2
xp = cp

print("Now going to load in class")

inspiral_kwargs = {"func":"KerrEccEqFlux"}
sum_kwargs = {"pad_output":True}


Kerr_waveform = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=sum_kwargs,
        return_list=False,
    )

AAK_waveform = GenerateEMRIWaveform(
        "Pn5AAKWaveform",
        sum_kwargs=sum_kwargs,
        inspiral_kwargs = inspiral_kwargs,
        return_list=False,
    )


Kerr_TDI_Model = ResponseWrapper(
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


AAK_TDI_Model = ResponseWrapper(
        AAK_waveform,
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

# Define PSDs
# First, write PSD to a file.

PSD_filename = "tdi2_AE_w_background.npy"
kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE', 
                           tdi2=True, include_foreground=True, 
                           filename = run_direc + PSD_filename, **kwargs_PSD)

PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=cp)
a_vec = np.arange(0, 1.0, 0.1)
extra_a = np.array([0.99, 0.998])

a_vec = np.concatenate([a_vec,extra_a])
e0_vec = np.arange(0.01,0.81,0.01)

SNR_Kerr_vec=[]
SNR_AAK_vec=[]
SNR_AAK_Kerr_ratio_vec = []

np.save(data_direc + "e0_vec.npy", e0_vec)

# Test
# e0_vec = np.arange(0.01,0.8,0.01)

traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
for spin in a_vec:
    SNR_Kerr_vec=[]
    SNR_AAK_vec=[]
    for eccentricity in e0_vec:
        spin = np.round(spin,5)
        p_sep = get_separatrix(spin, eccentricity, 1.0)
        print(f"p_sep = {p_sep}")
 
        print(f"For eccentricity = {eccentricity} at spin = {spin}")
        traj_args = [M, mu, spin, eccentricity, 1.0]
        index_of_p = 3

        try:    
            p_new = get_p_at_t(
                traj,
                T,
                traj_args,
            )            
        except ValueError:
            print("error with interpolant, breaking out of loop but adding nans ")
            n_missing = len(e0_vec) - len(SNR_Kerr_vec)
            SNR_Kerr_vec.extend([np.nan] * n_missing)
            SNR_AAK_vec.extend([np.nan] * n_missing)
            break
        print(f"value of p_new={p_new} to give T_obs = 2 years")
     
        params = [M,mu,spin,p_new,eccentricity, x_I0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]  

        waveform_Kerr = Kerr_TDI_Model(*params)
        waveform_AAK = AAK_TDI_Model(*params, nmodes = 50)
    
        SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
        SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)
    
        ratio_AAK_Kerr = SNR_AAK/SNR_Kerr

        SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
        SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))
        SNR_AAK_Kerr_ratio_vec.append(xp.asnumpy(ratio_AAK_Kerr))

    np.save(data_direc + "/pure_SNRs/" + f"SNR_Kerr_vec_{spin}.npy", SNR_Kerr_vec)
    np.save(data_direc + "/pure_SNRs/" + f"SNR_AAK_vec_{spin}.npy", SNR_AAK_vec)
    np.save(data_direc + "/ratio_SNRs/" + f"SNR_AAK_Kerr_vec_ratio_{spin}.npy", SNR_AAK_Kerr_ratio_vec)



