import cupy as cp
import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
sys.path.append("../")

# from scipy.signal import tukey       # I'm always pro windowing.  

from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.trajectory.ode.flux import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, get_p_at_t

# Import relevant EMRI packages
from lisatools.sensitivity import get_sensitivity
from lisatools.utils.utility import AET
from lisatools.detector import scirdv1
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import AE1SensitivityMatrix


# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

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

def compute_traj(M,mu,a,p0,e0,T = 4):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: two separate trajectories from ELQ and pex module 
    """
    
    # Compute value of p to give T year inspiral
    # Compute trajectories for ELQ and pex
    out_pn5 = traj_pn5(M, mu, a, p0, e0, 1.0, T=T)  # ELQ
    out_Kerr = traj_Kerr(M, mu, a, p0, e0, 1.0, T=T,  
                       new_t=out_pn5[0], upsample=True) # pex, NOTE: using ELQ time array. 


    return out_pn5, out_Kerr

def check_phasing(M,mu,a,p0,e0,T = 4):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0, 
            observation time T (optional)

    outputs: phasing information from the ELQ and pex traj modules

    Very useful for debugging and checking!  
    """

    out_pn5, out_Kerr = compute_traj(M,mu,a,p0,e0,T = T)

    print("Difference in p, final point", abs(out_pn5[1][-1] - out_Kerr[1][-1])) # semi-latus rectum
    print("Difference in e, final point", abs(out_pn5[2][-1] - out_Kerr[2][-1])) # Eccentricity
    print("Difference in Phi_phi, final point", abs(out_pn5[-3][-1] - out_Kerr[-3][-1])) # orbital phase in phi

    print("")
    return out_pn5, out_Kerr

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
    
    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1] 

    PSD = 2 * [xp.asarray(noise_PSD_AE(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR

def mismatch_function(sig1_t, dt, N_channels = 2):
    N_t = len(sig1_t[0])

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1] 

    PSD = 2 * [xp.asarray(noise_PSD_AE(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR
##======================Likelihood and Posterior (change this)=====================

# SNR comparisons for M = 1e5 and mu = 1e1
M = 1e6; mu = 1e1; a = 0.998; p0 = 150; e0 = 0.3; x_I0 = 1.0;
dist = 1.0; qS = 0.7; phiS = 0.7; qK = 0.7; phiK = 0.7; 
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

delta_t = 2.0;  # Sampling interval [seconds]
T = 4.0     # Evolution time [years]
## =================== SET UP PARAMETERS =====================

N_strains = 2

## ===================== CHECK TRAJECTORY ====================
# 
traj_Kerr = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
traj_pn5 = EMRIInspiral(func=PN5)  # Set up trajectory module, pn5 AAK

check_phasing(M,mu,a,p0,e0,T = 4)

quit()
print("Now going to load in class")

inspiral_kwargs = {"func":"KerrEccEqFlux"}
sum_kwargs = {"pad_output":True}


Kerr_waveform = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=sum_kwargs,
        return_list=True,
    )

AAK_waveform = GenerateEMRIWaveform(
        "Pn5AAKWaveform",
        sum_kwargs=sum_kwargs,
        return_list=True,
    )

## ============= USE THE LONG WAVELENGTH APPROXIMATION, VOMIT ================ ##
params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 

waveform_AAK = AAK_waveform(*params, nmodes = 50) 
waveform_Kerr = Kerr_waveform(*params)

N_t = len(waveform_AAK[0])

# Compute signal in frequency domain
Kerr_strain_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in waveform_Kerr])
AAK_strain_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in waveform_AAK])

freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)
PSD_AET = 2*[noise_PSD_AE(freq_np)]

# Compute optimal matched filtering SNR
SNR2_Kerr_FEW = xp.asarray([inner_prod(Kerr_TDI_fft[i],Kerr_TDI_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])
SNR2_AAK_FEW = xp.asarray([inner_prod(AAK_TDI_fft[i],AAK_TDI_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])

SNR_Kerr_FEW = xp.asnumpy(xp.sum(SNR2_Kerr_FEW)**(1/2))
SNR_AAK_FEW = xp.asnumpy(xp.sum(SNR2_AAK_FEW)**(1/2))

print("SNR for Kerr_FEW is",SNR_Kerr_FEW)
print("SNR for AAK_FEW is",SNR_AAK_FEW)


a_vec = np.arange(0, 1.0, 0.1)
extra_a = np.array([0.99, 0.998])

# a_vec = np.concatenate([a_vec,extra_a])
# e0_start = np.array([0.01])
e0_vec = np.arange(0.01,0.71,0.01)
# e0_vec = np.arange(0.4,0.71,0.01)

# e0_vec = np.concatenate([e0_start,e0_vec])
SNR_Kerr_vec=[]
SNR_AAK_vec=[]

# data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/SNR_data/M1e6_mu1e1"
# data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/SNR_data/M1e7_mu1e2/"
data_direc = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/SNR_data/M1e5_mu1/"
# np.save(data_direc + "e0_vec.npy", e0_vec)

a_vec = [0.998]
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
            if spin >= 0.9:
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
                    #bounds = None
                    bounds=[25,28] # Good for M = 1e5, mu = 1
                )
            else:
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
        except ValueError:
            print("error with interpolant, continuing ")
            break
        print(f"value of p_new={p_new} to give T_obs = 2 years")
     
        params = [M,mu,spin,p_new,eccentricity,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]  

        waveform_Kerr = Kerr_TDI_Model(*params)
        waveform_AAK = AAK_TDI_Model(*params, nmodes = 50)
    
        SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
        SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)
    
        SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
        SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))

    np.save(data_direc + f"SNR_Kerr_vec_{spin}.npy", SNR_Kerr_vec)
    np.save(data_direc + f"SNR_AAK_vec_{spin}.npy", SNR_AAK_vec)



