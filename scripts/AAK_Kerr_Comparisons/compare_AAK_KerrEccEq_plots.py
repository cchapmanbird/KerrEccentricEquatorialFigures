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

YRSID_SI = 31558149.763545603

np.random.seed(1234)
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
##======================Likelihood and Posterior (change this)=====================

M = 1e6; mu = 10; a = 0.9; p0 = 8.54; e0 = 0.01; x_I0 = 1.0;
dist = 1.0; qS = 0.7; phiS = 0.7; qK = 0.7; phiK = 0.7; 
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]
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
    bounds=[6, 15]
)


print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))

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

## ============= USE THE LONG WAVELENGTH APPROXIMATION, VOMIT ================ ##
params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 

Raw_AAK = AAK_waveform(*params, dt = delta_t, T = T) 
Raw_Kerr = Kerr_waveform(*params, dt = delta_t, T = T) 

waveform_AAK = AAK_TDI_Model(*params, nmodes = 50) 
waveform_Kerr = Kerr_TDI_Model(*params)

N_t = len(waveform_AAK[0])

# Compute signal in frequency domain
Kerr_TDI_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in waveform_Kerr])
AAK_TDI_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in waveform_AAK])

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

aa = xp.asarray([inner_prod(AAK_TDI_fft[i],AAK_TDI_fft[i], N_t, delta_t, PSD_AET[i]) for i in range(N_channels)])
bb = xp.asarray([inner_prod(Kerr_TDI_fft[i],Kerr_TDI_fft[i], N_t, delta_t, PSD_AET[i]) for i in range(N_channels)])
ab = xp.asarray([inner_prod(AAK_TDI_fft[i],Kerr_TDI_fft[i], N_t, delta_t, PSD_AET[i]) for i in range(N_channels)])

overlap = 0.5*xp.sum(ab/(np.sqrt(aa*bb)))
mismatch = 1 - overlap

t = np.arange(0,N_t*delta_t, delta_t)
waveform_AAK_I_np = xp.asnumpy(waveform_AAK[0])
waveform_Kerr_I_np = xp.asnumpy(waveform_Kerr[0])

waveform_AAK_hp_np = xp.asnumpy(Raw_Kerr.real)
waveform_Kerr_hp_np = xp.asnumpy(Raw_AAK.real)

import matplotlib.pyplot as plt
MAKE_RESPONSE_PLOT = False
if MAKE_RESPONSE_PLOT == True: 
    plt.plot(t[5000:5250], -waveform_AAK_I_np[5000:5250], label = "AAK -- SNR = {}".format(np.round(SNR_AAK_FEW,5)))
    plt.plot(t[5000:5250],waveform_Kerr_I_np[5000:5250], label = "Kerr -- SNR = {}".format(np.round(SNR_Kerr_FEW,5)))
    
    plt.ylabel(r'Waveform strain (channel A)')
else:
    plt.plot(t[5000:5250], -waveform_AAK_hp_np[5000:5250], label = "AAK -- SNR = {}".format(np.round(SNR_AAK_FEW,5)))
    plt.plot(t[5000:5250],waveform_Kerr_hp_np[5000:5250], label = "Kerr -- SNR = {}".format(np.round(SNR_Kerr_FEW,5)))
    plt.ylabel(r'Waveform strain hp')
    
    plt.xlabel(r'Time [seconds]')
    plt.title(f"(M, mu, a, p0, e0) = ({M},{mu},{a},{p0},{e0})")
    plt.legend(fontsize = 16)
    plt.savefig("plots/waveform_plots/waveform_plot_start.png",bbox_inches='tight')
    plt.clf()

    # plt.plot(t[-600:], waveform_AAK_I_np[-600:], label = "AAK -- SNR = {}".format(np.round(SNR_AAK_FEW,5)))
    # plt.plot(t[-600:],waveform_Kerr_I_np[-600:], label = "Kerr -- SNR = {}".format(np.round(SNR_Kerr_FEW,5)))
    # plt.xlabel(r'Time [seconds]')
    # plt.ylabel(r'Waveform strain (channel II)')
    # plt.title(f"(M, mu, a, p0, e0) = ({M},{mu},{a},{p0},{e0})")
    # plt.legend(fontsize = 16)
    # plt.savefig("plots/waveform_plots/waveform_plot_end.png",bbox_inches='tight')
    # plt.clf()

quit()
