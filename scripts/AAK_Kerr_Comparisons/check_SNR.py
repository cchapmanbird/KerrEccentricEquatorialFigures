import cupy as cp
import numpy as np
import os 
import sys
sys.path.append("../")

# from scipy.signal import tukey       # I'm always pro windowing.  

from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform, AAKWaveformBase, KerrEquatorialEccentric,KerrEquatorialEccentricWaveformBase
from few.trajectory.inspiral import EMRIInspiral
from few.summation.directmodesum import DirectModeSum 
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.aakwave import AAKSummation

from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

YRSID_SI = 31558149.763545603

np.random.seed(1234)

def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).
    
    args:
        f (float): LISA-band frequency of the signal
    
    Returns:
        The output sensitivity strain Sn(f)
    """
    
    #Defining supporting functions
    L = 2.5e9 #m
    fstar = 19.09e-3 #Hz
    
    P_OMS = (1.5e-11**2)*(1+(2e-3/f)**4) #Hz-1
    P_acc = (3e-15**2)*(1+(0.4e-3/f)**2)*(1+(f/8e-3)**4) #Hz-1
    
    #S_c changes depending on signal duration (Equation 14 in 1803.01944)
    #for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    #log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1 
    
    A=9e-45
    Sc = A*f**(-7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fk-f)))
    sensitivity_LWA = (10/(3*L**2))*(P_OMS+4*(P_acc)/((2*np.pi*f)**4))*(1 + 6*f**2/(10*fstar**2))+Sc
    return sensitivity_LWA
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

    PSD = 2 * [xp.asarray(sensitivity_LWA(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR
##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    """
    # Intrinsic Parameters
    M_val = float(params[0])
    mu_val = float(params[1])
    
    a_val =  float(params[2])            
    p0_val = float(params[3])
    e0_val = float(params[4])
    xI0_val = 1.0 
    
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

    # Secondary charge 
    gamma_val = float(params[12]) 

    if a_val < 0:
        a_val *= -1.0
        xI0_val *= -1.0

    # Propose new waveform model
    waveform_prop = Waveform_model(M_val, mu_val, a_val, p0_val, e0_val, 
                                  xI0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0_val, Phi_theta0_val, Phi_r0_val, gamma_val, 
                                    mich=True, dt=delta_t, T=T)  # EMRI waveform across A, E and T.


    # Taper and then zero pad. 
    EMRI_w_pad_prop = [zero_pad(waveform_prop[i]) for i in range(N_channels)]

    # Compute in frequency domain
    EMRI_fft_prop = [xp.fft.rfft(item) for item in EMRI_w_pad_prop]

    # Compute (d - h| d- h)
    diff_f = [data_f[k] - EMRI_fft_prop[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f[k],diff_f[k],N_t,delta_t,PSD[k]) for k in range(N_channels)])
    
    # Return log-likelihood value as numpy val. 
    llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod))) 
    return (llike_val_np)

M = 1e6; mu = 10; a = 0.9; p0 = 8.54; e0 = 0.3; x_I0 = 1.0;
dist = 1.0; qS = 0.7; phiS = 0.7; qK = 0.7; phiK = 0.7; 
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

use_gpu = True
xp = cp
mich = True
# define trajectory
func = "KerrEccentricEquatorial"
insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "use_rk4": False,
    "func": func,
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

## ===================== CHECK TRAJECTORY ====================
# 
traj = EMRIInspiral(func=func, inspiral_kwargs = insp_kwargs)  # Set up trajectory module, pn5 AAK

# Compute trajectory 
if a < 0:
    a *= -1.0 
    x_I0 *= -1.0


t_traj, *out_GR = traj(M, mu, a, p0, e0, x_I0, 0.0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

print("Final value in semi-latus rectum", out_GR[0][-1])

traj_args_GR = [M, mu, a, out_GR[1][0], x_I0]
index_of_p = 3
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
p_new = 30
# p_new = get_p_at_t(
#     traj,
#     T,
#     traj_args_GR,
#     index_of_p=3,
#     index_of_a=2,
#     index_of_e=4,
#     index_of_x=5,
#     xtol=2e-12,
#     rtol=8.881784197001252e-16,
#     bounds=[25, 28],
# )

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", out_GR[0][-1])
print("Separatrix : ", get_separatrix(a, out_GR[1][-1], x_I0))

import time

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "err": 1e-10,  # To be set within the class
        "use_rk4": False,
        "integrate_phases":True,
        'func': 'KerrEccentricEquatorial'
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is availabel for this type of summation
    "pad_output": True,
}
    

amplitude_kwargs = {
    "specific_spins":[0.8, 0.9, 0.95],
    "use_gpu": True
    }
Waveform_model_AAK = GenerateEMRIWaveform(
AAKWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
AAKSummation, # Define the interpolation for the amplitudes
inspiral_kwargs=inspiral_kwargs,
sum_kwargs=sum_kwargs,
use_gpu=use_gpu,
return_list=True,
frame="detector"
)
    
Waveform_model_Kerr = GenerateEMRIWaveform(
KerrEquatorialEccentricWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
AmpInterpKerrEqEcc, # Define the interpolation for the amplitudes
InterpolatedModeSum, # Define the type of summation
ModeSelector, # Define the type of mode selection
inspiral_kwargs=inspiral_kwargs,
sum_kwargs=sum_kwargs,
amplitude_kwargs=amplitude_kwargs,
use_gpu=use_gpu,
return_list=True,
frame='detector'
)

## ============= USE THE LONG WAVELENGTH APPROXIMATION, VOMIT ================ ##
nmodes = 
specific_modes = [(2,2,n) for n in range(-2,2)]
params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 

waveform_AAK = Waveform_model_AAK(*params, T = T, dt = delta_t, mich = True)  # Generate h_plus and h_cross
waveform_Kerr = Waveform_model_Kerr(*params, T = T, dt = delta_t, mich = True)  # Generate h_plus and h_cross

N_t = len(zero_pad(waveform_AAK[0]))

freq_bin_np = np.fft.rfftfreq(N_t, delta_t)
freq_bin_np[0] = freq_bin_np[1]

PSD = 2*[cp.asarray(sensitivity_LWA(freq_bin_np))]

N_channels = 2

waveform_AAK_fft = [xp.fft.rfft(zero_pad(waveform_AAK[i])) for i in range(N_channels)]
waveform_Kerr_fft = [xp.fft.rfft(zero_pad(waveform_Kerr[i])) for i in range(N_channels)]

SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)

print("Truth waveform, final SNR for Kerr = ",SNR_Kerr)
print("Truth waveform, final SNR for AAK = ",SNR_AAK)

os.chdir('/home/ad/burkeol/work/Kerr_Systematics/test_few/diagnostics/data_file/eccentricity')
e0_vec = np.arange(0.1,0.5,0.02)

SNR_Kerr_vec=[]
SNR_AAK_vec=[]

import matplotlib.pyplot as plt
for eccentricity in e0_vec:
    params = [M,mu,a,p0,eccentricity,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]  

    waveform_Kerr = Waveform_model_Kerr(*params, mich = True, dt = delta_t, T = T)
    waveform_AAK = Waveform_model_AAK(*params, mich = True, dt = delta_t, T = T)
    
    SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
    SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)
    
    SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
    SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))


np.save("e0_vec.npy", e0_vec)
np.save("SNR_Kerr_vec.npy", SNR_Kerr_vec)
np.save("SNR_AAK_vec.npy", SNR_AAK_vec)

plt.plot(e0_vec,SNR_Kerr_vec, label = 'Kerr amplitudes')
plt.plot(e0_vec,SNR_AAK_vec, label = 'AAK amplitudes')
plt.grid()
plt.xlabel(r'Eccentricity $e_{0}$')
plt.ylabel(r'SNR')
plt.title("(M,mu,a,p0, T) = (1e6, 10, 0.9, 8.58, 2 years)")
plt.legend()
plt.savefig("Eccentricity_Plot_strong_field.pdf",bbox_inches = 'tight')
plt.clf()

p0 = 14.0 # Weak field

SNR_Kerr_vec=[]
SNR_AAK_vec=[]
for eccentricity in e0_vec:
    params = [M,mu,a,p0,eccentricity,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]  

    waveform_Kerr = Waveform_model_Kerr(*params, mich = True, dt = delta_t, T = T)
    waveform_AAK = Waveform_model_AAK(*params, mich = True, dt = delta_t, T = T)
    
    SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
    SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)
    
    SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
    SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))

plt.plot(e0_vec,SNR_Kerr_vec, label = 'Kerr amplitudes')
plt.plot(e0_vec,SNR_AAK_vec, label = 'AAK amplitudes')
plt.grid()
plt.xlabel(r'Eccentricity $e_{0}$')
plt.ylabel(r'SNR')
plt.title("(M,mu,a,p0, T) = (1e6, 10, 0.9, 14.0, 2 years)")
plt.legend()
plt.savefig("Eccentricity_Plot_weak_field.pdf",bbox_inches = 'tight')

