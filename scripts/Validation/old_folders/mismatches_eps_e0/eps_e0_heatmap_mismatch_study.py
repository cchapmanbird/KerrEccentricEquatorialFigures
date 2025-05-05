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

M = 1e6; mu = 10.0; a = 0.998; p0 = 10.628; e0 = 0.1; x_I0 = 1.0
dist = 5.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2;  # Usual extrinsic parameters
Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

delta_t = 5.0; T = 4.0

# New extrinsic params

# Test case 1
qS = 1.3; phiS = 0.5; qK = 0.3; phiK = 1.8
# Test case 2
#qS = ; phiS = 0.5; qK = 0.3; phiK = 1.8

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

# ================== PLOT THE A CHANNEL ===================

# Now try to compute mismatches as we change e0. 
import time

start = time.time()
# Production runs 
eps_vec = np.arange(-6,0,0.2) 
e0_vec = np.arange(0.0,0.8,0.02)
# e0_vec = np.arange(0.5,0.8,0.1)

# Test runs
# eps_vec = np.arange(-5,-1.0,2) 
# e0_vec = np.arange(0.0,0.8,0.4)

M_mu_vec = [[1e5,1.0], [1e6, 1e1], [1e7, 1e2]]
labels = [["1e5", "1"], ["1e6", "10"], ["1e7", "100"]]

# M_mu_vec = [[1e7, 1e2]]
# labels = [["1e7", "100"]]

# test mass-ratio? 
# M_mu_vec = [[1e6,1e1], [1e6, 1e2], [1e6, 1e3]]
# labels = [["1e6", "1e1"], ["1e6", "1e2"], ["1e6", "1e3"]]
# M_mu_vec = [[1e6,10.0]]

fig,ax = plt.subplots(1,3, figsize = (16,7), sharey = True)
# labels = [["1e6","10"]]
k = 0
for M_mu in (M_mu_vec):
    mismatch_vals = np.zeros((len(e0_vec), len(eps_vec)))
    # Initialize heat map storage
    for i, e_val in tqdm(enumerate(e0_vec)):

        # Compute semi-latus rectum giving plunge for T = 4 years for each e0
        traj_args = [M_mu_vec[k][0], M_mu_vec[k][1], 0.998, e_val, 1.0]
        index_of_p = 3
        # Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
        try:
            p_plunge = get_p_at_t(
                traj,
                4.0,
                traj_args,
                bounds=None
            )
            print(f"for M = {M_mu_vec[k][0]} and mu = {M_mu_vec[k][1]} the value of p0 ={p_plunge} for T = {T} yr observation")
        except ValueError:
            mismatch_vals[i,j] = np.nan
            break

        params = [M_mu_vec[k][0],M_mu_vec[k][1], 0.998, p_plunge, e_val, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
        truth_waveform = EMRI_TDI_Model(*params, eps = 0.0)
        for j, eps_val in enumerate(eps_vec):
            waveform_eps = EMRI_TDI_Model(*params, eps = 10**eps_val) # Compute waveform with less modes
            mismatch_vals[i,j] = xp.asnumpy(mismatch_function(waveform_eps,truth_waveform, PSD_AET, delta_t,N_channels=2)) 

    # if mismatch_vals[i,j] == 0.0:
    #     mismath_vals[i,j] = np.nan
    mismatch_vals[mismatch_vals == 0] = np.nan
    # mismatch_vec_np = cp.asnumpy(mismatch_vals)
    mismatch_vals = mismatch_vals.T
    # Create meshgrid
    EPS, E0 = np.meshgrid(e0_vec, eps_vec)

    # Use pcolormesh for smooth gradients

    from matplotlib.colors import LogNorm
    cmap = plt.get_cmap("viridis").copy()  # Copy the viridis colormap
    cmap.set_bad("grey")  # Set NaN values to appear as grey


    pcm = ax[k].pcolormesh(EPS, E0, mismatch_vals, shading='auto', cmap=cmap, norm=LogNorm(vmin=np.nanmin(mismatch_vals), vmax=np.nanmax(mismatch_vals)))

    # Define contour levels at 0.1 and 1 radians
    contour_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Add white contour lines on top of the heatmap
    contours = ax[k].contour(EPS, E0, mismatch_vals, levels=contour_levels, colors='white', linewidths=1, linestyles='dashed')

    # Add labels to contours
    ax[k].clabel(contours, fmt={1e-5: r"$10^{-5}$", 1e-4: r"$10^{-4}$", 1e-3: r"$10^{-3}$", 1e-2: r"$10^{-2}$", 1e-1: r"$10^{-1}$"}, colors='white', fontsize=15) 
 
    # Set labels and title
    ax[k].set_xlabel(r'$e_{0}$', fontsize=20)
    ax[0].set_ylabel(r'$\log_{10}\kappa$', fontsize=20)
    ax[k].set_title(rf'$(M/M_\odot, \mu/M_\odot, a) = ({labels[k][0]}, {labels[k][1]}, 0.998)$',fontsize = 14)
    # np.save("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/mismatches_eps_e0/data/mismatch_vals_"+labels[k][0] + "_" + labels[k][1], mismatch_vals)
    # np.save("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/mismatches_eps_e0/data/ecc_vec.npy", e0_vec)
    # np.save("/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/mismatches_eps_e0/data/eps_vec.npy", eps_vec)
    k+=1


# Add a single colorbar for all subplots
cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.15)
# cbar = fig.colorbar(pcm, orientation='horizontal', shrink=0.8, pad=0.15)
cbar.set_label(r'Mismatch', fontsize=25)
cbar.ax.tick_params(labelsize=20)  # Set tick label font size

plot_dir = "/home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/mismatches_eps_e0/heatmap_plots/"
plt.savefig(plot_dir + f"check_plot_extrinsic_params_qS_{qS}_phiS_{phiS}_qK_{qK}_phiK_{phiK}.png",bbox_inches="tight")
print("Finished run in ", time.time() - start, "seconds")
plt.show()