from bhpwave.waveform import KerrWaveform, KerrFrequencyWaveform
import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import SchwarzEccFlux, KerrEccEqFlux, PN5

from few.amplitude.romannet import RomanAmplitude
from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc

from few.utils.utility import ( 
    get_mismatch, 
    get_fundamental_frequencies, 
    get_separatrix, 
    get_mu_at_t, 
    get_p_at_t,
    )

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant, InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.summation.aakwave import AAKSummation
from few.utils.constants import *

from few.waveform import (
    FastSchwarzschildEccentricFlux, 
    SlowSchwarzschildEccentricFlux,
    FastKerrEccentricEquatorialFlux, 
    Pn5AAKWaveform,
    GenerateEMRIWaveform
)
from few.waveform.base import SphericalHarmonicWaveformBase, AAKWaveformBase


use_gpu = False

# keyword arguments for inspiral generator (EMRIInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        #"buffer_length": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    #"buffer_length": int(1e3),  # all of the trajectories will be well under len = 1000
   # "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    #"use_gpu": use_gpu,  # GPU is available for this type of summation
    "pad_output": False,
}


few = FastKerrEccentricEquatorialFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    #use_gpu=use_gpu,
)


BHPWave = KerrWaveform()


spins = np.array([-0.999,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.999])
mms = np.zeros(len(spins))
traj_module = EMRIInspiral(func=KerrEccEqFlux)

M = 1e6  # primary mass in solar masses
mu = 1e1 # secondary mass in solar masses
e0 = 0.0 # eccentricity is ignored for circular orbits
x0 = 1.  # inclination is ignored for circular orbits
qK = 0.8  # polar angle of Kerr spin angular momentum
phiK = 0.2  # azimuthal angle of Kerr spin angular momentum
theta = np.pi/3  # polar viewing angle
phi = np.pi/4  # azimuthal viewing angle
Phi_phi0 = 0.0 # initial azimuthal position of the secondary
Phi_theta0 = 0. # ignored for circular orbits
Phi_r0 = 0.0 # ignored for circular orbits
dt = 10.0  # time steps in seconds
T = 4.0  # waveform duration in years


for i in range(len(spins)):
    a = spins[i]
    # print(a)
    traj_args = [M, mu, a, e0, x0]
    traj_kwargs = {}
    index_of_p = 3

    # run trajectory
    p_new = get_p_at_t(
        traj_module,
        T,
        traj_args,
        index_of_p=3,
        index_of_a=2,
        index_of_e=4,
        index_of_x=5,
        traj_kwargs={},
        xtol=2e-12,
        rtol=8.881784197001252e-16,
        bounds=None,
    )
    
    fewWF = few(M, mu,a,p_new, e0,x0, theta, phi, dt=dt, T=T)
    BHPWF = BHPWave.source_frame(M, mu, a, p_new, theta, phi, Phi_phi0, dt=dt, T=T)
    mm = get_mismatch(fewWF, BHPWF)
    if(mm > 1.):
        mm = 1 - (mm-1)
    mms[i] = mm

np.savetxt("BHPWaveMismatchComparisonSpins.txt", spins)
np.savetxt("BHPWaveMismatchComparison.txt", mms)