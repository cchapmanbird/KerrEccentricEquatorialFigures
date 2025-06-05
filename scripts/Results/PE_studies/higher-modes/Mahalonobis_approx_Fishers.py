import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["mathtext.fontset"] = "custom"
#plt.rcParams["mathtext.rm"] = "Times New Roman"
#plt.rcParams["mathtext.it"] = "Times New Roman:italic"
#plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

from itertools import product
import os
import h5py
from tqdm import tqdm

#few utils
from few.utils.utility import get_p_at_t, get_mismatch, get_overlap
#few trajectory
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
#few waveform
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

from few.amplitude.romannet import RomanAmplitude
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms

#sef imports
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import generate_PSD, padding, inner_product
from stableemrifisher.fisher.derivatives import derivative

#LISAanalysistools imports
from fastlisaresponse import ResponseWrapper  # Response function 
from lisatools.detector import ESAOrbits, EqualArmlengthOrbits #ESAOrbits correspond to esa-trailing-orbits.h5, EqualArmlengthOrbits are equalarmlength-orbits.h5

from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

import few

use_gpu = True

if not use_gpu:
    
    #tune few configuration
    cfg_set = few.get_config_setter(reset=True)
    
    # Uncomment if you want to force CPU or GPU usage
    # Leave commented to let FEW automatically select the best available hardware
    #   - To force CPU usage:
    cfg_set.enable_backends("cpu")
    #   - To force GPU usage with CUDA 12.x
    #cfg_set.enable_backends("cuda12x", "cpu")
    #   - To force GPU usage with CUDA 11.x
    # cfg_set.enable_backends("cuda11x", "cpu")
    #
    cfg_set.set_log_level("info");
else:
    pass #let the backend decide for itself.
    
#fixed parameters
T_LISA = 1.0 #observation time, years
dt = 10.0 #sampling interval, seconds

m1 = 1.5e6 #MBH mass in solar masses (source frame)
m2 = 15.0 #secondary mass in solar masses (source frame)
x0 = 1.0 #inclination, must be = 1.0 for equatorial model

# initial phases
Phi_phi0 = 0.0 #azimuthal phase
Phi_theta0 = 0.0 #polar phase
Phi_r0 = 0.0 #radial phase

# define the extrinsic parameters
qK = np.pi / 3  # polar spin angle
phiK = np.pi / 4  # azimuthal viewing angle
qS = np.pi / 5  # polar sky angle
phiS = np.pi / 6  # azimuthal viewing angle
dist = 1.0  # distance in Gpc. We'll adjust this later to fix the SNR as 100.0

filename = f'Maha_ae_grid_Mtot_{m1+m2}'
if not os.path.exists(filename):
    os.mkdir(filename)

calculate_Fishers = True #calculate Fishers using the full model
calculate_derivatives = True #calculate derivatives using approximate models (for CV-bias calculation)
calculate_CV_biases = True #calculate the Cutler-Valisneri biases.

kerr_traj = EMRIInspiral(func=KerrEccEqFlux)

#grid parameters (p and e for now)
N = 10

try:
    with h5py.File(f"{filename}/data.h5", "r") as f:
        param_grid = f["gridpoints"][:]  # Read the dataset into a NumPy array
        p_range = f["p0"][:] + 0.5 #buffer
    try:
        with h5py.File(f"{filename}/data.h5", "r") as f:
            dist_range = f["dists"][:]  # Read the dataset into a NumPy array
    except KeyError:
        pass #will be handled later
        
except FileNotFoundError:
    
    a_range = np.linspace(0.1,0.9,N)
    e_range = np.linspace(0.1,0.5,N) #eccentricity above 0.5 is extremely expensive already...
    
    # Generate the Cartesian product of all parameter values
    param_grid = np.array(list((product(a_range, e_range))))
    
    p_range = []
    for i in tqdm(range(len(param_grid))):
        a = param_grid[i,0]
        e0 = param_grid[i,1]
        
        p0 = get_p_at_t(traj_module=kerr_traj, t_out=T_LISA, traj_args=[m1, m2, a, e0, x0, Phi_phi0, Phi_theta0, Phi_r0])
        
        p_range.append(p0)

        #check p for plunge trajectories
        #t, p, e, x, pp, pt, pr = kerr_traj(m1, m2, a, p0, e0, x0, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T_LISA, dt=dt)
    
        #print(t[-1] - (T_LISA*YRSID_SI))
        
    p_range = np.array(p_range)

    print(param_grid.shape, p_range.shape)
    
    #save as an h5py file
    with h5py.File(f"{filename}/data.h5", "w") as f:
        f.create_dataset("gridpoints", data=param_grid)
        f.create_dataset("p0", data=p_range) #save plunging traj p0
        
    p_range += 0.5 #buffer
    
    #with h5py.File(f"{filename}/p_e_grid_data.h5", "r") as f:
    #    loaded_data = f["dataset_name"][:]  # Read the dataset into a NumPy array
    
    #print(loaded_data)
    
    plt.scatter(param_grid[:,0],param_grid[:,1])
    plt.xlabel('a range',fontsize=16)
    plt.ylabel('e0 range',fontsize=16)
    plt.savefig(f"{filename}/grid.png",dpi=300,bbox_inches='tight')
    plt.show()
    
#initialize waveform model
sum_kwargs = {
    "pad_output": True, # True if expecting waveforms smaller than LISA observation window.
}

max_step_days = 10.0 #max trajectory step size in days

inspiral_kwargs = {
    "err":1e-11, #default = 1e-11
    "max_step_size":max_step_days*24*60*60, #in seconds
}

Waveform_model = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            sum_kwargs=sum_kwargs,
            inspiral_kwargs=inspiral_kwargs,
            return_list=False,
            )

#setup LISA response
tdi_gen ="2nd generation"# "2nd generation"#

order = 20  # interpolation order (should not change the result too much)
tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AET",
)  # could do "AET"

index_lambda = 8
index_beta = 7

# with longer signals we care less about this
t0 = 10000.0  # throw away on both ends when our orbital information is weird

EMRI_TDI = ResponseWrapper(
                        waveform_gen=Waveform_model,
                        Tobs=T_LISA,
                        t0=t0,
                        dt=dt,
                        index_lambda=index_lambda,
                        index_beta=index_beta,
                        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
                        use_gpu=use_gpu,
                        is_ecliptic_latitude=False,  # False if using polar angle (theta)
                        remove_garbage="zero",  # removes the beginning of the signal that has bad information
                        **tdi_kwargs_esa,
                        )
                        
#calculate and save SNR-adjusted dist

#calculate Fishers at each grid point
channels = [A1TDISens, E1TDISens, T1TDISens]
noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels]

param_names = ['M','mu','a','p0','e0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0'] #M, mu are primary and secondary masses in the SEF code respectively. This will be made consistent with FEW in a later version.

sef_kwargs = {'EMRI_waveform_gen':EMRI_TDI, #EMRI waveform model with TDI response
              'param_names': param_names, #params to be varied
              'der_order':4, #derivative order
              'Ndelta':12, #number of stable points
              'stats_for_nerds': False, #true if you wanna print debugging info
              'stability_plot': True, #true if you wanna plot stability surfaces
              'use_gpu':use_gpu,
              'filename': filename, #filename
              'noise_model':get_sensitivity,
              'channels':channels,
              'noise_kwargs':noise_kwargs,
              #'suffix':i #suffix for ith EMRI source will be added inside for loop
             }

emri_kwargs = {'T': T_LISA, 'dt': dt}

try:
    with h5py.File(f"{filename}/data.h5", "r") as f:
        dist_range = f["dists"][:]  # Read the dataset into a NumPy array

except:
    
    dist_range = []
    for i in tqdm(range(len(param_grid))):
        a = param_grid[i][0]
        e0 = param_grid[i][1]
        p0 = p_range[i]
    
        param_list = [m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
                
        #calculate Fisher
        sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
        SNR_before = sef.SNRcalc_SEF()
        dist_fact = SNR_before/100.0 #adjust distance such that SNR = 100.0
        
        param_list[6] *= dist_fact
        dist_range.append(dist*dist_fact)
    
        del sef
    
        sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
        SNR_after = sef.SNRcalc_SEF()
    
        print("SNR before: ", SNR_before, "SNR_after: ", SNR_after)
    
    #save as an h5py file
    with h5py.File(f"{filename}/data.h5", "a") as f:
        f.create_dataset("dists", data=np.array(dist_range))
        
def logmasstransform(Fisher, m1, m2, index_of_m1 = 0, index_of_m2 = 1):
    """ transform m1, m2 -> lnm1, lnm2. Fisher transformation: https://en.wikipedia.org/wiki/Fisher_information """
    
    J = np.eye(len(Fisher))
    J[index_of_m1,index_of_m1] = m1
    J[index_of_m2,index_of_m2] = m2

    return J.T@Fisher@J

def cutlervallis(waveform_truth, waveform_approx, Fisher_truth, partial_approx, params_truth, PSD_func, dt, use_gpu=False):
    """
    Calculate best-fit param points using the Cutler-Vallisneri linear-bias approximation.
    
    params:
    
        waveform_truth (ndarray) : the time-series waveform from the true template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        waveform_approx (ndarray) : the time-series waveform from the approximate template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        partial_approx (ndarray) : the time-series partial derivative of the approximate template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        Fisher_truth (ndarray) : the Fisher matrix (log-mass units) at params_truth. ndarray of shape 'd x d'
        params_truth (ndarray) : the array of true params. ndarray of shape 'd'
        PSD_func (ndarray) : the frequency-domain LISA noise sensitivity curve. ndarray of shape 'N x L' where L is the length of the frequency series. 

    returns:
    
        CV_bias (ndarray): params_truth + np.linalg.inv(Fisher_truth) @ inner_product(partial_approx, waveform_truth - waveform_approx)
    """

    if use_gpu:
        xp = cp
    else:
        xp = np
        
    waveform_truth = padding(waveform_truth, waveform_approx, use_gpu = use_gpu) #make waveform_truth the same length as waveform_approx

    delta_wave = waveform_truth - waveform_approx
    
    # calculate all the column-wise inner products
    
    inn_prods = []
    
    for j in range(len(Fisher_truth)):
        
        print("delta_wave.shape, partial_approx.shape: ", delta_wave.shape, partial_approx[j].shape)
                      
        inn_prod_j = inner_product(partial_approx[j], delta_wave, PSD=PSD_func, dt=dt, use_gpu = use_gpu) #should be a scalar

        inn_prod_j = xp.asarray(inn_prod_j)
        
        print(f'inner_prod at j = {j}: ', inn_prod_j)
        
        if use_gpu:
            inn_prod_j = xp.asnumpy(inn_prod_j)
        
        inn_prods.append(inn_prod_j)
        
    inn_prods = np.array(inn_prods)
        
    # calculate all the param shifts
    
    delta_param_all = []
    
    for i in range(len(Fisher_truth)):
            
        delta_param_i = np.linalg.inv(Fisher_truth)[i,:]@inn_prods

        delta_param_all.append(delta_param_i)
        
    print('delta_param_all: ', delta_param_all)
        
    return params_truth + np.array(delta_param_all)
    
if calculate_CV_biases:
    
    #alternate model analysis.
    
    alternate_models = [
        #"leqmFastKerr",
        #"l2m2FastKerr",
        #"l2FastKerr",
        #"errFastKerr",
        "AAK",
    ]
    
    for j in range(len(alternate_models)):
        alt_mod = alternate_models[j]
    
        if alt_mod == "l2m2FastKerr":
            # only use the dominant l=2 modes
            specific_modes = []
            lmax = 2
            for l in range(2, lmax+1):
                for m in range(2, 3): #FEW will handle m, n < 0 case
                    for n in range(0, 31):
                        specific_modes.append((l, m, n))
                        
            mode_selector_kwargs = {"mode_selection": specific_modes}
            #emri_kwargs_alt["include_minus_mkn"] = False

            #instantiate alternate waveform model
            Waveform_model_alt = GenerateEMRIWaveform(
                    "FastKerrEccentricEquatorialFlux",
                    sum_kwargs=sum_kwargs,
                    inspiral_kwargs=inspiral_kwargs,
                    mode_selector_kwargs=mode_selector_kwargs,
                    return_list=False,
                    )
    
        elif alt_mod == "l2FastKerr":
            # only use the dominant l=2 modes
            specific_modes = []
            lmax = 2
            for l in range(2, lmax+1):
                for m in range(0, l+1): #FEW will handle m, n < 0 case
                    for n in range(0, 31):
                        specific_modes.append((l, m, n))
                        
            mode_selector_kwargs = {"mode_selection": specific_modes}
            #emri_kwargs_alt["include_minus_mkn"] = False

            #instantiate alternate waveform model
            Waveform_model_alt = GenerateEMRIWaveform(
                    "FastKerrEccentricEquatorialFlux",
                    sum_kwargs=sum_kwargs,
                    inspiral_kwargs=inspiral_kwargs,
                    mode_selector_kwargs=mode_selector_kwargs,
                    return_list=False,
                    )
            
        elif alt_mod == "leqmFastKerr":
            # only use modes with l = m
            specific_modes = []
            lmax = 10
            for l in range(2, lmax+1):
                for m in range(l, l+1): #FEW will handle m, n < 0 case
                    for n in range(0, 31):
                        specific_modes.append((l, m, n))
                        
            mode_selector_kwargs = {"mode_selection": specific_modes}
            #emri_kwargs_alt["include_minus_mkn"] = False

            #instantiate alternate waveform model
            Waveform_model_alt = GenerateEMRIWaveform(
                    "FastKerrEccentricEquatorialFlux",
                    sum_kwargs=sum_kwargs,
                    inspiral_kwargs=inspiral_kwargs,
                    mode_selector_kwargs=mode_selector_kwargs,
                    return_list=False,
                    )
    
        elif alt_mod == "errFastKerr":
            mode_selector_kwargs = {"eps": 1e-2} #instead of the default 1e-5
            #instantiate alternate waveform model
            Waveform_model_alt = GenerateEMRIWaveform(
                    "FastKerrEccentricEquatorialFlux",
                    sum_kwargs=sum_kwargs,
                    inspiral_kwargs=inspiral_kwargs,
                    mode_selector_kwargs=mode_selector_kwargs,
                    return_list=False,
                    )

        elif alt_mod == 'AAK':
            #use the 5PN AAK Kludge
            waveform_kwargs = {}
            inspiral_kwargs["func"] = "KerrEccEqFlux" #still use relativistic trajectories - only difference is in Amplitudes. 
            Waveform_model_alt = GenerateEMRIWaveform(
                "Pn5AAKWaveform",
                sum_kwargs=sum_kwargs,
                inspiral_kwargs=inspiral_kwargs,
                return_list=False
                )
    
        filename_bias = os.path.join(filename,alt_mod)
    
        EMRI_TDI_alt = ResponseWrapper(
                            waveform_gen=Waveform_model_alt,
                            Tobs=T_LISA,
                            t0=t0,
                            dt=dt,
                            index_lambda=index_lambda,
                            index_beta=index_beta,
                            flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
                            use_gpu=use_gpu,
                            is_ecliptic_latitude=False,  # False if using polar angle (theta)
                            remove_garbage="zero",  # removes the beginning of the signal that has bad information
                            **tdi_kwargs_esa,
                            )
    
        param_names = ['M','mu','a','p0','e0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0']
        
        #update sef_kwargs
        sef_kwargs = {'EMRI_waveform_gen':EMRI_TDI_alt, #EMRI waveform model with TDI response
                  'param_names': param_names, #params to be varied
                  'der_order':4, #derivative order
                  'Ndelta':12, #number of stable points
                  'stats_for_nerds': False, #true if you wanna print debugging info
                  'stability_plot': True, #true if you wanna plot stability surfaces
                  'use_gpu':use_gpu,
                  'filename': filename_bias,
                  'noise_model':get_sensitivity,
                  'channels':channels,
                  'noise_kwargs':noise_kwargs,
                  #'suffix':i #suffix for ith EMRI source will be added inside for loop
                  }

        if calculate_Fishers:

            for i in tqdm(range(len(param_grid))):

                print(f"Calculated Fishers for {alt_mod}")
                
                try:
                    #skip Fishers which are already calculated.
                    with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                        _ = f["Fisher"][:]
                    continue

                except FileNotFoundError:
                    a = param_grid[i][0]
                    e0 = param_grid[i][1]
                    p0 = p_range[i]
                    dist = dist_range[i]

                    sef_kwargs['suffix'] = i

                    param_list = [m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
            
                    #calculate Fisher
                    sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
                    sef()

            print("Transforming to log-mass")
            
            for i in range(len(param_grid)):
            
                try:
                    with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                        _ = f["Fisher_transformed"][:]
                    continue
                except KeyError:
                    with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                        Fisher = f["Fisher"][:]
                    Fisher_transform = logmasstransform(Fisher, m1, m2)
                    with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "a") as f: #'a' mode for appending existing datasets.
                        f.create_dataset("Fisher_transformed", data=Fisher_transform)

            print("Checking log-mass Fisher positive-definiteness...")

            check_flag = True
            
            for i in tqdm(range(len(param_grid))):
            
                with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                    Fisher = f["Fisher"][:]
            
                with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                    Fisher_transformed = f["Fisher_transformed"][:]

                #if you want to exclude sus parameters for troubleshooting
                rows_to_remove = []
                for j in range(len(param_names)):
                    if param_names[j] in []:#'dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0']:
                        rows_to_remove.append(j)
            
                rows_to_remove = np.array(rows_to_remove)
                
                if len(rows_to_remove) > 0:
                    Fisher_transformed = np.delete(np.delete(Fisher_transformed,rows_to_remove,axis=0),rows_to_remove,axis=1)
                    Fisher = np.delete(np.delete(Fisher,rows_to_remove,axis=0),rows_to_remove,axis=1)
            
                if (np.linalg.eigvals(Fisher_transformed) < 0.0).any():
                    check_flag = False
                    print('positive definiteness failed for: ', i)
                    
            if check_flag:
                print("positive-definiteness check passed!")
    
        if calculate_derivatives:
                
            for i in tqdm(range(len(param_grid))):
                
                print(f"Calculating derivatives for {alt_mod}")
            
                try:
                    #do not calculate derivatives that have already been calculated.
                    with h5py.File(f"{filename_bias}/derivatives_{i}.h5", "r") as f:
                        _ = f["derivatives"][:]
                    continue
    
                except FileNotFoundError:
                    a = param_grid[i][0]
                    e0 = param_grid[i][1]
                    p0 = p_range[i]
                    dist = dist_range[i]
                    
                    sef_kwargs['suffix'] = i
                
                    param_list = [m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
                
                    sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
                    rho = sef.SNRcalc_SEF()
                    sef.SNR2 = rho**2
                    print(f'Waveform Generated. SNR: {rho}')
            
                    if sef.filename != None:
                        if not os.path.exists(sef.filename):
                            os.makedirs(sef.filename)
                
                    sef.Fisher_Stability() #calculate stable deltas
                    
                    print(f'calculating derivatives for {alt_mod} at {param_list}...')
                    
                    Fisher = np.zeros((sef.npar,sef.npar), dtype=np.float64)
                    dtv = []
                    for k in range(sef.npar):
                    
                        if sef.param_names[k] in list(sef.minmax.keys()):
                            if sef.wave_params[sef.param_names[k]] <= sef.minmax[sef.param_names[k]][0]:
                                dtv.append(derivative(sef.waveform_generator, sef.wave_params, sef.param_names[k], sef.deltas[sef.param_names[k]], kind="forward", waveform=sef.waveform, order=sef.order, use_gpu=sef.use_gpu, waveform_kwargs=sef.waveform_kwargs))
                            elif sef.wave_params[sef.param_names[k]] > sef.minmax[sef.param_names[k]][1]:
                                dtv.append(derivative(sef.waveform_generator, sef.wave_params, sef.param_names[k],sef.deltas[sef.param_names[k]], kind="backward", waveform=sef.waveform, order=sef.order, use_gpu=sef.use_gpu, waveform_kwargs=sef.waveform_kwargs))
                            else:
                                dtv.append(derivative(sef.waveform_generator, sef.wave_params, sef.param_names[k],sef.deltas[sef.param_names[k]],use_gpu=sef.use_gpu, waveform=sef.waveform, order=sef.order, waveform_kwargs=sef.waveform_kwargs))
                        else:
                            dtv.append(derivative(sef.waveform_generator, sef.wave_params, sef.param_names[k], sef.deltas[sef.param_names[k]],use_gpu=sef.use_gpu, waveform=sef.waveform, order=sef.order, waveform_kwargs=sef.waveform_kwargs))
            
                    print("Finished derivatives")
        
                    if sef.use_gpu:
                        dtv = cp.asnumpy(cp.asarray(dtv)) #h5py only stores numpy
                        
                    #Jacobian transform to log-masses
                    for k in range(sef.npar):
                        if sef.param_names[k] == 'M':
                            dtv[k] *= m1 #to obtain \partial_{log m1}h
                        elif sef.param_names[k] == 'mu':
                            dtv[k] *= m2 #to obtain \partial_{log m2}h
                            
                    with h5py.File(f"{filename_bias}/derivatives_{i}.h5", "w") as f:
                        f.create_dataset("derivatives", data=dtv)
    
        #calculate the CV biases
        print(f"Calculating Cutler-Vallisneri Biases for model {alt_mod}")
    
        for i in tqdm(range(len(param_grid))):
            a = param_grid[i][0]
            e0 = param_grid[i][1]
            p0 = p_range[i]
            dist = dist_range[i]
            
            params_truth = [m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
            
            params_truth_in = np.array([m1, m2, a, p0, e0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_r0]) #params from which Fisher is calculated.
            params_truth_in_transformed = np.array([np.log(m1), np.log(m2), a, p0, e0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_r0]) #params from which the transformed Fisher is calculated.
            
            #import the Fisher matrix
            with h5py.File(f"{filename_bias}/Fisher_{i}.h5", "r") as f:
                Fisher_transformed = f["Fisher_transformed"][:]
            
            try:
                raise FileNotFoundError
                #with h5py.File(f"{filename_bias}/biased_params_{i}.h5","r") as f:
                #    biased_params = f["biased_params"][:]
            except FileNotFoundError:
                sef = StableEMRIFisher(*params_truth, **emri_kwargs, **sef_kwargs)
                rho = sef.SNRcalc_SEF() #generates the PSD and the approximate waveform
        
                PSD_func = sef.PSD_funcs
                waveform_approx = sef.waveform
            
                #import the derivatives
                with h5py.File(f"{filename_bias}/derivatives_{i}.h5", "r") as f:
                    partial_approx = f["derivatives"][:]
                    
                rows_to_remove = []
                for _ in range(len(param_names)):
                    if param_names[_] in []:#'qK','phiK']:
                        rows_to_remove.append(_)
        
                rows_to_remove = np.array(rows_to_remove)
        
                if len(rows_to_remove) > 0:
                    Fisher_transformed = np.delete(np.delete(Fisher_transformed,rows_to_remove,axis=0),rows_to_remove,axis=1)
                    partial_approx = np.delete(partial_approx, rows_to_remove, axis=0)
                    params_truth_in_transformed = np.delete(params_truth_in_transformed, rows_to_remove, axis=0)
        
                #print(partial_approx[0][0][20000:40000])
                #print(partial_approx[0][1][20000:40000])
                #print(partial_approx[0][2][20000:40000])
                
                #calculate the waveform using the full template
                waveform_truth = EMRI_TDI(*params_truth, **emri_kwargs)
        
                biased_params = cutlervallis(waveform_truth=waveform_truth, waveform_approx=waveform_approx,
                                             Fisher_truth=Fisher_transformed, partial_approx=partial_approx,
                                             params_truth=params_truth_in_transformed, PSD_func = PSD_func, dt = dt,
                                              use_gpu=use_gpu)
        
                if use_gpu:
                    biased_params = cp.asnumpy(cp.asarray(biased_params))
        
                with h5py.File(f"{filename_bias}/biased_params_{i}.h5", "w") as f:
                    f.create_dataset("biased_params", data=biased_params) #saved in log-masses!!!
        
            print('biased params: ', alt_mod, biased_params)
        
            #calculate the sigma contour as the Mahalanobis distance between the true and biased params (https://en.wikipedia.org/wiki/Mahalanobis_distance):
            
            sigma_contour = np.sqrt((biased_params - params_truth_in_transformed)@Fisher_transformed@(biased_params - params_truth_in_transformed))
            sigma_contour /= np.sqrt(len(params_truth_in)) #scaled Mahalonobis distance for dimensional generality
            
            print('sigma contours: ', alt_mod, sigma_contour)
        
            with h5py.File(f"{filename_bias}/biased_params_{i}.h5", "a") as f: #"a" for appending datapoints/
                f.create_dataset("sigma_contours", data=sigma_contour) #in log-masses!!!
                

