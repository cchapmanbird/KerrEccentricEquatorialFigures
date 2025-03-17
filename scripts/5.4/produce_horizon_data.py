# Description: Script to generate horizon data for a given set of parameters
# python produce_horizon_data.py -Tobs 2.0 -dt 5.0 -t kerr kerr pn5 -wf kerr aak aak -outname new_few -grids e0 M -qs 1e-5

import argparse
import os
import GPUtil
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
print("PID:",os.getpid())
import time
parser = argparse.ArgumentParser(description="horizon redshift")
parser.add_argument("-dev", "--dev", help="GPU device", required=False, type=int, default=None)
parser.add_argument("-Tobs", "--Tobs", help="Observation Time in years", required=False, default=1.0, type=float)
parser.add_argument("-Ms", "--Ms", help="masses", required=False, nargs='*', default=1e6, type=float)
parser.add_argument("-qs", "--qs", help="mass ratio", required=False, nargs='*', default=5e-5, type=float)
parser.add_argument("-e0s", "--e0s", help="initial eccentricity", required=False, nargs='*', default=0.0, type=float)
parser.add_argument("-spins", "--spins", help="dimensionless spin", required=False, nargs='*', default=0.99, type=float)
parser.add_argument("-dt", "--dt", help="sampling interval delta t", required=False, type=float, default=5.0)
parser.add_argument("-SNR", "--SNR", help="SNR", required=False, type=float, default=20.0)
parser.add_argument("-outname", "--outname", help="output name", required=False, type=str, default="")
parser.add_argument("-t", "--traj", help="trajectory", required=False, nargs='*', type=str, default="kerr")
parser.add_argument("-wf", "--wf", help="waveform", required=False, nargs='*', type=str, default="kerr")
parser.add_argument("-grids", "--grids", help="parameters to iterate over", required=False, nargs=2, default=['e0', 'M'], type=str)


args = vars(parser.parse_args())

dev = args['dev']

# if dev is not None:
#     os.system("CUDA_VISIBLE_DEVICES="+str(args['dev']))
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args['dev'])
#     os.system("echo $CUDA_VISIBLE_DEVICES")

import sys, os

import matplotlib.pyplot as plt

import matplotlib.lines as mlines
import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist

#sys.path.append('/data/asantini/emris/DirtyEMRI/DataAnalysis/LISAanalysistools/')

from lisatools.diagnostic import *
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens, T2TDISens
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer

from few.trajectory.ode.flux import SchwarzEccFlux, KerrEccEqFlux
from few.trajectory.ode.pn5 import PN5
from few.waveform.waveform import GenerateEMRIWaveform, AAKWaveformBase, FastKerrEccentricEquatorialFlux, FastSchwarzschildEccentricFlux
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix
from few.utils.globals import get_first_backend

from fastlisaresponse import ResponseWrapper

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

import h5py

#will have to remove this
import pysco
pysco.plot.default_plotting()

from scipy.signal import get_window
from tqdm import tqdm

SEED = 26011996

def get_free_gpus(n_gpus=1):
    '''
    Get the IDs of free GPUs.

    Parameters
    ----------
    n_gpus : int
        Number of free GPUs to return.
    
    Returns
    -------
    free_gpus : list
        List of IDs of free GPUs.
    '''

    free_gpus = GPUtil.getAvailable(order='first', limit=n_gpus, maxLoad=0.001, maxMemory=0.001)
    return free_gpus


try:
    import cupy as xp
    # set GPU device
    if dev is None:
        free_gpus = get_free_gpus(n_gpus=1)
        if not free_gpus:
            gpu_available = False
        else:
            dev = free_gpus[0]
    else:
        os.system("CUDA_VISIBLE_DEVICES="+str(dev))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        print("Using GPU", dev)
        gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

import warnings
warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = True

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

np.random.seed(SEED)
xp.random.seed(SEED)


cosmo = Planck18 #FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def get_redshift(distance):
    return float(z_at_value(cosmo.luminosity_distance, distance * u.Gpc ))

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

class wave_gen_windowed:
    """
    Generate a waveform and apply a window to it
    """
    def __init__(self, wave_gen, window_fn=('tukey', 0.005)):
        self.wave_gen = wave_gen
        self.window_fn = window_fn

    def __call__(self, *args, kwargs={}):
        wave = self.wave_gen(*args, **kwargs)
        if isinstance(wave, list):
            window = xp.asarray(get_window(self.window_fn, len(wave[0])))
            wave = [wave[i] * window for i in range(len(wave))]
        else:
            window = xp.asarray(get_window(self.window_fn, len(wave)))
            wave = wave * window

        return wave   


def generate_data(
        Tobs,
        dt,
        fixed_params,
        grids,
        outname,
        traj,
        wave_gen,
        snr_thr=20.0,
        emri_kwargs={},):
    """
    Generate data for horizon plot
    """
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI

    tdi_gen = "2nd generation" #"1st generation" or "2nd generation"

    order = 25  # interpolation order (should not change the result too much)
    orbits = EqualArmlengthOrbits(use_gpu=use_gpu) # ESAOrbits(use_gpu=use_gpu)

    tdi_kwargs_esa = dict(
        orbits=orbits,
        order=order,
        tdi=tdi_gen,
        tdi_chan="AET",
    )  # could do "AET

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird

    resp_gen = ResponseWrapper(
        wave_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,#"zero",  # removes the beginning of the signal that has bad information
        #n_overide=int(1e5),  # override the number of points (should be larger than the number of points in the signal)
        **tdi_kwargs_esa,
    )

    resp_gen = wave_gen_windowed(resp_gen, window_fn=('tukey', 0.005))

    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(-0.99999, 0.99999),  # qS
                1: uniform_dist(0.0, 2 * np.pi),  # phiS
                2: uniform_dist(-0.99999, 0.99999),  # qK
                3: uniform_dist(0.0, 2 * np.pi),  # phiK
                4: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                5: uniform_dist(0.0, 2 * np.pi),  # Phi_theta0
                6: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    def get_p0(M, mu, a, e0, x0, Tobs):
        # fix p0 given T
        try:
            p0 = get_p_at_t(traj,Tobs * 0.999,[M, mu, a, e0, x0],bounds=[get_separatrix(a,e0,x0)+0.1, 150.0])
        except Exception as e:
            print(e)
            breakpoint()   
        #print("new p0 fixed by Tobs, p0=", p0, traj(M, mu, a, p0, e0, x0, T=10.0)[0][-1]/YRSID_SI)
        return p0
    
    # inner_kw = dict(dt=dt,
    #                 psd="A1TDISens",
    #                 psd_args=(),
    #                 psd_kwargs={'stochastic_params':(Tobs,)},
    #                 ) 
    
    inner_kw = dict(dt=dt,psd="AET2SensitivityMatrix",psd_args=(),psd_kwargs={"stochastic_params": (Tobs,)})#,use_gpu=use_gpu) 
    inner_kw = dict(dt=dt,psd=[A2TDISens, E2TDISens, T2TDISens], psd_args=(),psd_kwargs={"stochastic_params": (Tobs,)})
    
    def get_snr(inp, emri_kwargs={}):
        data_channels = resp_gen(*inp, kwargs=emri_kwargs)

    
        return snr([data_channels[0], data_channels[1], data_channels[2]], **inner_kw)

        
    
    def get_snr_avg(M, mu, spin, e0, x0, Tobs, avg_n=1, emri_kwargs={}):
        p0 = get_p0(M, mu, spin, e0, x0, Tobs)
        prior_draw = priors['emri'].rvs(avg_n) #random draw from prior for the extrinsic parameters

        prior_draw[:, 0] = np.arccos(prior_draw[:, 0])  # qS
        prior_draw[:, 2] = np.arccos(prior_draw[:, 2]) # qK

        injection = np.empty((avg_n, 7))
        injection[:, 0] = M
        injection[:, 1] = mu
        injection[:, 2] = spin
        injection[:, 3] = p0
        injection[:, 4] = e0
        injection[:, 5] = x0  
        injection[:, 6] = 1.0 #distance
        
        injection = np.concatenate((injection, prior_draw), axis=1)

        data_channels = resp_gen(*injection[0], kwargs=emri_kwargs)

        ffth = xp.fft.rfft(data_channels[0])*dt
        fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)

        PSD_arr = get_sensitivity(fft_freq, sens_fn='A2TDISens', **inner_kw['psd_kwargs'])
        plt.figure()
        try:
            plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
            plt.loglog(fft_freq.get(), PSD_arr.get())
        except:
            plt.plot(fft_freq, (xp.abs(ffth)**2))
            plt.loglog(fft_freq, PSD_arr)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [Hz$^{-1}$]")

        savepath = savename[:-3] + '_tmp_plots/' + f"M_{M:.1e}_mu_{mu:.1e}_a_{spin:.1e}_e0_{e0:.1e}_x0_{x0:.1e}.pdf"

        plt.savefig(savepath)

        snr_here = np.mean(np.array([get_snr(inp, emri_kwargs) for inp in injection]))
        assert np.isfinite(snr_here)
        return snr_here

    with h5py.File(outname, 'w') as f:
        f.attrs['Tobs'] = Tobs
        f.attrs['dt'] = dt
        for key, val in fixed_params.items():
            f.attrs[key] = val
        
        for key, val in grids.items():
            f.attrs[key] = val

        params_all = fixed_params.copy()

        keys_grid = list(grids.keys())
        key_lines, vals_lines = keys_grid[0], grids[keys_grid[0]]
        key_x, vals_x = keys_grid[1], grids[keys_grid[1]]

        mumin, mumax = 1, 1e7
        
        for line_element in tqdm(vals_lines):
            print(key_lines, line_element)
            z = np.zeros((len(vals_x),))
            for i, x_element in enumerate(vals_x):
                params_all[key_lines] = line_element
                params_all[key_x] = x_element

                M = params_all['M']
                q = params_all['q']
                spin = params_all['spin']
                e0 = params_all['e0']
                #x0 = params_all['x0']
                x0 = np.sign(spin) * 1.0 if spin != 0.0 else 1.0
                spin = np.abs(spin)
                print("M, q, spin, e0, x0", M, q, spin, e0, x0)
                mu = q * M
                if mu < mumin or mu > mumax:
                    z[i] = np.nan
                    continue
                snr_here = get_snr_avg(M, mu, spin, e0, x0, Tobs, avg_n=300, emri_kwargs=emri_kwargs)
            
                d_L = snr_here / snr_thr
                z[i] = get_redshift(d_L)

            f.create_dataset(key_lines + f'_{line_element}', data=z)

if __name__ == '__main__':
    Tobs = args['Tobs']
    Ms = args['Ms']
    qs = args['qs']
    spins = args['spins']
    e0s = args['e0s']
    dt = args['dt']
    snr_thr = args['SNR']
    base_outname = args['outname']
    traj_module = args['traj']
    wf_module = args['wf']
    grid_keys = args['grids']

    inspiral_func_all = dict(zip(['kerr', 'schwarzschild', 'pn5'], [KerrEccEqFlux, SchwarzEccFlux, PN5]))
    args_all = dict(zip(['kerr', 'schwarzschild', 'aak'], [[FastKerrEccentricEquatorialFlux,], [FastSchwarzschildEccentricFlux,], [AAKWaveformBase, EMRIInspiral, AAKSummation]]))


    print("generating different " + grid_keys[0] + " lines for " + grid_keys[1] + " grid")

    params_all = ['M', 'q', 'spin', 'e0']
    fixed_params_keys = [key for key in params_all if key not in grid_keys]
    fixed_params_all = {}
    for key, el in zip(params_all, [Ms, qs, spins, e0s]):
        if isinstance(el, float):
            el = [el]
        fixed_params_all[key] = el

    #e0_grid = np.arange(0.15, 0.76, 0.15)
    e0_grid = [0.1, 0.4, 0.6, 0.75]
    spin_grid = [-0.99, -0.5, 0.0, 0.5, 0.99]
    #np.concatenate((np.arange(0.0, 0.99, 0.1), np.array([0.99])))
    q_grid = [1e-5, 1e-4, 1e-3]

    Mmin, Mmax = 9e5, 5e7
    nmasses = 20
    M_grid =10**np.linspace(np.log10(Mmin), np.log10(Mmax), num=nmasses)

    grids_all = {
        'M': M_grid,
        'q': q_grid,
        'spin': spin_grid,
        'e0': e0_grid,
    }

    grids = {key: grids_all[key] for key in grid_keys}
    assert len(grids) == 2

    fixed_params_all.pop(grid_keys[0])
    fixed_params_all.pop(grid_keys[1])


    for traj_here, amp_here in zip(traj_module, wf_module):
        
        assert traj_here in ['kerr', 'schwarzschild', 'pn5'], traj_here
        assert amp_here in ['kerr', 'schwarzschild', 'aak'], amp_here
        
        if traj_here == 'schwarzschild':
            fixed_params['spin'] = [0.0]

        outname = base_outname + '_traj_' + traj_here + '_wf_' + amp_here + '_'

        inspiral_func = inspiral_func_all[traj_here]
        traj = EMRIInspiral(func=inspiral_func)
        args = args_all[amp_here]


        best_backend = get_first_backend(FastKerrEccentricEquatorialFlux.supported_backends())
        backend = best_backend if use_gpu else 'cpu'

        sum_kwargs = {
            "force_backend": backend, # GPU is available for this type of summation
            "pad_output": True
        }

        inspiral_kwargs={
                "err": 1e-10,
                "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
                "max_init_len": int(1e4),  # dense stepping trajectories
                "func": inspiral_func
            }
        
        wave_gen = GenerateEMRIWaveform(
            *args,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            return_list=False,
            force_backend=backend,
            frame="detector"
        )
        waveform_kwargs = {
            "T": Tobs,
            "dt": dt,
        }

        #wave_gen = wave_gen_windowed(wave_gen, window_fn=('tukey', 0.005))
        savename = './horizon_data/' + outname + 'T_%.1f' % Tobs
        fixed_params = {}
        
        key1, key2 = fixed_params_keys

        for val1 in fixed_params_all[key1]:
            fixed_params[key1] = val1
            for val2 in fixed_params_all[key2]:
                fixed_params[key2] = val2

                savename = savename + '_%s_%.1e' % (key1, val1) + '_%s_%.1e' % (key2, val2)
                os.makedirs(savename + '_tmp_plots', exist_ok=True)
                savename = savename + '.h5'

                if not os.path.exists(savename):
                    print("Generating data") 
                    try:
                        generate_data(
                            Tobs,
                            dt,
                            fixed_params,
                            grids,
                            savename,
                            traj,
                            wave_gen,
                            snr_thr=snr_thr,
                            emri_kwargs=waveform_kwargs,
                        )

                        fixed_params = {} #reset dictionary
                    except ValueError as e:
                        print(e)
                        print("Error in generating data")
                        print(fixed_params)
                        if os.path.exists(savename):
                            print("Removing file")
                            os.remove(savename)
                else:
                    print("Data already exists, skipping")

        print("done")