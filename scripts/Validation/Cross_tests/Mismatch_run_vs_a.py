import os
import numpy as np
from matplotlib import pyplot as plt
import h5py
import json
import matplotlib.ticker as ticker


import few
for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
    print(f" - Backend '{backend}': {'available' if few.has_backend(backend) else 'unavailable'}")

breakpoint()
# cfg_set = few.get_config_setter(reset=True)
# # cfg_set.enable_backends("cpu")
# cfg_set.enable_backends("cuda12x", "cpu")
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.fdutils import GetFDWaveformFromFD, GetFDWaveformFromTD
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.constants import YRSID_SI, MTSUN_SI
from few.utils.utility import get_mismatch
import multiprocessing
num_threads =  multiprocessing.cpu_count()


Kerr_ecc_wave = FastKerrEccentricEquatorialFlux()


from bhpwave.waveform import KerrWaveform
from  bhpwave.trajectory.inspiral import InspiralGenerator
from bhpwave.waveform import scaled_amplitude
import multiprocessing
Zach_gen_Kerr = KerrWaveform()
num_threads_BHPWave = multiprocessing.cpu_count()
traj_BHPWave = InspiralGenerator(trajectory_data=None)
print("num_threads_BHPWave:", num_threads_BHPWave, "num_threads:", num_threads)

KerrCirc_wave_path = "/mnt/beegfs/hkhalvati/data_for_KerrEcc_comparison/vs_spin_waveform_full_newmass/"
dir_list = os.listdir(KerrCirc_wave_path)
sorted_files = sorted( [f for f in dir_list if f.startswith("Kerr_wave_")], 
                      key=lambda x: float(x.split("_")[3].split("a")[1]))  

dt = 10.0
e0 = 0.0
x0 = 1.0
dist = 1.0
Phi_phi0 = 0.0


lmax = 10
specific_modes = []
for l in range(2,lmax+1):
    for m in range(0,l+1):
        specific_modes += [(l,m,0)]


lmax = 9
specific_modes_BHPWave = []
for l in range(2,lmax+1):
    for m in range(0,l+1):
        specific_modes_BHPWave += [(l,m)]





output_file = "./Data/mismatch_across_models_full_newmass_lamx9BHPWave_2.json"
if os.path.exists(output_file):
    os.remove(output_file)
    print("old file removed")

mis_data = {}




for ii, f0 in enumerate(sorted_files):
    with h5py.File(KerrCirc_wave_path + f0, "r") as f:
        # print(f.keys())
        # print(f.attrs.keys())
        att_list = list(f.attrs.keys())
        # for att in att_list:
        #     print(att, f.attrs[att])
        wave_KerrCirc_lmax10 = f["Kerr_wave"][:]
        M = f.attrs["M"]
        mu = f.attrs["mu"]
        a0 = f.attrs["a0"]
        p0 = f.attrs["p0"]
        T_obs = f.attrs["T_obs"]
        theta = f.attrs["theta"]
        phi = f.attrs["phi"]
        print("a0:", a0, "p0:", p0, "M:", M, "mu:", mu, "T_obs:", T_obs)
    MM = M+mu
    mumu = M*mu/(M+mu) # for the new mass convention in the KerrEccentric
    zach_scaled_amp = scaled_amplitude(mu, dist)
    Zach_source = zach_scaled_amp * Zach_gen_Kerr.source_frame(MM, mumu, a0, p0, theta, phi, Phi_phi0 , dt=dt, T = T_obs)# ,select_modes=specific_modes_BHPWave,  num_threads=num_threads_BHPWave)
    waveform_KerrEcc = Kerr_ecc_wave(M, mu, a0, p0, e0, x0, theta, phi,dt=dt, T=T_obs, dist = dist, mode_selection=specific_modes)
    waveform_KerrEcc = waveform_KerrEcc.get()
    print("number of modes kept:",Kerr_ecc_wave.num_modes_kept)

    mis_BHPWave_kerrecc = get_mismatch(Zach_source, waveform_KerrEcc, use_gpu=True)
    mis_kerrcir10_kerrecc = get_mismatch(wave_KerrCirc_lmax10, waveform_KerrEcc, use_gpu=True)
    mis_kerrcir10_BHPWave = get_mismatch(wave_KerrCirc_lmax10, Zach_source, use_gpu=True)
    print("mismatch with BHPWave:", mis_BHPWave_kerrecc)
    print("mismatch with KerrCirc l=10:", mis_kerrcir10_kerrecc,'\n')
    mis_data[f"a0:{a0}"] = { "a0": float(a0),
        "p0": float(p0),
        "M": float(M),
        "mu": float(mu),
        "T_obs": float(T_obs),
        "num_modes_kept": Kerr_ecc_wave.num_modes_kept,
        "mismatch_BHPWave_vs_KerrEcc": float(mis_BHPWave_kerrecc),
        "mismatch_KerrCirc_l10_vs_KerrEcc": float(mis_kerrcir10_kerrecc),
        "mismatch_KerrCirc_l10_vs_BHPWave": float(mis_kerrcir10_BHPWave),
    }
    with open(output_file, "w") as outfile:
        json.dump(mis_data, outfile, indent=4)
    print("mismatch data saved to:", output_file)
