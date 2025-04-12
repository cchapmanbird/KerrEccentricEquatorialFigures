# import os
# # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import multiprocessing
num_threads =  multiprocessing.cpu_count()
print(f"Number of threads: {num_threads}")
import few
for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
    print(f" - Backend '{backend}': {"available" if few.has_backend(backend) else "unavailable"}")









import numpy as np
from matplotlib import pyplot as plt

import few
# cfg_set = few.get_config_setter(reset=True)
# # cfg_set.enable_backends("cpu")
# cfg_set.enable_backends("cuda12x", "cpu")
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.fdutils import GetFDWaveformFromFD, GetFDWaveformFromTD
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.constants import YRSID_SI, MTSUN_SI
from few.utils.utility import get_mismatch



Kerr_ecc_wave = FastKerrEccentricEquatorialFlux()


from bhpwave.waveform import KerrWaveform
from  bhpwave.trajectory.inspiral import InspiralGenerator
from bhpwave.waveform import scaled_amplitude
import multiprocessing
Zach_gen_Kerr = KerrWaveform()
num_threads_BHPWave = multiprocessing.cpu_count()
traj_BHPWave = InspiralGenerator(trajectory_data=None)
print("num_threads_BHPWave:", num_threads_BHPWave, "num_threads:", num_threads)







####### Loading the saved data from KerrCir version of FEW ---> https://github.com/Hassankh92/FastEMRIWaveforms_KerrCircNonvac
###### Download the data from :
#  https://perimeter-my.sharepoint.com/:f:/g/personal/hkhalvati_perimeterinstitute_ca/EpClG00fwZVNsch7WU2OrGQBdtY8cctLCEdKmemo2S7fZw?e=KbP3cX

KerrCirc_wave_path = "/mnt/beegfs/hkhalvati/data_for_KerrEcc_comparison/"
import h5py
import os

# file_list = os.listdir(KerrCirc_wave_path)

file_list = ["Kerr_wave_l10_a0.99_p6.858234_T1_dt2.h5", "Kerr_wave_l10_a0.99_p10.031492_T4_dt2.h5"]

for file1 in file_list:
    with h5py.File(KerrCirc_wave_path + file1, "r") as f:
        print(f.keys())
        print(f.attrs.keys())
        att_list = list(f.attrs.keys())
        for att in att_list:
            print(att, f.attrs[att])
        wave_KerrCirc_lmax10 = f["Kerr_wave"][:]
        M = f.attrs["M"]
        mu = f.attrs["mu"]
        a0 = f.attrs["a0"]
        p0 = f.attrs["p0"]
        T_obs = f.attrs["T_obs"]
        theta = f.attrs["theta"]
        phi = f.attrs["phi"]

    dt = 2.0
    e0 = 0.0
    x0 = 1.0
    dist = 1.0
    Phi_phi0 = 0.0


    zach_scaled_amp = scaled_amplitude(mu, dist)
    Zach_source = zach_scaled_amp * Zach_gen_Kerr.source_frame(M, mu, a0, p0, theta, phi, Phi_phi0 , dt=dt, T = T_obs)# ,mode_selection=specific_modes,  num_threads=num_threads_BHPWave)
    waveform_KerrEcc = Kerr_ecc_wave(M, mu, a0, p0, e0, x0, theta, phi,dt=dt, T=T_obs, dist = dist,eps = 1e-16)
    waveform_KerrEcc = waveform_KerrEcc.get()
    print("number of modes kept:",Kerr_ecc_wave.num_modes_kept)

    mis_BHPWave_kerrecc = get_mismatch(Zach_source, waveform_KerrEcc, use_gpu=True)
    mis_kerrcir10_kerrecc = get_mismatch(wave_KerrCirc_lmax10, waveform_KerrEcc, use_gpu=True)
    print("mismatch with BHPWave:", mis_BHPWave_kerrecc)
    print("mismatch with KerrCirc l=10:", mis_kerrcir10_kerrecc)

    t_arr = np.arange(len(waveform_KerrEcc.real))*dt # in seconds
    print(f"len of waveforms: KerrEcc: {len(waveform_KerrEcc)}, KerrCirc: {len(wave_KerrCirc_lmax10)}, BHPWave: {len(Zach_source)}")

    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14
    text_fontsize = 12

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    # First subplot
    axes[0].plot(t_arr[:8000], waveform_KerrEcc[:8000].real, '-k', label='KerrEccEq, a = 0.99', rasterized=True)
    axes[0].plot(t_arr[:8000], wave_KerrCirc_lmax10[:8000].real, '--r', label='KerrCircEq, a = 0.99, lmax=10', rasterized=True)
    axes[0].plot(t_arr[:8000], Zach_source[:8000].real, ':g', label='BHPWAVE, a = 0.99', rasterized=True)
    axes[0].set_xlabel('t [s]', fontsize=label_fontsize)
    axes[0].set_ylabel(r'$h_{\plus}$', fontsize=label_fontsize)
    axes[0].set_ylim(-6e-22, 6e-22)
    axes[0].legend(fontsize=legend_fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[0].grid(True)
    # Second subplot
    axes[1].plot(t_arr[-500:], waveform_KerrEcc[-500:].real, '-k', label='Relativistic circular Kerr', rasterized=True)
    axes[1].plot(t_arr[-500:], wave_KerrCirc_lmax10[-500:].real, '--r', label='KerrCircEq, lmax=10', rasterized=True)
    axes[1].plot(t_arr[-500:], Zach_source[-500:].real, ':g', label='BHPWAVE', rasterized=True)
    axes[1].set_xlabel('t [s]', fontsize=label_fontsize)
    axes[1].set_ylim(-6e-22, 6e-22)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    axes[1].grid(True)
    textstr = f"mismatch with BHPWave: ${mis_BHPWave_kerrecc:.2e}$"
    textstr += f"\nmismatch with KerrCirc lmax=10: ${mis_kerrcir10_kerrecc:.2e}$"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.05, 0.05, textstr, transform=axes[0].transAxes, fontsize=text_fontsize,
                verticalalignment='bottom', bbox=props)
    fig.suptitle(f"Waveforms comparison for {T_obs:.1f} year long inspiral", fontsize=title_fontsize, y=0.94)
    plt.tight_layout()
    plt.savefig(f"waveforms_comparison_Tobs{T_obs}.pdf")









# ####### Loading the saved data from KerrCir version of FEW ---> https://github.com/Hassankh92/FastEMRIWaveforms_KerrCircNonvac
# ###### Download the data from :
# #  https://perimeter-my.sharepoint.com/:f:/g/personal/hkhalvati_perimeterinstitute_ca/EpClG00fwZVNsch7WU2OrGQBdtY8cctLCEdKmemo2S7fZw?e=KbP3cX

# KerrCirc_wave_path = "/mnt/beegfs/hkhalvati/data_for_KerrEcc_comparison/"
# import h5py
# import os

# file1 = "Kerr_wave_l10_a0.99_p10.031492_T4_dt2.h5"   
# with h5py.File(KerrCirc_wave_path + file1, "r") as f:
#     print(f.keys())
#     print(f.attrs.keys())
#     att_list = list(f.attrs.keys())
#     for att in att_list:
#         print(att, f.attrs[att])
#     wave_KerrCirc_lmax10 = f["Kerr_wave"][:]
#     M = f.attrs["M"]
#     mu = f.attrs["mu"]
#     a0 = f.attrs["a0"]
#     p0 = f.attrs["p0"]
#     T_obs = f.attrs["T_obs"]
#     theta = f.attrs["theta"]
#     phi = f.attrs["phi"]

# dt = 2.0
# e0 = 0.0
# x0 = 1.0
# dist = 1.0
# Phi_phi0 = 0.0






# zach_scaled_amp = scaled_amplitude(mu, dist)
# Zach_source = zach_scaled_amp * Zach_gen_Kerr.source_frame(M, mu, a0, p0, theta, phi, Phi_phi0 , dt=dt, T = T_obs)# ,mode_selection=specific_modes,  num_threads=num_threads_BHPWave)



# waveform_KerrEcc = Kerr_ecc_wave(M, mu, a0, p0, e0, x0, theta, phi,dt=dt, T=T_obs, dist = dist,eps = 1e-16)
# waveform_KerrEcc = waveform_KerrEcc.get()
# print("number of modes kept:",Kerr_ecc_wave.num_modes_kept)


# mis_BHPWave_kerrecc = get_mismatch(Zach_source, waveform_KerrEcc, use_gpu=True)
# mis_kerrcir10_kerrecc = get_mismatch(wave_KerrCirc_lmax10, waveform_KerrEcc, use_gpu=True)
# print("mismatch with BHPWave:", mis_BHPWave_kerrecc)
# print("mismatch with KerrCirc l=10:", mis_kerrcir10_kerrecc)


# t_arr = np.arange(len(waveform_KerrEcc.real))*dt # in seconds
# len(waveform_KerrEcc), len(wave_KerrCirc_lmax10), len(Zach_source)




# title_fontsize = 20
# label_fontsize = 16
# tick_fontsize = 14
# legend_fontsize = 14
# text_fontsize = 12


# fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# # First subplot
# axes[0].plot(t_arr[:8000], waveform_KerrEcc[:8000].real, '-k', label='KerrEccEq, a = 0.99',rasterized=True)
# axes[0].plot(t_arr[:8000], wave_KerrCirc_lmax10[:8000].real, '--r', label='KerrCircEq, a = 0.99, lmax=10',rasterized=True)
# axes[0].plot(t_arr[:8000], Zach_source[:8000].real, ':g', label='BHPWAVE, a = 0.99',rasterized=True)
# # axes[0].plot(t_arr[:8000], wave_KerrCirc_lmax15[:8000].real, '-g', label='KerrCircEq, a = 0.99, lmax=15')
# axes[0].set_xlabel('t [s]', fontsize=label_fontsize)
# axes[0].set_ylabel(r'$h_{\plus}$', fontsize=label_fontsize)
# axes[0].set_ylim(-6e-22, 6e-22)
# # axes[0].set_title("Beginning of wave", fontsize=title_fontsize)
# axes[0].legend(fontsize=legend_fontsize)
# axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axes[0].grid(True)

# # Second subplot
# axes[1].plot(t_arr[-500:], waveform_KerrEcc[-500:].real, '-k', label='Relativistic circular Kerr ',rasterized=True)
# axes[1].plot(t_arr[-500:], wave_KerrCirc_lmax10[-500:].real, '--r', label='KerrCircEq, lmax=10',rasterized=True)
# axes[1].plot(t_arr[-500:], Zach_source[-500:].real, ':g', label='BHPWAVE',rasterized=True)
# # axes[1].plot(t_arr[-500:], wave_KerrCirc_lmax15[-500:].real, '-g', label='KerrCircEq, lmax=15')
# axes[1].set_xlabel('t [s]', fontsize=label_fontsize)
# # axes[1].set_ylabel(r'$h_{\plus}$', fontsize=label_fontsize)
# axes[1].set_ylim(-6e-22, 6e-22)
# # axes[1].set_title("End of wave", fontsize=title_fontsize)
# # axes[1].legend(fontsize=legend_fontsize)
# axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axes[1].tick_params(axis='y', left=False, labelleft=False)

# axes[1].grid(True)

# textstr = f"mismatch with BHPWave: ${mis_BHPWave_kerrecc:.2e}$"
# textstr += f"\nmismatch with KerrCirc lmax=10: ${mis_kerrcir10_kerrecc:.2e}$"
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# axes[0].text(0.05, 0.05, textstr, transform=axes[0].transAxes, fontsize=text_fontsize,
#              verticalalignment='bottom', bbox=props)

# fig.suptitle(f"Waveforms comparison for {T_obs:.1f} year long inspiral", fontsize=title_fontsize, y=0.94)


# plt.tight_layout()

# plt.savefig(f"waveforms_comparison_Tobs{T_obs}.pdf")
