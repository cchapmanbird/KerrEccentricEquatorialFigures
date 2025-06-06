import numpy as np
import matplotlib.pyplot as plt
import few
from scipy.interpolate import CubicSpline
from seaborn import color_palette
from few.utils.geodesic import get_separatrix

cpal = color_palette('colorblind')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams['font.size'] = 12

data = np.loadtxt(few.get_file_manager().get_file("LPA.txt"), skiprows=1)
data[:, 1] = data[:, 1] ** 2
get_sensitivity = CubicSpline(*data.T)

def mismatch(a, b):
    a_f = np.fft.rfft(a, axis=0)[1:]
    b_f = np.fft.rfft(b, axis=0)[1:]

    frs = np.fft.rfftfreq(a.shape[0], d=5)[1:]
    df = frs[1] - frs[0]
    psdh = get_sensitivity(frs)
    
    a_b = ((a_f.conj() * b_f).real / psdh[:,None]).sum()
    a_a = ((a_f.conj() * a_f).real / psdh[:,None]).sum()
    b_b = ((b_f.conj() * b_f).real / psdh[:,None]).sum()
    return 1 - a_b / np.sqrt(a_a * b_b)


def wave_comp_plot(fp, fpt, a):
    waves = np.load(fp)
    tms = np.arange(waves.shape[1])*5
    scott_wave = waves[:2].copy().T
    our_wave = waves[2:4].copy().T
    our_wave_tric = waves[4:6].copy().T
    
    trajectory = np.loadtxt(fpt).T

    traj_seps = get_separatrix(a, trajectory[2], trajectory[3])

    misms_logsp_p = []
    lengths_logsp_p = (trajectory[0][5:] / 5).astype(np.int32)
    for lens in lengths_logsp_p:   
        lens = int(lens)
        misms_logsp_p.append(mismatch(scott_wave[:lens], our_wave[:lens]))

    misms_logsp_tric = []
    lengths_logsp_p_tric = (trajectory[0][5:] / 5).astype(np.int32)
    for lens in lengths_logsp_p_tric:   
        lens = int(lens)
        misms_logsp_tric.append(mismatch(scott_wave[:lens], our_wave_tric[:lens]))

    fig = plt.figure(figsize=(5.5,5.5), dpi=150)
    plt.subplot(221)
    plt.plot(tms, waves[0], label='SAH', c=cpal[0])
    plt.plot(tms, waves[2], label='FEW', c=cpal[1], ls='--')
    plt.xlim(0, 9999)
    plt.xlabel('Time [s]')
    plt.ylabel(r'$h_+$ (source frame)')
    plt.legend(loc='upper center', ncols=2,frameon=False)

    plt.subplot(222)
    plt.plot(tms - 6.309e7, waves[0], label='SAH', c=cpal[0])
    plt.plot(tms - 6.309e7, waves[2], label='FEW', c=cpal[1], ls='--')
    # plt.xlim(6.309e7, 6.3091e7)
    plt.xlim(0, 999)
    plt.xlabel(r'Time$\,-\,6.309\times10^{7}$ [s]')
    plt.tick_params(axis='y', labelleft=False)

    plt.subplot(223)
    plt.semilogy((trajectory[0])[5:] / 3.155e7, misms_logsp_p, c=cpal[2], label='FEW (Bic+Li)')
    plt.semilogy((trajectory[0])[5:] / 3.155e7, misms_logsp_tric, c=cpal[3], label='FEW (Tric)')
    plt.xlabel(r'Time [y]')
    plt.ylabel(r'$\mathcal{M}$')
    plt.legend(loc='upper left', frameon=False,)

    plt.subplot(224)
    plt.loglog((trajectory[1] - traj_seps)[5:], misms_logsp_p, c=cpal[2])
    plt.loglog((trajectory[1] - traj_seps)[5:], misms_logsp_tric, c=cpal[3])
    plt.xlabel(r'$p - p_\mathrm{sep}$')
    plt.tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(fp[:-4]+'_mismatch.pdf', bbox_inches='tight')

# a0.998 first
wave_comp_plot('./a0.998waves.npy', './trajectory_1_a0.998.txt', 0.998)

# # a0.5 second
# wave_comp_plot('./a0.5waves.npy', './trajectory_2_a0.5.txt', 0.5)
