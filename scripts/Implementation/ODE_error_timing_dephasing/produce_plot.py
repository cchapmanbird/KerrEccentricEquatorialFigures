import numpy as np
import matplotlib.pyplot as plt
import h5py
from seaborn import color_palette

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
cpal = color_palette("colorblind", 4)

f = h5py.File("ODEerror_timing_dephasing_data.h5", "r")
Ntest = f.attrs["Ntest"]
mass_ratio_vec = f.attrs["mass_ratio"][:]
err_vec = f.attrs['err_vec'][:]

plt.figure(figsize=(4.5, 7))

lses = ['o-', 'D-', 's-']

for k,mass_ratio in enumerate(mass_ratio_vec):
    lsh = lses[k]
    phase_difference = f[f'mass_ratio_{mass_ratio}']['phase_difference'][:]
    timing = f[f'mass_ratio_{mass_ratio}']['timing'][:]
    N_points = f[f'mass_ratio_{mass_ratio}']['N_points'][:]

    # Plot 1: Mean phase difference
    plt.subplot(3, 1, 1)
    # plt.title(f'Average over {Ntest} random initial conditions')
    median_phase_diff = np.median(np.log10(phase_difference), axis=0)
    sigma_phase_diff = np.std(np.log10(phase_difference), axis=0)

    plt.semilogx(err_vec, median_phase_diff, lsh, c=cpal[k])
    plt.fill_between(err_vec, median_phase_diff - sigma_phase_diff, median_phase_diff + sigma_phase_diff, alpha=0.3,color=cpal[k])
    
    if k == 0:
        plt.axhline(0, color='gray', lw=1., linestyle='--', label='1.0 rad')
        # plt.axhline(-1, color='k', linestyle='-', label='0.1 rad')
        plt.ylabel(r'$\log_{10} \Delta \Phi_\phi$', fontsize=14)
        plt.tick_params(axis='x', labelbottom=False)
        # plt.gca().set_yticks([-5, -4, -3, -2, -1, 0])
        # plt.gca().set_yticklabels([r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        # plt.legend()

    # Plot 2: Mean timing
    plt.subplot(3, 1, 2)
    median_timing = np.median(timing*1000, axis=0)
    sigma_timing = np.std(timing*1000, axis=0)

    plt.semilogx(err_vec, median_timing, lsh, label=rf'$10^{{{int(np.log10(mass_ratio)):d}}}$',c=cpal[k])
    plt.fill_between(err_vec, median_timing - sigma_timing, median_timing + sigma_timing, alpha=0.3,color=cpal[k])

    if k == 2:
        plt.legend(frameon=False)
        plt.ylabel('CPU wall-time [ms]', fontsize=14)
        plt.tick_params(axis='x', labelbottom=False)


    # Plot 3: Mean number of points
    plt.subplot(3, 1, 3)
    median_N_points = np.median(N_points, axis=0)
    sigma_N_points = np.std(N_points, axis=0)
    plt.semilogx(err_vec, median_N_points, lsh, label=f'mass ratio={mass_ratio}',c=cpal[k])
    plt.fill_between(err_vec, median_N_points - sigma_N_points, median_N_points + sigma_N_points, alpha=0.3,color=cpal[k])
    # plt.semilogx(err_vec, N_points, '-o', label=f'mass ratio={mass_ratio}')
    if k == 2:
        plt.xlabel(r'$\sigma_\mathrm{tol}$', fontsize=14)
        plt.ylabel('Trajectory length \n (points)', fontsize=14)

plt.tight_layout()

plt.savefig(f'Trajectory_timing_ODEerror_dephasing.pdf', bbox_inches='tight')
plt.show()