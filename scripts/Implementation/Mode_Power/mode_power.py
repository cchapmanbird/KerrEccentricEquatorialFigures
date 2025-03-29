import numpy as np
import os 
import sys
sys.path.append("../")


# Import relevant EMRI packages
from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc 

from few.utils.utility import get_separatrix 
from few.utils.ylm import GetYlms

import matplotlib.pyplot as plt

np.random.seed(1234)

amp_module = AmpInterpKerrEqEcc()  # Amplitue module (KerrEccEq)
ylm_gen = GetYlms(assume_positive_m=True, use_gpu=False) # Spin weighted spherical harmonics

# Current code projects the teukolsky coefficients onto spherical harmonic basis. No need for
# spin weighted spheroidal harmonics. 

theta, phi = np.pi/4, np.pi/4  # Viewing angles

m0mask = amp_module.m0mask # 
# maximum number of lmodes and nmodes. 
# Symmetry in m modes, no symmetry in n modes (unless circular, trivial)
l_max = 10 #amp_module.lmax
n_max = amp_module.nmax

# Define parameters 
a = 0.99 # Spin parameter 
e0 = 0.7 # Eccentricity
p_sep = get_separatrix(a,e0,1.0)

# We want to be strong field. This is where we really see the harmonics shine. 
p0 = p_sep + 0.1

# Extract Teukolsky modes for all available modes. ordered (l,m,n). No k modes, zero inclination.
Amplitudes_all_lmn = amp_module.get_amplitudes(a, np.asarray([p0]), np.asarray([e0]), np.asarray([1.]))

# Spin weighted spheroidal harmonics
ylms = ylm_gen(amp_module.unique_l, amp_module.unique_m, theta, phi).copy()[amp_module.inverse_lm]

# Build true |H_lmn|^{2} 
power = (np.abs(np.concatenate([Amplitudes_all_lmn, np.conj(Amplitudes_all_lmn[:, m0mask])], axis=1)* ylms)** 2)

# Total power of the harmonics summed over all (l,m,n)
power_tot = sum(power[0])

# power_tot = 0.270012
print("For a = {}, e0 = {}, p0 = {}, total power = {}".format(a,e0,p0,np.round(power_tot,6)))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Parameters

n_harmonic_range = np.arange(-(n_max), n_max, 1) # n modes

# Placeholder for the sum over l and for fixed m and n. 

def power_sum_over_l_fixed_mn_fun(l_max):
    """
    args: l_max, integer
    outputs: An array for each m = 0, 1, ..., lmax indexed by 
             negative n indices and positive n indices. 

    Shape [max_m_modes, 2 * n_modes]

    """
    power_compare_sum_over_l_fixed_mn = []    
    m_values = np.arange(0, l_max + 1, 1)  # m modes 
    for m_choice in m_values: 
        # Fix m and n and sum over l >= m_modes until lmax is reached
        # this is done for EVERY choice of m and l. 
        power_compare_sum_over_l_fixed_mn.append(
            np.array([np.sum([(abs(Amplitudes_all_lmn[:, amp_module.special_index_map[(l_index, m_choice, n)]] * 
                                ylm_gen(np.array([l_index]), np.array([m_choice]), theta, phi)[0])**2)/power_tot 
                            for l_index in range(2, l_max + 1) if m_choice <= l_index]) 
                    for n in range(-n_max, n_max)])
        )    
    # append to list
    power_compare_sum_over_l_fixed_mn = np.array(power_compare_sum_over_l_fixed_mn)
    return power_compare_sum_over_l_fixed_mn 


m_values = np.arange(0, l_max + 1, 1)  # m modes 
power_compare_sum_over_l_fixed_mn = power_sum_over_l_fixed_mn_fun(l_max)
# Plotting
fig, ax = plt.subplots(figsize=(12, 10))

# Normalize the power values for colormap using LogNorm
norm = LogNorm(vmin=1e-10, vmax=1.0)

for i, m_choice in enumerate(m_values):
    power_values = power_compare_sum_over_l_fixed_mn[i, :]
    # Set a threshold, similarly done in the FEW paper
    threshold = 1e-10
    # Create a mask for values above the threshold
    mask = power_values >= threshold
    # Create an array where values below the threshold are set to a very small value
    filtered_powers = np.where(mask, power_values, threshold)
    # Plot vertical bars
    bars = ax.bar(x=[m_choice] * len(n_harmonic_range), height=[1] * len(n_harmonic_range), bottom=n_harmonic_range - 0.5, width=0.8, color=plt.cm.viridis(norm(filtered_powers)))

    # Make bars with power below the threshold invisible by setting their height to zero
    # makes the plots less clutterful  
    for j, bar in enumerate(bars):
        if not mask[j]:
            bar.set_height(0)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$\sum_{n}\sum_{l=2}^{10}|H_{lmn}|^2 \left/ \right.\sum_{lmn}|H_{lmn}|^2$', fontsize = 20)
ax.set_ylabel('n index', fontsize=20)
ax.set_xlabel('m index', fontsize=20)
ax.set_title(f'Parameters: (a,p0,e0) = ({a}, {np.round(p0,4)}, {e0})', fontsize=20)
ax.set_xticks(m_values)
ax.set_xticklabels([f'$m = {m}$' for m in m_values])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig("plots/lmax_{}_mode_bar_chart_sum_over_l_mn_a_{}_e0_{}.pdf".format(l_max,a,e0),bbox_inches="tight")
plt.clf()
quit()
plt.show()

print("Now going to check for convergence")

# Here we indicate the truth number of modes 
# and test against an approximate number of modes
l_truth = 10
l_approx = 4

power_compare_sum_over_l_10_mn = power_sum_over_l_fixed_mn_fun(l_truth)
power_compare_sum_over_l_approx_mn = power_sum_over_l_fixed_mn_fun(l_approx)

length_approx = power_compare_sum_over_l_approx_mn.shape[0]
m_values = np.arange(0, length_approx, 1)

# Compute relative difference
# Notice that these are normalised by P_tot, over lmax = 10 modes. 

rel_difference = abs(power_compare_sum_over_l_10_mn[0:length_approx,:] - power_compare_sum_over_l_approx_mn)

# Normalize the power values for colormap using LogNorm
threshold = 1e-10
# Ignore differences < 1e-10
norm = LogNorm(vmin=threshold, vmax=1.0)
fig, ax = plt.subplots(figsize=(12, 10))
for i, m_choice in enumerate(m_values):
    diff_values = rel_difference[i, :]

    # Create a mask for values above the threshold
    mask = diff_values >= threshold
    # Create an array where values below the threshold are set to a very small value (or any value that won't affect the log scale)
    filtered_diff_values = np.where(mask, diff_values, threshold)
    # Plot vertical bars
    bars = ax.bar(x=[m_choice] * len(n_harmonic_range), height=[1] * len(n_harmonic_range), bottom=n_harmonic_range - 0.5, width=0.8, color=plt.cm.viridis(norm(filtered_diff_values)))

    # Make bars with power below the threshold invisible by setting their height to zero
    for j, bar in enumerate(bars):
        if not mask[j]:
            bar.set_height(0)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar_string = r'$\sum_{{mn}}\left(\sum_{{l=1}}^{{10}}|H_{{lmn}}|^2 -\sum_{{l=1}}^{}|H_{{lmn}}|^2\right)$'.format(l_approx)
cbar.set_label(cbar_string, fontsize=20)

ax.set_ylabel('n index', fontsize=20)
ax.set_xlabel('m index', fontsize=20)
ax.set_title(f'Parameters: (a,p0,e0) = ({a}, {np.round(p0,4)}, {e0})', fontsize=20)
ax.set_xticks(m_values)
ax.set_xticklabels([f'$m = {m}$' for m in m_values])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig("KerrEquatorialCodes/ModeChecking/Plots/DiffPlots/diff_{}_{}_mode_bar_chart_sum_over_l_mn_a_{}_e0_{}.pdf".format(l_truth,l_approx,a,e0),bbox_inches="tight")
plt.show()

quit()
