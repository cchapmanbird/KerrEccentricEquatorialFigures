import numpy as np
import matplotlib.pyplot as plt
import corner
import os, glob
import warnings
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
plt.rcParams['font.size'] = 20
SNR_data_direc = "../SNR_data/"
file_list = os.listdir()

from seaborn import color_palette
import h5py
from functools import reduce

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
cpal = color_palette("colorblind", 6)


os.chdir(SNR_data_direc + "M1e6_mu1e1/pure_SNRs/")
# # Step 1: Get all relevant filenames
files = glob.glob("*.npy")  # Adjust path as needed

# # Step 2: Extract and sort spin values
# spin_values = sorted(set(float(f.split("_")[-1].replace(".npy", "")) for f in files))

spin_values = np.concatenate([np.round(np.arange(0,1.0,0.1),3),np.array([0.99,0.998])])
# Step 3: Compute SNR ratios
SNR_ratios_M1e6_mu1e1 = []
ordered_spins = []  # To keep track of the order

for spin in spin_values:
    kerr_file = f"SNR_Kerr_vec_{spin}.npy"
    aak_file = f"SNR_AAK_vec_{spin}.npy"
 
    # Ensure both files exist before proceeding
    # print(file_list)
    if kerr_file in files or aak_file in file_list:

        SNR_Kerr = np.load(kerr_file)
        SNR_AAK = np.load(aak_file)

        # Compute ratio (avoid division by zero)
        # ratio = SNR_Kerr / np.where(SNR_AAK != 0, SNR_AAK, np.nan)
        ratio = SNR_AAK/SNR_Kerr

        SNR_ratios_M1e6_mu1e1.append(ratio[0:])
        ordered_spins.append(spin)
# Convert to arrays
SNR_ratios_array_M1e6_mu1e1 = np.array(SNR_ratios_M1e6_mu1e1)
ordered_spins_array = np.array(ordered_spins)
e0_vec = np.arange(0.01, 0.81, 0.01)

# Create the heatmap
fig, ax = plt.subplots(figsize=(7, 5))
cax = ax.imshow(
    SNR_ratios_array_M1e6_mu1e1,
    aspect='auto',
    origin='lower',
    extent=[e0_vec[0], e0_vec[-1], ordered_spins_array[0], ordered_spins_array[-1]],
    cmap='cividis'
)

# Axis labels and title
ax.set_xlabel(r'Eccentricity $e_0$', fontsize=20)
ax.set_ylabel(r'Spin $a$', fontsize=20)
# ax.set_title(r'Heatmap of $\rho_{AAK}/\rho_{Kerr}$', fontsize=20)

# Add colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label(r'${\rho}_{\rm AAK}/{\rho}_{\rm Kerr}$', fontsize=16)

# Add a white tick at value 1.0
# tick_value = 1.0
# cbar.set_ticks(list(cbar.get_ticks()) + [tick_value])
# cbar.ax.yaxis.set_tick_params(color='white', width=2)
# for t in cbar.ax.get_yticklabels():
#     if np.isclose(float(t.get_text()), tick_value):
#         t.set_color('white')
#         t.set_fontweight('bold')

# Optional: Add contour at ratio = 1.0 to separate over/underestimation
# X, Y mesh for contour
E0, SPIN = np.meshgrid(e0_vec, ordered_spins_array)
CS = ax.contour(E0, SPIN, SNR_ratios_array_M1e6_mu1e1, levels=[1.0], colors='white', linewidths=1.5, )#
ax.clabel(CS, CS.levels, fmt="1", fontsize=14)
CS = ax.contour(E0, SPIN, SNR_ratios_array_M1e6_mu1e1, levels=[1.1], colors='white', linewidths=1.5, linestyles='--')
ax.clabel(CS, CS.levels, fmt="1.1", fontsize=14)
CS = ax.contour(E0, SPIN, SNR_ratios_array_M1e6_mu1e1, levels=[0.9], colors='white', linewidths=1.5, linestyles='-.')
ax.clabel(CS, CS.levels, fmt="0.9", fontsize=14)
# ax.contour(E0, SPIN, SNR_ratios_array_M1e6_mu1e1, levels=[0.9], colors='white', linewidths=1.5, linestyles='--')
plt.tight_layout()
plt.savefig("heatmap_SNR_M1e6_mu1e1.png", dpi=300)


