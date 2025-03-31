import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

# Need to convert seaborn colors to matplotlib colors
colorblind_palette = sns.color_palette('colorblind')
colorblind_hex = [mcolors.to_hex(color) for color in colorblind_palette]

# General Plot Settings
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# Load the data
dephasings = np.loadtxt('DownsampledFluxesData.txt')
# Plot Historgrams of Dephasings
bins = np.arange(-4,4.5,0.25)

plt.hist(dephasings[:,0], bins=bins, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="1/2 grid points")
plt.hist(dephasings[:,1], bins=bins, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="1/4 grid points")
plt.hist(dephasings[:,2], bins=bins, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="1/8 grid points")

# Compute medians
median1 = np.median(dephasings[:,0])
median2 = np.median(dephasings[:,1])
median3 = np.median(dephasings[:,2])

# Add vertical lines at the medians
plt.axvline(median1, color=colorblind_hex[0], linestyle='dashed', linewidth=1)
plt.axvline(median2, color=colorblind_hex[1], linestyle='dashed', linewidth=1)
plt.axvline(median3, color=colorblind_hex[2], linestyle='dashed', linewidth=1)


plt.xlabel(r'$\log_{10}(\Delta \Phi_{\phi})$')
plt.ylabel("Count")
plt.legend()
plt.savefig("DownsampledFluxesHistogram.pdf")


