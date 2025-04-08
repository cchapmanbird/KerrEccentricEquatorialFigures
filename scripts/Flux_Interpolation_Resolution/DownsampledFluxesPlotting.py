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


# Plot Historgrams of Dephasings when downsampling by 2, 4, and 8
bins1 = np.arange(-5,3.5,0.25)

plt.hist(dephasings[:,0], bins=bins1, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="1/2 grid points")
plt.hist(dephasings[:,1], bins=bins1, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="1/4 grid points")
plt.hist(dephasings[:,2], bins=bins1, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="1/8 grid points")

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
plt.savefig("DownsampledFluxesHistogram1.pdf")

# Plot Historgrams of Dephasings when downsampling in u, w, and z
bins2 = np.arange(-6,1,0.25)

plt.hist(dephasings[:,3], bins=bins2, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="downsample in $u$")
plt.hist(dephasings[:,4], bins=bins2, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="downsample in $w$")
plt.hist(dephasings[:,5], bins=bins2, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="downsample in $z$")

# Compute medians
median1 = np.median(dephasings[:,3])
median2 = np.median(dephasings[:,4])
median3 = np.median(dephasings[:,5])

# Add vertical lines at the medians
plt.axvline(median1, color=colorblind_hex[0], linestyle='dashed', linewidth=1)
plt.axvline(median2, color=colorblind_hex[1], linestyle='dashed', linewidth=1)
plt.axvline(median3, color=colorblind_hex[2], linestyle='dashed', linewidth=1)


plt.xlabel(r'$\log_{10}(\Delta \Phi_{\phi})$')
plt.ylabel("Count")
#plt.title("Dephasing after 4 years as a function of downsampling factor")
plt.legend()
plt.savefig("DownsampledFluxesHistogram2.pdf")

# Plot both histograms together
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create 1x2 subplots
ax1 = axs[0]
ax2 = axs[1]

ax1.hist(dephasings[:,0], bins=bins1, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="downsample by $2$")
ax1.hist(dephasings[:,1], bins=bins1, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="downsample by $4$")
ax1.hist(dephasings[:,2], bins=bins1, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="downsample by $8$")

# Compute medians
median1 = np.median(dephasings[:,0])
median2 = np.median(dephasings[:,1])
median3 = np.median(dephasings[:,2])

# Add vertical lines at the medians
ax1.axvline(median1, color=colorblind_hex[0], linestyle='dashed', linewidth=1.5)
ax1.axvline(median2, color=colorblind_hex[1], linestyle='dashed', linewidth=1.5)
ax1.axvline(median3, color=colorblind_hex[2], linestyle='dashed', linewidth=1.5)


ax1.set_xlabel(r'$\log_{10}(\Delta \Phi_{\phi})$')
ax1.set_ylabel("Count")
#plt.title("Dephasing after 4 years as a function of downsampling factor")
ax1.legend()

bins2 = np.arange(-6,1,0.25)

ax2.hist(dephasings[:,3], bins=bins2, histtype='stepfilled', facecolor='none', edgecolor=colorblind_hex[0], linewidth=2, label="$1/2$ points in $u$")
ax2.hist(dephasings[:,4], bins=bins2, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[1], linewidth=2, label="$1/2$ points in $w$")
ax2.hist(dephasings[:,5], bins=bins2, histtype='stepfilled',facecolor='none', edgecolor=colorblind_hex[2], linewidth=2, label="$1/2$ points in $z$")

# Compute medians
median1 = np.median(dephasings[:,3])
median2 = np.median(dephasings[:,4])
median3 = np.median(dephasings[:,5])

# Add vertical lines at the medians
ax2.axvline(median1, color=colorblind_hex[0], linestyle='dashed', linewidth=1.5)
ax2.axvline(median2, color=colorblind_hex[1], linestyle='dashed', linewidth=1.5)
ax2.axvline(median3, color=colorblind_hex[2], linestyle='dashed', linewidth=1.5)


ax2.set_xlabel(r'$\log_{10}(\Delta \Phi_{\phi})$')
ax2.set_ylabel("Count")
ax2.legend()


plt.tight_layout()
plt.savefig('DownsampledFluxesHistogram2.pdf')





