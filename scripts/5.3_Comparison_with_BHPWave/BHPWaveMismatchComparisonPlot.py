import matplotlib.pyplot as plt
import numpy as np
from seaborn import color_palette

spins = np.loadtxt("BHPWaveMismatchComparisonSpins.txt")
mms = np.loadtxt("BHPWaveMismatchComparison.txt")


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# seaborn colorblind palette
cpal = color_palette("colorblind")

plt.plot(spins,mms,color=cpal[0],rasterized=True)
plt.xlabel(r'Primary Spin $(a)$')
plt.ylabel(r'Mismatch $\mathcal{M}$')
#plt.yscale('log')
plt.grid(True)
plt.savefig("5-3a-ComparisonWithBHPWave.pdf")