import matplotlib.pyplot as plt
import numpy as np
from seaborn import color_palette

# use computer modern font

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# seaborn colorblind palette
cpal = color_palette("colorblind")

# please rasterize scatter plots
plt.scatter(np.random.uniform(size=10), np.random.uniform(size=10), color=cpal[0], rasterized=True)

# save as pdf
plt.savefig("example.pdf")

