#!/usr/bin/env python

"""
Script for plotting flux timing results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from few.utils.utility import get_separatrix
# from seaborn import color_palette

# use computer modern font

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# # seaborn colorblind palette
# cpal = color_palette("colorblind")

"""
WARNING:    Plot generation requires access to the raw data grids which are stored
            in feather files. So you need pyarrow and pandas also installed. For now
            reach out to @znasipak for access to the files
"""
data_path = "/Users/znasipak/Library/CloudStorage/OneDrive-UniversityofSouthampton/FluxData/data"

# Load Inner Data
data_info = os.path.join(data_path, 'flux_grid_params_inner_v3.feather')
df_info = pd.read_feather(data_info)

info_ival = 4
data_dir = os.path.join(data_path, df_info['filename'][info_ival])
data = pd.read_feather(data_dir)

NU = np.int64(df_info['nu'][info_ival])
NW = np.int64(df_info['nw'][info_ival])
NZ = np.int64(df_info['nz'][info_ival])

timing_data_full = data["cost"].to_numpy().reshape(NU, NW, NZ)
p_data_full = data["p0"].to_numpy().reshape(NU, NW, NZ)
e_data_full = data["e0"].to_numpy().reshape(NU, NW, NZ)

# Load Outer Data
data_info_outer = os.path.join(data_path, 'flux_grid_params_outer.feather')
df_info_outer = pd.read_feather(data_info_outer)

data_dir_outer = os.path.join(data_path, df_info_outer['filename'][0])
data_outer = pd.read_feather(data_dir_outer)

NU_outer = np.int64(df_info_outer['nu'][0])
NW_outer = np.int64(df_info_outer['nw'][0])
NZ_outer = np.int64(df_info_outer['nz'][0])

timing_data_outer_full = data_outer["cost"].to_numpy().reshape(NU_outer, NW_outer, NZ_outer)
p_data_outer_full = data_outer["p0"].to_numpy().reshape(NU_outer, NW_outer, NZ_outer)
e_data_outer_full = data_outer["e0"].to_numpy().reshape(NU_outer, NW_outer, NZ_outer)

# Get max timing
time_max = timing_data_full.max()

# Choose index of a slices to plot
a_iter_list = [0, 18, 42, NZ-1] # must be even!
fig_num = len(a_iter_list)

# Choose aspect ratio for each subfigure
aspect_ratio = 2/3

# Create subplots
fig, ax = plt.subplots(fig_num, 1, figsize=(6, 6*fig_num*aspect_ratio))

# Iterate over different slices
for i, a_iter in enumerate(a_iter_list):
    # Inner grid data
    timing_data = timing_data_full[:,:,a_iter]/60 # convert from secs to mins
    timing_data[timing_data == 0.] = 0.5 # I didn't store circular orbit timing because it is so fast to evaluate. Make it nonzero
    
    # Get the value of a
    aval = data['a0'].to_numpy().reshape(NU, NW, NZ)[0,0,a_iter] 
    # Take this slice of the data with constant aval
    p_data = p_data_full[:,:,a_iter]
    e_data = e_data_full[:,:,a_iter]
    # Calculate LSO for this slice
    pLSO_data = get_separatrix(aval, e_data.flatten(), 1).reshape(e_data.shape)

    # Outer grid data
    a_iter_outer = a_iter // 2 # Outer grid has half the resolution in a
    timing_data_outer = timing_data_outer_full[:,:,a_iter_outer]/60
    timing_data_outer[timing_data_outer == 0.] = 0.5
    
    # double check that we are taking the same aval slice for the outer and inner grids
    aval_outer = data_outer["a0"].to_numpy().reshape(NU_outer, NW_outer, NZ_outer)[0,0,a_iter_outer]
    if aval_outer != aval:
        raise ValueError("Mismatch in aval")
    
    # Take this slice of the outer data with constant aval
    p_data_outer = p_data_outer_full[:,:,a_iter_outer]
    e_data_outer = e_data_outer_full[:,:,a_iter_outer]
    pLSO_data_outer = get_separatrix(aval, e_data_outer.flatten(), 1).reshape(e_data_outer.shape)
    
    # Generate scatter plots
    ax[i].scatter(p_data_outer - pLSO_data_outer, e_data_outer, c=timing_data_outer, norm=LogNorm(), vmax=time_max/60, rasterized=True)
    pos = ax[i].scatter(p_data - pLSO_data, e_data, c=timing_data, norm=LogNorm(), vmax=time_max/60, rasterized=True)
    # Set labels and ticks
    ax[i].set_title('$a={:.4f}$'.format(aval))
    if i == fig_num - 1:
        ax[i].set_xlabel('$p_0 - p_\\mathrm{LSO}$')
    ax[i].set_ylabel('$e_0$')
    ax[i].set_xscale('log')
    if i < fig_num - 1: # Only include x tick labels for bottom plot
        ax[i].tick_params(labelbottom=False)  

# Add space for colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(pos, cax=cbar_ax)
cbar.set_label('Time (minutes)')

# Show/save plot
# plt.show()
plt.savefig('timing.pdf', dpi=100)