# %%
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from pandas import DataFrame as df

import json
import corner


label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %%
fname = 'lakshmi_timing_4.0yrInspErrDefault.json'
timing_data = json.load(open(fname, 'r'))
fname = "results/" + fname[:-5]
# %%
def cast_results_to_dataframe(input_data):
    output = []

    key_list = list(input_data[0]['parameters'].keys())
    key_list.append('duration')
    key_list.append('iterations')
    key_list.append('dt')
    key_list.append('mode_selection_threshold')
    key_list.append('fd_timing')
    key_list.append('td_timing')
    key_list.append('overlap')
    key_list.append('power_fd')
    key_list.append('power_td')

    for single in input_data:
        _output_list = list(single['parameters'].values())
        print(single['duration'])
        _output_list.append(single['duration'])
        _output_list.append(single['iterations'])
        for data in single['timing_results']:
            temp = _output_list.copy()
            temp.append(data['dt'])
            temp.append(data['mode_selection_threshold'])
            temp.append(data['fd_timing'])
            temp.append(data['td_timing'])
            temp.append(data['overlap'])
            temp.append(data['power_fd'])
            temp.append(data['power_td'])
            output.append(temp.copy())
        
    return df(output, columns=key_list), key_list

# %%
def corner_plot(dataframe, minmax=None, use_td=True, plot_type='timing', eps_value=1e-2, dt_value=10.0):
    eps_range = np.unique(dataframe['mode_selection_threshold'])
    dt_range = np.unique(dataframe['dt'])
    assert eps_value in eps_range
    assert dt_value in dt_range

    data_given_eps = dataframe[(dataframe['mode_selection_threshold'] == eps_value)&(dataframe['dt'] == dt_value)]
    # obtain keys from dataframe
    params = list(data_given_eps.keys())

    if plot_type == 'timing':
        timing_values = (data_given_eps['td_timing' if use_td else 'fd_timing'])
    if plot_type == 'ratio':
        timing_values = (data_given_eps['fd_timing']) / (data_given_eps['td_timing'])
    if plot_type == 'overlap':
        timing_values = np.log10(np.abs(1-data_given_eps['overlap']))
    if plot_type == 'power':
        timing_values = np.log10(np.abs(1-data_given_eps['power_td']))
    # remove outliers outside 99.9 percentile
    mask = (timing_values > np.percentile(timing_values, 0.1)) & (timing_values < np.percentile(timing_values, 99.9))
    timing_values = timing_values[mask]
    # remove outliers outside 99.9 percentile
    data_given_eps = data_given_eps[mask]
    
    if not minmax:
        vmin = min(timing_values)
        vmax = max(timing_values)
    else:
        vmin, vmax = minmax
    
    # transform data_given_eps to a dictionary
    data_given_eps = {param: data_given_eps[param].values for param in params}
    # create new keys like log10_mass_1, log10_mass_ratio, 
    data_given_eps['log10_mass_1'] = np.log10(data_given_eps['mass_1'])
    data_given_eps['log10_mass_ratio'] = np.log10(data_given_eps['mass_2'] / data_given_eps['mass_1'])
    labels = [r"$\log_{10} M_1$ [M$_\odot$]", r"$\log_{10} M_2/M_1$", r"$\log_{10} \chi$", r"$e_0$"]
    labels_dictionary = {
        'log10_mass_1': r"$\log_{10} M_1$ [M$_\odot$]",
        'log10_mass_ratio': r"$\log_{10} M_2/M_1$",
        'spin': r"$\chi$",
        'e0': r"$e_0$",}
    params = ['log10_mass_1', 'log10_mass_ratio','spin','e0']
    labels = [labels_dictionary[param] for param in params]
    num_params = len(params)

    fig, axes = plt.subplots(num_params, num_params, figsize=(7, 7))
    cmap = plt.get_cmap('coolwarm')
    for i in range(num_params):
        for j in range(num_params):
            if i > j:
                im = axes[i, j].scatter(data_given_eps[params[j]], data_given_eps[params[i]], alpha=1, s=10, c=timing_values, vmax=vmax, vmin=vmin, cmap=cmap)
                if j == 0:
                    axes[i, j].set_ylabel(labels[i])
                if i == num_params - 1:
                    axes[i, j].set_xlabel(labels[j])
            else:
                axes[i, j].remove()

    fig.subplots_adjust(right=0.8)
    cbar_axis = None#fig.add_axes([0.7, 0.1, 0.02, 0.65])
    
    if plot_type == 'timing':
        fig.colorbar(im, cax=cbar_axis,label=r'Speed [s]')
    if plot_type == 'ratio':
        fig.colorbar(im, cax=cbar_axis, label=r'Speed Ratio $\frac{FD}{TD}$')
    if plot_type == 'overlap':
        fig.colorbar(im, cax=cbar_axis, label=r'$\log_{10}$ Mismatch')
        
    fig.tight_layout()
    # plt.show()

data_df, param_names = cast_results_to_dataframe(timing_data)

# histogram timing
_min_fd, _max_fd = data_df['fd_timing'].min(), data_df['fd_timing'].max()
_min_td, _max_td = data_df['td_timing'].min(), data_df['td_timing'].max()
_min, _max = min([_min_fd, _min_td]), max([_max_fd, _max_td])
# _max = 1.0

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
dt = 5.0
shift_factor = 0.9  # Factor to slightly shift the bins for each histogram
for idx, (eps_val, pc) in enumerate(zip([1e-2, 1e-5], ['-', '--'])):
    print(f"eps_val: {eps_val}")
    data_td = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['td_timing']
    data_fd = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['fd_timing']
    eps_val_log10 = int(np.log10(eps_val))
    
    # Shift the bins slightly for each histogram
    fact = np.random.uniform(-0.01, 0.01) 
    lb = np.logspace(np.log10(_min*(1-fact)), np.log10(_max*(1+fact)), 100)
    
    ax.hist(data_td, density=True, bins=lb, histtype='step', label=rf"Time Domain, $k = 10^{{{eps_val_log10}}}$", linestyle=pc, color="C0")
    ax.hist(data_fd, density=True, bins=lb, histtype='step', label=rf"Frequency Domain, $k = 10^{{{eps_val_log10}}}$", linestyle=pc, color="C1")
    print(f"Median TD timing for eps={eps_val}: {np.median(data_td):.4f} s")
    print(f"Median FD timing for eps={eps_val}: {np.median(data_fd):.4f} s")
    # Find and print the parameters for the minimum and maximum TD and FD timings
    min_td_idx = data_td.idxmin()
    max_td_idx = data_td.idxmax()
    min_fd_idx = data_fd.idxmin()
    max_fd_idx = data_fd.idxmax()

    print(f"TD min: {data_td[min_td_idx]:.4f} s, params: {data_df.loc[min_td_idx, param_names[:5]]}")
    print(f"TD max: {data_td[max_td_idx]:.4f} s, params: {data_df.loc[max_td_idx, param_names[:5]]}")
    print(f"FD min: {data_fd[min_fd_idx]:.4f} s, params: {data_df.loc[min_fd_idx, param_names[:5]]}")
    print(f"FD max: {data_fd[max_fd_idx]:.4f} s, params: {data_df.loc[max_fd_idx, param_names[:5]]}")
    ax.set_xscale('log')
    ax.set_xlabel('Speed [s]', fontsize=label_fontsize)
    # ax.set_title(rf"$\Delta t = ${dt} s", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=label_fontsize)
plt.tight_layout()
plt.savefig(fname + 'timing_dt_5.png', dpi=300)

# histogram overlap
_min, _max = np.abs(1-data_df['overlap']).min(), np.abs(1-data_df['overlap']).max()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
dt = 5.0
shift_factor = 0.9  # Factor to slightly shift the bins for each histogram
for idx, (eps_val, pc) in enumerate(zip([1e-2, 1e-5], ['tab:blue', 'tab:orange', 'tab:green'])):
    data_td = np.abs(1-data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['overlap'])
    eps_val_log10 = int(np.log10(eps_val))
    
    # Shift the bins slightly for each histogram
    lb = np.logspace(np.log10(_min), np.log10(_max), 100)
    
    ax.hist(data_td, density=True, bins=lb, histtype='step', label=rf"TD, $k= 10^{{{eps_val_log10}}}$", linestyle='--', color=pc)
    ax.hist(data_fd, density=True, bins=lb, histtype='step', label=rf"FD, $k= 10^{{{eps_val_log10}}}$", color=pc)
    ax.set_xscale('log')
    ax.set_xlabel('Mismatch', fontsize=label_fontsize)
    # ax.set_title(rf"$\Delta t = ${dt} s", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=label_fontsize)
plt.savefig(fname + 'overlap_dt_5.png', dpi=300)



# Overlap vs Mass
_min, _max = np.abs(1-data_df['overlap']).min(), np.abs(1-data_df['overlap']).max()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
dt = 5.0
shift_factor = 0.9  # Factor to slightly shift the bins for each histogram
for idx, (eps_val, pc) in enumerate(zip([1e-2, 1e-5], ['tab:blue', 'tab:orange', 'tab:green'])):
    data_td = np.abs(1-data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['overlap'])
    
    mass_1 = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['mass_1']
    eps_val_log10 = int(np.log10(eps_val))
    ax.plot(mass_1, data_td, '.')
    # ax.hist(data_td, density=True, bins=lb, histtype='step', label=rf"TD, $\mathcal{k}= 10^{{{eps_val_log10}}}$", linestyle='--', color=pc)
    # ax.hist(data_fd, density=True, bins=lb, histtype='step', label=rf"FD, $\mathcal{k}= 10^{{{eps_val_log10}}}$", color=pc)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Mismatch', fontsize=label_fontsize)
    ax.set_xlabel(r"$M_1$ [M$_\odot$]", fontsize=label_fontsize)
    # ax.set_xlabel('Mismatch', fontsize=label_fontsize)
    # ax.set_title(rf"$\Delta t = ${dt} s", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=label_fontsize)
plt.savefig(fname + 'mismatch_M_dt_5.png', dpi=300)

for variable in ["mass_1", "spin", "e0"]:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    dt = 5.0
    shift_factor = 0.9  # Factor to slightly shift the bins for each histogram
    for idx, (eps_val, pc) in enumerate(zip([1e-2], ['tab:blue', 'tab:orange', 'tab:green'])):
        data_td = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['td_timing']
        data_fd = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)]['fd_timing']
        mass_1 = data_df[(data_df['mode_selection_threshold'] == eps_val) & (data_df['dt'] == dt)][variable]
        eps_val_log10 = int(np.log10(eps_val))
        ax.plot(mass_1, data_td, 'P',alpha=0.1, label='TD speed')
        ax.plot(mass_1, data_fd, 'X',alpha=0.1, label='FD speed')
        # ax.hist(data_td, density=True, bins=lb, histtype='step', label=rf"TD, $\mathcal{k}= 10^{{{eps_val_log10}}}$", linestyle='--', color=pc)
        # ax.hist(data_fd, density=True, bins=lb, histtype='step', label=rf"FD, $\mathcal{k}= 10^{{{eps_val_log10}}}$", color=pc)
        
        if variable == 'mass_1':
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Speed [s]', fontsize=label_fontsize)
        # ax.set_xlabel(r"$M_1$ [M$_\odot$]", fontsize=label_fontsize)
        ax.set_xlabel(variable, fontsize=label_fontsize)
        # ax.set_title(rf"$\Delta t = ${dt} s", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.legend(fontsize=label_fontsize)
    plt.savefig(fname + 'timing_' + variable + '_dt_5.png', dpi=300)

# %%
# corner_plot(data_df, eps_value=1e-5, dt_value=5.0)
# plt.savefig(fname + 'corner_td.png', dpi=300)

corner_plot(data_df, eps_value=1e-5, dt_value=5.0, plot_type='power')
plt.savefig(fname + 'corner_power.png', dpi=300)

corner_plot(data_df, eps_value=1e-5, dt_value=5.0, plot_type='ratio')
plt.savefig(fname + 'corner_ratio.png', dpi=300)

corner_plot(data_df, eps_value=1e-5, dt_value=5.0, plot_type='overlap')
plt.savefig(fname + 'corner_overlap.png', dpi=300)