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

# %%
timing_data = json.load(open('timing_test_results.json', 'r'))

# %%
def cast_results_to_dataframe(input_data):
    output = []

    key_list = list(input_data[0]['parameters'].keys())
    key_list.append('duration')
    key_list.append('iterations')
    key_list.append('dt')
    key_list.append('eps')
    key_list.append('fd_timing')
    key_list.append('td_timing')

    for single in input_data:
        _output_list = list(single['parameters'].values())
        _output_list.append(single['duration'])
        _output_list.append(single['iterations'])
        for data in single['timing_results']:
            temp = _output_list.copy()
            temp.append(data['dt'])
            temp.append(data['eps'])
            temp.append(data['fd_timing'])
            temp.append(data['td_timing'])
            output.append(temp.copy())
        
    return df(output, columns=key_list), key_list

# %%
def corner_plot(dataframe, minmax=None, use_td=True, plot_ratio=False, eps_value=1e-2, dt_value=10.0):
    eps_range = np.unique(dataframe['eps'])
    dt_range = np.unique(dataframe['dt'])
    assert eps_value in eps_range
    assert dt_value in dt_range

    data_given_eps = dataframe[(dataframe['eps'] == eps_value)&(dataframe['dt'] == dt_value)]
    # obtain keys from dataframe
    params = list(data_given_eps.keys())

    if not plot_ratio:
        timing_values = (data_given_eps['td_timing' if use_td else 'fd_timing'])
    else:
        timing_values = (data_given_eps['fd_timing']) / (data_given_eps['td_timing'])
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
    params = ['log10_mass_1', 'e0']
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
    
    if plot_ratio:
        fig.colorbar(im, cax=cbar_axis, label=r'Speed Ratio $\frac{FD}{TD}$')
    else:
        fig.colorbar(im, cax=cbar_axis,label=r'Speed [s]')
    fig.tight_layout()
    # plt.show()

data_df, param_names = cast_results_to_dataframe(timing_data)

# %%
_min_fd, _max_fd = data_df['fd_timing'].min(), data_df['fd_timing'].max()
_min_td, _max_td = data_df['td_timing'].min(), data_df['td_timing'].max()
_min, _max = min([_min_fd, _min_td]), max([_max_fd, _max_td])

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
dt = 5.0
shift_factor = 0.9  # Factor to slightly shift the bins for each histogram
for idx, (eps_val, pc) in enumerate(zip([1e-2, 1e-3, 1e-4], ['tab:blue', 'tab:orange', 'tab:green'])):
    data_td = data_df[(data_df['eps'] == eps_val) & (data_df['dt'] == dt)]['td_timing']
    data_fd = data_df[(data_df['eps'] == eps_val) & (data_df['dt'] == dt)]['fd_timing']
    eps_val_log10 = int(np.log10(eps_val))
    
    # Shift the bins slightly for each histogram
    fact = np.random.uniform(-0.01, 0.01) 
    lb = np.logspace(np.log10(_min*(1-fact)), np.log10(_max*(1-fact)), 100)
    
    ax.hist(data_td, density=True, bins=lb, histtype='step', label=rf"TD, $\epsilon = 10^{{{eps_val_log10}}}$", linestyle='--', color=pc)
    ax.hist(data_fd, density=True, bins=lb, histtype='step', label=rf"FD, $\epsilon = 10^{{{eps_val_log10}}}$", color=pc)
    ax.set_xscale('log')
    ax.set_xlabel('Speed [s]', fontsize=label_fontsize)
    ax.set_title(rf"$\Delta t = ${dt} s", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=label_fontsize)
plt.savefig('timing_test_results_dt_5.pdf', dpi=150)

# %%
corner_plot(data_df, eps_value=1e-2, dt_value=5.0)
plt.savefig('timing_test_results_corner_td.pdf', dpi=150)

corner_plot(data_df, eps_value=1e-2, dt_value=5.0, plot_ratio=True)
plt.savefig('timing_test_results_corner_ratio.pdf', dpi=150)



