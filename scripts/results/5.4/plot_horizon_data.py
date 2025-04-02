# plot horizon data for the Kerr eccentric equatorial case
# python plot_horizon_data.py -Tobs 2.0 -q 1.0e-4 -spin 0.99 -zaxis e0 -base AE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

from seaborn import color_palette

from scipy.interpolate import interp1d
import argparse

import h5py

import re

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

import pysco 
pysco.plot.default_plotting()


parser = argparse.ArgumentParser(description="Plot horizon data")
parser.add_argument("-Tobs", "--Tobs", type=float, default=1.0, help="Observation time in years")
parser.add_argument("-q", "--q", type=float, default=1.0e-5, help="Mass ratio")
parser.add_argument("-e0", "--e0", type=float, default=5.0e-1, help="initial eccentricity")
parser.add_argument("-spin", "--spin", type=float, default=9.9e-5, help="spin")
parser.add_argument("-zaxis", "--zaxis", type=str, default='e0', help="z-axis")
parser.add_argument("-base", "--base_name", type=str, default='horizon', help="base name of the data file")
parser.add_argument("-interp", "--interp", action='store_true', default=False, help="use interpolation")
parser.add_argument("-cpal", "--cpal", type=str, default='colorblind', help="color palette")
parser.add_argument("-ec", "--every_color", type=int, default=1, help="how many colors to skip")
parser.add_argument("--hide_aak", action='store_true', default=False, help="hide AAK data")
parser.add_argument("-min", "--min", type=float, default=-100.0, help="minimum value for the z-axis")
parser.add_argument("-max", "--max", type=float, default=100.0, help="maximum value for the z-axis")

args = vars(parser.parse_args())

# seaborn colorblind palette
cpal = color_palette(args['cpal'])[::args['every_color']]#, as_cmap=True)
#cpal = color_palette("dark:#5A9_r")[::2]

datadir = './horizon_data/'
plotdir = './figures/'

def extract_numeric_value(zaxis, key):
    match = re.search(zaxis + r"_(-?\d+\.?\d*)", key)
    return float(match.group(1)) if match else 0


def format_parameter_strings(param_list):
    """
    Format parameter strings that contain scientific values to have consistent decimal notation.
    Handles strings in the format '*_VALUE' and '*_VALUE_sigma'.
    
    Args:
        param_list (list): List of parameter strings to format
    
    Returns:
        list: List of formatted parameter strings
    """
    result = []
    
    for param in param_list:
        # Split the string to extract components
        parts = param.split('_')
        
        # Check if this is a base parameter or a sigma parameter
        is_sigma = parts[-1] == 'sigma'
        
        # Extract the numerical value
        if is_sigma:
            # For sigma parameters, the value is the second-to-last part
            value_str = parts[-2]
            prefix = '_'.join(parts[:-2])  # Everything before the value
            suffix = '_sigma'
        else:
            # For base parameters, the value is the last part
            value_str = parts[-1]
            prefix = '_'.join(parts[:-1])  # Everything before the value
            suffix = ''
        
        # Convert the value to a float and then to decimal notation
        try:
            value = float(value_str)
            
            # Format small numbers in decimal notation
            # For very small numbers, use a fixed number of decimal places
            if value < 0.01:
                decimal_places = 6  # Adjust this as needed
                formatted_value = f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
            else:
                # For larger numbers, use fewer decimal places
                formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
            
            # Reconstruct the parameter string
            formatted_param = f"{prefix}_{formatted_value}{suffix}"
            result.append(formatted_param)
        except ValueError:
            # If conversion fails, keep the original string
            result.append(param)
    
    return result

def get_mantissa_exponent(value):
    exponent = np.floor(np.log10(abs(value)))  # Get exponent
    mantissa = value / 10**exponent  # Get mantissa
    return mantissa, int(exponent)


def convert_tick_labels(z_axis, z_values):  
    """
    Convert tick labels for the colorbar based on the z-axis parameter.
    
    Args:
        z_axis (str): The z-axis parameter ('e0', 'spin', or 'q').
        z_values (list): The list of z-axis values.
    
    Returns:
        list: The converted tick labels.
    """
    if z_axis == 'e0':
        return [f"{float(z):.2f}" for z in z_values]
    elif z_axis == 'spin':
        return [f"{float(z):.3f}" for z in z_values]
    elif z_axis == 'q':
        #return [f"{float(z):.1e}" for z in z_values]
        # return in the form "n x 10^m". I need the mantissa and the exponent
        labels = []
        for z in z_values:
            mantissa, exponent = get_mantissa_exponent(z)
            if int(mantissa) != 1:
                labels.append(r"%i \times 10^{%i}" % (mantissa, exponent))
            else:
                labels.append(r"$10^{%i}$" % (exponent))
        return labels
    else:
        raise ValueError("Invalid z-axis parameter")

def add_plot(M_detector, data, data_sigma, ls, colors='k', fill=False, interp=False, interp_kwargs={}, plot_kwargs={}, fig=None, axs=None, use_gpr=True):
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True)

    if isinstance(colors, str):
        colors = [colors] * len(data.keys())

    logMgrid = np.linspace(4, 8, 1000)
    Mgrid = np.logspace(4, 8, 1000)
    zmin = 0.0

    for key, color in zip(data.keys(), colors):
        z_here = data[key][:]
        sigma_here = data_sigma[key + '_sigma'][:]
        nan_mask = np.isnan(z_here) # remove NaNs added for points outside the grids
        z_here = z_here[~nan_mask]
        sigma_here = sigma_here[~nan_mask]
        M_detector_here = M_detector[~nan_mask]

        #breakpoint()

        M_source = to_M_source(M_detector_here, z_here)
        if interp:
            if use_gpr:
                kernel = 1.0 * RBF(length_scale=1e-4, length_scale_bounds=(1e-10, 1e10)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-30, 1e10))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=sigma_here**2 * 1e-4,)

                gp.fit(np.log10(M_source)[:, None], z_here)                
                x = logMgrid

                # gp.fit(M_source[:, None], z_here)
                # x = Mgrid

                y, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
                x = 10**x
            else:
                x = Mgrid
                interpolant = interp(M_source, z_here, **interp_kwargs) 
                y, sigma = interpolant(x), None

        else:
            x = M_source
            y, sigma = z_here, None

        if fill:
            #axs.semilogx(x, y, ls=ls, color='k', label=key, **plot_kwargs)
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)
            axs.fill_between(x, zmin, y, alpha=0.3, zorder=1, hatch='', color=color, rasterized=True)
            zmin = y
        else:
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)
            if sigma is not None:   
                axs.fill_between(x, y-sigma, y+sigma, alpha=0.3, zorder=1, hatch='', color=color, rasterized=True)
        
        # if interp:
        #     axs.semilogx(M_source, z_here, marker='x', color=color, **plot_kwargs)

    plt.xlabel(r'$M_{\rm source} \, [M_\odot]$')
    plt.ylabel(r'$\bar{z}$')

    return fig, axs


def to_logM_source(logM, z):
    return logM - np.log(1+z) 

def to_M_source(M, z):
    return M / (1+z)

def pastel_map(cmap, c=0.2, n=6):
    """
    Create a lighter version of a colormap
    Arguments:
    cmap : colormap
    c : scale factor for the lighter colors
    n : number of colors to return
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = ((1. - c) * cmap(np.linspace(0., 1., n)) + c * np.ones((n, 4)))
    return colors
    

if __name__ == '__main__':
    Tobs = args['Tobs']
    q = args['q']
    e0 = args['e0']
    spin = args['spin']
    zaxis = args['zaxis']
    base_name = args['base_name']

    print('z-axis: ', zaxis)

    if zaxis == 'e0':
        datastring = 'T_{:.1f}_q_{:.1e}_spin_{:.1e}.h5'.format(Tobs, q, spin)
        zaxis_plot = r'$e_0$'

    elif zaxis == 'spin':
        datastring = 'T_{:.1f}_q_{:.1e}_e0_{:.1e}.h5'.format(Tobs, q, e0)
        zaxis_plot = r'$a$'
    
    elif zaxis == 'q':
        datastring = 'T_{:.1f}_spin_{:.1e}_e0_{:.1e}.h5'.format(Tobs, spin, e0)   
        zaxis_plot = r'$q$' 

    else:
        raise ValueError("z-axis must be either 'e0', 'spin', or 'q'")
    
    savename = plotdir + base_name + '_' + zaxis + '_' + datastring[:-3] + '.pdf'
    
    kerr_kerr_data = base_name + '_traj_kerr_wf_kerr_' + datastring
    kerr_aak_data = base_name + '_traj_kerr_wf_aak_' + datastring
    pn5_aak_data = base_name + '_traj_pn5_wf_aak_' + datastring

    linestyles = ['-', '--', '-.']
    #labels = ['Kerr 0PA', 'Kerr trajectory, AAK amplitudes', 'PN5 trajectory, AAK amplitudes']
    labels = ['Kerr-Kerr', 'Kerr-AAK', 'PN5-AAK']
    handles = []

    interp = args['interp']
    interp_kwargs = dict(fill_value='extrapolate')
    fill = False
    zmax = 0.0

    what_to_plot = [kerr_kerr_data]
    if not args['hide_aak']:
        what_to_plot += [kerr_aak_data, pn5_aak_data]

    for i, datafile in enumerate(what_to_plot):
        ls = linestyles[i]
        label = labels[i]
        data = {}
        attr = {}
        try:
            with h5py.File(datadir + datafile, 'r') as f:
                for key in f.attrs.keys():
                    attr[key] = f.attrs[key]
                for key in f.keys():
                    data[key] = f[key][:]

        except Exception as e:
            print("Error reading file: ", datafile)
            print(e)
            continue
            
        #sort the data by increasing zaxis keyword
        unformatted_keys = list(data.keys())
        keys = format_parameter_strings(list(data.keys()))
        data = {key:data[unformatted_keys[i]] for i, key in enumerate(keys)}
        #sorted_keys = np.sort(keys)
        sorted_keys = sorted(keys, key=lambda x: extract_numeric_value(zaxis, x))

        # keep only the keys that are in the sorted list between args['min'] and args['max']
        sorted_keys = [key for key in sorted_keys if key in data.keys() and args['min'] <= float(key.split('_')[1]) <= args['max']]


        data = {key:data[key] for key in sorted_keys}
    
        #split the keys into two lists: one without the 'sigma' keys and one with
        keys_no_sigma = [key for key in sorted_keys if 'sigma' not in key]
        keys_sigma = [key for key in sorted_keys if 'sigma' in key]
        data_z = {key:data[key] for key in keys_no_sigma}
        data_sigma = {key:data[key] for key in keys_sigma}

        zmax = max(zmax, max([data_z[key][np.isfinite(data_z[key])].max() for key in data_z.keys()]) + 0.2)

        #cpal = pastel_map('Blues_r', c=0.2, n=len(data.keys())+1)[::-1]

        M_detector = attr['M'] # detector-frame mass
        plot_kwargs = dict(zorder=10-i, rasterized=True)
        if i == 0: # create the figure
            fig, ax = add_plot(M_detector, data_z, data_sigma, ls, colors=cpal, fill=fill, interp=interp, interp_kwargs=interp_kwargs, plot_kwargs=plot_kwargs, use_gpr=True)
            z_values = np.array(list(attr[zaxis]))
            zmask = np.logical_and(z_values >= args['min'],z_values <= args['max'])
            z_values = z_values[zmask]
            
            print('z_values: ', z_values)
            nlines = len(z_values)

        else:
            add_plot(M_detector, data_z, data_sigma, ls, colors=cpal, fill=fill, interp=interp, interp_kwargs=interp_kwargs, plot_kwargs=plot_kwargs, fig=fig, axs=ax, use_gpr=True)

        handle = mlines.Line2D([], [], color='gray', linestyle=ls, label=label)
        handles += [handle]

    boundaries = np.arange(nlines+1)  # Create boundaries for the colorbar
    cmap = mpl.colors.ListedColormap(cpal[:nlines])  # One fewer color than boundaries

    # Create the norm - BoundaryNorm places colors between the boundaries
    norm = mpl.colors.BoundaryNorm(boundaries, nlines)

    # Create the ScalarMappable
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the colorbar
    cbar_ax = fig.add_axes([0.95, 0.09, 0.03, 0.8])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, orientation='vertical', 
                        cax=cbar_ax, extend='both', drawedges=True)

    tick_positions = [(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)]
    cbar.set_ticks(tick_positions, labels=convert_tick_labels(zaxis,z_values))
    cbar.set_label(zaxis_plot, fontsize=18, labelpad=22)
    cbar.ax.yaxis.label.set_rotation(0)


    #ax.legend(handles=handles, ncols=3, bbox_to_anchor=(1.038, 1.1))
    #fig.tight_layout() 
    ax.xaxis.set_tick_params(pad=6)
    ax.set_xlim(9e4, 5e7)
    ax.set_ylim(0., zmax)
    plt.savefig(savename, dpi=300)
    #breakpoint()


    