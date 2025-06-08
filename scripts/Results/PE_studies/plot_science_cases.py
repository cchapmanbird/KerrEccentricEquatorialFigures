import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set the GPU to use
os.environ["OMP_NUM_THREADS"] = "1" # set the number of threads to use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

from seaborn import color_palette

import logging
import h5py


# to be changed for consistency across the paper
custom_rcParams = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'font.weight':'medium',
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r"\usepackage{amsmath}",
        'font.size': 20,
        'figure.figsize': (7, 7),
        'figure.titlesize': 22,
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': [-2, 4],
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.top': True,
        'xtick.major.size': 5,
        'xtick.minor.size': 3,
        'xtick.major.width': 0.8,
        'xtick.minor.visible': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 20,
        'ytick.right': True,
        'ytick.major.size': 5,
        'ytick.minor.size': 3,
        'ytick.major.width': 0.8,
        'ytick.minor.visible': True,
        'ytick.direction': 'in',
        'ytick.labelsize': 20,
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.fontsize': 20,
        'legend.scatterpoints' : 3,
        #'lines.color': 'k',
        'lines.linewidth': 1.5,
        'patch.linewidth': 1,
        'hatch.linewidth': 1,
        'grid.linestyle': 'dashed',
        'savefig.dpi' : 300,
        'savefig.format' : 'pdf',
        'savefig.bbox' : 'tight',
        'savefig.transparent' : True,
    }

# Set the custom rcParams
plt.rcParams.update(custom_rcParams)

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import YRSID_SI
from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
from few.summation.interpolatedmodesum import CubicSplineInterpolant


LIGHT_IMRI_PARAMS = dict(
    M=1e5, 
    mu=1e3, 
    a=0.95, 
    p0=74.94184, 
    e0=0.85, 
    x_I0=1.0,
    dist=2.0, 
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=5.0,
    Phi_theta0=0.0,
    Phi_r0=4.5,
    dt=2.0,
    label = "Light_imri",
    )

HEAVY_IMRI_PARAMS = dict(
    M=1e7,
    mu=100_000,
    a=0.95,
    p0=23.6015,
    e0=0.85,
    x_I0=1.0,
    dist=7.25,
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=0.0,
    Phi_theta0=0.0,
    Phi_r0=6.0,
    dt = 5.0,
    label = "Heavy_imri"
    )

STRONG_FIELD_EMRI_PARAMS = dict(
    M=1e7,
    mu=1e1,
    a=0.998,
    p0=2.12,
    e0=0.425,
    x_I0=1.0,
    dist=5.465,
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=2.0,
    Phi_theta0=0.0,
    Phi_r0=3.0,
    dt=5.0,
    label = "Strong-field_emri"
)

PROGRADE_EMRI_PARAMS = dict(
    M=1e6,
    mu=1e1,
    a=0.998,
    p0=7.7275,
    e0=0.730,
    x_I0=1.0,
    dist=7.660,
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=2.0,
    Phi_theta0=0.0,
    Phi_r0=3.0,
    dt=5.0,
    label = "Prograde_emri"
)

RETROGRADE_EMRI_PARAMS = dict(
    M=1e5,
    mu=1e1,
    a=0.5,
    p0=26.19,
    e0=0.8,
    x_I0=-1.0,
    dist=1.5667,
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=2.0,
    Phi_theta0=0.0,
    Phi_r0=3.0,
    dt=2.0,
    label = "Retrograde_emri"
)


def process_label(label):
    """
    Process the label to remove underscores and replace with spaces.
    """
    label = label.replace("_", " ")
    label = label.replace("imri", "IMRI")
    label = label.replace("emri", "EMRI")
    #label = label.replace(" ", "\,")
    return label

def format_title(time_in_seconds):
    """
    Format the title to be more readable.
    """
    if time_in_seconds < 60:
        N = int(time_in_seconds)
        title = f"{N} seconds"
    elif time_in_seconds < 3600:
        N = int(time_in_seconds/60)
        title = f"{N} minutes"
    elif time_in_seconds < 86400:
        N = int(time_in_seconds/3600)
        title = f"{N} hours"
    elif time_in_seconds < 31557600:
        N = int(time_in_seconds/86400)
        title = f"{N} days"
    elif time_in_seconds < 3155760000:
        N = int(time_in_seconds/31557600)
        title = f"{N} months"
    else:
        N = int(time_in_seconds/3155760000)
        title = f"{N} years"

    if np.isclose(N, 1):
        title = title.replace("s", "")
        title =  title.split(" ")[1]
    return title
        

def get_params_from_dict(params_dict):
    """
    Get parameters from a dictionary.
    """
    M = params_dict["M"]
    mu = params_dict["mu"]
    a = params_dict["a"]
    p0 = params_dict["p0"]
    e0 = params_dict["e0"]
    x_I0 = params_dict["x_I0"]
    dist = params_dict["dist"]
    qS = params_dict["qS"]
    phiS = params_dict["phiS"]
    qK = params_dict["qK"]
    phiK = params_dict["phiK"]
    Phi_phi0 = params_dict["Phi_phi0"]
    Phi_theta0 = params_dict["Phi_theta0"]
    Phi_r0 = params_dict["Phi_r0"]
    dt = params_dict["dt"]
    label = params_dict["label"]

    return label, [M, mu, a, p0, e0, x_I0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0], dt

def produce_plot(logger,
                 figsize=(10, 6), 
                 use_gpu=False, 
                 first_nseconds=0,
                 last_nseconds=0,
                 seconds_to_plunge=0,
                 inspiral_kwargs={},
                 plot_hx=True,
                 spline=False,
                 savename='waveform_plots/science_cases'):
    
    cpal = color_palette("colorblind")
    # Set the color palette
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cpal)
    
    Tobs = 2.0 # 2.0 years

    LEFT_END_SECONDS = first_nseconds
    RIGHT_START_SECONDS = Tobs * YRSID_SI - seconds_to_plunge 
    RIGHT_END_SECONDS = Tobs * YRSID_SI - seconds_to_plunge + last_nseconds #-1 * last_nseconds

    logger.info(f"plotting the first {LEFT_END_SECONDS} seconds of the waveform")
    logger.info(f"plotting {RIGHT_END_SECONDS - RIGHT_START_SECONDS} seconds of the waveform, {seconds_to_plunge} seconds before plunge")

    backend = "gpu" if use_gpu else "cpu"
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=figsize, sharex='col', sharey='row')
    
    Kerr_waveform = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True),
        inspiral_kwargs = inspiral_kwargs,
        force_backend=backend,
        return_list=False,
    )
    traj = EMRIInspiral(func='KerrEccEqFlux') 


    def get_p0(M, mu, a, e0, x0, Tobs):
        # fix p0 given T
        left_bound = None#get_separatrix(a,e0,x0)+0.1
        right_bound = 200.0
        try:
            #try:
            T_factor = 1.0
            p0 = get_p_at_t(traj, Tobs * T_factor, [M, mu, a, e0, x0], bounds=[left_bound, right_bound])
            # except:
            #     left_bound = max(left_bound, 3.41)
            #     p0 = get_p_at_t(traj, Tobs * 0.999, [M, mu, a, e0, x0], bounds=[left_bound, right_bound])
        except Exception as e:
            logger.info(e)
            p0 = None   
        #logger.info("new p0 fixed by Tobs, p0=", p0, traj(M, mu, a, p0, e0, x0, T=10.0)[0][-1]/YRSID_SI)
        return p0

    def load_waveform(source_dict, use_cached=True):
        #save the waveform to a file to edit it also offline on the laptop
        # Get the parameters from the dictionary
        label, params, dt = get_params_from_dict(source_dict)

        filename = 'waveform_files/{}.h5'.format(label)
        
        # Check if the file exists
        if os.path.exists(filename) and use_cached:
            logger.info(f"Loading waveform from file")
            with h5py.File(filename, 'r') as f:
                # Load the waveform data
                t = f['t'][:]
                hp = f['hp'][:] #h+
                hc = f['hc'][:] #hx
        else:
            # Generate the waveform
            logger.info(f"Generating waveform")
            #tweak the parameters to get the right p0
            logger.info(f"p0 before: {params[3]}")
            params[3] = get_p0(params[0], params[1], params[2], params[4], params[5], Tobs)
            logger.info(f"p0 after: {params[3]}")
            wf = Kerr_waveform(*params, dt=dt, T=Tobs, mode_selection_threshold=0.0)

            if hasattr(wf, "get"): # convert from cupy to numpy
                wf = wf.get()
            #breakpoint()
            hp = wf.real
            hc = -1 * wf.imag # few is h+ - i hx
            t = np.arange(len(hp)) * dt

            # Save the waveform data to a file
            with h5py.File(filename, 'w') as f:
                f.create_dataset('t', data=t)
                f.create_dataset('hp', data=hp)
                f.create_dataset('hc', data=hc)
            logger.info(f"Saved waveform to {filename}")

        tmp = plt.figure(); plt.plot(t, hp); plt.xlabel("Time [s]"); plt.ylabel("h+"); plt.savefig("waveform_files/{}.pdf".format(label)); plt.close(tmp)
        
        # #spline the waveform using cubic interpolation and a smaller dt
        # spline_dt = 0.1
        # t_spline = np.arange(t[0], t[-1], spline_dt)
        # hp_spline = CubicSplineInterpolant(t, hp)(t_spline)
        # hc_spline = CubicSplineInterpolant(t, hc)(t_spline)

        return t, hp, hc

    def plot_row(axs, source_dict):
        #breakpoint()
        logger.info(f"Plotting {source_dict['label']}")

        t, hp, hc = load_waveform(source_dict, use_cached=True)

        left_end = np.ceil(LEFT_END_SECONDS / source_dict['dt'])
        right_start = np.floor(RIGHT_START_SECONDS / source_dict['dt'])
        right_end = np.floor(RIGHT_END_SECONDS / source_dict['dt'])
        LEFT_END = int(left_end)
        RIGHT_START = int(right_start)
        RIGHT_END = int(right_end)

        t_start = t[:LEFT_END]
        t_end = t[RIGHT_START:RIGHT_END]

        hp_start = hp[:LEFT_END]
        hp_end = hp[RIGHT_START:RIGHT_END]
        hc_start = hc[:LEFT_END]
        hc_end = hc[RIGHT_START:RIGHT_END]

        if spline:
            spline_dt = 0.1

            # Spline the waveform using cubic interpolation and a smaller dt
            t_start_spline = np.arange(t_start[0], t_start[-1], spline_dt)
            hp_start_spline = CubicSplineInterpolant(t_start, hp_start)(t_start_spline)
            hc_start_spline = CubicSplineInterpolant(t_start, hc_start)(t_start_spline)
            t_start = t_start_spline
            hp_start = hp_start_spline.get() if hasattr(hp_start_spline, "get") else hp_start_spline
            hc_start = hc_start_spline.get() if hasattr(hc_start_spline, "get") else hc_start_spline
            
            # Spline the waveform using cubic interpolation and a smaller dt
            t_end_spline = np.arange(t_end[0], t_end[-1], spline_dt)
            hp_end_spline = CubicSplineInterpolant(t_end, hp_end)(t_end_spline)
            hc_end_spline = CubicSplineInterpolant(t_end, hc_end)(t_end_spline)
            t_end = t_end_spline
            hp_end = hp_end_spline.get() if hasattr(hp_end_spline, "get") else hp_end_spline
            hc_end = hc_end_spline.get() if hasattr(hc_end_spline, "get") else hc_end_spline

        # Plot the waveform
        axs[0].plot(t_start, hp_start, label=r"$h_+$")
        
        axs[1].plot(t_end, hp_end)
        if plot_hx:
            axs[0].plot(t_start, hc_start, ls='--', label=r"$h_\times$", rasterized=True)   
            axs[1].plot(t_end, hc_end, ls='--', rasterized=True)    

        # Set the x-axis limits
        axs[0].set_xlim(-1 , LEFT_END_SECONDS)
        #axs[1].set_xlim(Tobs  * YRSID_SI + RIGHT_START_SECONDS, Tobs * 1.00000 * YRSID_SI)
        #axs[1].set_xlim(Tobs  * YRSID_SI - RIGHT_START_SECONDS, Tobs * YRSID_SI - RIGHT_START_SECONDS + RIGHT_END_SECONDS)
        axs[1].set_xlim(RIGHT_START_SECONDS, RIGHT_END_SECONDS)

        #set the y-axis label
        axis_label = process_label(source_dict['label'])
        axs[0].set_ylabel(axis_label)
        
        #enforce y axis to be symmetric around 0
        ymax = max(np.max(np.abs(hp_start)), np.max(np.abs(hp_end)))
        axs[0].set_ylim(-1.2 * ymax, 1.2 * ymax)

    
    def fill_figure(axs, all_dicts):
        # Plot the waveforms
        for i, source_dict in enumerate(all_dicts):
            plot_row(axs[i], source_dict)
        
        #set the legend for the first row
        # if plot_hx: 
        axs[0,0].legend(loc='upper left', handlelength=1.2)
        # Set the x-axis label
        axs[-1,0].set_xlabel("Time [s]")
        axs[-1,1].set_xlabel("Time [s]")

    all_dicts = [
                PROGRADE_EMRI_PARAMS, 
                STRONG_FIELD_EMRI_PARAMS, 
                HEAVY_IMRI_PARAMS,
                LIGHT_IMRI_PARAMS, 
                RETROGRADE_EMRI_PARAMS
                ]#[::-1]
    fill_figure(axes, all_dicts)

    #remove x ticks from the top row
    # for i in range(2):
    #     for ax in axes[i]:
    #         ax.set_xticks([])
    #         ax.set_xlabel("")
    
    # add a title to the first row
    title_start = format_title(LEFT_END_SECONDS)
    title_end = format_title(seconds_to_plunge)
    axes[0,0].set_title("First " + title_start)
    axes[0,1].set_title( title_end + " from plunge")
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig(savename)
    plt.close(fig)

if __name__ == "__main__":

    use_gpu = True
    inspiral_kwargs = {'flux_output_convention':'ELQ'}
    savename = 'waveform_plots/time_domain/time_snapshots.pdf'
    figsize = (18, 15)
    plot_hx = False
    spline = True

    if plot_hx:
        savename = savename.replace(".pdf", "_hx.pdf")

    first_nmonths = 0
    first_nweeks = 0
    first_ndays = 0
    first_nhours = 6
    first_nseconds = ((first_nmonths * 30 + first_nweeks * 7 + first_ndays) * 24 + first_nhours) * 3600

    last_nmonths = 0
    last_nweeks = 0
    last_ndays = 0
    last_nhours = 1
    last_nseconds = ((last_nmonths * 30 + last_nweeks * 7 + last_ndays) * 24 + last_nhours) * 3600

    hours_to_plunge = 6
    seconds_to_plunge = hours_to_plunge * 3600

    logger = logging.getLogger("science_cases")
    level = logging.INFO
    logger.setLevel(level)
    if (len(logger.handlers) < 2):
        formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
        
        shandler = logging.StreamHandler(sys.stdout)
        shandler.setLevel(level)
        shandler.setFormatter(formatter)
        logger.addHandler(shandler)
    logger.info(f"Using GPU: {use_gpu}")
    logger.info(f"Saving figure to: {savename}")

    produce_plot(logger=logger,
                figsize=figsize, 
                use_gpu=use_gpu, 
                first_nseconds=first_nseconds,
                last_nseconds=last_nseconds,
                seconds_to_plunge=seconds_to_plunge,
                inspiral_kwargs=inspiral_kwargs,
                plot_hx=plot_hx,
                spline=spline,
                savename=savename)

