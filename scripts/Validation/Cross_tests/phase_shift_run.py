import numpy as np
from matplotlib import pyplot as plt
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux, SchwarzEccFlux
from few.utils.utility import get_p_at_t
from few.utils.constants import MTSUN_SI, YRSID_SI
from numpy import pi

import json
import os
from bhpwave.waveform import KerrWaveform
from  bhpwave.trajectory.inspiral import InspiralGenerator


import multiprocessing
num_threads = multiprocessing.cpu_count()
from  bhpwave.trajectory.inspiral import InspiralGenerator
traj_BHPWave = InspiralGenerator(trajectory_data=None)
# traj_l30_Has = EMRIInspiral(func="Relativistic_Kerr_Circ_Flux") # my flux and lmax = 30
traj_few = EMRIInspiral(func = SchwarzEccFlux, integrate_constants_of_motion=False) 
traj_Kerr_ecc = EMRIInspiral(func= KerrEccEqFlux, integrate_constants_of_motion=False) 


output_file = "final_phase_across_models.json"
if os.path.exists(output_file):
    os.remove(output_file)
    print("old file removed")




M = 1e6
mu = 1e1
dt = 1.0
dtz = 40.0 # this is to have less number of point for Zach's traj
T = 4.0
e0 = 0.0
Y0 = 1.0
x0 = 1.0
n_points_interp = 500

from bhpwave.constants import Modot_GC1_to_S
Mt2st_BHPWave = (M*Modot_GC1_to_S)
Mt2st_FEW = M*MTSUN_SI
print("mass in second from Zach: ", Mt2st_BHPWave,'\n', "mass in second from FEW: ", Mt2st_FEW,'\n', "fracional error of the two", (Mt2st_BHPWave-Mt2st_FEW)/Mt2st_FEW)




##### loading saved data form the old FEW Kerr circular version
path_amp_plots = os.getcwd() + '/'#"/home/hkhalvati/Downloads/KerrEccentricEquatorialFigures/scripts/Results/Cross_tests/"
traj_KerrCirc_result = np.loadtxt(path_amp_plots + "Traj_KerrCirc.txt")


a_arr = np.unique(traj_KerrCirc_result[:,0])

all_data = {}

for aa in a_arr:
    #### mask the KerrCirc data for spin aa:
    mask_traj_has = traj_KerrCirc_result[:,0] == aa
    t_KerrCirc =  traj_KerrCirc_result[mask_traj_has][:,1]
    p_KerrCirc =  traj_KerrCirc_result[mask_traj_has][:,2]
    Phi_phi_KerrCirc = traj_KerrCirc_result[mask_traj_has][:,3]
    p0 = p_KerrCirc[0]
    print(f"from the KerrCirc data file -> p0:{p0} for a = {aa}")

    ###### Run the BHPWave traj for the same spin
    BHPWave_traj_results = traj_BHPWave(M, mu, aa, p0, dt=dtz, T=T, num_threads=num_threads)
    Phi_phi_BHPWave = BHPWave_traj_results.inspiral_data.phase
    t_BHPWave = BHPWave_traj_results.inspiral_data.time
    p_BHPWave = BHPWave_traj_results.inspiral_data.radius
    print("BHPWave traj p0:", p_BHPWave[0], "BHPWave traj final p:", p_BHPWave[-1])

    # -------- Notice that the trajectories are not in the coordinate time, so the t is t/M and to make it into sec we have to use t*M*MTSUN_SI--------------
    KerrEcc_result = traj_Kerr_ecc(M,mu,aa,p0,e0,Y0,in_coordinate_time=False, dt=dt, T=T,max_init_len=int(1e5) ,err=1e-17)
    Phi_phi_KerrEcc = KerrEcc_result[4]
    t_KerrEcc = KerrEcc_result[0]
    p_KerrEcc = KerrEcc_result[1]
    print("KerrEcc traj p0:", p_KerrEcc[0], "KerrEcc traj final p:", p_KerrEcc[-1])

    

   
    spline_Phi_KerrCirc = CubicSplineInterpolant(t_KerrCirc, Phi_phi_KerrCirc) #  spline for Has's traj
    spline_Phi_BHPWave = CubicSplineInterpolant(t_BHPWave, Phi_phi_BHPWave) #  spline for Zach's traj
    spline_Phi_KerrEcc = CubicSplineInterpolant(t_KerrEcc, Phi_phi_KerrEcc) #  spline for Kerr's traj

    final_time_KerrCirc = t_KerrCirc[-1]#/YRSID_SI*Mt2st_FEW
    final_time_BHPWave = t_BHPWave[-1]#/YRSID_SI*Mt2st_FEW
    final_time_KerrEcc = t_KerrEcc[-1]#/YRSID_SI*Mt2st_FEW 
    print(f"t finals in years: KerrCirc = {t_KerrCirc[-1]/YRSID_SI*Mt2st_FEW}, BHPWave = {t_BHPWave[-1]/YRSID_SI*Mt2st_FEW}, KerrEcc = {t_KerrEcc[-1]/YRSID_SI*Mt2st_FEW}")


    #### interpolate here:
    init = 0
    fin = min(t_KerrCirc[-1], t_BHPWave[-1], t_KerrEcc[-1]) 
    # t_spline = np.linspace(init,fin,n_points_interp)
    x = np.linspace(0, 1, n_points_interp)  # Uniform parameter space
    t_spline = init + (fin - init) * (1 - (1 - x)**3)

    Phi_phi_spl_KerrCirc = spline_Phi_KerrCirc(t_spline)
    Phi_phi_spl_BHPWave = spline_Phi_BHPWave(t_spline)
    Phi_phi_spl_KerrEcc = spline_Phi_KerrEcc(t_spline)


    Phi_phi_final_KerrCirc = Phi_phi_spl_KerrCirc[-1]
    Phi_phi_final_BHPWave = Phi_phi_spl_BHPWave[-1]
    Phi_phi_final_KerrEcc = Phi_phi_spl_KerrEcc[-1]
    print(f"final phase for KerrCirc:{Phi_phi_final_KerrCirc}, BHPWave:{Phi_phi_final_BHPWave}, KerrEcc:{Phi_phi_final_KerrEcc} \n\n")
    all_data[f"spin_{aa:.4f}"] = {
        "KerrCirc": {
             "spin": float(aa),
            "M": float(M),
            "mu": float(mu),
            "p0": float(p0),
            "t_final": float(final_time_KerrCirc),
            "p_final": float(p_KerrCirc[-1]),
            "Phi_phi_final": float(Phi_phi_final_KerrCirc)},
        "BHPWave": {
             "spin": float(aa),
            "M": float(M),
            "mu": float(mu),
            "p0": float(p0),
            "t_final": float(final_time_BHPWave),
            "p_final": float(p_BHPWave[-1]),
            "Phi_phi_final": float(Phi_phi_final_BHPWave)},
        "KerrEcc": {
             "spin": float(aa),
            "M": float(M),
            "mu": float(mu),
            "p0": float(p0),
            "t_final": float(final_time_KerrEcc),
            "p_final": float(p_KerrEcc[-1]),
            "Phi_phi_final": float(Phi_phi_final_KerrEcc)}
    }

    if aa==0.0:
        print(" \n\n Spin zero case \n\n")
        ### Schwarzchild case from the new style traj FEW
        Schw_result = traj_few(M,mu,aa,p0,e0,Y0,in_coordinate_time=False, dt=dt, T=T,max_init_len=int(1e5),err=1e-17)
        t_Schw = Schw_result[0]
        p_Schw = Schw_result[1]
        Phi_phi_Schw = Schw_result[4]
        spline_Phi_Schw = CubicSplineInterpolant(t_Schw, Phi_phi_Schw) #  spline for FEW's traj frm new version
        
        final_time_Schw = t_Schw[-1]#/YRSID_SI*Mt2st_FEW
        final_time_KerrCirc = t_KerrCirc[-1]#/YRSID_SI*Mt2st_FEW
        final_time_BHPWave = t_BHPWave[-1]#/YRSID_SI*Mt2st_FEW
        final_time_KerrEcc = t_KerrEcc[-1]#/YRSID_SI*Mt2st_FEW 
        
        print(f"t finals in years: KerrCirc = {t_KerrCirc[-1]/YRSID_SI*Mt2st_FEW}, BHPWave = {t_BHPWave[-1]/YRSID_SI*Mt2st_FEW}, KerrEcc = {t_KerrEcc[-1]/YRSID_SI*Mt2st_FEW}, Schw = {t_Schw[-1]/YRSID_SI*Mt2st_FEW}")
        fin = min(t_KerrCirc[-1], t_BHPWave[-1], t_KerrEcc[-1], t_Schw[-1]) 
        # t_spline = np.linspace(init,fin,n_points_interp)
        x = np.linspace(0, 1, n_points_interp)  # Uniform parameter space
        t_spline = init + (fin - init) * (1 - (1 - x)**3)
        Phi_phi_spl_Schw = spline_Phi_Schw(t_spline)
        Phi_phi_spl_KerrCirc = spline_Phi_KerrCirc(t_spline)
        Phi_phi_spl_BHPWave = spline_Phi_BHPWave(t_spline)
        Phi_phi_spl_KerrEcc = spline_Phi_KerrEcc(t_spline)
        
        Phi_phi_final_Schw = Phi_phi_spl_Schw[-1]
        Phi_phi_final_KerrCirc = Phi_phi_spl_KerrCirc[-1]
        Phi_phi_final_BHPWave = Phi_phi_spl_BHPWave[-1]
        Phi_phi_final_KerrEcc = Phi_phi_spl_KerrEcc[-1]
        print(f"final phase for Schw:{Phi_phi_final_Schw}, KerrCirc:{Phi_phi_final_KerrCirc}, BHPWave:{Phi_phi_final_BHPWave}, KerrEcc:{Phi_phi_final_KerrEcc} \n\n end of spin zero case\n\n")
        all_data[f"spin_{aa:.4f}"]["Schw"] = {
             "spin": float(aa),
            "M": float(M),
            "mu": float(mu),
            "p0": float(p0),
            "t_final": float(final_time_Schw),
            "p_final": float(p_Schw[-1]),
            "Phi_phi_final": float(Phi_phi_final_Schw)}

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)
        print("data saved in json file")

    









