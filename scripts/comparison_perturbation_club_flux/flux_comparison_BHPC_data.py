import numpy as np
from few.trajectory.ode import KerrEccEqFlux
from few.utils.geodesic import get_separatrix

fewFlux = KerrEccEqFlux()

#Import BHPC data
data_a0p9pro = np.loadtxt('pert_club_flux_data/dIdt_q0.90inc0.dat')
data_a0p7pro = np.loadtxt('pert_club_flux_data/dIdt_q0.70inc0.dat')
data_a0p5pro = np.loadtxt('pert_club_flux_data/dIdt_q0.50inc0.dat')
data_a0p3pro = np.loadtxt('pert_club_flux_data/dIdt_q0.30inc0.dat')
data_a0p1pro = np.loadtxt('pert_club_flux_data/dIdt_q0.10inc0.dat')

data_a0p9ret = np.loadtxt('pert_club_flux_data/dIdt_q-0.90inc0.dat')
data_a0p7ret = np.loadtxt('pert_club_flux_data/dIdt_q-0.70inc0.dat')
data_a0p5ret = np.loadtxt('pert_club_flux_data/dIdt_q-0.50inc0.dat')
data_a0p3ret = np.loadtxt('pert_club_flux_data/dIdt_q-0.30inc0.dat')
data_a0p1ret = np.loadtxt('pert_club_flux_data/dIdt_q-0.10inc0.dat')

#Function to compute relative difference with Few Flux
def pdotedotcompare(a, x, data):
    M = 1 #1e6
    mu = 1 # 1e1
    fewFlux.add_fixed_parameters(M, mu, a)
    
    length=data.shape[0]

    pepdotedotvals= np.zeros((length,12))
    for i in range(length):
        
        p=data[i][1]
        e=data[i][2]

        lmax=data[i][-3]
        DeltaEinf=data[i][-2]
        DeltaEh=data[i][-1]
        
        pLSO=get_separatrix(a,e,x)

        # Attempt to call the flux function and check for exception indicating outside of interpolant range
        def is_not_allowed_value(p,e,x):
            try:
                pdot, edot, xIdot, Omega_phi, Omega_theta, Omega_r = fewFlux([p, e, x])
        
                # If no ValueError, set the variable to False
                error_occurred = False
            except Exception:
        
                # If a ValueError occurs, set the variable to True
                error_occurred = True

            return error_occurred

        if abs(DeltaEinf)>10**(-7) or abs(DeltaEh)>10**(-7) or is_not_allowed_value(p,e,x): #Throw away ryuichi's data with larger mode sum error or data outside interpolant
            
            pepdotedotvals[i][0]=p #Store p value for reference and move on
            pepdotedotvals[i][1]=e #Store e value for reference and move on
            pepdotedotvals[i][2]=(mu/M)*(data[i][13]+data[i][16]) #Store total pdot value
            pepdotedotvals[i][3]=(mu/M)*(data[i][14]+data[i][17]) #Store total edot value

        else:
            
            pdot, edot, xIdot, Omega_phi, Omega_theta, Omega_r = fewFlux([p, e, x])

            pepdotedotvals[i][0]=p #Store p value for reference and move on
            pepdotedotvals[i][1]=e #Store e value for reference and move on
        
            pepdotedotvals[i][2]=(mu/M)*(data[i][13]+data[i][16]) #Store total pdot value
            pepdotedotvals[i][3]=(mu/M)*(data[i][14]+data[i][17]) #Store total edot value

            pepdotedotvals[i][4]= pdot #Store total pdot value from FEW
            pepdotedotvals[i][5]= edot #Store total edot value from FEW

            pepdotedotvals[i][6] = np.log10(abs(1 - pdot/pepdotedotvals[i][2])) #Store log10 of pdot relative diff
            pepdotedotvals[i][7] = np.log10(abs(1 - edot/pepdotedotvals[i][3])) #Store log10 of edot relative diff

            pepdotedotvals[i][8] = lmax #Store BHPC's lmax for later reference
            pepdotedotvals[i][9] = np.log10(abs(DeltaEinf)) #Store BHPC's DeltaEinf for later reference on log scale
            pepdotedotvals[i][10] = np.log10(abs(DeltaEh)) #Store BHPC's DeltaEh for later reference on log scale

            pepdotedotvals[i][11] = p-pLSO #Store as distance from separatrix for later.

    #Now filter out the entries outside of the domain of the interpolant by deleting rows where the last seven entries are zero
    filtered_array = pepdotedotvals[~np.all(pepdotedotvals[:, -8:] == 0, axis=1)]

    return filtered_array

#Compare the prograde data - save
compare_a0p9pro = pdotedotcompare(0.9, 1, data_a0p9pro)
compare_a0p7pro = pdotedotcompare(0.7, 1, data_a0p7pro)
compare_a0p5pro = pdotedotcompare(0.5, 1, data_a0p5pro)
compare_a0p3pro = pdotedotcompare(0.3, 1, data_a0p3pro)
compare_a0p1pro = pdotedotcompare(0.1, 1, data_a0p1pro)

np.savetxt("compare_a0p9pro.txt", compare_a0p9pro)
np.savetxt("compare_a0p7pro.txt", compare_a0p7pro)
np.savetxt("compare_a0p5pro.txt", compare_a0p5pro)
np.savetxt("compare_a0p3pro.txt", compare_a0p3pro)
np.savetxt("compare_a0p1pro.txt", compare_a0p1pro)

#Compare the retrograde data - save
compare_a0p9ret = pdotedotcompare(0.9, -1, data_a0p9ret)
compare_a0p7ret = pdotedotcompare(0.7, -1, data_a0p7ret)
compare_a0p5ret = pdotedotcompare(0.5, -1, data_a0p5ret)
compare_a0p3ret = pdotedotcompare(0.3, -1, data_a0p3ret)
compare_a0p1ret = pdotedotcompare(0.1, -1, data_a0p1ret)

np.savetxt("compare_a0p9ret.txt", compare_a0p9ret)
np.savetxt("compare_a0p7ret.txt", compare_a0p7ret)
np.savetxt("compare_a0p5ret.txt", compare_a0p5ret)
np.savetxt("compare_a0p3ret.txt", compare_a0p3ret)
np.savetxt("compare_a0p1ret.txt", compare_a0p1ret)
