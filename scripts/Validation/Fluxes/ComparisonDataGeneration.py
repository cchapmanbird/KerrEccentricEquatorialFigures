import numpy as np
from few.trajectory.inspiral import EMRIInspiral

from few.trajectory.ode import KerrEccEqFlux
rhsFlux = KerrEccEqFlux()

from few.trajectory.ode import PN5, SchwarzEccFlux

for name,ode_compare in [("SchwarzEccFlux", SchwarzEccFlux()), ("PN5", PN5())]:
    rhsPN = ode_compare

    M = 1e6  # Solar masses
    mu = 1e1  # Solar masses
    # At large p, spin is irrelevant (hopefully)
    a = 0.998
    if name == "SchwarzEccFlux":
        a=0.0
        ps = np.linspace(10, 40, 50)
        es = np.linspace(0.01, 0.7, 50)
    if name == "PN5":
        ps = np.linspace(20, 200, 50)
        es = np.linspace(0.01, 0.9, 50)

    
    x = 1

    rhsFlux.add_fixed_parameters(M, mu, a)

    rhsPN.add_fixed_parameters(M, mu, a)


    pdotsRelDiff = np.zeros((len(ps), len(es)))
    edotsRelDiff = np.zeros((len(ps), len(es)))

    FluxRelDiff = np.zeros((len(ps), len(es),2))

    for i in range(len(es)):
        for j in range(len(ps)):
            pdot, edot, xIdot, Omega_phi, Omega_theta, Omega_r = rhsFlux([ps[j], es[i], x])
            pdotPN, edotPN, xIdotPN, Omega_phiPN, Omega_thetaPN, Omega_rPN = rhsPN([ps[j], es[i], x])
            pdotsRelDiff[i, j] = np.log10(abs(1 - pdot/pdotPN))
            edotsRelDiff[i, j] = np.log10(abs(1 - edot/edotPN))


    np.savetxt(name+ "_ComparisonPdot.txt", pdotsRelDiff)
    np.savetxt(name+"_ComparisonEdot.txt", edotsRelDiff)
    np.savetxt(name+"_ComparisonPs.txt", ps)
    np.savetxt(name+"_ComparisonEs.txt", es)
