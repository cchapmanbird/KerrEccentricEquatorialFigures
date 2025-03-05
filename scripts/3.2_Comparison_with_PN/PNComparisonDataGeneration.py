import numpy as np
from few.trajectory.inspiral import EMRIInspiral

from few.trajectory.ode import KerrEccEqFlux
rhsFlux = KerrEccEqFlux()

from few.trajectory.ode import PN5
rhsPN = PN5()

M = 1e6  # Solar masses
mu = 1e1  # Solar masses
# At large p, spin is irrelevant (hopefully)
a = 0.998
x = 1

rhsFlux.add_fixed_parameters(M, mu, a)

rhsPN.add_fixed_parameters(M, mu, a)


ps = np.linspace(40, 200, 50)
es = np.linspace(0.01, 0.9, 50)

pdotsRelDiff = np.zeros((len(ps), len(es)))
edotsRelDiff = np.zeros((len(ps), len(es)))

FluxRelDiff = np.zeros((len(ps), len(es),2))

for i in range(len(es)):
    for j in range(len(ps)):
        pdot, edot, xIdot, Omega_phi, Omega_theta, Omega_r = rhsFlux([ps[j], es[i], x])
        pdotPN, edotPN, xIdotPN, Omega_phiPN, Omega_thetaPN, Omega_rPN = rhsPN([ps[j], es[i], x])
        pdotsRelDiff[i, j] = np.log10(abs(1 - pdot/pdotPN))
        edotsRelDiff[i, j] = np.log10(abs(1 - edot/edotPN))


np.savetxt("PNComparisonPdot.txt", pdotsRelDiff)
np.savetxt("PNComparisonEdot.txt", edotsRelDiff)
np.savetxt("PNComparisonPs.txt", ps)
np.savetxt("PNComparisonEs.txt", es)
