#BEGINS: Functions for spherical mode projection

import numpy as np
from scipy.linalg import eig_banded

#Assuming s=-2, l>2, -l<=m<=l with waveform amplitudes in mind.
#Using l for spherical modes, j for spheroidal.

#The diagonals of the banded matrix system for the spheroidal to spherical mode mixing coefficients. 
def d(l, m, gamma):
    return (
        -2 
        + l * (1 + l) 
        - (8 * m * gamma) / (l + l**2) 
        - (1/3) * (1 + (2 * (-3 + l) * (4 + l) * (l + l**2 - 3 * m**2)) / 
        (l * (-3 + l + 4 * l**2 * (2 + l)))) * gamma**2
    )

def dk1(l, m, gamma):
    return -(
        (4 * np.sqrt(
            ((-1 + l) * (3 + l) * (1 + 2 * l) * (1 + l - m) * (1 + l + m)) /
            (3 + 2 * l)
        ) * gamma * (l * (2 + l) + m * gamma)) /
        (l * (1 + l) * (2 + l) * (1 + 2 * l))
    )

def dk2(l, m, gamma):
    sqrt_term = np.sqrt(
        ((-1 + l) * l * (3 + l) * (4 + l) * (1 + l - m) * (2 + l - m) * (1 + l + m) * (2 + l + m)) /
        ((1 + l)**2 * (2 + l)**2 * (1 + 2 * l) * (3 + 2 * l)**2 * (5 + 2 * l))
    )
    return -sqrt_term * gamma**2

# Compute mode mixing coefficients
def spheroidal_in_spherical(j, m, gamma, num):
    lmin = max(2, abs(m))
    
    a_band = np.zeros((3, num))
    for i in range(num):
        a_band[0, i] = d(lmin + i, m, gamma)
        a_band[1, i] = dk1(lmin + i, m, gamma)
        a_band[2, i] = dk2(lmin + i, m, gamma)
        
    eigenvals, eigenvects = eig_banded(a_band, lower=True)
    eigenvect = np.transpose(eigenvects)
    
    # enforce sign convention that j=l mode is positive - adapted from Zach's spheroidal package as this is neater than toolkit way.
    sign = np.sign(eigenvect[int(j - lmin)][int(j - lmin)])
    
    return sign * eigenvect[int(j - lmin)]

# Takes a list of spheroidal modes (j,m) at a point in parameter space from j=max(2, abs(m)) to some jmax. Computes the (l,m) spherical mode by summing the mode mixing formula up to jmax.
def sphericalmodefromspheroidal(l, m, gamma, spheroidalmodes):
    
    if isinstance(spheroidalmodes,np.ndarray) and spheroidalmodes.ndim==1:
        n = len(spheroidalmodes)
    else:
        print("Invalid. spheroidalmodes must be a one dimensional numpy array of modes (beginning at spheroidal l=2) for a given m value. ")
        return None

    ljmin = max(2, abs(m))
    jmax = n + 1

    #Vary number of terms by accuracy requirements? For now compute minumum number of terms +9:
    num_min = l-(ljmin-1)
    num=num_min + 9

    sphericalmode = 0

    #Ensuring we compute enough coefficients to include the relevant l mode based of automatic number of terms included.
    for i in range(n):
        sphericalmode += (spheroidal_in_spherical(ljmin+i, m, gamma, num)[l-ljmin])*spheroidalmodes[i]

    return sphericalmode

#ENDS: Functions for spherical mode projection



#BEGINS: Data for plot style 1: p=100, relative difference plot in a,e plane

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=0 containting entries of [e,a,Re[Amp],Im[Amp]] for p=100, prograde.
PNp1l2m2n0 = np.loadtxt('5PN_e10_mathematica/p1l2m2n0.csv', delimiter=',')
PNp1l3m2n0 = np.loadtxt('5PN_e10_mathematica/p1l3m2n0.csv', delimiter=',')
PNp1l4m2n0 = np.loadtxt('5PN_e10_mathematica/p1l4m2n0.csv', delimiter=',')
PNp1l5m2n0 = np.loadtxt('5PN_e10_mathematica/p1l5m2n0.csv', delimiter=',')
PNp1l6m2n0 = np.loadtxt('5PN_e10_mathematica/p1l6m2n0.csv', delimiter=',')

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=1 containting entries of [e,a,Re[Amp],Im[Amp]] for p=100, prograde.
PNp1l2m2n1 = np.loadtxt('5PN_e10_mathematica/p1l2m2n1.csv', delimiter=',')
PNp1l3m2n1 = np.loadtxt('5PN_e10_mathematica/p1l3m2n1.csv', delimiter=',')
PNp1l4m2n1 = np.loadtxt('5PN_e10_mathematica/p1l4m2n1.csv', delimiter=',')
PNp1l5m2n1 = np.loadtxt('5PN_e10_mathematica/p1l5m2n1.csv', delimiter=',')
PNp1l6m2n1 = np.loadtxt('5PN_e10_mathematica/p1l6m2n1.csv', delimiter=',')

from few.utils.geodesic import get_fundamental_frequencies

#l=2,m=2,n=0 spherical mode:
PNp1l2m2n0spherical=np.zeros(np.shape(PNp1l2m2n0));

for i in range(0,np.shape(PNp1l2m2n0)[0]):
    inputmodesRe=np.array([PNp1l2m2n0[i][2],PNp1l3m2n0[i][2],PNp1l4m2n0[i][2],PNp1l5m2n0[i][2],PNp1l6m2n0[i][2]])
    inputmodesIm=np.array([PNp1l2m2n0[i][3],PNp1l3m2n0[i][3],PNp1l4m2n0[i][3],PNp1l5m2n0[i][3],PNp1l6m2n0[i][3]])

    a=PNp1l2m2n0[i][1]
    p=float(100)
    e=PNp1l2m2n0[i][0]
    x=1
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, p, e, x)
    omega=2*phifreq #+0*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp1l2m2n0spherical[i][0]=PNp1l2m2n0[i][0]
    PNp1l2m2n0spherical[i][1]=PNp1l2m2n0[i][1]

    PNp1l2m2n0spherical[i][2]=outputmodeRe
    PNp1l2m2n0spherical[i][3]=outputmodeIm   


#l=2,m=2,n=1 spherical mode:
PNp1l2m2n1spherical=np.zeros(np.shape(PNp1l2m2n1));

for i in range(0,np.shape(PNp1l2m2n1)[0]):
    inputmodesRe=np.array([PNp1l2m2n1[i][2],PNp1l3m2n1[i][2],PNp1l4m2n1[i][2],PNp1l5m2n1[i][2],PNp1l6m2n1[i][2]])
    inputmodesIm=np.array([PNp1l2m2n1[i][3],PNp1l3m2n1[i][3],PNp1l4m2n1[i][3],PNp1l5m2n1[i][3],PNp1l6m2n1[i][3]])

    a=PNp1l2m2n1[i][1]
    p=float(100)
    e=PNp1l2m2n1[i][0]
    x=1
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, p, e, x)
    omega=2*phifreq +1*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp1l2m2n1spherical[i][0]=e
    PNp1l2m2n1spherical[i][1]=a

    PNp1l2m2n1spherical[i][2]=outputmodeRe
    PNp1l2m2n1spherical[i][3]=outputmodeIm 

from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
amp_module = AmpInterpKerrEccEq()

def compareamps(dataPN,n):
    
    length=dataPN.shape[0]

    reldiffs= np.zeros((length,9))
    for i in range(length):
        
        e=dataPN[i][0]
        a=dataPN[i][1]

        ampval = amp_module(a, float(100), e, 1, specific_modes=[(2, 2, n)])[(2,2,n)][0]
        
        reampPN=-dataPN[i][2] #Extra minus sign it seems from convention
        imampPN=-dataPN[i][3] #Extra minus sign it seems from convention
        
        reampfew=np.real(ampval)
        imampfew=np.imag(ampval)

        rediff=np.log10(abs(1-(reampfew/reampPN)))
        imdiff=np.log10(abs(1-(imampfew/imampPN)))
        magdiff = np.log10(np.abs(1 - (reampfew + 1j*imampfew) / (reampPN + 1j*imampPN)))

        reldiffs[i][0]=e
        reldiffs[i][1]=a
        reldiffs[i][2]=rediff
        reldiffs[i][3]=imdiff
        reldiffs[i][4]=magdiff

        reldiffs[i][5]=reampfew #Store both PN amps and few amps for inspection if need be
        reldiffs[i][6]=imampfew
        reldiffs[i][7]=reampPN
        reldiffs[i][8]=imampPN


    return reldiffs

p1l2m2n0diffs=compareamps(PNp1l2m2n0spherical,0)
p1l2m2n1diffs=compareamps(PNp1l2m2n1spherical,1)

np.savetxt("p1l2m2n0diffs.txt", p1l2m2n0diffs)
np.savetxt("p1l2m2n1diffs.txt", p1l2m2n1diffs)

#ENDS: Data for plot style 1: p=100, relative difference plot in a,e plane



#BEGINS:Data for plot style 2: relative difference plot in p,e plane, large ps, a=0.998, prograde

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=0 containting entries of [p,e,Re[Amp],Im[Amp]] for a=0.998, prograde.
PNp2l2m2n0 = np.loadtxt('5PN_e10_mathematica/p2l2m2n0.csv', delimiter=',')
PNp2l3m2n0 = np.loadtxt('5PN_e10_mathematica/p2l3m2n0.csv', delimiter=',')
PNp2l4m2n0 = np.loadtxt('5PN_e10_mathematica/p2l4m2n0.csv', delimiter=',')
PNp2l5m2n0 = np.loadtxt('5PN_e10_mathematica/p2l5m2n0.csv', delimiter=',')
PNp2l6m2n0 = np.loadtxt('5PN_e10_mathematica/p2l6m2n0.csv', delimiter=',')

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=1 containting entries of [p,e,Re[Amp],Im[Amp]] for a=0.998. prograde.
PNp2l2m2n1 = np.loadtxt('5PN_e10_mathematica/p2l2m2n1.csv', delimiter=',')
PNp2l3m2n1 = np.loadtxt('5PN_e10_mathematica/p2l3m2n1.csv', delimiter=',')
PNp2l4m2n1 = np.loadtxt('5PN_e10_mathematica/p2l4m2n1.csv', delimiter=',')
PNp2l5m2n1 = np.loadtxt('5PN_e10_mathematica/p2l5m2n1.csv', delimiter=',')
PNp2l6m2n1 = np.loadtxt('5PN_e10_mathematica/p2l6m2n1.csv', delimiter=',')

#Array for given spheroidal l mode (4,5,6,7,8,9) ,m=4,n=2 containting entries of [p,e,Re[Amp],Im[Amp]] for a=0.998. prograde.
PNp2l4m4n2 = np.loadtxt('5PN_e10_mathematica/p2l4m4n2.csv', delimiter=',')
PNp2l5m4n2 = np.loadtxt('5PN_e10_mathematica/p2l5m4n2.csv', delimiter=',')
PNp2l6m4n2 = np.loadtxt('5PN_e10_mathematica/p2l6m4n2.csv', delimiter=',')
PNp2l7m4n2 = np.loadtxt('5PN_e10_mathematica/p2l7m4n2.csv', delimiter=',')
PNp2l8m4n2 = np.loadtxt('5PN_e10_mathematica/p2l8m4n2.csv', delimiter=',')
PNp2l9m4n2 = np.loadtxt('5PN_e10_mathematica/p2l9m4n2.csv', delimiter=',')

#l=2,m=2,n=0 spherical mode:
PNp2l2m2n0spherical=np.zeros(np.shape(PNp2l2m2n0));

for i in range(0,np.shape(PNp2l2m2n0)[0]):
    inputmodesRe=np.array([PNp2l2m2n0[i][2],PNp2l3m2n0[i][2],PNp2l4m2n0[i][2],PNp2l5m2n0[i][2],PNp2l6m2n0[i][2]])
    inputmodesIm=np.array([PNp2l2m2n0[i][3],PNp2l3m2n0[i][3],PNp2l4m2n0[i][3],PNp2l5m2n0[i][3],PNp2l6m2n0[i][3]])

    a=float(0.998)
    p=PNp2l2m2n0[i][0]
    e=PNp2l2m2n0[i][1]
    x=1
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, p, e, x)
    omega=2*phifreq #+0*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp2l2m2n0spherical[i][0]=p
    PNp2l2m2n0spherical[i][1]=e

    PNp2l2m2n0spherical[i][2]=outputmodeRe
    PNp2l2m2n0spherical[i][3]=outputmodeIm   


#l=2,m=2,n=1 spherical mode:
PNp2l2m2n1spherical=np.zeros(np.shape(PNp2l2m2n1));

for i in range(0,np.shape(PNp2l2m2n1)[0]):
    inputmodesRe=np.array([PNp2l2m2n1[i][2],PNp2l3m2n1[i][2],PNp2l4m2n1[i][2],PNp2l5m2n1[i][2],PNp2l6m2n1[i][2]])
    inputmodesIm=np.array([PNp2l2m2n1[i][3],PNp2l3m2n1[i][3],PNp2l4m2n1[i][3],PNp2l5m2n1[i][3],PNp2l6m2n1[i][3]])

    a=float(0.998)
    p=PNp2l2m2n1[i][0]
    e=PNp2l2m2n1[i][1]
    x=1
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, p, e, x)
    omega=2*phifreq +1*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp2l2m2n1spherical[i][0]=p
    PNp2l2m2n1spherical[i][1]=e

    PNp2l2m2n1spherical[i][2]=outputmodeRe
    PNp2l2m2n1spherical[i][3]=outputmodeIm 

#l=6,m=4,n=2 spherical mode:
PNp2l6m4n2spherical=np.zeros(np.shape(PNp2l6m4n2));

for i in range(0,np.shape(PNp2l6m4n2)[0]):
    inputmodesRe=np.array([PNp2l4m4n2[i][2],PNp2l5m4n2[i][2],PNp2l6m4n2[i][2],PNp2l7m4n2[i][2],PNp2l8m4n2[i][2],PNp2l9m4n2[i][2]])
    inputmodesIm=np.array([PNp2l4m4n2[i][3],PNp2l5m4n2[i][3],PNp2l6m4n2[i][3],PNp2l7m4n2[i][3],PNp2l8m4n2[i][3],PNp2l9m4n2[i][3]])

    a=float(0.998)
    p=PNp2l6m4n2[i][0]
    e=PNp2l6m4n2[i][1]
    x=1
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, p, e, x)
    omega=4*phifreq +2*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(6, 4, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(6, 4, gamma, inputmodesIm)

    PNp2l6m4n2spherical[i][0]=p
    PNp2l6m4n2spherical[i][1]=e

    PNp2l6m4n2spherical[i][2]=outputmodeRe
    PNp2l6m4n2spherical[i][3]=outputmodeIm 

def compareamps2(dataPN,l,m,n):
    
    length=dataPN.shape[0]

    reldiffs= np.zeros((length,9))
    for i in range(length):

        p=dataPN[i][0]
        e=dataPN[i][1]
        a=float(0.998)

        ampval = amp_module(a, p, e, 1, specific_modes=[(l, m, n)])[(l,m,n)][0]
        
        reampPN=-dataPN[i][2] #Extra minus sign it seems from convention
        imampPN=-dataPN[i][3] #Extra minus sign it seems from convention
        
        reampfew=np.real(ampval)
        imampfew=np.imag(ampval)

        rediff=np.log10(abs(1-(reampfew/reampPN)))
        imdiff=np.log10(abs(1-(imampfew/imampPN)))
        magdiff = np.log10(np.abs(1 - (reampfew + 1j*imampfew) / (reampPN + 1j*imampPN)))

        reldiffs[i][0]=p
        reldiffs[i][1]=e
        reldiffs[i][2]=rediff
        reldiffs[i][3]=imdiff
        reldiffs[i][4]=magdiff

        reldiffs[i][5]=reampfew #Store both PN amps and few amps for inspection if need be
        reldiffs[i][6]=imampfew
        reldiffs[i][7]=reampPN
        reldiffs[i][8]=imampPN

    return reldiffs

p2l2m2n0diffs=compareamps2(PNp2l2m2n0spherical,2,2,0)
p2l2m2n1diffs=compareamps2(PNp2l2m2n1spherical,2,2,1)
p2l6m4n2diffs=compareamps2(PNp2l6m4n2spherical,6,4,2)

np.savetxt("p2l2m2n0diffs.txt", p2l2m2n0diffs)
np.savetxt("p2l2m2n1diffs.txt", p2l2m2n1diffs)
np.savetxt("p2l6m4n2diffs.txt", p2l6m4n2diffs)

#ENDS: Data for plot style 2


#BEGINS:Data for plot style 3: same as plot 2 but for smaller ps

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=0 containting entries of [p-plso,e,Re[Amp],Im[Amp]] for a=0.998, prograde.
PNp3l2m2n0 = np.loadtxt('5PN_e10_mathematica/p3l2m2n0.csv', delimiter=',')
PNp3l3m2n0 = np.loadtxt('5PN_e10_mathematica/p3l3m2n0.csv', delimiter=',')
PNp3l4m2n0 = np.loadtxt('5PN_e10_mathematica/p3l4m2n0.csv', delimiter=',')
PNp3l5m2n0 = np.loadtxt('5PN_e10_mathematica/p3l5m2n0.csv', delimiter=',')
PNp3l6m2n0 = np.loadtxt('5PN_e10_mathematica/p3l6m2n0.csv', delimiter=',')

#Array for given spheroidal l mode (2,3,4,5,6) ,m=2,n=1 containting entries of [p-plso,e,Re[Amp],Im[Amp]] for a=0.998. prograde.
PNp3l2m2n1 = np.loadtxt('5PN_e10_mathematica/p3l2m2n1.csv', delimiter=',')
PNp3l3m2n1 = np.loadtxt('5PN_e10_mathematica/p3l3m2n1.csv', delimiter=',')
PNp3l4m2n1 = np.loadtxt('5PN_e10_mathematica/p3l4m2n1.csv', delimiter=',')
PNp3l5m2n1 = np.loadtxt('5PN_e10_mathematica/p3l5m2n1.csv', delimiter=',')
PNp3l6m2n1 = np.loadtxt('5PN_e10_mathematica/p3l6m2n1.csv', delimiter=',')

from few.utils.geodesic import get_separatrix

#l=2,m=2,n=0 spherical mode:
PNp3l2m2n0spherical=np.zeros(np.shape(PNp3l2m2n0));

for i in range(0,np.shape(PNp3l2m2n0)[0]):
    inputmodesRe=np.array([PNp3l2m2n0[i][2],PNp3l3m2n0[i][2],PNp3l4m2n0[i][2],PNp3l5m2n0[i][2],PNp3l6m2n0[i][2]])
    inputmodesIm=np.array([PNp3l2m2n0[i][3],PNp3l3m2n0[i][3],PNp3l4m2n0[i][3],PNp3l5m2n0[i][3],PNp3l6m2n0[i][3]])

    a=float(0.998)
    dp=PNp3l2m2n0[i][0]
    e=PNp3l2m2n0[i][1]
    x=1
    
    plso=get_separatrix(a,e,x)
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, plso+dp, e, x)
    omega=2*phifreq #+0*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp3l2m2n0spherical[i][0]=dp
    PNp3l2m2n0spherical[i][1]=e

    PNp3l2m2n0spherical[i][2]=outputmodeRe
    PNp3l2m2n0spherical[i][3]=outputmodeIm   


#l=2,m=2,n=1 spherical mode:
PNp3l2m2n1spherical=np.zeros(np.shape(PNp3l2m2n1));

for i in range(0,np.shape(PNp3l2m2n1)[0]):
    inputmodesRe=np.array([PNp3l2m2n1[i][2],PNp3l3m2n1[i][2],PNp3l4m2n1[i][2],PNp3l5m2n1[i][2],PNp3l6m2n1[i][2]])
    inputmodesIm=np.array([PNp3l2m2n1[i][3],PNp3l3m2n1[i][3],PNp3l4m2n1[i][3],PNp3l5m2n1[i][3],PNp3l6m2n1[i][3]])

    a=float(0.998)
    dp=PNp3l2m2n1[i][0]
    e=PNp3l2m2n1[i][1]
    x=1

    plso=get_separatrix(a,e,x)
    
    #spheroidicity gamma=a omega
    phifreq, thetafreq, radfreq=get_fundamental_frequencies(a, plso+dp, e, x)
    omega=2*phifreq +1*radfreq
    gamma=a*omega
    
    outputmodeRe=sphericalmodefromspheroidal(2, 2, gamma, inputmodesRe)
    outputmodeIm=sphericalmodefromspheroidal(2, 2, gamma, inputmodesIm)

    PNp3l2m2n1spherical[i][0]=dp
    PNp3l2m2n1spherical[i][1]=e

    PNp3l2m2n1spherical[i][2]=outputmodeRe
    PNp3l2m2n1spherical[i][3]=outputmodeIm 

def compareamps3(dataPN,n):
    
    length=dataPN.shape[0]

    reldiffs= np.zeros((length,10))

# Attempt to call the amp function and check for exception indicating outside of interpolant range (as getting a value error somewhere, maybe separatrix finder failing?)
    def is_not_allowed_value(p,e,x):
        try:
            ampval = amp_module(a, p, e, x, specific_modes=[(2, 2, n)])[(2,2,n)][0]
            # If no ValueError, set the variable to False
            error_occurred = False
        except Exception:
            # If a ValueError occurs, set the variable to True
            error_occurred = True
        return error_occurred
            
    for i in range(length):

        dp=dataPN[i][0]
        e=dataPN[i][1]
        a=float(0.998)
        plso=get_separatrix(a,e,1)

        if is_not_allowed_value(plso+dp,e,1):
            
            reldiffs[i][2]=0
            reldiffs[i][3]=0
        
            reldiffs[i][4]=0 
            reldiffs[i][5]=0
            reldiffs[i][6]=0
            reldiffs[i][7]=0
            reldiffs[i][8]=0
            

        else:
            ampval = amp_module(a, plso+dp, e, 1, specific_modes=[(2, 2, n)])[(2,2,n)][0]
        
            reampPN=-dataPN[i][2] #Extra minus sign it seems from convention
            imampPN=-dataPN[i][3] #Extra minus sign it seems from convention
        
            reampfew=np.real(ampval)
            imampfew=np.imag(ampval)

            rediff=np.log10(abs(1-(reampfew/reampPN)))
            imdiff=np.log10(abs(1-(imampfew/imampPN)))
            magdiff = np.log10(np.abs(1 - (reampfew + 1j*imampfew) / (reampPN + 1j*imampPN)))

            reldiffs[i][0]=plso+dp
            reldiffs[i][1]=e
            reldiffs[i][2]=rediff
            reldiffs[i][3]=imdiff
            reldiffs[i][4]=magdiff
        
            reldiffs[i][5]=reampfew #Store both PN amps and few amps for inspection if need be
            reldiffs[i][6]=imampfew
            reldiffs[i][7]=reampPN
            reldiffs[i][8]=imampPN
            reldiffs[i][9]=plso 

    #Now filter out the entries outside of the domain of the interpolant by deleting rows where the few amps are zero
    filtered_array = reldiffs[~np.all(reldiffs[:, -6:] == 0, axis=1)]

    return filtered_array

p3l2m2n0diffs=compareamps3(PNp3l2m2n0spherical,0)
p3l2m2n1diffs=compareamps3(PNp3l2m2n1spherical,1)

np.savetxt("p3l2m2n0diffs.txt", p3l2m2n0diffs)
np.savetxt("p3l2m2n1diffs.txt", p3l2m2n1diffs)

#ENDS: Data for plot style 3