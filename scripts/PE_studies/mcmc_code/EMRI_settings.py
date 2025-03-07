###
# This is the main file that should be edited. 
###
try:
    import cupy as cp
    import numpy as np
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False


# Extreme point for TDI2, SNR = 20. EMRI
# M = 1e6; mu = 10; a = 0.998; p0 = 7.7275; e0 = 0.73; x_I0 = 1.0
# dist = 7.66; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# IMRI source, TDI2, SNR = 30,

#M = 1e5; mu = 70; a = 0.998; p0 = 44.321; e0 = 0.5; x_I0 = 1.0
#dist = 7.28; 
#qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
#Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

#delta_t = 5.0; T = 2.0;




# MEGA IMRI -- Pints. TDI2, SNR = 
# M = 1e7; mu = 100_000; a = 0.95; p0 = 23.6015; e0 = 0.85; x_I0 = 1.0
# dist = 7.25; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 4.0

# delta_t = 10.0; T = 2.0;


# Extreme point for TDI2, SNR = 20. EMRI -- Simple source 

# M = 1e6; mu = 10; a = 0.8; p0 = 8.8; e0 = 0.3; x_I0 = 1.0
# dist = 1.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# delta_t = 10.0; T = 2.0;

#74.9418273973385

# Light IMRI, SNR = 443
# M = 1e5; mu = 1e3; a = 0.95; p0 = 74.94184; e0 = 0.85; x_I0 = 1.0
# dist = 2.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# delta_t = 5.0; T = 2.0;

# bullshit IMRI, SNR = 212 
# M = 1e5; mu = 1e4; a = 0.95; p0 = 133.4623; e0 = 0.85; x_I0 = 1.0
# dist = 10.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# Huge mass-ratio q = 1e-6, SNR = 30
M = 1e7; mu = 1e1; a = 0.998; p0 = 2.12; e0 = 0.425; x_I0 = 1.0
dist = 5.465; 
qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

delta_t = 10.0; T = 2.0;