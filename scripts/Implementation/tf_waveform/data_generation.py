import numpy as np
from few.waveform import FastKerrEccentricEquatorialFlux
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI, MTSUN_SI
from few.utils.utility import get_p_at_t
from scipy.signal.windows import tukey
import h5py

wf = FastKerrEccentricEquatorialFlux()
traj = EMRIInspiral(func="KerrEccEqFlux")

# pars = [
#     1e6,
#     3e1,
#     0.998,
#     7.,
#     0.75,
#     1.,
#     np.pi/3,
#     0.,
# ]

# dist = 1.

# T = 12 * 3600 * 366 * 2 / YRSID_SI
# dt = 5.

# p_goal = get_p_at_t(traj, T, [pars[0], pars[1], pars[2], pars[4], pars[5]])
# pars[3] = p_goal
# print(p_goal)

# trudge = traj(*pars[:6], dt=dt, T=T)
# frequencies = traj.inspiral_generator.eval_integrator_derivative_spline(trudge[0], order=1)[:,3:] / 2 / np.pi

# wave = wf(*pars, T=T, dt=dt, dist=dist, eps=1e-6)

# pars_last_hr = np.array(traj(*pars[:6], dt=dt, T=T - 3600 / YRSID_SI))[:,-1]

# pars = [
#     1e6,
#     3e1,
#     0.998,
#     pars_last_hr[1],
#     pars_last_hr[2],
#     1.,
#     np.pi/3,
#     0.,
# ]

# dt_ups = 0.5

# wave_ups = wf(*pars, T=T, dt=dt_ups, dist=dist, eps=1e-6)


pars = [
    1e5,
    3e1,
    0.998,
    7.,
    0.85,
    -1.,
    np.pi/3,
    0.,
]

dist = 1.

T = 12 * 3600 * 366 * 2 / YRSID_SI
dt = 5.

p_goal = get_p_at_t(traj, T, [pars[0], pars[1], pars[2], pars[4], pars[5]])
pars[3] = p_goal
print(p_goal)

wave2 = wf(*pars, T=T, dt=dt, eps=1e-6, dist=dist)

pars_last_hr = np.array(traj(*pars[:6], dt=dt, T=T - 3600 / YRSID_SI))[:,-1]

pars = [
    1e5,
    3e1,
    0.998,
    pars_last_hr[1],
    pars_last_hr[2],
    -1.,
    np.pi/3,
    0.,
]

dt_ups = 0.5

wave2_ups = wf(*pars, T=T, dt=dt_ups, dist=dist, eps=1e-6)

with h5py.File("waveform.h5", "w") as f:
    # f.create_dataset("prograde_waveform", data=wave)
    # f.create_dataset("prograde_waveform_last_hr_ups", data=wave_ups)
    f.create_dataset("retrograde_waveform", data=wave2)
    f.create_dataset("retrograde_waveform_last_hr_ups", data=wave2_ups)
    f.attrs["T"] = T
    f.attrs["dt"] = dt
    f.attrs["dt_ups"] = dt_ups
