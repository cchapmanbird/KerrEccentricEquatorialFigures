import numpy as np
from few.utils.utility import p_to_y
import warnings
try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc, AmpInterpKerrEqEcc
from few.amplitude.romannet import RomanAmplitude

# initialize ROMAN class
amp = RomanAmplitude(max_init_len=10000)  # max_init_len creates memory buffers
amp_kerr = AmpInterpKerrEqEcc(max_init_len=10000)

p = np.linspace(10.0, 14.0, 100)
e = np.linspace(0.0, 0.7, 100)

p_all, e_all = np.meshgrid(p, e)

teuk_modes = amp(0., p_all.flatten(), e_all.flatten(), 1.)

amp_norm = amp.amp_norm_spline.ev(
                p_to_y(p_all.flatten(), e_all.flatten()), e_all.flatten()
            )

amp_for_norm = np.sum(
    np.abs(
        np.concatenate(
            [teuk_modes, np.conj(teuk_modes[:, amp.m0mask])],
            axis=1,
        )
    )
    ** 2,
    axis=1,
) ** (1 / 2)

factor = amp_norm / amp_for_norm
teuk_modes = teuk_modes * factor[:, np.newaxis]

# (2, 2, 0)
specific_modes = [(ll, 2, 0) for ll in range(2, 3)]

spin=0.0
# notice this returns a dictionary with keys as the mode tuple and values as the mode values at all trajectory points
kerr_teuk_modes = amp_kerr(spin, p_all.flatten(), e_all.flatten(), 1., specific_modes=specific_modes)

# we can find the index to these modes to check
inds = np.array([amp.special_index_map[lmn] for lmn in specific_modes])

first_check = np.allclose(kerr_teuk_modes[(2, 2, 0)], teuk_modes[:, inds[0]])


# plot the contours of the amplitudes
kerr_amp_to_plot = np.sum([np.abs(kerr_teuk_modes[el])**2 for el in specific_modes], axis=0)
schw_amp_to_plot = np.sum([np.abs(teuk_modes[:, inds[el]])**2 for el in range(len(specific_modes))], axis=0)
# breakpoint()
amp_to_plot = np.abs(schw_amp_to_plot - kerr_amp_to_plot).reshape((p.shape[0], e.shape[0]))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.figure()
plt.contourf(p_all, e_all, amp_to_plot, norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label(r'Amplitude  Difference')
plt.xlabel("$p$")
plt.ylabel("$e$")
plt.title(r"Amplitude $\sum_l A_{lmn}$ " + f"difference \n between Roman and Bicubic \n of m=2, n=0 mode for spin={spin}")
plt.tight_layout()
plt.savefig("amplitude_220.png")