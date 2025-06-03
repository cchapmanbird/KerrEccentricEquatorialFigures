import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq, AmpInterpSchwarzEcc
from few.utils.mappings.pn import xI_to_Y
from few.amplitude.romannet import RomanAmplitude
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.summation.directmodesum import DirectModeSum
from few.summation.aakwave import AAKSummation
from few.utils.constants import MRSUN_SI, Gpc
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum

from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux
from tqdm import tqdm

from few.utils.baseclasses import (
    SchwarzschildEccentric,
    KerrEccentricEquatorial,
    BackendLike,
)
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase, BackendLike

from few.utils.mappings.schwarzecc import (
    schwarzecc_p_to_y,
)
from few.utils.ylm import GetYlms
from few.utils.constants import MRSUN_SI, Gpc
from few.utils.citations import REFERENCE

from typing import Union, Optional, TypeVar, Generic
import time

from few.utils.utility import get_p_at_t



InspiralModule = TypeVar("InspiralModule", bound=ParallelModuleBase)
"""Used for type hinting the Inspiral generator classes."""

AmplitudeModule = TypeVar("AmplitudeModule", bound=ParallelModuleBase)
"""Used for type hinting the Amplitude generator classes."""

SumModule = TypeVar("SumModule", bound=ParallelModuleBase)
"""Used for type hinting the Sum classes."""

ModeSelectorModule = TypeVar("ModeSelectorModule", bound=ParallelModuleBase)
"""Used for type hinting the Mode selector classes."""

WaveformModule = TypeVar("WaveformModule", bound=ParallelModuleBase)
"""Used for type hinting Waveform Generator classes"""

class TimingBase(
    ParallelModuleBase,
    Generic[InspiralModule, AmplitudeModule, SumModule, ModeSelectorModule],
):
    """Base class for waveforms built with amplitudes expressed in a spherical harmonic basis.

    This class contains the methods required to build the core waveform for Kerr equatorial eccentric
    (to be upgraded to Kerr generic once that is available). Stock waveform classes constructed in
    this basis can subclass this class and implement their own "__call__" method to fill in the
    relevant data.

    Args:
        inspiral_module: Class object representing the module for creating the inspiral.
            This returns the phases and orbital parameters. See :ref:`trajectory-label`.
        amplitude_module: Class object representing the module for creating the amplitudes.
            This returns the complex amplitudes of the modes. See :ref:`amplitude-label`.
        sum_module: Class object representing the module for summing the final waveform from the
            amplitude and phase information. See :ref:`summation-label`.
        mode_selector_module: Class object representing the module for selecting modes that contribute
            to the waveform. See :ref:`utilities-label`.
        inspiral_kwargs: Optional kwargs to pass to the inspiral generator. Default is {}.
        amplitude_kwargs: Optional kwargs to pass to the amplitude generator. Default is {}.
        sum_kwargs: Optional kwargs to pass to the sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the Ylm generator. Default is {}.
        mode_selector_kwargs: Optional kwargs to pass to the mode selector module. Default is {}.
        normalize_amps: If True, normalize the amplitudes at each step of the trajectory. This option should
            be used alongside ROMAN networks that have been trained with normalized amplitudes.
            Default is False.
    """

    normalize_amps: bool
    """Whether to normalize amplitudes to flux at each step from trajectory"""

    inspiral_kwargs: dict
    """Keyword arguments passed to the inspiral generator call function"""

    inspiral_generator: InspiralModule
    """Instance of the trajectory module"""

    amplitude_generator: AmplitudeModule
    """Instance of the amplitude module"""

    create_waveform: SumModule
    """Instance of the summation module"""

    ylm_gen: GetYlms
    """Instance of the Ylm module"""

    mode_selector: ModeSelectorModule
    """Instance of the mode selector module"""

    def __init__(
        self,
        /,  # force use of keyword arguments for readability
        inspiral_module: type[InspiralModule],
        amplitude_module: type[AmplitudeModule],
        sum_module: type[SumModule],
        mode_selector_module: type[ModeSelectorModule],
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        normalize_amps: bool = False,
        force_backend: BackendLike = None,
    ):
        ParallelModuleBase.__init__(self, force_backend=force_backend)

        self.normalize_amps = normalize_amps
        self.inspiral_kwargs = {} if inspiral_kwargs is None else inspiral_kwargs
        self.inspiral_generator = inspiral_module(
            **self.inspiral_kwargs
        )  # The inspiral generator does not rely on backend adjustement

        self.amplitude_generator = self.build_with_same_backend(
            amplitude_module, kwargs=amplitude_kwargs
        )
        self.create_waveform = self.build_with_same_backend(
            sum_module, kwargs=sum_kwargs
        )
        self.ylm_gen = self.build_with_same_backend(GetYlms, kwargs=Ylm_kwargs)

        # selecting modes that contribute at threshold to the waveform
        self.mode_selector = self.build_with_same_backend(
            mode_selector_module,
            args=[self.l_arr_no_mask, self.m_arr_no_mask, self.n_arr_no_mask],
            kwargs=mode_selector_kwargs,
        )

    def _generate_waveform(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        xI0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        dist: Optional[float] = None,
        Phi_phi0: float = 0.0,
        Phi_r0: float = 0.0,
        dt: float = 10.0,
        T: float = 1.0,
        mode_selection_threshold: float = 1e-5,
        show_progress: bool = False,
        batch_size: int = -1,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
        include_minus_m: bool = True,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Call function for waveform models built in the spherical harmonic basis.

        This function will take input parameters and produce waveforms. It will use all of the modules preloaded to
        compute desired outputs.

        args:
            M: Mass of larger black hole in solar masses.
            mu: Mass of compact object in solar masses.
            a: Dimensionless spin parameter of larger black hole.
            p0: Initial (osculating) semilatus rectum of inspiral trajectory.
            e0: Initial (osculating) eccentricity of inspiral trajectory.
            theta: Polar viewing angle in radians (:math:`-\pi/2\leq\Theta\leq\pi/2`).
            phi: Azimuthal viewing angle in radians.
            *args: extra args for trajectory model.
            dist: Luminosity distance in Gpc. Default is None. If None,
                will return source frame.
            Phi_phi0: Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_r0: Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            dt: Time between samples in seconds (inverse of
                sampling frequency). Default is 10.0.
            T: Total observation time in years.
                Default is 1.0.
            mode_selection_threshold: Controls the fractional accuracy during mode
                filtering. Raising this parameter will remove modes. Lowering
                this parameter will add modes. Default that gives a good overalp
                is 1e-5.
            show_progress: If True, show progress through
                amplitude/waveform batches using
                `tqdm <https://tqdm.github.io/>`_. Default is False.
            batch_size (int, optional): If less than 0, create the waveform
                without batching. If greater than zero, create the waveform
                batching in sizes of batch_size. Default is -1.
            mode_selection: Determines the type of mode
                filtering to perform. If None, perform our base mode filtering
                with mode_selection_threshold as the fractional accuracy on the total power.
                If 'all', it will run all modes without filtering. If a list of
                tuples (or lists) of mode indices
                (e.g. [(:math:`l_1,m_1,n_1`), (:math:`l_2,m_2,n_2`)]) is
                provided, it will return those modes combined into a
                single waveform.
            include_minus_m: If True, then include -m modes when
                computing a mode with m. This only effects modes if :code:`mode_selection`
                is a list of specific modes. Default is True.

        Returns:
            The output waveform.

        Raises:
            ValueError: user selections are not allowed.

        """

        if xI0 < 0.0:
            a = -a
            xI0 = -xI0
            theta = np.pi - theta
            phi = -phi

        if dist is not None:
            if dist <= 0.0:
                raise ValueError("Luminosity distance must be greater than zero.")

            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

        else:
            dist_dimensionless = 1.0

        # makes sure viewing angles are allowable
        theta, phi = self.sanity_check_viewing_angles(theta, phi)

        a, xI0 = self.sanity_check_init(M, mu, a, p0, e0, xI0)
        
        # Time the trajectory generation
        start_time = time.perf_counter()
        
        # get trajectory
        (t, p, e, xI, Phi_phi, Phi_theta, Phi_r) = self.inspiral_generator(
            M,
            mu,
            a,
            p0,
            e0,
            xI0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=0.0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )
        
        # Log the time taken
        trajectory_time = time.perf_counter() - start_time
        print(f"Trajectory generation took {trajectory_time} seconds.")
        
        start_time = time.perf_counter()
        # makes sure p and e are generally within the model
        self.sanity_check_traj(a, p, e, xI)
     
        if self.normalize_amps:
            # get the vector norm
            amp_norm = self.amplitude_generator.amp_norm_spline.ev(
                schwarzecc_p_to_y(p, e), e
            )  # TODO: handle this grid parameter change, fix to Schwarzschild for now
            amp_norm = self.xp.asarray(amp_norm)

        self.end_time = t[-1]

        # convert for gpu
        t = self.xp.asarray(t)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        xI = self.xp.asarray(xI)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_theta = self.xp.asarray(Phi_theta)
        Phi_r = self.xp.asarray(Phi_r)

        # get ylms only for unique (l,m) pairs
        # then expand to all (lmn with self.inverse_lm)
        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi)[self.inverse_lm]
        # if mode selector is predictive, run now to avoid generating amplitudes that are not required
        if self.mode_selector.is_predictive:
            # overwrites mode_selection so it's now a list of modes to keep, ready to feed into amplitudes
            mode_selection = self.mode_selector(
                M, mu, a * xI0, p0, e0, 1.0, theta, phi, T, mode_selection_threshold
            )  # TODO: update this if more arguments are required

        # split into batches

        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [self.xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = self.xp.split(self.xp.arange(len(t)), split_inds)

        # select tqdm if user wants to see progress
        iterator = enumerate(inds_split_all)
        iterator = (
            tqdm(iterator, desc="time batch", total=len(inds_split_all))
            if show_progress
            else iterator
        )

        for i, inds_in in iterator:
            # get subsections of the arrays for each batch
            t_temp = t[inds_in]
            p_temp = p[inds_in]
            e_temp = e[inds_in]
            xI_temp = xI[inds_in]
            Phi_phi_temp = Phi_phi[inds_in]
            Phi_theta_temp = Phi_theta[inds_in]
            Phi_r_temp = Phi_r[inds_in]

            if self.normalize_amps:
                amp_norm_temp = amp_norm[inds_in]

            # if we aren't requesting a subset of modes, compute them all now
            if not isinstance(mode_selection, (list, self.xp.ndarray)):
                # amplitudes
                teuk_modes = self.xp.asarray(
                    self.amplitude_generator(a, p_temp, e_temp, xI_temp)
                )

                # normalize by flux produced in trajectory
                if self.normalize_amps:
                    amp_for_norm = self.xp.sum(
                        self.xp.abs(
                            self.xp.concatenate(
                                [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])],
                                axis=1,
                            )
                        )
                        ** 2,
                        axis=1,
                    ) ** (1 / 2)

                    # normalize
                    factor = amp_norm_temp / amp_for_norm
                    teuk_modes = teuk_modes * factor[:, np.newaxis]

            # different types of mode selection
            # sets up ylm and teuk_modes properly for summation
            if isinstance(mode_selection, str):
                # use all modes
                if mode_selection == "all":
                    self.ls = self.l_arr[: teuk_modes.shape[1]]
                    self.ms = self.m_arr[: teuk_modes.shape[1]]
                    self.ns = self.n_arr[: teuk_modes.shape[1]]

                    keep_modes = self.xp.arange(teuk_modes.shape[1])
                    temp2 = keep_modes * (keep_modes < self.num_m0) + (
                        keep_modes + self.num_m_1_up
                    ) * (keep_modes >= self.num_m0)

                    ylmkeep = self.xp.concatenate([keep_modes, temp2])
                    ylms_in = ylms[ylmkeep]
                    teuk_modes_in = teuk_modes

                else:
                    raise ValueError("If mode selection is a string, must be `all`.")

            # get a specific subset of modes
            elif isinstance(mode_selection, (list, self.xp.ndarray)):
                if len(mode_selection) == 0:
                    raise ValueError("If mode selection is a list, cannot be empty.")

                if self.normalize_amps:
                    assert isinstance(mode_selection, list)

                    # compute all amplitudes
                    teuk_modes = self.xp.asarray(
                        self.amplitude_generator(a, p_temp, e_temp, xI_temp)
                    )

                    amp_for_norm = self.xp.sum(
                        self.xp.abs(
                            self.xp.concatenate(
                                [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])],
                                axis=1,
                            )
                        )
                        ** 2,
                        axis=1,
                    ) ** (1 / 2)

                    keep_inds = self.xp.asarray(
                        [
                            self.amplitude_generator.special_index_map[md]
                            for md in mode_selection
                        ]
                    )

                    # filter modes and normalize
                    factor = amp_norm_temp / amp_for_norm
                    teuk_modes = teuk_modes[:, keep_inds] * factor[:, np.newaxis]

                else:
                    # generate only the required modes with the amplitude module
                    teuk_modes = self.amplitude_generator(
                        a, p_temp, e_temp, xI_temp, specific_modes=mode_selection
                    )

                # unpack the dictionary
                if isinstance(teuk_modes, dict):
                    teuk_modes_in = self.xp.asarray(
                        [
                            teuk_modes[lmn] if lmn[1] >= 0 else (-1)**lmn[0] * teuk_modes[lmn].conj()  # here, we reverse the symmetry transformation due to later assumptions.
                            for lmn in mode_selection
                        ]
                    ).T
                else:
                    teuk_modes_in = teuk_modes

                # for removing opposite m modes
                fix_include_ms = self.xp.full(2 * len(mode_selection), False)
                if isinstance(mode_selection, list):
                    keep_modes = self.xp.zeros(len(mode_selection), dtype=self.xp.int32)
                    for jj, lmn in enumerate(mode_selection):
                        l, m, n = tuple(lmn)

                        # keep modes only works with m>=0
                        if m < 0:
                            lmn_in = (l, -m, -n)
                        else:
                            lmn_in = (l, m, n)
                        keep_modes[jj] = self.xp.int32(self.lmn_indices[lmn_in])

                        if not include_minus_m:
                            if m > 0:
                                # minus m modes blocked
                                fix_include_ms[len(mode_selection) + jj] = True
                            elif m < 0:
                                # positive m modes blocked
                                fix_include_ms[jj] = True
                else:
                    keep_modes = mode_selection
                    m_temp = abs(self.m_arr[mode_selection])
                    for jj, m_here in enumerate(m_temp):
                        if not include_minus_m:
                            if m_here > 0:
                                # minus m modes blocked
                                fix_include_ms[len(mode_selection) + jj] = True
                            elif m_here < 0:
                                # positive m modes blocked
                                fix_include_ms[jj] = True

                self.ls = self.l_arr[keep_modes]
                self.ms = self.m_arr[keep_modes]
                self.ns = self.n_arr[keep_modes]

                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = self.xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]

                # remove modes if include_minus_m is False
                ylms_in[fix_include_ms] = 0.0 + 1j * 0.0

            # mode selection based on input module
            else:
                fund_freq_args = (
                    M,
                    0.0,
                    p_temp,
                    e_temp,
                    self.xp.zeros_like(e_temp),
                    t_temp,
                )
                modeinds = [self.l_arr, self.m_arr, self.n_arr]
                (
                    teuk_modes_in,
                    ylms_in,
                    self.ls,
                    self.ms,
                    self.ns,
                ) = self.mode_selector(
                    teuk_modes,
                    ylms,
                    modeinds,
                    fund_freq_args=fund_freq_args,
                    mode_selection_threshold=mode_selection_threshold,
                )

            # store number of modes for external information
            self.num_modes_kept = teuk_modes_in.shape[1]

            # prepare phases for summation modules
            if not self.inspiral_generator.dense_stepping:
                # prepare phase spline coefficients
                phase_information_in = self.xp.asarray(
                    self.inspiral_generator.integrator_spline_phase_coeff
                )[:, [0,2], :]

                # flip azimuthal phase for retrograde inspirals
                if a > 0:
                    phase_information_in[:, 0] *= self.xp.sign(xI0)

                if self.inspiral_generator.integrate_backwards:
                    phase_information_in[:, :, 0] += self.xp.array(
                        [Phi_phi[-1] + Phi_phi[0], Phi_r[-1] + Phi_r[0]]
                    )

                phase_t_in = (
                    self.inspiral_generator.integrator_spline_t
                )
            else:
                phase_information_in = self.xp.asarray(
                    [Phi_phi_temp, Phi_theta_temp, Phi_r_temp]
                )
                if self.inspiral_generator.integrate_backwards:
                    phase_information_in[0] += self.xp.array([Phi_phi[-1] + Phi_phi[0]])
                    phase_information_in[1] += self.xp.array(
                        [Phi_theta[-1] + Phi_theta[0]]
                    )
                    phase_information_in[2] += self.xp.array([Phi_r[-1] + Phi_r[0]])

                # flip azimuthal phase for retrograde inspirals
                if a > 0:
                    phase_information_in[0] *= self.xp.sign(xI0)

                phase_t_in = None
            # Log the time taken
            amplitude_time = time.perf_counter() - start_time
            print(f"Amplitude generation took {amplitude_time} seconds.")
            start_time = time.perf_counter()
            # create waveform
            waveform_temp = self.create_waveform(
                t_temp,
                teuk_modes_in,
                ylms_in,
                phase_t_in,
                phase_information_in,
                self.ls,
                self.ms,
                self.ns,
                M,
                a,
                p,
                e,
                xI,
                dt=dt,
                T=T,
                include_minus_m=include_minus_m,
                integrate_backwards=self.inspiral_generator.integrate_backwards,
                **kwargs,
            )
            
            # if batching, need to add the waveform (block if/else disabled since waveform is not already defined)
            # if i > 0:
            #     waveform = self.xp.concatenate([waveform, waveform_temp])

            # # return entire waveform
            # else:
            #     waveform = waveform_temp
            summation_time = time.perf_counter() - start_time
            print(f"Waveform summation took {summation_time} seconds.")
            waveform = waveform_temp

        return trajectory_time, amplitude_time, summation_time, waveform


class TimingWaveform(
    TimingBase, KerrEccentricEquatorial
):
    """Prebuilt model for fast Kerr eccentric equatorial flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.ampinterp2d.AmpInterp2D` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs : Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

    """

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}

        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = KerrEccEqFlux

        # inspiral_kwargs = augment_ODE_func_name(inspiral_kwargs)

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array(
                    [0, 1, 2, 3, 4, 6, 7, 8, 9]
                )

        KerrEccentricEquatorial.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["lmax", "nmax", "ndim"]
            },
            force_backend=force_backend,
        )
        TimingBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpKerrEccEq,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            **{
                key: value for key, value in kwargs.items() if key in ["normalize_amps"]
            },
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        return False

    def __call__(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        xI: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """
        Generate the waveform.

        Args:
            M: Mass of larger black hole in solar masses.
            mu: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            xI: Initial cosine of the inclination angle.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

        """
        return self._generate_waveform(
            M,
            mu,
            a,
            p0,
            e0,
            xI,
            theta,
            phi,
            *args,
            **kwargs,
        )



# parameters
T = 4.0  # years
dt = 5.0  # seconds
t_vec = np.arange(0, T * 366 * 24 * 3600, dt)
M = 1e7
mu = 10.
a = 0.9
traj_module = EMRIInspiral(func=KerrEccEqFlux)
e0 = 0.6
p0 = get_p_at_t(
                traj_module,
                T * 0.9999,
                [
                    M,
                    mu,
                    a,
                    e0,
                    1.0,
                ],
                index_of_a=2,
                index_of_p=3,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-6,
                rtol=8.881784197001252e-6,
            )

xI = 1.0
dist = 1.0
theta = 1.0471975511965976  # qS
phi = 1.0471975511965976    # phiS
qK = 1.0471975511965976
phiK = 1.0471975511965976
Phi_phi0 = 1.0471975511965976
Phi_theta0 = 0.0
Phi_r0 = 1.0471975511965976
#import json file

import matplotlib.pyplot as plt

for outtype in ["fd", "td"]:
    print("------------------------")
    print("output type", outtype)
    wave_timing = TimingWaveform(sum_kwargs={"output_type": outtype, "pad_output": True, "odd_len":True})#inspiral_kwargs={"err": 1e-8})

    trajectory_time, amplitude_time, summation_time, wave = wave_timing(M, mu, a,  p0, e0, 1.0, theta, phi, dist=dist, T=T, dt=dt, mode_selection_threshold=1e-5)
    
    print("Now actual timing")
    tic = time.perf_counter()
    trajectory_time, amplitude_time, summation_time, wave = wave_timing(M, mu, a,  p0, e0, 1.0, theta, phi, dist=dist, T=T, dt=dt, mode_selection_threshold=1e-5)
    toc = time.perf_counter()
    # Save speed and percentage information to a markdown file
    with open(f"results/ref_source_timing_results_{outtype}.md", "a") as f:
        f.write(f"## Output type: {outtype}\n")
        f.write(f"- **Total speed:** {toc - tic:.6f} s (sum: {trajectory_time + amplitude_time + summation_time:.6f} s)\n")
        f.write(f"- **Trajectory speed:** {trajectory_time:.6f} s ({trajectory_time/(toc - tic)*100:.2f}%)\n")
        f.write(f"- **Amplitude speed:** {amplitude_time:.6f} s ({amplitude_time/(toc - tic)*100:.2f}%)\n")
        f.write(f"- **Summation speed:** {summation_time:.6f} s ({summation_time/(toc - tic)*100:.2f}%)\n\n")
    if outtype == "td":
        plt.figure(); plt.plot(t_vec[:wave.size]/86400, wave.real.get()); plt.savefig("waveform.png")


import json
with open("lakshmi_timing_4.0yrInspErrDefault.json", "r") as f:
    test_timing = json.load(f)

for el in test_timing:
    
    M = el["parameters"]["mass_1"] 
    if M < 0.9e7:
        continue
    mu = el["parameters"]["mass_2"]
    if mu/M > 5e-6:
        continue
    print("M", M, "mu", mu, "mu/M", mu/M)
    a = el["parameters"]["spin"]
    p0 = el["parameters"]["p0"]
    e0 = el["parameters"]["e0"]
    if e0< 0.1:
        continue
    print("*******************************")
    p0_test = get_p_at_t(
                traj_module,
                T * 0.99,
                [
                    M,
                    mu,
                    a,
                    e0,
                    1.0,
                ],
                index_of_a=2,
                index_of_p=3,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=1e-10,
                rtol=1e-10,
            )
    print("p0", p0, "p0_test", p0_test)
    print("fd",el["timing_results"][1]["fd_timing"], "td", el["timing_results"][1]["td_timing"])
    for outtype in ["fd", "td"]:
        print("------------------------")
        print("output type", outtype)
        wave_timing = TimingWaveform(sum_kwargs={"output_type": outtype})#inspiral_kwargs={"err": 1e-8})

        trajectory_time, amplitude_time, summation_time, wave = wave_timing(M, mu, a,  p0, e0, 1.0, theta, phi, dist=dist, T=T, dt=dt, mode_selection_threshold=1e-5)
        
        print("Now actual timing")
        tic = time.perf_counter()
        trajectory_time, amplitude_time, summation_time, wave = wave_timing(M, mu, a,  p0, e0, 1.0, theta, phi, dist=dist, T=T, dt=dt, mode_selection_threshold=1e-5)
        toc = time.perf_counter()
        # Save speed and percentage information to a markdown file
        print(f"## Output type: {outtype}\n")
        print(f"- **Total speed:** {toc - tic:.6f} s (sum: {trajectory_time + amplitude_time + summation_time:.6f} s)\n")
        # print(f"- **Trajectory speed:** {trajectory_time:.6f} s ({trajectory_time/(toc - tic)*100:.2f}%)\n")
        # print(f"- **Amplitude speed:** {amplitude_time:.6f} s ({amplitude_time/(toc - tic)*100:.2f}%)\n")
        # print(f"- **Summation speed:** {summation_time:.6f} s ({summation_time/(toc - tic)*100:.2f}%)\n\n")
