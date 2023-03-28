from __future__ import annotations

import random

import numpy as np


from abc import ABC
from cmath import exp
from dataclasses import astuple
from typing import Any, NamedTuple


from tqdm import trange

from typing import Callable, Sequence

from scipy.optimize import nnls  # type: ignore



from dataclasses import dataclass
from typing import Collection, Protocol

from mne.io.brainvision.brainvision import read_raw_brainvision  # type: ignore


from scipy.signal import hilbert, butter, lfilter, filtfilt
import matplotlib.pyplot as plt




def complex_randn() -> complex:
    """Generate random complex number with Re and Im sampled from N(0, 1)"""
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


def complex2vec(z: complex):
    """Convert complex number to 2d vector"""
    return np.array([[z.real], [z.imag]])


def vec2complex(v) -> complex:
    """Convert 2d vector to a complex number"""
    return v[0, 0] + 1j * v[1, 0]


def complex2mat(z: complex):
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return np.array([[z.real, -z.imag], [z.imag, z.real]])


def mat2complex(M) -> complex:
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return M[0, 0] + 1j * M[1, 0]
















class SignalGenerator(Protocol):
    def step(self) -> complex:
        """Generate single noise sample"""
        ...


@dataclass
class MatsudaParams:
    """Single oscillation Matsuda-Komaki model parameters"""

    A: float
    freq: float
    sr: float

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)


@dataclass
class SingleRhythmModel:
    mp: MatsudaParams
    sigma: float
    x: complex = 0

    def step(self) -> complex:
        """Update model state and generate measurement"""
        self.x = self.mp.Phi * self.x + complex_randn() * self.sigma
        return self.x

    def psd_onesided(self, f: float) -> float:
        """
        Theoretical PSD for Matsuda-Komaki multivariate AR process
        Notes
        -----
        Implementation follows eq. (6.38) from [1] for the first state component, i.e.
        it effectively computes p_{11} from (6.38) for the single-rhythm MK model
        N.B.: eq. (6.38) is for two-sided spectrum. To match the default of scipy.signal.welch,
        we return onesided spectrum, which is double the original
        References
        ----------
        .. [1] Kitagawa, Genshiro. 2010. Introduction to Time Series Modeling.
        0 ed. Chapman and Hall/CRC. https://doi.org/10.1201/9781584889229.
        """
        phi = 2 * np.pi * self.mp.freq / self.mp.sr
        psi = 2 * np.pi * f / self.mp.sr
        A = self.mp.A

        denom = (
            np.abs(1 - 2 * A * np.cos(phi) * np.exp(-1j * psi) + A**2 * np.exp(-2j * psi)) ** 2
        )
        num = 1 + A**2 - 2 * A * np.cos(phi) * np.cos(psi)
        return self.sigma**2 / self.mp.sr * num / denom * 2


def gen_ar_noise_coefficients(alpha: float, order: int):
    """
    Parameters
    ----------
    order : int
        Order of the AR model
    alpha : float in the [-2, 2] range
        Alpha as in '1/f^alpha' PSD profile
    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.
    """
    a: list[float] = [1]
    for k in range(1, order + 1):
        a.append((k - 1 - alpha / 2) * a[-1] / k)  # AR coefficients as in [1], eq. (116)
    return -np.array(a[1:])


class ArNoiseModel:
    """
    Generate 1/f^alpha noise with truncated autoregressive process, as described in [1]
    Parameters
    ----------
    x0 : np.ndarray of shape(order,)
        Initial conditions vector for the AR model
    order : int
        Order of the AR model
    alpha : float in range [-2, 2]
        Alpha as in '1/f^alpha'
    sigma : float, >= 0
        White noise standard deviation (see [1])
    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.
    """

    def __init__(
        self, x0: np.ndarray, sr: float, order: int = 1, alpha: float = 1, sigma: float = 1
    ):
        assert len(x0) == order, f"x0 length must match AR order; got {len(x0)=}, {order=}"
        self.a = gen_ar_noise_coefficients(alpha, order)
        self.x = x0
        self.sigma = sigma
        self.sr = sr

    def step(self) -> float:
        """Make one step of the AR process"""
        y_next = self.a @ self.x + np.random.randn() * self.sigma
        self.x = np.concatenate([[y_next], self.x[:-1]])  # type: ignore
        return float(y_next)

    def psd_onesided(self, f: float) -> float:
        return theor_psd_ar(f, self.sigma, ar_coef=self.a, sr=self.sr) * 2 / self.sr


def theor_psd_ar(f: float, s: float, ar_coef: Collection[float], sr: float) -> float:
    denom = 1 - sum(a * np.exp(-2j * np.pi * f / sr * m) for m, a in enumerate(ar_coef, 1))
    return s**2 / np.abs(denom) ** 2


class RealNoise:
    def __init__(self, single_channel_eeg, s: float):
        self.single_channel_eeg = single_channel_eeg
        self.ind = 0
        self.s = s

    def step(self) -> float:
        n_samp = len(self.single_channel_eeg)
        if self.ind >= len(self.single_channel_eeg):
            raise IndexError(f"Index {self.ind} is out of bounds for data of length {n_samp}")
        self.ind += 1
        return self.single_channel_eeg[self.ind] * self.s


def prepare_real_noise(
    raw_path: str, s: float = 1, minsamp: int = 0, maxsamp: int | None = None
) -> tuple[RealNoise, float]:
    raw = read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None, verbose="ERROR")

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop], s), raw.info["sfreq"]


def collect(signal_generator: SignalGenerator, n_samp: int):
    return np.array([signal_generator.step() for _ in range(n_samp)])


























"""
Vector general-case kalman filter implementations
Examples
--------
Get smoothed data
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from kalman_experiments.kalman.wrappers import PerturbedP1DMatsudaKF
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> from kalman_experiments.model_selection import fit_kf_parameters
Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + np.random.randn(len(gt_states))
>>> kf = PerturbedP1DMatsudaKF(mp, q_s=1, psi=np.zeros(0), r_s=10, lambda_=0).KF
>>> y = normalize_measurement_dimensions(meas)
>>> x, P = apply_kf(kf, y)
>>> x_n, P_n, J = apply_kalman_interval_smoother(kf, x, P)
>>> res = plt.plot([xx[0] for xx in x], label="fp", linewidth=4)
>>> res = plt.plot([xxn[0] for xxn in x_n], label="smooth", linewidth=4)
>>> res = plt.plot(np.real(gt_states), label="gt", linewidth=2)
>>> l = plt.legend()
>>> plt.show()
"""
import numpy as np




class DifferenceKF:
    """
    'Alternative approach' implementation for KF with colored noise from [1]
    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix (see eq.(1) in [2])
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model (see eq.(2) in [2]); maps state to
        measurements
    Psi : np.ndarray of shape(n_meas, n_meas)
        Measurement noise transfer matrix (see eq. (3) in [2])
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model (cov for e_{k-1}
        in eq. (3) in [2])
    References
    ----------
    .. [1] Chang, G. "On kalman filter for linear system with colored measurement
    noise". J Geod 88, 1163–1170, 2014 https://doi.org/10.1007/s00190-014-0751-7
    """

    def __init__(self, Phi, Q, H, Psi, R):
        n_states = Phi.shape[0]
        n_meas = H.shape[0]

        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x = np.zeros((n_states, 1))  # posterior state (after update)
        # self.P = np.eye(n_states) * 1e-3  # posterior cov (after update)
        self.P = np.eye(n_states)  # posterior cov (after update)

        self.y_prev = np.zeros((n_meas, 1))

    def predict(self, x, P):
        x_ = self.Phi @ x  # eq. (26) from [1]
        P_ = self.Phi @ P @ self.Phi.T + self.Q  # eq. (27) from [1]
        return x_, P_

    def update(self, y, x_, P_):
        A = self.Psi @ self.H
        B = self.H @ self.Phi
        P, H, R = self.P, self.H, self.R

        z = y - self.Psi @ self.y_prev  # eq. (35) from [1]
        n = z - self.H @ x_ + A @ self.x  # eq. (37) from [1]
        Sigma = H @ P_ @ H.T + A @ P @ A.T + R - B @ P @ A.T - A @ P @ B.T  # eq. (38) from [1]
        Pxn = P_ @ self.H.T - self.Phi @ P @ A.T  # eq. (39) from [1]

        K = Pxn / Sigma  # eq. (40) from [1]
        self.x = x_ + K * n  # eq. (41) from [1]
        self.P = P_ - K * Sigma @ K.T  # eq. (42) from [1]
        self.y_prev = y
        return self.x, self.P

    def update_no_meas(self, x_, P_):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        self.y_prev = self.H @ x_
        return x_, P_

    def step(self, y):
        x_, P_ = self.predict(self.x, self.P)
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)


class SimpleKF:
    """
    Standard Kalman filter implementation
    Implementation follows eq. (2, 3) from [1]
    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model; maps state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model
    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
    Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
    Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.
    """

    def __init__(
        self, Phi, Q, H, R, x_0 = None, P_0 = None
    ):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R

        n_states = Phi.shape[0]
        self.x = np.zeros((n_states, 1)) if x_0 is None else x_0  # posterior state (after update)
        self.P = np.eye(n_states) if P_0 is None else P_0  # posterior cov (after update)

    def reset_state(self):
        self.x = np.zeros_like(self.x)  # posterior state (after update)
        # self.P = np.eye(n_states) * 1e-3  # posterior cov (after update)
        self.P = np.eye(len(self.x))  # posterior cov (after update)

    def predict(self, x, P):
        x_ = self.Phi @ x
        P_ = self.Phi @ P @ self.Phi.T + self.Q
        return x_, P_

    def update(self, y, x_, P_):
        n, Sigma = self._compute_innovations(y, x_, P_)
        Pxn = P_ @ self.H.T
        K = np.linalg.solve(Sigma, Pxn.T).T

        self.x = x_ + K @ n
        self.P = P_ - K @ Sigma @ K.T
        return self.x, self.P

    def update_no_meas(self, x_, P_):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        return x_, P_

    def step(self, y):
        x_, P_ = self.predict(self.x, self.P)
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)

    def _compute_innovations(self, y, x_, P_):
        n = y - self.H @ x_
        Sigma = self.H @ P_ @ self.H.T + self.R
        return n, Sigma


class PerturbedPKF(SimpleKF):
    """
    Perturbed P implementation from [1] for KF with augmented state space
    Parameters
    ----------
    Phi : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented state transfer matrix (see eq. (9) in [3])
    Q : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented process noise covariance matrix (see eq.(9) in [3])
    H : np.ndarray of shape(n_meas, n_aug_states)
        Augmented matrix of the measurements model (see eq.(9) in [3]); maps
        augmented state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Measurements covariance matrix, usually of zeroes, see notes
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3].
    Notes
    -----
    R is added for possible regularization and normally must be a zero matrix,
    since the measurement errors are incorporated into the augmented state
    vector
    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
    Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
    Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.
    """

    def __init__(self, Phi, Q, H, R, lambda_: float = 1e-6):
        super().__init__(Phi, Q, H, R)
        self.lambda_ = lambda_

    def update(self, y, x_, P_):
        super().update(y, x_, P_)
        self.P += np.eye(len(self.P)) * self.lambda_
        return self.x, self.P


def apply_kf(KF: SimpleKF, y):
    n = len(y) - 1
    x = [None] * (n + 1)  # type: ignore  # x^t_t
    P = [None] * (n + 1)  # type: ignore  # P^t_t
    x[0], P[0] = KF.x, KF.P
    for t in range(1, n + 1):
        x[t], P[t] = KF.step(y[t])
    return x, P


def apply_kalman_interval_smoother(
    KF: SimpleKF, x, P):
    n = len(x) - 1
    x_n= [None] * (n + 1)  # type: ignore  # x^n_t
    P_n= [None] * (n + 1)  # type: ignore  # P^n_t
    x_n[n], P_n[n] = x[n], P[n]
    J = [None] * (n + 1)  # type: ignore
    for t in range(n, 0, -1):
        x_n[t - 1], P_n[t - 1], J[t - 1] = smoother_step(KF, x[t - 1], P[t - 1], x_n[t], P_n[t])

    return x_n, P_n, J

'''

def apply_kalman_interval_smoother(
    KF: SimpleKF, states_fp
):
    n = len(states_fp) - 1
    # x_n: list[Vec] = [np.zeros_like(states_fp[0].x)] * (n + 1)  # pyright: ignore  # x^n_t
    # P_n: list[Cov] = [np.zeros_like(states_fp[0].P)] * (n + 1)  # pyright: ignore  # P^n_t
    states_n = [Gaussian(np.zeros_like(states_fp[0]), np.zeros_like(states_fp[1]))] * (n + 1)
    states_n[n] = states_fp[n]
    J = [np.empty_like(states_fp[1])] * (n + 1)  # pyright: ignore
    for t in range(n, 0, -1):
        states_n[t - 1], J[t - 1] = smoother_step(KF, states_fp[t - 1], states_n[t])

    return states_n, J

'''

'''
def smoother_step(KF, state: Gaussian, state_n: Gaussian):
    """
    Make one Kalman Smoother step
    TODO: Update docstring
    Parameters
    ----------
    x : Vec
        State estimate after KF update step after the forward pass, i.e.
        x^{t-1}_{t-1} in eq (6.47) in [1]
    P : Cov
        State covariance after KF update step after the forward pass, i.e.
        P^{t-1}_{t-1} in eq. (6.48) in [1]
    x_n : Vec
        Smoothed state estimate for the time instaint following the one being
        currently processed, i.e. x^{n}_{t} in eq. (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for the time instant following the one being
        currently processed, i.e. P^{n}_{t} in eq. (6.47) in [1]
    Returns
    -------
    x_n : Vec
        Smoothed state estimate for one timestep back, i.e. x^{n}_{t-1} in eq.
        (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for one timestep back, i.e. P^{n}_{t-1} in
        eq. (6.48) in [1]
    J : Mat
        J_{t-1} in eq. (6.49) in [1]
    Notes
    -----
    Code here follows slightly different notation than in em_step(); e.g. here
    x_n is a state vector for a single time instant compared to an array of
    state vectors in em_step().
    References
    ----------
    [1] .. Shumway, Robert H., and David S. Stoffer. 2011. Time Series Analysis
    and Its Applications. Springer Texts in Statistics. New York, NY: Springer
    New York. https://doi.org/10.1007/978-1-4419-7865-3.
    """
    state_ = KF.predict(state[0],state[1])

    # P * Phi^T * P_^{-1}; solve is better than inv
    J = np.linalg.solve(state_.P, KF.Phi @ state.P).T

    x = state.x + J @ (state_n.x - state_.x)
    P = state.P + J @ (state_n.P - state_.P) @ J.T
    state_n = Gaussian(x, P)

    return state_n, J

'''



def smoother_step(KF: SimpleKF, x, P, x_n, P_n):
    """
    Make one Kalman Smoother step
    Parameters
    ----------
    x : Vec
        State estimate after KF update step after the forward pass, i.e.
        x^{t-1}_{t-1} in eq (6.47) in [1]
    P : Cov
        State covariance after KF update step after the forward pass, i.e.
        P^{t-1}_{t-1} in eq. (6.48) in [1]
    x_n : Vec
        Smoothed state estimate for the time instaint following the one being
        currently processed, i.e. x^{n}_{t} in eq. (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for the time instant following the one being
        currently processed, i.e. P^{n}_{t} in eq. (6.47) in [1]
    Returns
    -------
    x_n : Vec
        Smoothed state estimate for one timestep back, i.e. x^{n}_{t-1} in eq.
        (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for one timestep back, i.e. P^{n}_{t-1} in eq. (6.48) in [1]
    J : Mat
        J_{t-1} in eq. (6.49) in [1]
    Notes
    -----
    Code here follows slightly different notation than in em_step(); e.g. here
    x_n is a state vector for a single time instant compared to an array of
    state vectors in em_step().
    References
    ----------
    [1] .. Shumway, Robert H., and David S. Stoffer. 2011. Time Series Analysis
    and Its Applications. Springer Texts in Statistics. New York, NY: Springer
    New York. https://doi.org/10.1007/978-1-4419-7865-3.
    """
    x_, P_ = KF.predict(x, P)

    J = np.linalg.solve(P_, KF.Phi @ P).T  # P * Phi^T * P_^{-1}; solve is better than inv

    x_n = x + J @ (x_n - x_)
    P_n = P + J @ (P_n - P_) @ J.T

    return x_n, P_n, J








"""
Kalman filter implementations
References
----------
.. [1] Matsuda, Takeru, and Fumiyasu Komaki. “Time Series Decomposition into
Oscillation Components and Phase Estimation.” Neural Computation 29, no. 2
(February 2017): 332–67. https://doi.org/10.1162/NECO_a_00916.
.. [2] Chang, G. "On kalman filter for linear system with colored measurement
noise". J Geod 88, 1163–1170, 2014 https://doi.org/10.1007/s00190-014-0751-7
.. [3] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
https://doi.org/10.1109/TAES.2012.6178086.
.. [4] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings of
the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.
"""


class Gaussian(NamedTuple):
    mu: np.ndarray[np.floating[Any]]  # column vector of shape(n, 1)
    Sigma: np.ndrray[np.floating[Any]] 


class OneDimKF(ABC):
    """Single measurement Kalman filter abstraction"""

    KF: Any

    def predict(self, X: Gaussian) -> Gaussian:
        return Gaussian(*self.KF.predict(X.mu, X.Sigma))

    def update(self, y: float, X_: Gaussian) -> Gaussian:
        y_arr = np.array([[y]])
        return Gaussian(*self.KF.update(y=y_arr, x_=X_.mu, P_=X_.Sigma))

    def update_no_meas(self, X_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""
        return Gaussian(*self.KF.update_no_meas(x_=X_.mu, P_=X_.Sigma))

    def step(self, y: float | None) -> Gaussian:
        """Predict and update in one step"""
        X_ = self.predict(Gaussian(self.KF.x,self.KF.P))
        return self.update_no_meas(X_) if y is None else self.update(y, X_)


class Difference1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(1) colored noise
    Using Matsuda's model for oscillation prediction, see [1], and a difference
    scheme to incorporate AR(1) 1/f^a measurement noise, see [2]. Wraps
    DifferenceKF to avoid trouble with properly arranging matrix and
    vector shapes.
    Parameters
    ----------
    mp : MatsudaParams
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : float
        Coefficient of the AR(1) process modelling 1/f^a colored noise;
        see eq. (3) in [2]; 0.5 corresponds to 1/f noise, 0 -- to white noise,
        1 -- to Brownian motion, see [4]. In between values are also allowed.
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    See also
    --------
    gen_ar_noise_coefficients : generate psi
    """

    def __init__(self, mp: MatsudaParams, q_s: float, psi: float, r_s: float):
        A, f, sr = astuple(mp)
        Phi = complex2mat(A * exp(2 * np.pi * f / sr * 1j))
        Q = np.eye(2) * q_s**2
        H = np.array([[1, 0]])
        Psi = np.array([[psi]])
        R = np.array([[r_s**2]])
        self.KF = DifferenceKF(Phi=Phi, Q=Q, H=H, Psi=Psi, R=R)


class PerturbedP1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(n_ar) colored noise
    Using Matsuda's model for oscillation prediction, see [1], and AR(n) to
    make account for 1/f^a measurement noise. Previous states for
    AR(n_ar) are included via state-space augmentation with the Perturbed P
    stabilization technique, see [3]. Wraps PerturbedPKF to avoid trouble with
    properly arranging matrix and vector shapes.
    Parameters
    ----------
    mp : MatsudaParams
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]
    See also
    --------
    kalman_experiments.models.gen_ar_noise_coefficients : generate psi
    """

    def __init__(
        self,
        mp: MatsudaParams,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lambda_: float = 1e-6,
    ):
        ns = len(psi)  # number of noise states

        Phi_blocks = [
            [complex2mat(mp.A * exp(2 * np.pi * mp.freq / mp.sr * 1j)), np.zeros([2, ns])],
        ]
        if ns:
            Phi_blocks.append([np.zeros([1, 2]), psi[np.newaxis, :]])
            Phi_blocks.append([np.zeros([ns - 1, 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])])
        Phi = np.block(Phi_blocks)  # pyright: ignore
        Q_blocks = [[np.eye(2) * q_s**2, np.zeros([2, ns])]]
        if ns:
            Q_noise = np.zeros([ns, ns])
            Q_noise[0, 0] = r_s**2
            Q_blocks.append([np.zeros([ns, 2]), Q_noise])
        Q = np.block(Q_blocks)  # pyright: ignore

        H_noise = [1] + [0] * (ns - 1) if ns else []
        H = np.array([[1, 0] + H_noise])
        if ns:
            R = np.array([[0]])
        else:
            R = np.array([[r_s**2]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)
        self.mp = mp
        self.psi = psi
        self.lambda_ = lambda_
        self.q_s = q_s
        self.r_s = r_s

    def __repr__(self) -> str:
        return (
            f"PerturbedP1DMatsudaKF(mp={self.mp}, q_s={self.q_s:.2f},"
            f" psi={self.psi}, r_s={self.r_s:.2f}, lambda_={self.lambda_})"
        )


class PerturbedP1DMatsudaSmoother(OneDimKF):
    def __init__(
        self,
        mp: MatsudaParams,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lag: int = 0,
        lambda_: float = 1e-6,
    ):
        lag = 0 if lag < 0 else lag
        n_aug_x = lag * 2
        ns = len(psi)  # number of noise states

        Phi = np.block(
            [  # pyright: ignore
                [complex2mat(mp.Phi), np.zeros([2, ns + n_aug_x])],
                [np.eye(n_aug_x), np.zeros([n_aug_x, 2 + ns])],
                [np.zeros([1, n_aug_x + 2]), psi[np.newaxis, :]],
                [np.zeros([ns - 1, n_aug_x + 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])],
            ]
        )
        Q_noise = np.zeros([ns, ns])
        Q_noise[0, 0] = r_s**2
        Q = np.block(
            [  # pyright: ignore
                [np.eye(2) * q_s**2, np.zeros([2, ns + n_aug_x])],
                [np.zeros([n_aug_x, n_aug_x + ns + 2])],
                [np.zeros([ns, n_aug_x + 2]), Q_noise],
            ]
        )

        H = np.array([[1, 0] + [0] * n_aug_x + [1] + [0] * (ns - 1)])
        R = np.array([[0]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)
        self.lag = lag
'''

def apply_kf(kf: OneDimKF, signal, delay: int):
    """Convenience function to filter all signal samples at once with KF"""
    res = []
    # AR_ORDER = 2
    if delay > 0:
        assert hasattr(kf, "lag"), "Smoothing is not implemented for this KF"
        assert kf.lag <= delay  # pyright: ignore
        for y in signal:
            state = kf.step(y)
            res.append(vec2complex(state.mu[delay * 2 : (delay + 1) * 2]))
    else:
        k = 0
        for y in signal:
            state = kf.step(y)
            # envs = np.abs([vec2complex(state.mu[i * 2:(i + 1) * 2]) for i in range(5)])
            # rho, _ = yule_walker(envs, order=AR_ORDER)
            #     print(f"{rho=}, {envs=}")
            # envs_ar = list(envs[:AR_ORDER])
            # new_env = envs[0]
            for _ in range(abs(delay)):
                state = kf.predict(state)
                # new_env = rho.dot(envs_ar)
                # new_env += de
                # de += de2
                # envs_ar = [new_env] + envs_ar[:-1]
                # if not k % 500:
                #     print(f"{envs_ar=}")
            k += 1
            pred = vec2complex(state.mu[:2])
            # pred /= np.abs(pred)
            # pred *= new_env
            res.append(pred)
    return np.array(res)

'''







class KFParams(NamedTuple):
    A: float
    f: float
    q_s: float
    r_s: float
   


def fit_kf_parameters(
    meas,
    KF: PerturbedP1DMatsudaKF,
    n_iter: int = 800,
    tol: float = 1e-3,
    flim: tuple[float | None, float | None] = (None, None),
) -> PerturbedP1DMatsudaKF:

    AMP_EPS = 1e-4
    sr = KF.mp.sr
    prev_freq = KF.mp.freq
    model_error = np.inf
    for _ in (pb := trange(n_iter, desc="Fitting KF parameters")):
        (amp, freq, q_s, r_s, x_0, P_0), nll = em_step(meas, KF.KF)
        amp = min(amp, 1 - AMP_EPS)
        freq *= sr / (2 * np.pi)

        if flim[0] is not None:
            freq = max(freq, flim[0])
        if flim[1] is not None:
            freq = min(freq, flim[1])

        pb.set_description(
            f"Fitting KF parameters: nll={nll:.2f},"
            f"f={freq:.4f}, A={amp:.4f}, {q_s:.4f}, {r_s:.2f}"
        )

        mp = MatsudaParams(amp, freq, sr)
        KF = PerturbedP1DMatsudaKF(mp, q_s, KF.psi, r_s, KF.lambda_)
        KF.KF.x = x_0
        KF.KF.P = P_0
        model_error = abs(freq - prev_freq)
        prev_freq = freq
        if model_error < tol:
            break
    else:
        print(f"Did't converge after {n_iter} iterations; {model_error=}")
    return KF


def em_step(meas,KF: SimpleKF) -> tuple[KFParams, float]:
    n = len(meas)
    Phi, A, Q, R = KF.Phi, KF.H, KF.Q, KF.R
    assert n, "Measurements must be nonempty"

    y = normalize_measurement_dimensions(meas)
    x, P = apply_kf(KF, y)
    nll = compute_kf_nll(y, x, P, KF)
    x_n, P_n, J = apply_kalman_interval_smoother(KF, x, P)
    P_nt = estimate_adjacent_states_covariances(Phi, Q, A, R, P, J)

    S = compute_aux_em_matrices(x_n, P_n, P_nt)
    freq, Amp, q_s, r_s = params_update(S, Phi, n)
    x_0_new = x_n[0]
    P_0_new = P_n[0]

    return KFParams(Amp, freq, q_s, r_s, x_0_new, P_0_new), nll


def normalize_measurement_dimensions(meas):
    # prepend nan for to simplify indexing; 0 index is for x and P prior to the measurements
    n = len(meas)
    y = [np.array([[np.nan]])] * (n + 1)
    for t in range(1, n + 1):
        y[t] = meas[t - 1, np.newaxis, np.newaxis]
    return y


def estimate_adjacent_states_covariances(
    Phi, Q, A, R, P, J):
    # estimate P^n_{t-1,t-2}
    n = len(P) - 1
    P_ = Phi @ P[n - 1] @ Phi.T + Q  # P^{n-1}_n
    K = np.linalg.solve(A @ P_ @ A.T + R, A @ P[n]).T  # K_n, eq. (6.23) in [1]
    P_nt = [None] * (n + 1)  # type: ignore  # P^n_{t-1, t-2}
    P_nt[n - 1] = (np.eye(K.shape[0]) - K @ A) @ Phi @ P[n - 1]  # P^n_{n, n-1}, eq.(6.55) in [1]

    for t in range(n, 1, -1):
        P_nt[t - 2] = (
            P[t - 1] @ J[t - 2].T + J[t - 1] @ (P_nt[t - 1] - Phi @ P[t - 1]) @ J[t - 2].T
        )
    return P_nt


def compute_aux_em_matrices(x_n, P_n, P_nt):
    n = len(x_n) - 1
    S = {
        "11": np.zeros_like(P_n[0], dtype=np.longdouble),
        "10": np.zeros_like(P_nt[0], dtype=np.longdouble),
        "00": np.zeros_like(P_n[0], dtype=np.longdouble),
    }
    for t in range(1, n + 1):
        S["11"] += x_n[t] @ x_n[t].T + P_n[t]
        S["10"] += x_n[t] @ x_n[t - 1].T + P_nt[t - 1]
        S["00"] += x_n[t - 1] @ x_n[t - 1].T + P_n[t - 1]
    return S


def params_update(S, Phi, n: int) -> tuple[float, float, float, float]:
    A = S["00"][0, 0] + S["00"][1, 1]
    B = S["10"][0, 0] + S["10"][1, 1]
    C = S["10"][1, 0] - S["10"][0, 1]
    D = S["11"][0, 0] + S["11"][1, 1]
    f = max(C / B, 0)
    Amp = np.sqrt(B**2 + C**2) / A
    q_s = np.sqrt(max(0.5 * (D - Amp**2 * A) / n, 1e-6))
    r_s = np.sqrt(
        (S["11"][2, 2] - 2 * S["10"][2, :] @ Phi.T[:, 2] + (Phi[2, :] @ S["00"] @ Phi.T[:, 2])) / n
    )
    return float(f), float(Amp), float(q_s), float(r_s)


def compute_kf_nll_single_channel(y, x, P, KF: SimpleKF) -> float:
    """
    Parameters
    ----------
    y : list[Vec]
        normalized measurements
    x : list[Vec]
        posterior state estimates
    P : list[Cov]
        posterior state covariance estimates
    """
    n = len(y) - 1
    negloglikelihood = 0
    nss = 0
    dets = 0
    for t in range(30, n + 1):
        x_, P_ = KF.predict(x[t - 1], P[t - 1])
        eps = y[t] - KF.H @ x_
        Sigma = KF.H @ P_ @ KF.H.T + KF.R
        nss += eps[0] ** 2 / Sigma[0, 0]
        dets += np.log(Sigma[0, 0])
    N = (n + 1 - 30)
    sigma_ml = nss / N

    negloglikelihood = 0.5 * (N * np.log(sigma_ml) + dets)  # concentrated nll
    # negloglikelihood = 0.5 * (nss + dets)
    return float(negloglikelihood)


def compute_kf_nll(y, x, P, KF: SimpleKF) -> float:
    """
    Parameters
    ----------
    y : list[Vec]
        normalized measurements
    x : list[Vec]
        posterior state estimates
    P : list[Cov]
        posterior state covariance estimates
    KF : SimpleKF
    """
    n = len(y) - 1
    negloglikelihood = 0
    nss = 0
    dets = 0
    for t in range(30, n + 1):
        x_, P_ = KF.predict(x[t - 1], P[t - 1])
        eps = y[t] - KF.H @ x_
        Sigma = KF.H @ P_ @ KF.H.T + KF.R
        tmp = np.linalg.solve(Sigma, eps)  # Sigma inversion
        nss += eps.T @ tmp
        dets += np.log(np.linalg.det(Sigma))
    N = (n + 1 - 30)
    sigma_ml = nss / N

    negloglikelihood = 0.5 * (N * np.log(sigma_ml) + dets)  # concentrated nll
    # negloglikelihood = 0.5 * (nss + dets)
    return float(negloglikelihood)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def nll_opt_wrapper(x, meas, sr, psi, lambda_):
    A = logistic(x[0])
    f = np.exp(x[1])
    q_s = np.exp(x[2])
    # r_s = np.exp(x[3])
    y = normalize_measurement_dimensions(meas)
    mp = MatsudaParams(A, f, sr)
    # KF = PerturbedP1DMatsudaKF(mp, q_s, psi, r_s, lambda_)
    KF = PerturbedP1DMatsudaKF(mp, q_s, psi, 1, lambda_)
    x, P = apply_kf(KF.KF, y)
    return compute_kf_nll(y, x, P, KF.KF)












PsdFunc = Callable[[float], float]


def get_psd_val_from_est(f, freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Utility function to get estimated psd value by frequency
    If using welch for psd estimation, make sure it was called with
    `return_onesided=True` (default)
    """
    ind = np.argmin((freqs - f) ** 2)
    return psd[ind]


def estimate_sigmas(
    basis_psd_funcs: list[PsdFunc], data_psd_func: PsdFunc, freqs: Sequence[float]
) -> np.ndarray:
    A: list[list[float]] = []
    b = [1] * len(freqs)
    for row, f in enumerate(freqs):
        b_ = data_psd_func(f)
        A.append([])
        for func in basis_psd_funcs:
            A[row].append(func(f) / b_)
    return nnls(np.array(A), np.array(b))[0]



def to_db(arr):
    return 10 * np.log10(arr)











import scipy.optimize as opt

'''
def ideal_envelope(fc,srate, y, bandwidth = 4.0):
    
    b,a = butter(3,[fc-bandwidth/2.0,fc+bandwidth/2.0], fs = srate, btype = 'bandpass')
    
    filtered = filtfilt(b,a,y)
    
    hilb = hilbert(filtered)
    
    envelope = np.abs(hilb)
    
    return envelope'''


def ideal_rhythm(fc,srate, y, bandwidth = 4.0):
    
    b,a = butter(3,[fc-bandwidth/2.0,fc+bandwidth/2.0], fs = srate, btype = 'bandpass')
    
    filtered = filtfilt(b,a,y)
    
    
    return filtered





#q_s_beta_rate - how much higher

def grid_optimizer(signal, good_idx, freq_alpha, freq_beta, ar, srate,log_num = 10, q_s_alpha_bounds = [-2,-9],q_s_r_bounds = [-2,-9], q_s_beta_rate = 10.0):
    
    Num = 16
    q_s_alpha_range = np.logspace(q_s_alpha_bounds[0],q_s_alpha_bounds[1],num = Num)
    q_s_r_range = np.logspace(q_s_r_bounds[0],q_s_r_bounds[1],num = Num)
    
    
    
    corr_mem = np.zeros([Num, Num])
    lat_mem = np.zeros([Num, Num])
    
    
    
    for i, q_s_alpha in enumerate(q_s_alpha_range):
        for j, q_s_r in enumerate(q_s_r_range):
            
            q_s_beta = q_s_alpha*q_s_beta_rate
            
            
            mp_alpha = MatsudaParams(A=0.999, freq=freq_alpha, sr=srate) # x[0]
            mp_beta = MatsudaParams(A=0.999, freq=freq_beta, sr=srate) # x [1]
            
            kf = AlphaBetaKF(mp_alpha=mp_alpha, q_s_alpha=q_s_alpha, mp_beta=mp_beta, q_s_beta=q_s_beta, r_s=q_s_r, psi=ar)

            filtered = apply_kf_alpha_beta(kf, signal) 
            
            
            alpha_envelope = np.abs(filtered[0])[good_idx]
            beta_envelope = np.abs(filtered[1])[good_idx]
            
            

            
            gt_alpha_envelope = ideal_envelope(freq_alpha, srate, signal)[good_idx]
            gt_beta_envelope = ideal_envelope(freq_beta, srate, signal)[good_idx]
            
                
            
            
            
            alpha_cc_cum = np.zeros(99)
            for k,bias in enumerate(range(1,100)):
            
            
                gt_alpha_envelope_trim = gt_alpha_envelope[:-bias]
                gt_beta_envelope_trim = gt_beta_envelope[:-bias]
                
                       
                gt_alpha_envelope_trim -= np.mean(gt_alpha_envelope_trim)
                gt_beta_envelope_trim -= np.mean(gt_beta_envelope_trim)
            
            
                alpha_envelope_trim = alpha_envelope[bias:]
                beta_envelope_trim = beta_envelope[bias:]
            
            
                alpha_envelope_trim -= np.mean(alpha_envelope_trim)
                beta_envelope_trim -= np.mean(beta_envelope_trim)
                
                
                
            
            
                alpha_cc_cum[k] = np.sum(alpha_envelope_trim*gt_alpha_envelope_trim)/(np.linalg.norm(alpha_envelope_trim)*np.linalg.norm(gt_alpha_envelope_trim))
                #beta_cc_cum[i] = np.sum(beta_envelope*gt_beta_envelope)/(np.linalg.norm(beta_envelope)*np.linalg.norm(gt_beta_envelope))
            
            #################  
            
            latency = np.argmax(alpha_cc_cum)
            
   
   
            alpha_cc = np.max(alpha_cc_cum)
            
            corr_mem[i,j] = alpha_cc
            lat_mem[i,j] = latency
            


            print(q_s_alpha, q_s_r, ':')
            print(alpha_cc,latency)
            print()
            
            
            
            #alpha_gt = ideal_rhythm(freq_alpha,srate,signal)[good_idx][:-bias]
            #alpha_kf = np.real(filtered[0])[good_idx][bias:]
            #alpha_gt -= np.mean(a
            
            
            
            
            
    return corr_mem, lat_mem
    
    



#bias = 20 ms

def optimize_kf_params(signal,srate,freq_alpha,freq_beta, good_idx, ar, bias = 10):
    
    gt_alpha_envelope = ideal_envelope(freq_alpha, srate, signal)[good_idx]
    gt_beta_envelope = ideal_envelope(freq_beta, srate, signal)[good_idx]
    
    
    gt_alpha_envelope = 0#gt_alpha_envelope[:-bias]
    gt_beta_envelope = 0#gt_beta_envelope[:-bias]
    
    
    gt_alpha_envelope -= np.mean(gt_alpha_envelope)
    gt_beta_envelope -= np.mean(gt_beta_envelope)
    
    
    #x0 = [0.99,0.99,1e0,1e0,1e0]
    
    
    
    
    #initial_simplex = np.array([[0.99,0.99,1e0,1e0,1e6],
    #                            [0.99,0.99,1e0,1e0,1e-12],
    #                            [0.99,0.99,1e0,1e6,1e-12],
    #                            [0.99,0.99,1e0,1e-12,1e6],
    #                            [0.99,0.99,1e0,1e6,1e6],
    #                            [0.999,0.999,1e0,1e-12,1e-12]])
    
    #res = opt.minimize(fun_to_opt, x0, args=(signal,freq_alpha,freq_beta,ar,
    #                                         srate,good_idx,bias,gt_alpha_envelope, 
    #                                         gt_beta_envelope), method='Nelder-Mead', 
    #                   bounds = [[0.95,0.9999],[0.95,0.9999],[0.0,None],[0.0,None],[0.0,None]],options = {'maxiter': None, 'initial_simplex': initial_simplex})

    
    initial_simplex = np.array([[1e-12,1e6,1e-12],
                                [1e6,1e5,1e-12],
                                [1e-12,1e4,1e6],
                                [1e6,1e3,1e6]])
    
    res = opt.minimize(fun_to_opt,x0 = [1e0,1e0,1e0],args=(signal,freq_alpha,freq_beta,ar,
                                             srate,good_idx,bias,gt_alpha_envelope, 
                                             gt_beta_envelope), method='Nelder-Mead', 
                       bounds = [[0.0,None],[0.0,None],[0.0,None]],options = {'maxiter': None, 'initial_simplex': initial_simplex})

    
    
    return res.x



#x - [Aalpha, Abeta, q_s_alpha, q_s_beta, r_s
def fun_to_opt(x, signal, freq_alpha, freq_beta,ar, srate, good_idx, bias, gt_alpha_envelope, gt_beta_envelope):
    
    mp_alpha = MatsudaParams(A=0.999, freq=freq_alpha, sr=srate) # x[0]
    mp_beta = MatsudaParams(A=0.999, freq=freq_beta, sr=srate) # x [1]
    
    kf = AlphaBetaKF(mp_alpha=mp_alpha, q_s_alpha=x[0], mp_beta=mp_beta, q_s_beta=x[1], r_s=x[2], psi=ar)

    filtered = apply_kf_alpha_beta(kf, signal) 
    
    '''
    alpha_envelope = np.abs(filtered[0])
    beta_envelope = np.abs(filtered[1])
    
    
    
    
    
    alpha_envelope = alpha_envelope[good_idx][bias:]
    beta_envelope = beta_envelope[good_idx][bias:]
    
    
    
    

    alpha_envelope -= np.mean(alpha_envelope)
    beta_envelope -= np.mean(beta_envelope)
    
    
    
    alpha_cc = np.sum(alpha_envelope*gt_alpha_envelope)/(np.linalg.norm(alpha_envelope)*np.linalg.norm(gt_alpha_envelope))
    beta_cc = np.sum(beta_envelope*gt_beta_envelope)/(np.linalg.norm(beta_envelope)*np.linalg.norm(gt_beta_envelope))
    
    '''
    ########################
        
    alpha_envelope = np.abs(filtered[0])[good_idx]
    beta_envelope = np.abs(filtered[1])[good_idx]
    
    

    
    gt_alpha_envelope = ideal_envelope(freq_alpha, srate, signal)[good_idx]
    gt_beta_envelope = ideal_envelope(freq_beta, srate, signal)[good_idx]
    
        
    
    
    
    alpha_cc_cum = np.zeros(99)
    for i,bias in enumerate(range(1,100)):
    
    
        gt_alpha_envelope_trim = gt_alpha_envelope[:-bias]
        gt_beta_envelope_trim = gt_beta_envelope[:-bias]
        
               
        gt_alpha_envelope_trim -= np.mean(gt_alpha_envelope_trim)
        gt_beta_envelope_trim -= np.mean(gt_beta_envelope_trim)
    
    
        alpha_envelope_trim = alpha_envelope[bias:]
        beta_envelope_trim = beta_envelope[bias:]
    
    
        alpha_envelope_trim -= np.mean(alpha_envelope_trim)
        beta_envelope_trim -= np.mean(beta_envelope_trim)
        
        
        
    
    
        alpha_cc_cum[i] = np.sum(alpha_envelope_trim*gt_alpha_envelope_trim)/(np.linalg.norm(alpha_envelope_trim)*np.linalg.norm(gt_alpha_envelope_trim))
        #beta_cc_cum[i] = np.sum(beta_envelope*gt_beta_envelope)/(np.linalg.norm(beta_envelope)*np.linalg.norm(gt_beta_envelope))
    
    #################  
    
    latency = np.argmax(alpha_cc_cum)
    
    
    target_latency = 0.0#10.0 #samples
    dif_latency = np.abs(target_latency-latency)
    
    beta_cc = 0
    alpha_cc = np.max(alpha_cc_cum)
    

    
    loss = -alpha_cc*50.0-beta_cc*0.0+dif_latency*1.0
    
    #alpha_gt = ideal_rhythm(freq_alpha,srate,signal)[good_idx][:-bias]
    #alpha_kf = np.real(filtered[0])[good_idx][bias:]
    #alpha_gt -= np.mean(alpha_gt)
    #alpha_kf -= np.mean(alpha_kf)
    
    #loss = -np.sum(alpha_kf*alpha_gt)/(np.linalg.norm(alpha_kf)*np.linalg.norm(alpha_gt))
    

    print(loss)
    
    return loss




























class AlphaBetaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(n_ar) colored noise
    Using Matsuda's model for oscillation prediction, see [1], and AR(n) to
    make account for 1/f^a measurement noise. Previous states for
    AR(n_ar) are included via state-space augmentation with the Perturbed P
    stabilization technique, see [3]. Wraps PerturbedPKF to avoid trouble with
    properly arranging matrix and vector shapes.
    Parameters
    ----------
    mp : MatsudaParams
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]
    See also
    --------
    kalman_experiments.models.gen_ar_noise_coefficients : generate psi
    """

    def __init__(
        self,
        mp_alpha: MatsudaParams,
        q_s_alpha: float,
        mp_beta: MatsudaParams,
        q_s_beta: float,
        psi: np.ndarray,
        r_s: float,
        lambda_: float = 1e-6,
    ):
        ns = len(psi)  # number of noise states

        Phi_blocks = [
            [
                complex2mat(mp_alpha.A * exp(2 * np.pi * mp_alpha.freq / mp_alpha.sr * 1j)),
                np.zeros([2, ns + 2]),
            ],
            [
                np.zeros([2, 2]),
                complex2mat(mp_beta.A * exp(2 * np.pi * mp_beta.freq / mp_beta.sr * 1j)),
                np.zeros([2, ns]),
            ],
        ]
        if ns:
            Phi_blocks.append([np.zeros([1, 4]), psi[np.newaxis, :]])
            Phi_blocks.append([np.zeros([ns - 1, 4]), np.eye(ns - 1), np.zeros([ns - 1, 1])])
        Phi = np.block(Phi_blocks)  # pyright: ignore
        Q_blocks = [
            [np.eye(2) * q_s_alpha**2, np.zeros([2, ns + 2])],
            [np.zeros([2, 2]), np.eye(2) * q_s_beta**2, np.zeros([2, ns])],
        ]
        if ns:
            Q_noise = np.zeros([ns, ns])
            Q_noise[0, 0] = r_s**2
            Q_blocks.append([np.zeros([ns, 4]), Q_noise])
        Q = np.block(Q_blocks)  # pyright: ignore

        H_noise = [1] + [0] * (ns - 1) if ns else []
        H = np.array([[1, 0, 1, 0] + H_noise]) #danger!!!!!!!!!!!!!!!
        if ns:
            R = np.array([[0]])
        else:
            R = np.array([[r_s**2]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)
        self.mp_alpha = mp_alpha
        self.psi = psi
        self.lambda_ = lambda_
        self.q_s_alpha = q_s_alpha
        self.q_s_beta = q_s_beta
        self.r_s = r_s


def apply_kf_alpha_beta(kf: AlphaBetaKF, signal) :
    """Convenience function to filter all signal samples at once with KF"""
    res_alpha = []
    res_beta = []
    # AR_ORDER = 2
    k = 0
    for y in signal:
        state = kf.step(y)
        k += 1
        pred_alpha = vec2complex(state.mu[:2])
        pred_beta = vec2complex(state.mu[2:4])
        res_alpha.append(pred_alpha)
        res_beta.append(pred_beta)
    return (np.array(res_alpha), np.array(res_beta))


def lfilter_kf_alpha_beta(kf:AlphaBetaKF, chunk):
    Len = chunk.shape[0]
    alpha_env = np.zeros(Len)
    beta_env = np.zeros(Len)
    
    states_alpha = np.zeros((Len,2))
    states_beta = np.zeros((Len,2))
    
    for i in range(Len):
        state = kf.step(chunk[i]).mu
        
        states_alpha[i,0] = state[0,0]
        states_beta[i,0] = state[2,0]
        states_alpha[i,1] = state[1,0]
        states_beta[i,1] = state[3,0]
        
        alpha_env[i] = np.sqrt(state[0,0]*state[0,0]+state[1,0]*state[1,0])
        beta_env[i] = np.sqrt(state[2,0]*state[2,0]+state[3,0]*state[3,0])
    return alpha_env, beta_env, states_alpha,states_beta


def _get_ideal_H(n_fft, fs, band, delay=0):
    """
    Estimate ideal delayed analytic filter freq. response
    :param n_fft: length of freq. grid
    :param fs: sampling frequency
    :param band: freq. range to apply band-pass filtering
    :param delay: delay in samples
    :return: freq. response
    """
    w = np.arange(n_fft)
    H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
    H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
    return H


def _cLS(X, Y, lambda_=0):
    """
    Complex valued Least Squares with L2 regularisation
    """
    reg = lambda_ * np.eye(X.shape[1])
    b = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X.conj()) + reg), X.T.conj()), Y)
    return b




class CFIRBand:
    def __init__(self, band, fs, delay_ms=0, n_taps=500, n_fft=2000, reg_coeff=0):
        """
        Complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param smoother: smoother class instance to smooth output signal
        :param delay_ms: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :param reg_coeff: least squares L2 regularisation coefficient
        """
        H = _get_ideal_H(n_fft, fs, band, int(delay_ms * fs / 1000))
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        self.b = _cLS(F, H, reg_coeff)
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b) - 1)
        #self.smoother = smoother

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, zi=self.zi)
        #y = self.smoother.apply(np.abs(y))
        return y


class ExponentialSmoother:
    def __init__(self, factor):
        self.a = [1, -factor]
        self.b = [1 - factor]
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1,))

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, zi=self.zi)
        return y



from scipy.signal import firwin
'''
def ideal_envelope(fc,srate, y, bandwidth = 4.0):
    
    n_taps = 250
    #n_fft = 2500
    #reg_coeff = 0
    #H = _get_ideal_H(n_fft, srate, [fc-bandwidth/2.0,fc+bandwidth/2.0], 0)
    #F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
    #b = _cLS(F, H, reg_coeff)
    
    
    b= firwin(numtaps = n_taps, cutoff = [fc-bandwidth/2.0,fc+bandwidth/2.0], pass_zero=False, fs=srate)

    
    
    a = np.array([1.])
    
    y = hilbert(lfilter(b, a, y))
    
    envelope = np.zeros(np.shape(y)[0])
     
    envelope[:-n_taps//2] =np.abs(y)[n_taps//2:]
    #envelope[n_taps//2:] = np.abs(y)[:-n_taps//2]
    
    
    
    return envelope

'''

#from scipy.signal import bytter, hilbert,filtfilt
def ideal_envelope(fc,srate, y, bandwidth = 4.0):
    
    b,a = butter(3,[fc-bandwidth/2.0,fc+bandwidth/2.0], fs = srate, btype = 'bandpass')
    
    filtered = filtfilt(b,a,y)
    
    hilb = hilbert(filtered)
    
    envelope = np.abs(hilb)
    
    return envelope


#x - gt, y - delayed signal
def xcor(x,y,bias, bad_idx):
    
    #??????
    x[:500] = 0
    x[-500:] = 0
    y[:500] = 0
    y[-500:] = 0
    #y[bad_idx] = 0
    cor = np.zeros(bias+1)
    for i, b in enumerate(np.arange(bias+1)):
        #if b<0:
        #    cor[i] = np.sum(x[:b]*y[-b:])/(np.linalg.norm(x[:b])*np.linalg.norm(y[-b:]))
        if b == 0:
            cor[i] = np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))

        else:
            cor[i] = np.sum(x[:-b] * y[b:]) / (np.linalg.norm(x[:-b]) * np.linalg.norm(y[b:]))
        #if b >0:
        #    cor[i] = np.sum(x[b:]*y[:-b])/(np.linalg.norm(x[b:])*np.linalg.norm(y[:-b]))
        
    return cor


#return phase envelope loss and phase loss
def phase_envelope_loss(srate,fc,envelope,y, bad_idx):
    
    gt_env = ideal_envelope(fc,srate,y)
    
    #gt_env /= np.linalg.norm(gt_env)
    #envelope /= np.linalg.norm(envelope)
    
    bias =int(round(srate))
    envecor= xcor(gt_env,envelope,bias, bad_idx)
    plt.figure()
    plt.plot(gt_env/np.linalg.norm(gt_env))
    plt.plot(envelope/np.linalg.norm(envelope))
    plt.figure()
    plt.plot(envecor)
    plt.show()
    #envecor= np.correlate(gt_env,envelope,mode = 'same')
    
    #plt.figure()
    #plt.plot(envecor)
    
    cor = np.max(envecor)
    lat= np.argmax(envecor)#-bias
    
    return cor, lat
    












