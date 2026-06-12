"""Physics-based battery energy storage system (BESS) thermal model."""

import collections
from collections.abc import Callable
from typing import overload

from river import base


@overload
def C_to_K(T: float) -> float: ...


@overload
def C_to_K(T: list[float]) -> list[float]: ...


def C_to_K(T: float | list[float]) -> float | list[float]:
    """Convert Celsius temperature (or list thereof) to Kelvin."""
    if isinstance(T, list):
        return [C_to_K(x) for x in T]
    return T + 273.15


@overload
def K_to_C(T: float) -> float: ...


@overload
def K_to_C(T: list[float]) -> list[float]: ...


def K_to_C(T: float | list[float]) -> float | list[float]:
    """Convert Kelvin temperature (or list thereof) to Celsius."""
    if isinstance(T, list):
        return [K_to_C(x) for x in T]
    return T - 273.15


def calcul_Q(P: float) -> float:
    """Estimate heat generation (kW) from demanded power P."""
    return 0.0003667 * abs(P) ** 2 + 0.005 * abs(P)


def bess_model(
    T_bat_0: float,
    P: float,
    Tout: float,
    q_fan: float,
    q_circ_fan: float,
    q_cool: float,
    q_heat: float,
    Ts: int,
) -> float:
    """Compute the next battery cell temperature using a lumped thermal model.

    Args:
        T_bat_0 (float): Current battery temperature in Celsius.
        P (float): Demanded power in kW (positive = charging).
        Tout (float): Ambient temperature in Celsius.
        q_fan (float): External ventilation fan duty fraction.
        q_circ_fan (float): Internal circulation fan duty fraction.
        q_cool (float): Cooling unit activation flag (0 or 1).
        q_heat (float): Heating unit activation flag (0 or 1).
        Ts (int): Sampling time in seconds.

    Returns:
        float: Estimated battery temperature at the next time step (Celsius).

    """
    # Model constants
    cp = 1.012  # kJ/s
    cp_b = 4
    Vb_max = 1000 / 3600
    Vc_max = 1000 / 3600  # m3/s
    rho = 1.2  # kg/m3
    P_cool = -5  # kW
    P_heat = 2  # kW
    m_bat = 827  # kg
    q_inner_fans = 1.1

    # Heat emissions are higher during charging phase
    c_scale = 4 if P >= 0 else 1

    Q_bat = c_scale * calcul_Q(P)
    return T_bat_0 + Ts * (
        q_fan * Vb_max * rho * cp * (Tout - T_bat_0)
        + Vc_max * q_circ_fan * rho * cp * C_to_K(T_bat_0)
        + q_circ_fan * (P_cool * q_cool + P_heat * q_heat)
        + c_scale * Q_bat
        + q_inner_fans
        - (Vb_max * q_fan + Vc_max * q_circ_fan) * rho * cp * C_to_K(T_bat_0)
    ) / (m_bat * cp_b)


class BESS(base.Transformer):
    """River transformer that augments a sample with modelled temp diff."""

    def __init__(self, model: Callable[..., float] = bess_model) -> None:
        """Initialise with an optional custom thermal model callable."""
        self.buffer = collections.deque(maxlen=1)
        self.model = model

    def learn_one(self, x: dict) -> None:
        """Store the current sample for use as the previous-step reference."""
        self.buffer.append(x)

    def transform_one(self, x: dict) -> dict:
        """Add modelled temperature residual to the current sample."""
        if len(self.buffer) != 0:
            x_prev = self.buffer[-1]
            args = [
                x_prev["Avg. Cell Temperature"],
                x_prev["String Power"],
                x_prev["Ambient Temperature"],
                x_prev["Battery part Fan Vb1 Feedback"],
                x_prev["HVAC Battery part Circulation Fan V1"],
                x_prev["HVAC Battery part cooling"],
                x_prev["HVAC Battery part heating"],
                60,
            ]
            x["Diff Cell Temperature"] = x[
                "Avg. Cell Temperature"
            ] - self.model(*args)
        else:
            x["Diff Cell Temperature"] = 0
        return x
