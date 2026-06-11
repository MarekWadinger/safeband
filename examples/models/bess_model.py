import collections

from river import base


def C_to_K(T):
    if isinstance(T, list):
        return [C_to_K(x) for x in T]
    return T + 273.15


def K_to_C(T):
    if isinstance(T, list):
        return [K_to_C(x) for x in T]
    return T - 273.15


def calcul_Q(P):
    return 0.0003667 * abs(P) ** 2 + 0.005 * abs(P)


def bess_model(T_bat_0, P, Tout, q_fan, q_circ_fan, q_cool, q_heat, Ts):
    """Args:
        T_bat_0 (float): initial condition/temperature measurement
        P (float): demanded power
        Tout (float): Temperature outside
        *q (list[int]): [q_fan, q_circ_fan, q_cool, q_heat]
        Ts (int): Sampling time in seconds.

    Returns:
        float: new temperature

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
    def __init__(self, model=bess_model) -> None:
        self.buffer = collections.deque(maxlen=1)
        self.model = model

    def learn_one(self, x: dict):
        self.buffer.append(x)
        return self

    def transform_one(self, x: dict) -> dict:
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
