import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from numpy.polynomial import Polynomial

from psychro.config import (
    R, M_DRY_AIR, M_WATER, HOG, HOV, CP_G, CP_V
)
from psychro.thermodynamics.virial import (
    bgg, bgv, bvv, cggg, cggv, cgvv, cvvv,
    dbgg_dT, dbgv_dT, dbvv_dT, dcggg_dT, dcggv_dT, dcgvv_dT, dcvvv_dT
)
from psychro.thermodynamics.enthalpy import hf, hs2
from psychro.solvers.saturation import solve_saturation_point


def Tad_objective(
    T1: float,
    H1: float,
    hog: float,
    hov: float,
    R_val: float,
    Cpg: np.ndarray,
    Cpv: np.ndarray,
    Mwg: float,
    Mwv: float,
    P: float,
    Hs: float,
    hf_val: float,
    hs2_val: float
) -> float:
    """
    Función objetivo para resolver implícitamente T_ad con fsolve.
    """
    T1_val = float(T1[0]) if isinstance(T1, np.ndarray) else float(T1)

    xv = 1.0 / (1.0 + Mwv / (H1 * Mwg))
    xg = 1.0 - xv

    # Coeficientes Viriales y Derivadas a T1
    Bgg, Bgv, Bvv = bgg(T1_val), bgv(T1_val), bvv(T1_val)
    Cggg, Cggv, Cgvv, Cvvv = cggg(T1_val), cggv(T1_val), cgvv(T1_val), cvvv(T1_val)

    dBgg_dT, dBgv_dT, dBvv_dT = dbgg_dT(T1_val), dbgv_dT(T1_val), dbvv_dT(T1_val)
    dCggg_dT, dCggv_dT, dCgvv_dT, dCvvv_dT = dcggg_dT(T1_val), dcggv_dT(T1_val), dcgvv_dT(T1_val), dcvvv_dT(T1_val)

    # Integración exacta de Cp usando quad y Polynomial
    inteCpg, _ = quad(Polynomial(Cpg), 273.15, T1_val)
    inteCpv, _ = quad(Polynomial(Cpv), 273.15, T1_val)

    # Coeficientes de Mezcla Viriales
    Bm = (xg**2) * Bgg + 2.0 * xg * xv * Bgv + (xv**2) * Bvv
    Cm = (xg**3) * Cggg + 3.0 * (xg**2) * xv * Cggv + 3.0 * xg * (xv**2) * Cgvv + (xv**3) * Cvvv

    dBm_dT = (xg**2) * dBgg_dT + 2.0 * xg * xv * dBgv_dT + (xv**2) * dBvv_dT
    dCm_dT = (xg**3) * dCggg_dT + 3.0 * (xg**2) * xv * dCggv_dT + 3.0 * xg * (xv**2) * dCgvv_dT + (xv**3) * dCvvv_dT

    # Raíces para determinar Volumen Molar (Vm)
    V = np.roots([1.0, -(R_val * T1_val / P), -(R_val * T1_val / P * Bm), -(R_val * T1_val / P * Cm)])
    Vm = float(V[0].real)

    # Entalpia de entrada
    Hs1_ = (
        xg * (hog + inteCpg)
        + xv * (hov + inteCpv)
        + R_val * T1_val * 101325.0 / 1e6 * ((T1_val * dBm_dT - Bm) * (1.0 / Vm) + (T1_val / 2.0 * dCm_dT - Cm) * (1.0 / (Vm**2)))
    )
    Hs1 = Hs1_ / (xg * Mwg)

    # Balance de Energía Adiabático
    return Hs1 + (Hs - H1) * hf_val - hs2_val


def vm(T: float, H1: float, R_val: float, P: float, Mwg: float, Mwv: float) -> float:
    """Calcula el volumen húmedo específico (m3/kg)."""
    xvs = 1.0 / (1.0 + Mwv / (H1 * Mwg))
    xgs = 1.0 - xvs

    Bgg, Bgv, Bvv = bgg(T), bgv(T), bvv(T)
    Cggg, Cggv, Cgvv, Cvvv = cggg(T), cggv(T), cgvv(T), cvvv(T)

    Bm = (xgs**2) * Bgg + 2.0 * xgs * xvs * Bgv + (xvs**2) * Bvv
    Cm = (xgs**3) * Cggg + 3.0 * (xgs**2) * xvs * Cggv + 3.0 * xgs * (xvs**2) * Cgvv + (xvs**3) * Cvvv

    V = np.roots([1.0, -(R_val * T / P), -(R_val * T / P * Bm), -(R_val * T / P * Cm)])
    Vm = float(V[0].real)
    return Vm / (xgs * Mwg)


def calculate_adiabatic_saturation_grid(
    T_array_K: np.ndarray,
    P: float = 1.0,
    num_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera la malla de curvas de saturación adiabática (T_ad) y volumen húmedo.

    Retorna:
        (T_ad, H_1R, V_r): Matrices de temperaturas de saturación adiabática, humedades y volúmenes.
    """
    num_T = len(T_array_K)
    T_ad = np.zeros((num_T, num_points))
    H_1R = np.zeros((num_T, num_points))
    V_r = np.zeros((num_T, num_points))

    for i in range(num_T):
        T_i = T_array_K[i]
        Hs_i, xgs_i, xvs_i = solve_saturation_point(T_i, P)
        
        Hf = hf(T_i - 273.15)
        Hs2_val = hs2(HOG, HOV, R, T_i, CP_G, CP_V, xgs_i, xvs_i, M_DRY_AIR, P)
        
        x0 = T_i
        H1 = np.linspace(Hs_i, 1e-6, num=num_points)
        H_1R[i, :] = H1

        for k in range(num_points):
            fobj = fsolve(
                Tad_objective,
                x0,
                args=(H1[k], HOG, HOV, R, CP_G, CP_V, M_DRY_AIR, M_WATER, P, Hs_i, Hf, Hs2_val)
            )
            T_ad_val = float(fobj[0])
            T_ad[i, k] = T_ad_val
            x0 = T_ad_val
            V_r[i, k] = vm(T_ad_val, H1[k], R, P, M_DRY_AIR, M_WATER)

    return T_ad, H_1R, V_r