import numpy as np
from scipy.integrate import quad
from numpy.polynomial import Polynomial

from psychro.thermodynamics.virial import (
    bgg, bgv, bvv, cggg, cggv, cgvv, cvvv,
    dbgg_dT, dbgv_dT, dbvv_dT, dcggg_dT, dcggv_dT, dcgvv_dT, dcvvv_dT
)

def hf(T_C: float) -> float:
    """Calcula la entalpía de referencia / condensado (T_C en °C)."""
    hfo = -0.00115 + 4.21965 * T_C + 0.198652 * (T_C**2) - 2.93514e-6 * (T_C**3) + 1.003624e-7 * (T_C**4)
    landa = 2501.4 - 2.37341 * T_C + 0.000805207 * T_C - 1.69018e-5 * (T_C**3) + 1.81383e-8 * (T_C**4)
    return hfo - landa

def hs2(
    hog: float,
    hov: float,
    R: float,
    T: float,
    Cpg: np.ndarray,
    Cpv: np.ndarray,
    xg: float,
    xv: float,
    Mwg: float,
    P: float
) -> float:
    """
    Calcula la entalpía del aire húmedo a la salida considerando correcciones reales.
    
    Parámetros:
        hog, hov : Entalpías de referencia
        R        : Constante universal de los gases
        T        : Temperatura en Kelvin
        Cpg, Cpv : Arreglos de coeficientes polinomiales de Cp
        xg, xv   : Fracciones molares de gas seco y vapor
        Mwg      : Peso molecular del gas seco
        P        : Presión en atm
    """
    # Coeficientes Viriales y Derivadas a T
    Bgg, Bgv, Bvv = bgg(T), bgv(T), bvv(T)
    Cggg, Cggv, Cgvv, Cvvv = cggg(T), cggv(T), cgvv(T), cvvv(T)

    dBgg_dT, dBgv_dT, dBvv_dT = dbgg_dT(T), dbgv_dT(T), dbvv_dT(T)
    dCggg_dT, dCggv_dT, dCgvv_dT, dCvvv_dT = dcggg_dT(T), dcggv_dT(T), dcgvv_dT(T), dcvvv_dT(T)

    # Integración de Cp desde estado de referencia (273.15 K) hasta T
    inteCpg, _ = quad(Polynomial(Cpg), 273.15, T)
    inteCpv, _ = quad(Polynomial(Cpv), 273.15, T)

    # Coeficientes de Mezcla Viriales y sus derivadas
    Bm = (xg**2) * Bgg + 2.0 * xg * xv * Bgv + (xv**2) * Bvv
    Cm = (xg**3) * Cggg + 3.0 * (xg**2) * xv * Cggv + 3.0 * xg * (xv**2) * Cgvv + (xv**3) * Cvvv

    dBm_dT = (xg**2) * dBgg_dT + 2.0 * xg * xv * dBgv_dT + (xv**2) * dBvv_dT
    dCm_dT = (xg**3) * dCggg_dT + 3.0 * (xg**2) * xv * dCggv_dT + 3.0 * xg * (xv**2) * dCgvv_dT + (xv**3) * dCvvv_dT

    # Cálculo del Volumen Molar (Vm) a partir de las raíces del polinomio virial
    V = np.roots([1.0, -(R * T / P), -(R * T / P * Bm), -(R * T / P * Cm)])
    Vm = float(V[0].real)

    # Entalpía Residual de la Mezcla Real
    Hs2_ = (
        xg * (hog + inteCpg)
        + xv * (hov + inteCpv)
        + R * T * 101325.0 / 1e6 * ((T * dBm_dT - Bm) * (1.0 / Vm) + (T / 2.0 * dCm_dT - Cm) * (1.0 / (Vm**2)))
    )
    return Hs2_ / (xg * Mwg)