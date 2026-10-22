import numpy as np
from psychro.config import CONSTANTS

def Pv_antoine(T: float) -> float:
    """Calcula la presión de vapor del agua en atm (T en K)."""
    cfg = CONSTANTS["antoine_coefficients"]
    # Ecuación modificada de Antoine (retorna mm Hg)
    pv_mmHg = 10 ** (
        cfg["A"]
        - (cfg["B"] / T)
        - (cfg["C"] * np.log10(T))
        + (cfg["D"] * T)
        + (cfg["E"] * (T ** 2))
    )
    return pv_mmHg / 760.0  # Conversión a atm


def kappa(T_C: float) -> float:
    """Calcula la compresibilidad kappa (1/Pa) del condensado (T en °C)."""
    C = np.array([
        5.088496e1,
        6.163813e-1,
        1.459187e-3,
        2.008438e-5,
        -5.847727e-8,
        4.104110e-10,
        1.967348e-2,
    ])
    term_sum = sum(C[i] * (T_C ** i) for i in range(len(C) - 1))
    return (term_sum / (1.0 + C[6] * T_C)) * (10 ** -11)


def vvc(T: float) -> float:
    """Calcula el volumen molar del fluido que condensa en cm3/mol (T en K)."""
    C = np.array([
        -2403.36021,
        -1.40758895,
        0.1068287657,
        -2.914492351e-4,
        3.73497936e-6,
        -2.1203787e-10,
    ])
    term_sum = sum(C[i] * (T ** i) for i in range(len(C)))
    return (-61692.295 + 291.8088 * T) / term_sum


def kh(T: float) -> float:
    """Calcula la constante de Henry para el aire seco (T en K)."""
    coeffs = CONSTANTS["henry_coefficients"]
    
    kh_N2 = 1.0 / np.exp(coeffs["N2"][0] + (coeffs["N2"][1] / T) + (coeffs["N2"][2] * np.log(T)))
    kh_O2 = 1.0 / np.exp(coeffs["O2"][0] + (coeffs["O2"][1] / T) + (coeffs["O2"][2] * np.log(T)))
    kh_Ar = 1.0 / np.exp(coeffs["Ar"][0] + (coeffs["Ar"][1] / T) + (coeffs["Ar"][2] * np.log(T)))
    kh_CO2 = 1.0 / np.exp(coeffs["CO2"][0])
    kh_Ne = 1.0 / np.exp(coeffs["Ne"][0] + (coeffs["Ne"][1] / T) + (coeffs["Ne"][2] * np.log(T)))

    fracs = CONSTANTS["dry_air_fractions"]
    kh_aire = (
        (fracs["N2"] / kh_N2)
        + (fracs["O2"] / kh_O2)
        + (fracs["Ar"] / kh_Ar)
        + (fracs["CO2"] / kh_CO2)
        + (fracs["Ne"] / kh_Ne)
    )
    return kh_aire