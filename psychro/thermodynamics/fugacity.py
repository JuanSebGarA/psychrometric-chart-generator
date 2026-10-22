import numpy as np
from psychro.thermodynamics.pure_properties import Pv_antoine, kappa, vvc, kh
from psychro.thermodynamics.virial import (
    bgg, bvv, bgv,
    cggg, cggv, cgvv, cvvv
)

def fug(T: float, xgs: float, P: float, R: float) -> float:
    """
    Calcula el logaritmo natural del factor de corrección por fugacidad (ln(f_c)).

    Parámetros:
        T   : Temperatura en Kelvin
        xgs : Fracción molar del gas en saturación
        P   : Presión total en atm
        R   : Constante universal de los gases (atm cm3 / mol K)
    """
    k = kappa(T - 273.15) * 101325.0  # Convertir compresibilidad
    Pv = Pv_antoine(T)
    Vvc = vvc(T)
    Kh = kh(T)

    # Coeficientes viriales a temperatura T
    Bgg = bgg(T)
    Bvv = bvv(T)
    Bgv = bgv(T)
    Cggg = cggg(T)
    Cggv = cggv(T)
    Cgvv = cgvv(T)
    Cvvv = cvvv(T)

    # Términos de la ecuación de fugacidad
    term1 = (((1.0 + k * Pv) * (P - Pv) - (0.5 * k * ((P**2) - (Pv**2)))) / (R * T)) * Vvc
    term2 = np.log(1.0 - (Kh * xgs * P))
    term3 = ((2.0 * (xgs**3) * (2.0 - 3.0 * xgs) * (P**2)) / ((R * T)**2)) * (Bgg * Bgv)
    term4 = (((xgs**2) * P) / (R * T)) * Bgg
    term5 = (((2.0 * xgs**2) * P) / (R * T)) * Bgv
    term6 = ((P - Pv - ((xgs**2) * P)) / (R * T)) * Bgg
    term7 = ((3.0 * (xgs**2) * (1.0 + 2.0 * xgs) * (P**2)) / (2.0 * (R * T)**2)) * Cggv
    term8 = ((3.0 * (xgs**2) * (1.0 - xgs) * (P**2)) / ((R * T)**2)) * Cgvv
    term9 = ((((1.0 + 2.0 * xgs) * ((1.0 - xgs)**2) * (P**2)) - (Pv**2)) / (2.0 * (R * T)**2)) * Cvvv
    term10 = ((3.0 * (xgs**4) * (P**2)) / (2.0 * (R * T)**2)) * (Bgg**2)
    term11 = (((xgs**2) * (1.0 - 3.0 * xgs) * (1.0 - xgs) * (P**2)) / (R * T)**2) * (Bgg * Bvv)
    term12 = ((6.0 * (xgs**2) * ((1.0 - xgs)**2) * (P**2)) / ((R * T)**2)) * (Bvv * Bgv)
    term13 = (((2.0 * xgs**2) * (1.0 - 3.0 * xgs) * (1.0 - xgs) * (P**2)) / (R * T)**2) * (Bgv**2)
    term14 = (((xgs**3) * (P**2)) / (R * T)**2) * Cggg
    term15 = (((Pv**2) - ((1.0 + 3.0 * xgs) * ((1.0 - xgs)**3) * (P**2))) / (2.0 * (R * T)**2)) * (Bvv**2)

    ln_f = (
        term1 + term2 - term3 + term4 - term5 - term6 + term7
        - term8 - term9 - term10 - term11 + term12 - term13 + term14 - term15
    )
    return ln_f