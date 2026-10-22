import numpy as np
from psychro.config import R, M_DRY_AIR, M_WATER
from psychro.thermodynamics.pure_properties import Pv_antoine
from psychro.thermodynamics.fugacity import fug

def solve_saturation_point(T_K: float, P: float = 1.0) -> tuple[float, float, float]:
    """
    Resuelve mediante iteración de punto fijo la fracción molar y la humedad de saturación
    corregida por fugacidad para una temperatura T (K) y presión P (atm).

    Retorna:
        (Hs, xgs, xvs): Humedad de saturación, fracción molar de gas seco y fracción molar de vapor.
    """
    Pv_I = Pv_antoine(T_K)
    f = 1.0
    tol = 1e-6
    max_iter = 100

    xgs = (P - (f * Pv_I)) / P
    xvs = f * Pv_I / P

    for _ in range(max_iter):
        xgs = (P - (f * Pv_I)) / P
        xvs = f * Pv_I / P
        
        ln_f = fug(T_K, xgs, P, R)
        f_c = np.exp(ln_f)
        
        error = abs(f - f_c)
        if error < tol:
            break
        f = f_c

    Hs = (xvs * M_WATER) / (xgs * M_DRY_AIR)
    return Hs, xgs, xvs


def calculate_relative_humidity_array(
    T_array_K: np.ndarray, 
    phi_array: np.ndarray, 
    P: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la matriz de humedad absoluta para distintos valores de humedad relativa (phi).

    Retorna:
        (Hs, Hr): Humedad de saturación vectorizada y matriz Hr [len(T), len(phi)].
    """
    num_T = len(T_array_K)
    num_phi = len(phi_array)

    Hs = np.zeros(num_T)
    Hr = np.zeros((num_T, num_phi))

    for i in range(num_T):
        T_i = T_array_K[i]
        Pv_I_i = Pv_antoine(T_i)
        
        # Obtener punto de saturación real
        Hs_i, xgs_i, xvs_i = solve_saturation_point(T_i, P)
        Hs[i] = Hs_i

        # Calcular curvas para cada phi (Humedad Relativa)
        for j in range(num_phi):
            phi_j = phi_array[j]
            xv_ij = phi_j * xvs_i
            x_gas_ij = 1.0 - xv_ij
            
            ln_fH = fug(T_i, x_gas_ij, P, R)
            f_cH = np.exp(ln_fH)
            
            Hr[i, j] = (Hs_i * phi_j * (1.0 - (f_cH * Pv_I_i / P))) / (1.0 - (phi_j * Pv_I_i / P))

    return Hs, Hr