import numpy as np

# --- SEGUNDOS COEFICIENTES VIRIALES ---

def bvv(T: float) -> float:
    """Segundo coeficiente virial del Agua (cm3/mol, T en K)."""
    return T * (0.05820 - 0.012234 * np.exp(1734.29 / T))

def bgg(T: float) -> float:
    """Segundo coeficiente virial del Aire (cm3/mol, T en K)."""
    return 34.9568 - (0.668772e4 / T) - (0.210141e7 / (T**2)) + (0.924746e8 / (T**3))

def bgv(T: float) -> float:
    """Segundo coeficiente virial Cruzado Aire-Agua (cm3/mol, T en K)."""
    return 32.366097 - (14113.8 / T) - (1.245535e6 / (T**2)) - (2.34789e9 / (T**4))

# --- TERCEROS COEFICIENTES VIRIALES ---

def cggg(T: float) -> float:
    """Tercer coeficiente virial del Aire (cm6/mol2, T en K)."""
    return 1.25975e3 - (1.90905e5 / T) + (6.32467e7 / (T**2))

def cvvv(T: float) -> float:
    """Tercer coeficiente virial del Agua (cm6/mol2, T en K)."""
    return (T**2) * (
        752.82
        - (2.31788 * np.exp(3645.09 / T))
        + (1.49755 * np.exp(3468.58 / T))
        - (14.2446 * np.exp(1734.29 / T))
    ) * 10e-4

def cggv(T: float) -> float:
    """Tercer coeficiente virial Cruzado 1 Aire-Aire-Agua (cm6/mol2, T en K)."""
    return (
        482.737
        + (1.05678e5 / T)
        - (6.56394e7 / (T**2))
        - (2.94442e10 / (T**3))
        - (3.19317e12 / (T**4))
    )

def cgvv(T: float) -> float:
    """Tercer coeficiente virial Cruzado 2 Aire-Agua-Agua (cm6/mol2, T en K)."""
    return -1e6 * np.exp(-10.728876 + (3478.02 / T) - (3.83383e5 / (T**2)) + (3.3406e7 / (T**3)))

# --- DERIVADAS ANALÍTICAS RESPECTO A LA TEMPERATURA (dT) ---

def dbgg_dT(T: float) -> float:
    return (6687.72 / (T**2)) + (4202820.0 / (T**3)) - (277423800.0 / (T**4))

def dbgv_dT(T: float) -> float:
    return (14113.8 / (T**2)) + (2491070.0 / (T**3)) + (9391560000.0 / (T**5))

def dbvv_dT(T: float) -> float:
    return -0.012234 * np.exp(1734.29 / T) + 0.0582 + (21.21730386 * np.exp(1734.29 / T) / T)

def dcggg_dT(T: float) -> float:
    return (190905.0 / (T**2)) - (126493400.0 / (T**3))

def dcvvv_dT(T: float) -> float:
    return 0.001 * (T**2) * (
        (24704.267334 * np.exp(1734.29 / T) / (T**2))
        - (5194.371979 * np.exp(3468.58 / T) / (T**2))
        + (8448.8812092 * np.exp(3645.09 / T) / (T**2))
    ) + 0.002 * T * (
        -14.2446 * np.exp(1734.29 / T)
        + 1.49755 * np.exp(3468.58 / T)
        - 2.31788 * np.exp(3645.09 / T)
        + 752.82
    )

def dcggv_dT(T: float) -> float:
    return (-105678.0 / (T**2)) + (131278800.0 / (T**3)) + (88332600000.0 / (T**4)) + (12772680000000.0 / (T**5))

def dcgvv_dT(T: float) -> float:
    return -21.9032397126109 * (
        (-3478.02 / (T**2)) + (766766.0 / (T**3)) - (100218000.0 / (T**4))
    ) * np.exp((3478.02 / T) - (383383.0 / (T**2)) + (33406000.0 / (T**3)))