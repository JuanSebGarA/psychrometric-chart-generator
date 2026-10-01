import json
from pathlib import Path
import numpy as np

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "property_constants.json"

# Carga de constantes físicas desde JSON
with open(DATA_PATH, "r", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

# Constantes Universales
R = 0.08205746 * 1000  # atm * cm3 / (mol * K)
T_REF = 273.15         # K

# Masas Molares (kg/kmol)
M_DRY_AIR = CONSTANTS["molecular_weights"]["dry_air"]
M_WATER = CONSTANTS["molecular_weights"]["water"]

# Entalpías de Referencia
HOG = CONSTANTS["reference_enthalpies"]["hog"]
HOV = CONSTANTS["reference_enthalpies"]["hov"]

# Ponderación del Cp del Aire Seco mezcla N2, O2, Ar, CO2, Ne
cp_cfg = CONSTANTS["cp_coefficients"]
fracs = CONSTANTS["dry_air_fractions"]

CP_G = (
    np.array(cp_cfg["N2"]) * fracs["N2"] +
    np.array(cp_cfg["O2"]) * fracs["O2"] +
    np.array(cp_cfg["Ar"]) * fracs["Ar"] +
    np.array(cp_cfg["CO2"]) * fracs["CO2"] +
    np.array(cp_cfg["Ne"]) * fracs["Ne"]
)

# Cp del Vapor de Agua
CP_V = np.array(cp_cfg["VAPOR"])