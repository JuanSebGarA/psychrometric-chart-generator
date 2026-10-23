import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from psychro.solvers.saturation import calculate_relative_humidity_array
from psychro.solvers.adiabatic import calculate_adiabatic_saturation_grid


def build_matplotlib_chart(
    P: float = 1.0,
    T_min_C: float = 0.0,
    T_max_C: float = 50.0,
    num_T_points: int = 100
) -> plt.Figure:
    """
    Genera la carta psicrométrica estática con Matplotlib.
    """
    T_P = np.linspace(T_min_C, T_max_C, num_T_points)
    T_K = T_P + 273.15
    phi = np.linspace(0.1, 0.9, 9)

    Hs, Hr = calculate_relative_humidity_array(T_K, phi, P)
    T_ad, H_1R, _ = calculate_adiabatic_saturation_grid(T_K, P)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Curva de Saturación
    ax.plot(T_P, Hs, 'k-', linewidth=1.5, label='Saturación (100% Hr)')

    # Curvas de Humedad Relativa
    for j in range(len(phi)):
        ax.plot(T_P, Hr[:, j], 'k-', linewidth=0.8)

    # Líneas de Saturación Adiabática (Tad)
    for k in range(0, num_T_points, 4):
        ax.plot(T_ad[k, :] - 273.15, H_1R[k, :], 'b-', linewidth=0.8)

    # Configuración de Ejes
    ax.set_title('Carta Psicrométrica Agua-Aire')
    ax.set_xlabel('Temperatura °C')
    ax.set_ylabel('Humedad absoluta (kg/kg)')
    ax.set_xticks(np.arange(T_min_C, T_max_C + 5, 5))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim([0, 0.04])
    ax.set_xlim([T_min_C, T_max_C])
    ax.set_yticks(np.arange(0, 0.042, 0.002))
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig


def build_plotly_chart(
    P: float = 1.0,
    T_min_C: float = 0.0,
    T_max_C: float = 50.0,
    num_T_points: int = 100
) -> go.Figure:
    """
    Genera la carta psicrométrica interactiva con Plotly.
    """
    T_P = np.linspace(T_min_C, T_max_C, num_T_points)
    T_K = T_P + 273.15
    phi = np.linspace(0.1, 0.9, 9)

    Hs, Hr = calculate_relative_humidity_array(T_K, phi, P)
    T_ad, H_1R, _ = calculate_adiabatic_saturation_grid(T_K, P)

    fig = go.Figure()

    # Curva de Saturación Límite
    fig.add_trace(
        go.Scatter(
            x=T_P,
            y=Hs,
            mode='lines',
            line=dict(color='black', width=2),
            name='Saturación'
        )
    )

    # Curvas de Humedad Relativa (Hr)
    for j in range(len(phi)):
        fig.add_trace(
            go.Scatter(
                x=T_P,
                y=Hr[:, j],
                mode='lines',
                line=dict(color='black', width=0.8),
                name=f'{int(phi[j]*100)}% Hr',
                showlegend=False
            )
        )

    # Líneas de Saturación Adiabática (Tad)
    for k in range(0, num_T_points, 4):
        fig.add_trace(
            go.Scatter(
                x=T_ad[k, :] - 273.15,
                y=H_1R[k, :],
                mode='lines',
                line=dict(color='blue', width=0.8),
                showlegend=False
            )
        )

    # Formato de la Leyenda y Ejes
    fig.update_layout(
        title='Carta Psicrométrica Agua-Aire (Interactiva)',
        xaxis=dict(
            title='Temperatura °C',
            tickvals=list(np.arange(T_min_C, T_max_C + 5, 5)),
            range=[T_min_C, T_max_C],
            showgrid=True
        ),
        yaxis=dict(
            title='Humedad absoluta (kg/kg)',
            side='right',
            range=[0, 0.04],
            tickvals=list(np.arange(0, 0.042, 0.002)),
            showgrid=True
        ),
        width=900,
        height=500,
        showlegend=False
    )

    return fig