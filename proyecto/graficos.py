# -*- coding: utf-8 -*-
"""Gráficos interactivos (Plotly) equivalentes a los 4 gráficos del informe."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelo import Modelo

AZUL, ROJO, NARANJA, VERDE, MAGENTA, VIOLETA = "#2563eb", "#dc2626", "#ea580c", "#15803d", "#c026d3", "#7c3aed"
ORO = "#f5b800"


def _flechas(x, y, dx, dy, color, name, ancho=2, cabeza=0.22, showlegend=True):
    """Campo de flechas como una sola traza (segmentos separados por None)."""
    x, y, dx, dy = (np.asarray(v, float).ravel() for v in (x, y, dx, dy))
    xs, ys = [], []
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        xf, yf = xi + dxi, yi + dyi
        L = np.hypot(dxi, dyi)
        if L == 0:
            continue
        ux, uy = dxi / L, dyi / L
        h = cabeza * L
        for s in (+1, -1):
            ang = np.pi * 5 / 6 * s
            c, sn = np.cos(ang), np.sin(ang)
            bx, by = ux * c - uy * sn, ux * sn + uy * c
            xs += [xf, xf + h * bx, None]
            ys += [yf, yf + h * by, None]
        xs += [xi, xf, None]
        ys += [yi, yf, None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=ancho), name=name,
                      showlegend=showlegend, hoverinfo="skip")


def _base(fig, titulo, x="V1", y="V2", h=620):
    fig.update_layout(title=titulo, template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=60),
                      legend=dict(orientation="h", y=-0.16))
    fig.update_xaxes(title=x, scaleanchor="y", scaleratio=1, zeroline=False, gridcolor="rgba(0,0,0,.08)")
    fig.update_yaxes(title=y, zeroline=False, gridcolor="rgba(0,0,0,.08)")
    return fig


def _centroide(m: Modelo, size=16):
    return go.Scatter(x=[m.mu1], y=[m.mu2], mode="markers", name="Centroide",
                      marker=dict(symbol="star", size=size, color=ORO, line=dict(color="black", width=1)))


def _elipse(m: Modelo, k=None, **kw):
    t = np.linspace(0, 2 * np.pi, 600)
    k = m.c if k is None else k
    return go.Scatter(x=m.mu1 + m.sigma1 * np.sqrt(k) * np.cos(t), y=m.mu2 + m.sigma2 * np.sqrt(k) * np.sin(t),
                      mode="lines", **kw)


def grafico1(m, legit=None, fraud=None, simulado=False):
    fig = go.Figure()
    if legit is not None and len(legit):
        nombre = f"Legítimas ({'simuladas' if simulado else 'muestra'} {len(legit):,})".replace(",", ".")
        fig.add_trace(go.Scattergl(x=legit.V1, y=legit.V2, mode="markers", name=nombre,
                                   marker=dict(size=3.5, color=AZUL, opacity=.25)))
    if fraud is not None and len(fraud):
        fig.add_trace(go.Scattergl(x=fraud.V1, y=fraud.V2, mode="markers",
                                   name=f"Fraudes (muestra {len(fraud):,})".replace(",", "."),
                                   marker=dict(size=5, color=ROJO, opacity=.7)))
    fig.add_trace(_elipse(m, name=f"Frontera E_d = {m.c:g}", line=dict(color="black", width=2.5)))
    fig.add_trace(_centroide(m))
    return _base(fig, "Transacciones y frontera de normalidad", "Componente principal V1", "Componente principal V2", 650)


def grafico2(m, t0=None):
    fig = make_subplots(rows=1, cols=2, column_widths=[.5, .5], subplot_titles=("Frontera y tangentes v(t)", "Perfil de curvatura en [0, 2π]"))
    t = np.linspace(0, 2 * np.pi, 600)
    xe, ye = m.r(t)
    fig.add_trace(go.Scatter(x=xe, y=ye, mode="lines", name="Frontera r(t)", line=dict(color="navy", width=2.5)), 1, 1)
    fig.add_trace(_centroide(m, 13), 1, 1)
    ti = np.linspace(0, 2 * np.pi, 9)[:-1]
    xx, yy = m.r(ti); vx, vy = m.v(ti)
    sc = .65 * min(m.a, m.b) / np.hypot(vx, vy)
    fig.add_trace(_flechas(xx, yy, vx * sc, vy * sc, NARANJA, "Tangentes v(t)", 3), 1, 1)
    if t0 is not None:
        px, py = m.r(t0); vx0, vy0 = m.v(t0); ax_, ay_ = m.acel(t0)
        n = np.hypot(vx0, vy0)
        fig.add_trace(_flechas([px], [py], [vx0 / n * .9 * min(m.a, m.b) / 2], [vy0 / n * .9 * min(m.a, m.b) / 2], VERDE, "v(t₀) (dirección)", 4), 1, 1)
        fig.add_trace(_flechas([px], [py], [ax_ * .35], [ay_ * .35], MAGENTA, "a(t₀) (escala 0,35)", 4), 1, 1)
        fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="r(t₀)", marker=dict(size=11, color=VERDE, line=dict(color="white", width=2))), 1, 1)
    ts = np.linspace(0, 2 * np.pi, 800)
    fig.add_trace(go.Scatter(x=ts, y=m.kappa(ts), mode="lines", name="κ(t)", line=dict(color=VIOLETA, width=2.5)), 1, 2)
    fig.add_trace(go.Scatter(x=[0, np.pi, np.pi / 2, 3 * np.pi / 2], y=[m.kappa_max] * 2 + [m.kappa_min] * 2, mode="markers+text",
                             text=[f"máx {m.kappa_max:.4f}", f"máx {m.kappa_max:.4f}", f"mín {m.kappa_min:.4f}", f"mín {m.kappa_min:.4f}"],
                             textposition="top right", marker=dict(size=9, color="black"), name="Vértices", showlegend=False), 1, 2)
    if t0 is not None:
        fig.add_trace(go.Scatter(x=[t0], y=[m.kappa(t0)], mode="markers", marker=dict(size=12, color=VERDE, line=dict(color="white", width=2)), name="κ(t₀)", showlegend=False), 1, 2)
    fig.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="V1", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="V2", row=1, col=1)
    fig.update_xaxes(title_text="Parámetro t", tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                     ticktext=["0", "π/2", "π", "3π/2", "2π"], range=[0, 2 * np.pi], row=1, col=2)
    fig.update_yaxes(title_text="Curvatura κ(t)", row=1, col=2)
    return fig


def grafico3(m, punto=None):
    fig = go.Figure()
    niveles = [0.5, 1, 2, 3, 4, m.c, 8, 10]
    cmap = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58"]
    for k, col in zip(niveles, cmap):
        es_c = abs(k - m.c) < 1e-9
        fig.add_trace(_elipse(m, k, name=f"E_d = {k:g}" + (" (frontera)" if es_c else ""),
                              line=dict(color="black" if es_c else col, width=3 if es_c else 1.8, dash="dash" if es_c else "solid"),
                              hovertemplate=f"E_d = {k:g}<extra></extra>"))
    xs = m.mu1 + np.linspace(-1, 1, 11) * 1.15 * m.a
    ys = m.mu2 + np.linspace(-1, 1, 11) * 1.15 * m.b
    X, Y = np.meshgrid(xs, ys)
    gx, gy = m.grad_E(X, Y)
    n = np.hypot(gx, gy); n[n == 0] = 1
    esc = .42 * min(m.a, m.b) / 3
    fig.add_trace(_flechas(X, Y, gx / n * esc * 1.6, gy / n * esc * 1.6, ROJO, "Dirección de ∇E_d", 1.5, .3))
    if punto is not None:
        px, py = punto
        gpx, gpy = m.grad_E(px, py); gn = np.hypot(gpx, gpy) or 1
        fig.add_trace(_flechas([px], [py], [gpx / gn * m.a * .35], [gpy / gn * m.a * .35], VERDE, "∇E_d en el punto", 4))
        fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="Punto evaluado", marker=dict(size=11, color=VERDE, line=dict(color="white", width=2))))
    fig.add_trace(_centroide(m))
    return _base(fig, "Curvas de nivel y campo gradiente", h=680)


def grafico4(m):
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 600)
    xe, ye = m.r(t)
    fig.add_trace(go.Scatter(x=xe, y=ye, mode="lines", fill="toself", fillcolor="rgba(56,189,248,.25)",
                             line=dict(color="black", width=2.5), name="Región R: E_d ≤ c / contorno C"))
    xg = np.linspace(m.mu1 - m.a, m.mu1 + m.a, 15); yg = np.linspace(m.mu2 - m.b, m.mu2 + m.b, 15)
    Xg, Yg = np.meshgrid(xg, yg)
    ins = m.E_d(Xg, Yg) <= m.c
    Fx, Fy = m.grad_E(Xg, Yg)
    k = 0.62 * (2 * m.a / 14) / max(np.hypot(Fx[ins], Fy[ins]).max(), 1e-12)
    fig.add_trace(_flechas(Xg[ins], Yg[ins], Fx[ins] * k, Fy[ins] * k, AZUL, "Campo F = ∇E_d", 1.5, .3))
    tn = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    xn, yn = m.r(tn); nx, ny = m.normal_ext(tn)
    fig.add_trace(_flechas(xn, yn, .85 * nx, .85 * ny, MAGENTA, "Normales exteriores unitarias", 3.5, .3))
    to = np.pi / 4
    xo, yo = m.r(to); vx, vy = m.v(to); nv = np.hypot(vx, vy)
    fig.add_trace(_flechas([xo], [yo], [.9 * vx / nv], [.9 * vy / nv], VERDE, "Orientación positiva (antihoraria)", 5))
    fig.add_trace(_centroide(m, 14))
    return _base(fig, "Región R, campo F y normales exteriores", h=680)


def figura_matriz_confusion(cm):
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fila = [cm["tn"] + cm["fp"], cm["fn"] + cm["tp"]]
    txt = [[f"{z[i][j]:,}<br>({100 * z[i][j] / fila[i]:.2f}% de la fila)".replace(",", ".") for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=[[100 * z[i][j] / fila[i] for j in range(2)] for i in range(2)], zmin=0, zmax=100, x=["Dentro (E_d ≤ c)", "Fuera (E_d > c)"],
                               y=["Legítimas", "Fraudes"], text=txt, texttemplate="%{text}", colorscale=[[0, "#f1f5f9"], [1, "#93c5fd"]], textfont=dict(color="#0f172a", size=15),
                               showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10))
    return fig
