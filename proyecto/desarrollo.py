# -*- coding: utf-8 -*-
"""Desarrollo paso a paso: fórmulas en LaTeX con los valores actuales sustituidos.
Se recalcula en cada interacción (umbral c, μ, σ, punto evaluado, t₀, etc.)."""
import numpy as np
import streamlit as st

from modelo import Modelo


def n(x, d=4):
    """Número para LaTeX con coma decimal."""
    return f"{x:.{d}f}".replace(".", "{,}")


def p(x, d=4):
    """Número entre paréntesis si es negativo."""
    return f"({n(x, d)})" if x < 0 else n(x, d)


def _caja(titulo):
    c = st.container(border=True)
    c.markdown(f"#### 📐 {titulo}")
    return c


def _paso(c, titulo, *latex, nota=None):
    c.markdown(f"**{titulo}**")
    if nota:
        c.caption(nota)
    for l in latex:
        c.latex(l)


# ----------------------------------------------------------------------------- Resumen / modelo
def modelo(m: Modelo):
    c = _caja("Desarrollo: de los datos a la frontera")
    _paso(c, "Paso 1 · Parámetros de las transacciones legítimas",
          rf"\mu_1={n(m.mu1, 6)},\ \ \mu_2={n(m.mu2, 6)},\ \ \sigma_1={n(m.sigma1, 6)},\ \ \sigma_2={n(m.sigma2, 6)}")
    _paso(c, "Paso 2 · Distancia estandarizada (modelo diagonal)",
          r"E_d(V_1,V_2)=\frac{(V_1-\mu_1)^2}{\sigma_1^2}+\frac{(V_2-\mu_2)^2}{\sigma_2^2}",
          rf"E_d(V_1,V_2)=\frac{{(V_1-{p(m.mu1, 6)})^2}}{{{n(m.sigma1 ** 2, 6)}}}+\frac{{(V_2-{p(m.mu2, 6)})^2}}{{{n(m.sigma2 ** 2, 6)}}}")
    _paso(c, "Paso 3 · Frontera: E_d = c",
          rf"E_d\le c={n(m.c, 3)}\quad\Longrightarrow\quad \frac{{(V_1-\mu_1)^2}}{{\sigma_1^2\,c}}+\frac{{(V_2-\mu_2)^2}}{{\sigma_2^2\,c}}\le 1",
          nota="Dividiendo por c queda la ecuación canónica de una elipse de semiejes a y b.")
    _paso(c, "Paso 4 · Semiejes",
          rf"a=\sigma_1\sqrt c={n(m.sigma1, 4)}\cdot\sqrt{{{n(m.c, 3)}}}={n(m.a, 4)}",
          rf"b=\sigma_2\sqrt c={n(m.sigma2, 4)}\cdot\sqrt{{{n(m.c, 3)}}}={n(m.b, 4)}")


# ----------------------------------------------------------------------------- Unidad 1
def unidad1(m: Modelo, t0: float):
    x, y = m.r(t0); vx, vy = m.v(t0); ax, ay = m.acel(t0)
    a2, b2 = m.a ** 2, m.b ** 2
    c = _caja("Desarrollo paso a paso (con tus valores)")
    _paso(c, "Paso 1 · Función vectorial de la frontera",
          rf"\vec r(t)=\langle \mu_1+a\cos t,\ \mu_2+b\sin t\rangle=\langle {n(m.mu1, 4)}+{n(m.a, 4)}\cos t,\ {n(m.mu2, 4)}+{n(m.b, 4)}\sin t\rangle",
          rf"\vec r(t_0={n(t0, 2)})=\langle {n(m.mu1, 4)}+{n(m.a, 4)}\cos({n(t0, 2)}),\ {n(m.mu2, 4)}+{n(m.b, 4)}\sin({n(t0, 2)})\rangle=\langle {n(x, 4)},\ {n(y, 4)}\rangle")
    _paso(c, "Paso 2 · Velocidad y aceleración (derivadas)",
          rf"\vec v(t)=\vec r\,'(t)=\langle -a\sin t,\ b\cos t\rangle=\langle -{n(m.a, 4)}\sin t,\ {n(m.b, 4)}\cos t\rangle",
          rf"\vec v(t_0)=\langle {n(vx, 4)},\ {n(vy, 4)}\rangle,\qquad \lVert\vec v(t_0)\rVert=\sqrt{{({n(vx, 4)})^2+({n(vy, 4)})^2}}={n(float(np.hypot(vx, vy)), 4)}",
          rf"\vec a(t)=\vec r\,''(t)=\langle -a\cos t,\ -b\sin t\rangle\ \Rightarrow\ \vec a(t_0)=\langle {n(ax, 4)},\ {n(ay, 4)}\rangle",
          nota="La aceleración es −(r − μ): siempre apunta al centroide.")
    _paso(c, "Paso 3 · Curvatura",
          rf"\kappa(t)=\frac{{\lVert\vec v\times\vec a\rVert}}{{\lVert\vec v\rVert^3}}=\frac{{ab}}{{(a^2\sin^2t+b^2\cos^2t)^{{3/2}}}}=\frac{{{n(m.a * m.b, 4)}}}{{({n(a2, 4)}\sin^2t+{n(b2, 4)}\cos^2t)^{{3/2}}}}",
          rf"\kappa(t_0)={n(float(m.kappa(t0)), 6)}\qquad R(t_0)=\frac{{1}}{{\kappa(t_0)}}={n(float(1 / m.kappa(t0)), 4)}",
          rf"\kappa_{{\max}}=\frac{{a}}{{b^2}}=\frac{{{n(m.a, 4)}}}{{{n(b2, 4)}}}={n(m.kappa_max, 6)}\ \ (t=0,\pi)\qquad \kappa_{{\min}}=\frac{{b}}{{a^2}}=\frac{{{n(m.b, 4)}}}{{{n(a2, 4)}}}={n(m.kappa_min, 6)}\ \ (t=\pi/2,\,3\pi/2)")
    L, _ = m.longitud()
    _paso(c, "Paso 4 · Longitud de arco",
          rf"L=\int_0^{{2\pi}}\lVert\vec v(t)\rVert\,dt=\int_0^{{2\pi}}\sqrt{{{n(a2, 4)}\sin^2t+{n(b2, 4)}\cos^2t}}\,dt\approx {n(L, 4)}",
          nota="Integral elíptica de 2.ª especie: sin primitiva elemental, se resuelve numéricamente (scipy.integrate.quad).")


# ----------------------------------------------------------------------------- Unidad 2
def unidad2(m: Modelo, v1: float, v2: float, ang_deg: float):
    E = float(m.E_d(v1, v2)); gx, gy = (float(z) for z in m.grad_E(v1, v2))
    u = np.deg2rad(ang_deg); Du = gx * np.cos(u) + gy * np.sin(u)
    k1, k2 = 2 / m.sigma1 ** 2, 2 / m.sigma2 ** 2
    c = _caja("Desarrollo paso a paso (con tus valores)")
    _paso(c, "Paso 1 · Derivadas parciales de E_d",
          rf"\frac{{\partial E_d}}{{\partial V_1}}=\frac{{2(V_1-\mu_1)}}{{\sigma_1^2}}={n(k1, 6)}\,(V_1-{p(m.mu1, 6)})",
          rf"\frac{{\partial E_d}}{{\partial V_2}}=\frac{{2(V_2-\mu_2)}}{{\sigma_2^2}}={n(k2, 6)}\,(V_2-{p(m.mu2, 6)})",
          rf"\nabla E_d=\langle {n(k1, 4)}(V_1-{p(m.mu1, 4)}),\ {n(k2, 4)}(V_2-{p(m.mu2, 4)})\rangle")
    _paso(c, f"Paso 2 · Evaluación en el punto (V₁, V₂) = ({n(v1, 4).replace('{,}', ',')}, {n(v2, 4).replace('{,}', ',')})",
          rf"E_d=\frac{{({n(v1, 4)}-{p(m.mu1, 4)})^2}}{{{n(m.sigma1 ** 2, 4)}}}+\frac{{({n(v2, 4)}-{p(m.mu2, 4)})^2}}{{{n(m.sigma2 ** 2, 4)}}}={n(E, 4)}",
          rf"\nabla E_d=\langle {n(k1, 4)}\cdot({n(v1 - m.mu1, 4)}),\ {n(k2, 4)}\cdot({n(v2 - m.mu2, 4)})\rangle=\langle {n(gx, 4)},\ {n(gy, 4)}\rangle",
          rf"\lVert\nabla E_d\rVert={n(float(np.hypot(gx, gy)), 4)}",
          nota=("Resultado: " + ("**alerta** (E_d > c)" if E > m.c + 1e-5 else "**sobre la frontera**" if abs(E - m.c) <= 1e-5 else "**normal** (E_d ≤ c)")
                + f" porque E_d = {E:.4f} vs c = {m.c:.3f}."))
    _paso(c, f"Paso 3 · Derivada direccional con u = ⟨cos {ang_deg}°, sen {ang_deg}°⟩",
          rf"D_{{\vec u}}E_d=\nabla E_d\cdot\vec u={n(gx, 4)}\cos({ang_deg}^\circ)+{p(gy, 4)}\sin({ang_deg}^\circ)={n(float(Du), 4)}",
          nota="Máximo = ‖∇E_d‖ cuando u apunta como el gradiente; 0 si u es tangente a la curva de nivel; mínimo −‖∇E_d‖ en sentido opuesto.")
    H11, H22 = k1, k2
    _paso(c, "Paso 4 · Punto crítico y hessiana",
          rf"\nabla E_d=\vec 0\ \Longleftrightarrow\ V_1=\mu_1={n(m.mu1, 4)},\ V_2=\mu_2={n(m.mu2, 4)}\quad\Rightarrow\quad E_d(\vec\mu)=0",
          rf"H=\begin{{pmatrix}}\frac{{2}}{{\sigma_1^2}}&0\\0&\frac{{2}}{{\sigma_2^2}}\end{{pmatrix}}=\begin{{pmatrix}}{n(H11, 6)}&0\\0&{n(H22, 6)}\end{{pmatrix}},\qquad \det H={n(H11 * H22, 6)}>0",
          nota="det H > 0 y ∂²E/∂V₁² > 0 en todo ℝ² ⇒ superficie estrictamente convexa ⇒ el centroide es el mínimo global.")


# ----------------------------------------------------------------------------- Unidad 3
def unidad3(m: Modelo):
    fc, _ = m.flujo_contorno()
    c = _caja("Desarrollo paso a paso (con tus valores)")
    _paso(c, "Paso 1 · Masa de probabilidad (polares generalizadas)",
          r"V_1=\mu_1+\sigma_1 r\cos\theta,\quad V_2=\mu_2+\sigma_2 r\sin\theta,\quad |J|=\sigma_1\sigma_2\,r,\quad E_d=r^2\Rightarrow 0\le r\le\sqrt c",
          rf"M=\int_0^{{2\pi}}\!\!\int_0^{{\sqrt c}}\frac{{e^{{-r^2/2}}}}{{2\pi\sigma_1\sigma_2}}\,\sigma_1\sigma_2\,r\,dr\,d\theta=1-e^{{-c/2}}=1-e^{{-{n(m.c / 2, 4)}}}={n(m.masa, 6)}\ \ ({n(100 * m.masa, 3)}\,\%)")
    _paso(c, "Paso 2 · Área de la región y divergencia del campo",
          rf"\text{{Área}}(R)=\pi ab=\pi\cdot{n(m.a, 4)}\cdot{n(m.b, 4)}={n(m.area, 4)}",
          rf"\vec F=\nabla E_d\ \Rightarrow\ \nabla\!\cdot\!\vec F=\frac{{\partial^2E_d}}{{\partial V_1^2}}+\frac{{\partial^2E_d}}{{\partial V_2^2}}=\frac{{2}}{{\sigma_1^2}}+\frac{{2}}{{\sigma_2^2}}={n(2 / m.sigma1 ** 2, 6)}+{n(2 / m.sigma2 ** 2, 6)}={n(m.div_F, 6)}")
    _paso(c, "Paso 3 · Teorema de Green (forma de flujo)",
          rf"\Phi=\iint_R\nabla\!\cdot\!\vec F\,dA=\nabla\!\cdot\!\vec F\cdot\text{{Área}}(R)={n(m.div_F, 6)}\cdot{n(m.area, 4)}={n(m.flujo_area, 6)}",
          rf"\Phi=\oint_C\vec F\cdot\vec n\,ds=\int_0^{{2\pi}}\nabla E_d(\vec r(t))\cdot\langle b\cos t,\ a\sin t\rangle\,dt={n(fc, 6)}",
          nota=f"Diferencia entre los dos métodos: {abs(m.flujo_area - fc):.2e} → el teorema se verifica numéricamente. "
               "(Con n ds = ⟨b cos t, a sin t⟩ dt, la norma de v se cancela.)")


# ----------------------------------------------------------------------------- Clasificación
def clasificacion(cm: dict, m: Modelo):
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    rec, pre, esp = tp / (tp + fn), tp / (tp + fp), tn / (tn + fp)
    f1 = 2 * pre * rec / (pre + rec)
    c = _caja("Desarrollo de las métricas")
    _paso(c, "Regla de decisión", rf"\text{{alerta}}\iff E_d>c={n(m.c, 3)}")
    _paso(c, "Métricas a partir de la matriz de confusión",
          rf"\text{{Sensibilidad}}=\frac{{TP}}{{TP+FN}}=\frac{{{tp}}}{{{tp}+{fn}}}={n(100 * rec, 2)}\,\%",
          rf"\text{{Especificidad}}=\frac{{TN}}{{TN+FP}}=\frac{{{tn}}}{{{tn}+{fp}}}={n(100 * esp, 2)}\,\%",
          rf"\text{{Precisión}}=\frac{{TP}}{{TP+FP}}=\frac{{{tp}}}{{{tp}+{fp}}}={n(100 * pre, 2)}\,\%",
          rf"F_1=\frac{{2\cdot\text{{Prec}}\cdot\text{{Sens}}}}{{\text{{Prec}}+\text{{Sens}}}}={n(100 * f1, 2)}\,\%")
