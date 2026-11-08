# -*- coding: utf-8 -*-
"""Streamlit — Cálculo II · TPAG 2026 · Grupo 2
Frontera de normalidad y dinámica multivariable del error (V1, V2) — detección de fraude."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import desarrollo as D
import graficos as G
import simulacion
from modelo import (C_DEFAULT, CONF_INFORME, CORR_INFORME, COV_INFORME, PARAMS_INFORME, Modelo,
                    cargar_csv, clasificar, metricas, parametros_desde_df)

AQUI = Path(__file__).parent
CSV_LOCAL = AQUI / "data" / "creditcard.csv"

st.set_page_config(page_title="Frontera de normalidad · Cálculo II", page_icon="📐", layout="wide", initial_sidebar_state="collapsed")


def f(x, d=4):
    """Número con coma decimal (formato del informe)."""
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def entero(n):
    return f"{int(n):,}".replace(",", ".")


@st.cache_data(show_spinner="Leyendo CSV…")
def leer(origen):
    return cargar_csv(origen)


# ----------------------------------------------------------------------------- opciones (opcionales)
st.title("Frontera de normalidad y dinámica multivariable del error")
st.caption("Cálculo II · TPAG 2026 · Grupo N.º 2 · UNSL — Licenciatura en Análisis y Gestión de Datos  \n"
           "Plano de componentes principales (V₁, V₂) · detección de fraude con tarjetas")
st.caption("Todo funciona con los valores del informe; las opciones son **optativas** y, si las cambiás, fórmulas y resultados se recalculan solos.")
exp = st.expander("⚙️ Opciones (optativo): fuente de datos, umbral c y parámetros", expanded=False)
with exp:
    opciones = ["Subir CSV", "Parámetros del informe (sin CSV)"]
    if CSV_LOCAL.exists():
        opciones.insert(0, "CSV local (data/creditcard.csv)")
    fuente = st.radio("Fuente de datos", opciones, index=0 if CSV_LOCAL.exists() else opciones.index("Parámetros del informe (sin CSV)"))

    df = None
    if fuente.startswith("CSV local"):
        df = leer(str(CSV_LOCAL))
    elif fuente == "Subir CSV":
        up = st.file_uploader("creditcard.csv (columnas V1, V2, Class)", type=["csv", "gz", "zip"])
        if up is not None:
            try:
                df = leer(up)
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Subí el CSV del dataset (Kaggle *Credit Card Fraud Detection*) o elegí "
                    "«Parámetros del informe» para ver todo sin el archivo.")

    usar_informe = True
    if df is not None:
        usar_informe = st.toggle("Usar parámetros redondeados del informe", value=False,
                                 help="Apagado: se recalculan μ y σ desde el CSV (Class = 0, ddof = 1).")
    c = st.number_input("Umbral c (χ² con 2 g.l.)", 0.5, 20.0, C_DEFAULT, 0.001, format="%.3f",
                        help="5,991 ≈ percentil 95 de χ²₂. Masa teórica = 1 − e^(−c/2).")
    st.caption(f"Percentil equivalente: **{100 * (1 - np.exp(-c / 2)):.2f} %**")

    if df is not None:
        max_pts = st.slider("Máx. puntos por clase en el gráfico (solo visual)", 1000, 30000, 12000, 1000)
    st.divider()
    st.caption("Los conteos usan **todas** las filas válidas; el muestreo es solo para graficar.")

# ----------------------------------------------------------------------------- modelo
extra = {}
if df is not None:
    p = parametros_desde_df(df)
    extra = dict(cov12=p["cov12"], corr12=p["corr12"], n_legit=p["n_legit"])
    base = PARAMS_INFORME if usar_informe else {k: p[k] for k in ("mu1", "mu2", "sigma1", "sigma2")}
else:
    base = PARAMS_INFORME
    extra = dict(cov12=COV_INFORME, corr12=CORR_INFORME, n_legit=284315)
with exp:
    st.markdown("**Parámetros del modelo** (opcional)")
    manual = st.checkbox("Editar μ y σ a mano", value=False)
    if manual:
        q1, q2, q3, q4 = st.columns(4)
        base = dict(mu1=q1.number_input("μ₁", value=float(base["mu1"]), format="%.6f", step=0.001),
                    mu2=q2.number_input("μ₂", value=float(base["mu2"]), format="%.6f", step=0.001),
                    sigma1=q3.number_input("σ₁", value=float(base["sigma1"]), min_value=0.01, format="%.6f", step=0.01),
                    sigma2=q4.number_input("σ₂", value=float(base["sigma2"]), min_value=0.01, format="%.6f", step=0.01))
m = Modelo(**base, c=c)

# conteos
if df is not None:
    cm = clasificar(df, m)
    origen_cm = "calculado sobre todas las filas del CSV"
elif abs(c - C_DEFAULT) < 1e-9:
    cm = CONF_INFORME
    origen_cm = "publicado en el informe (dataset Kaggle completo)"
else:
    cm = None
    origen_cm = ""

# ----------------------------------------------------------------------------- encabezado
simulacion.render()

if df is None:
    st.info("Modo **sin CSV**: se usan los parámetros y conteos publicados en el informe. "
            "Para recalcular desde los datos, abrí «Opciones» y subí el CSV.")

tabs = st.tabs(["🏠 Resumen", "① Funciones vectoriales", "② Varias variables", "③ Integrales y Green",
                "④ Clasificación", "📄 Informe y descargas"])

# ----------------------------------------------------------------------------- resumen
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    L, errL = m.longitud()
    c1.metric("Semiejes a · b", f"{f(m.a)} · {f(m.b)}")
    c2.metric("Longitud de la frontera L", f(L))
    c3.metric("Área de R", f(m.area))
    c4.metric("Masa teórica retenida", f"{100 * m.masa:.2f} %".replace(".", ","))

    st.markdown(
        "El enfoque: en lugar de un clasificador supervisado (muy sensible al desbalance, ~0,17 % de fraudes), "
        "se **caracteriza la masa de operaciones legítimas** y se define una frontera elíptica de normalidad en el "
        "plano (V₁, V₂) con la distancia estandarizada de covarianza diagonal:")
    st.latex(r"E_d(V_1,V_2)=\frac{(V_1-\mu_1)^2}{\sigma_1^2}+\frac{(V_2-\mu_2)^2}{\sigma_2^2}\le c")

    D.modelo(m)
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        if df is not None:
            leg = df[df.Class == 0]; fr = df[df.Class == 1]
            fig = G.grafico1(m, leg.sample(min(len(leg), max_pts), random_state=42),
                             fr.sample(min(len(fr), max_pts), random_state=42))
        else:
            rng = np.random.default_rng(42)
            sim = pd.DataFrame({"V1": rng.normal(m.mu1, m.sigma1, 12000), "V2": rng.normal(m.mu2, m.sigma2, 12000)})
            fig = G.grafico1(m, sim, None, simulado=True)
        st.plotly_chart(fig, width="stretch")
        if df is None:
            st.caption("⚠️ Sin CSV los puntos son una **simulación normal ilustrativa** con los parámetros del informe, "
                       "no los datos reales (no se muestran fraudes).")
    with col_b:
        st.subheader("Parámetros del modelo")
        st.caption("Transacciones legítimas (Class = 0)" + (f", N = {entero(extra['n_legit'])}"))
        st.dataframe(pd.DataFrame({
            "": ["Media μ", "Desvío estándar σ"],
            "V₁": [f(m.mu1, 8), f(m.sigma1, 8)],
            "V₂": [f(m.mu2, 8), f(m.sigma2, 8)],
        }), hide_index=True, width="stretch")
        st.markdown(f"Correlación observada **r ≈ {f(extra['corr12'], 4)}** → muy baja, por eso se adopta el "
                    "**modelo diagonal** (aproximación diagonal de Mahalanobis), despreciando el término cruzado V₁V₂.")
        if df is not None and not usar_informe:
            st.caption("Parámetros recalculados desde el CSV (σ muestral, ddof = 1).")
        elif df is not None:
            st.caption("Parámetros redondeados publicados en el informe.")
        st.subheader("Tabla resumen de cálculos")
        res = m.resumen()
        res["Valor"] = res["Valor"].map(lambda v: f(v, 8))
        st.dataframe(res, hide_index=True, width="stretch", height=360)

# ----------------------------------------------------------------------------- unidad 1
with tabs[1]:
    st.header("Unidad 1 — Funciones vectoriales")
    st.markdown("La frontera $C=\\partial R$ se parametriza como una elipse alineada de semiejes "
                "$a=\\sigma_1\\sqrt c$, $b=\\sigma_2\\sqrt c$.")
    st.latex(r"\vec r(t)=\langle \mu_1+a\cos t,\ \mu_2+b\sin t\rangle,\quad t\in[0,2\pi]")
    st.latex(r"\vec v(t)=\langle -a\sin t,\ b\cos t\rangle\qquad \vec a(t)=-(\vec r(t)-\vec\mu)")
    st.latex(r"\kappa(t)=\frac{ab}{(a^2\sin^2t+b^2\cos^2t)^{3/2}}\qquad L=\int_0^{2\pi}\|\vec v(t)\|\,dt")

    t0 = st.slider("Parámetro t₀ (rad)", 0.0, float(2 * np.pi), float(np.pi / 4), 0.01, key="t0")
    px, py = m.r(t0); vx, vy = m.v(t0); ax_, ay_ = m.acel(t0)
    D.unidad1(m, t0)
    st.dataframe(pd.DataFrame({
        "r(t₀)": [f"({f(px, 4)} ; {f(py, 4)})"], "v(t₀)": [f"({f(vx, 4)} ; {f(vy, 4)})"],
        "‖v(t₀)‖": [f(np.hypot(vx, vy), 4)], "a(t₀)": [f"({f(ax_, 4)} ; {f(ay_, 4)})"], "κ(t₀)": [f(m.kappa(t0), 6)],
    }), hide_index=True, width="stretch")
    st.plotly_chart(G.grafico2(m, t0), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("κ máx = a/b² (t = 0, π)", f(m.kappa_max, 6))
    c2.metric("κ mín = b/a² (t = π/2, 3π/2)", f(m.kappa_min, 6))
    c3.metric("Longitud L (scipy.quad)", f(L, 6), f"error est. {errL:.1e}", delta_color="off")
    st.caption("L es una integral elíptica completa de 2.ª especie: no tiene primitiva elemental, por eso se resuelve numéricamente.")

# ----------------------------------------------------------------------------- unidad 2
with tabs[2]:
    st.header("Unidad 2 — Funciones de varias variables")
    st.latex(r"\nabla E_d=\Big\langle \tfrac{2(V_1-\mu_1)}{\sigma_1^2},\ \tfrac{2(V_2-\mu_2)}{\sigma_2^2}\Big\rangle"
             r"\qquad D_{\vec u}E_d=\nabla E_d\cdot\vec u")
    H = m.hessiana
    st.latex(r"H(E_d)=\begin{pmatrix}" + f"{f(H[0, 0], 6)}" + r"&0\\0&" + f"{f(H[1, 1], 6)}" +
             r"\end{pmatrix},\quad \det H=" + f"{f(m.det_H, 6)}" + r">0,\ \ \partial^2_{V_1}E_d>0")
    st.success("**Único punto crítico = centroide μ**, con E_d(μ) = 0. H es definida positiva en todo ℝ² → "
               "superficie estrictamente convexa → **mínimo global estricto**.")

    st.subheader("Evaluador interactivo")
    e1, e2, e3 = st.columns(3)
    v1 = e1.number_input("V₁", value=float(round(m.mu1 + m.a, 6)), format="%.6f", step=0.1)
    v2 = e2.number_input("V₂", value=float(round(m.mu2, 6)), format="%.6f", step=0.1)
    ang = e3.slider("Dirección u (grados)", 0, 359, 0)
    u = np.deg2rad(ang)
    E = float(m.E_d(v1, v2)); gx, gy = m.grad_E(v1, v2); Du = float(m.derivada_direccional(v1, v2, u))
    r1, r2, r3 = st.columns(3)
    r1.metric("E_d(V₁, V₂)", f(E, 4))
    r2.metric(f"D_u E_d  (u = {ang}°)", f(Du, 4))
    r3.metric("‖∇E_d‖ (tasa máx.)", f(float(np.hypot(gx, gy)), 4))
    st.markdown(f"**∇E_d** = ( {f(gx, 4)} ; {f(gy, 4)} )")
    if abs(E - m.c) < 1e-5:
        st.warning(f"⚠️ Sobre la frontera: E_d = {f(E, 3)} ≈ c = {f(m.c, 3)}")
    elif E > m.c:
        st.error(f"🚨 Alerta: E_d = {f(E, 3)} > c = {f(m.c, 3)} (fuera de la frontera)")
    else:
        st.success(f"✅ Normal: E_d = {f(E, 3)} ≤ c = {f(m.c, 3)} (dentro de la frontera)")
    st.caption(f"‖∇E_d‖ es la máxima tasa de cambio posible (en la dirección del gradiente). "
               f"En el vértice del eje mayor con u = ⟨1,0⟩: D_u E_d = 2√c/σ₁ = {f(2 * np.sqrt(m.c) / m.sigma1, 6)}.")
    D.unidad2(m, v1, v2, ang)
    st.plotly_chart(G.grafico3(m, (v1, v2)), width="stretch")

# ----------------------------------------------------------------------------- unidad 3
with tabs[3]:
    st.header("Unidad 3 — Integrales múltiples y campos vectoriales")
    st.subheader("Masa de probabilidad (coordenadas polares generalizadas)")
    st.latex(r"V_1=\mu_1+\sigma_1 r\cos\theta,\ V_2=\mu_2+\sigma_2 r\sin\theta,\quad J=\sigma_1\sigma_2\,r")
    st.latex(r"M=\int_0^{2\pi}\!\!\int_0^{\sqrt c}\frac{e^{-r^2/2}}{2\pi\sigma_1\sigma_2}\,\sigma_1\sigma_2 r\,dr\,d\theta=1-e^{-c/2}")
    mc1, mc2 = st.columns(2)
    mc1.metric("Masa teórica M", f"{100 * m.masa:.4f} %".replace(".", ","))
    if df is not None:
        legit_in = cm["tn"] / (cm["tn"] + cm["fp"])
        mc2.metric("Legítimas dentro de la elipse (empírico)", f"{100 * legit_in:.2f} %".replace(".", ","),
                   f"{100 * (legit_in - m.masa):+.2f} pp vs teoría".replace(".", ","), delta_color="off")
    elif cm:
        legit_in = cm["tn"] / (cm["tn"] + cm["fp"])
        mc2.metric("Legítimas dentro (empírico, informe)", f"{100 * legit_in:.2f} %".replace(".", ","),
                   f"{100 * (legit_in - m.masa):+.2f} pp vs teoría".replace(".", ","), delta_color="off")
    st.caption("La masa teórica supone normalidad bivariada con componentes independientes; la diferencia con el "
               "dato real refleja que V₁–V₂ no son exactamente normales.")

    st.subheader("Teorema de Green — forma de flujo (divergencia)")
    st.latex(r"\Phi=\oint_C \vec F\cdot\vec n\,ds=\iint_R \nabla\!\cdot\!\vec F\,dA,\qquad \vec F=\nabla E_d,\ \ "
             r"\nabla\!\cdot\!\vec F=\tfrac{2}{\sigma_1^2}+\tfrac{2}{\sigma_2^2}")
    fc, errf = m.flujo_contorno()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("div F = ΔE_d", f(m.div_F, 6))
    g2.metric("Área(R) = πab", f(m.area, 6))
    g3.metric("Φ = div F × Área", f(m.flujo_area, 8))
    g4.metric("Φ por integral de contorno", f(fc, 8))
    st.caption(f"Diferencia absoluta entre ambos métodos: {abs(m.flujo_area - fc):.2e} → se verifica el teorema.")
    D.unidad3(m)
    st.plotly_chart(G.grafico4(m), width="stretch")

# ----------------------------------------------------------------------------- clasificación
with tabs[4]:
    st.header("Clasificación sobre el dataset completo")
    if cm is None:
        st.warning("Con un umbral distinto de 5,991 hace falta el CSV para recalcular los conteos. "
                   "Subilo desde «Opciones» o volvé a c = 5,991 para ver los valores del informe.")
    else:
        st.caption(f"Conteos {origen_cm}. Regla: **alerta si E_d > c** (fuera de la elipse).")
        tabla = pd.DataFrame({
            "Categoría": ["Legítimas (Class = 0)", "Fraudes (Class = 1)", "Total"],
            "Dentro (E_d ≤ c)": [entero(cm["tn"]), entero(cm["fn"]), entero(cm["tn"] + cm["fn"])],
            "Fuera (E_d > c)": [entero(cm["fp"]), entero(cm["tp"]), entero(cm["fp"] + cm["tp"])],
            "Total filas": [entero(cm["tn"] + cm["fp"]), entero(cm["fn"] + cm["tp"]), entero(sum(cm.values()))],
        })
        cA, cB = st.columns([1.2, 1])
        with cA:
            st.dataframe(tabla, hide_index=True, width="stretch")
            st.plotly_chart(G.figura_matriz_confusion(cm), width="stretch")
        mt = metricas(cm)
        with cB:
            pc = lambda x: f"{100 * x:.4f} %".replace(".", ",")
            st.metric("Sensibilidad (recall)", pc(mt["sensibilidad"]), f"TP / (TP+FN) = {cm['tp']}/{cm['tp'] + cm['fn']}", delta_color="off")
            st.metric("Especificidad", pc(mt["especificidad"]))
            st.metric("Tasa de falsos positivos", pc(mt["fpr"]))
            st.metric("Precisión (PPV)", pc(mt["precision"]))
            st.metric("F₁", pc(mt["f1"]))
        D.clasificacion(cm, m)
        st.info("**Lectura:** la precisión es muy baja (~2 %): casi todas las alertas son legítimas inusuales. "
                "El modelo sirve como **filtro exploratorio de primer nivel** en 2D, no como bloqueador automático. "
                "Además, más de la mitad de los fraudes son indistinguibles de lo normal usando solo V₁ y V₂.")

# ----------------------------------------------------------------------------- informe
with tabs[5]:
    st.header("Informe, gráficos originales y descargas")
    st.markdown("Gráficos originales generados con el notebook de Colab (matplotlib):")
    nombres = [("grafico_1_datos_frontera.png", "1 · Datos y frontera"), ("grafico_2_curvatura_tangentes.png", "2 · Curvatura y tangentes"),
               ("grafico_3_niveles_gradiente.png", "3 · Niveles y gradiente"), ("grafico_4_region_flujo_normales.png", "4 · Región, campo y normales")]
    cols = st.columns(2)
    for i, (fn, tit) in enumerate(nombres):
        with cols[i % 2]:
            st.image(str(AQUI / "assets" / fn), caption=tit, width="stretch")

    st.subheader("Descargas")
    d1, d2, d3 = st.columns(3)
    d1.download_button("📄 Informe final (PDF)", (AQUI / "assets" / "informe.pdf").read_bytes(), "Informe_Calculo2_Grupo2.pdf", "application/pdf")
    d2.download_button("📊 resumen_calculos.csv", m.resumen().to_csv(index=False).encode(), "resumen_calculos.csv", "text/csv")
    if cm:
        tc = pd.DataFrame({"tn": [cm["tn"]], "fp": [cm["fp"]], "fn": [cm["fn"]], "tp": [cm["tp"]]})
        d3.download_button("📊 tabla_clasificacion.csv", tc.to_csv(index=False).encode(), "tabla_clasificacion.csv", "text/csv")

    st.divider()
    st.markdown("**Integrantes:** ADARO, Marcelo · FUNES, Samanta Celeste · GIL, Inés · GUIÑAZÚ, Leandro Sebastián · LEDEZMA, Ricardo Mariano  \n"
                "**Docente:** Mgtr. Ing. Martín Garciarena Ucelay · **Datos:** ULB/Kaggle, *Credit Card Fraud Detection* (284.807 transacciones)")
