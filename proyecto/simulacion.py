# -*- coding: utf-8 -*-
"""Sección de simulación: se generan datos sintéticos con parámetros editables y se aplica
exactamente el mismo modelo del trabajo. No modifica los resultados del resto de la app."""
import numpy as np
import pandas as pd
import streamlit as st

import graficos as G
from modelo import C_DEFAULT, PARAMS_INFORME, Modelo, clasificar, metricas


def _f(x, d=4):
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _ent(n):
    return f"{int(n):,}".replace(",", ".")


def render():
    with st.expander("🧪 Simulación: cambiá los parámetros y mirá qué pasa con la frontera", expanded=False):
        st.caption("Se generan transacciones **simuladas** (no son datos reales) y se les aplica el mismo modelo del trabajo: "
                   "μ y σ se estiman con las legítimas simuladas (ddof = 1), la frontera es E_d = c y se clasifica con E_d > c. "
                   "Esta sección es independiente: no cambia los resultados del informe de las pestañas.")
        if st.button("↺ Volver a los valores del informe", key="sim_reset"):
            for k in [k for k in st.session_state if k.startswith("sim_") and k != "sim_reset"]:
                del st.session_state[k]
            st.rerun()

        st.markdown("**Transacciones legítimas**")
        a1, a2, a3, a4 = st.columns(4)
        mu1 = a1.number_input("μ₁", value=float(round(PARAMS_INFORME["mu1"], 4)), step=0.1, format="%.4f", key="sim_mu1")
        mu2 = a2.number_input("μ₂", value=float(round(PARAMS_INFORME["mu2"], 4)), step=0.1, format="%.4f", key="sim_mu2")
        s1 = a3.number_input("σ₁", value=float(round(PARAMS_INFORME["sigma1"], 4)), min_value=0.1, step=0.1, format="%.4f", key="sim_s1")
        s2 = a4.number_input("σ₂", value=float(round(PARAMS_INFORME["sigma2"], 4)), min_value=0.1, step=0.1, format="%.4f", key="sim_s2")
        b1, b2, b3 = st.columns(3)
        rho = b1.slider("Correlación ρ entre V₁ y V₂", -0.95, 0.95, 0.0, 0.01, key="sim_rho",
                        help="El modelo diagonal supone ρ ≈ 0 (en el dataset real ρ ≈ 0,022). Subila para ver cómo se degrada.")
        n_leg = b2.slider("Cantidad de legítimas", 1000, 100000, 20000, 1000, key="sim_n")
        c = b3.number_input("Umbral c", min_value=0.5, max_value=20.0, value=C_DEFAULT, step=0.1, format="%.3f", key="sim_c")

        st.markdown("**Fraudes simulados**")
        d1, d2, d3, d4 = st.columns(4)
        n_f = d1.slider("Cantidad de fraudes", 0, 2000, 500, 50, key="sim_nf")
        sh1 = d2.slider("Desplazamiento V₁ (en σ₁)", -4.0, 4.0, -1.5, 0.1, key="sim_sh1")
        sh2 = d3.slider("Desplazamiento V₂ (en σ₂)", -4.0, 4.0, 1.0, 0.1, key="sim_sh2")
        esc = d4.slider("Dispersión relativa", 0.5, 4.0, 1.6, 0.1, key="sim_esc", help="1 = misma dispersión que las legítimas")
        seed = st.number_input("Semilla aleatoria", value=42, step=1, key="sim_seed")

        # --- datos simulados
        rng = np.random.default_rng(int(seed))
        cov = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
        leg = rng.multivariate_normal([mu1, mu2], cov, size=int(n_leg))
        fr = rng.multivariate_normal([mu1 + sh1 * s1, mu2 + sh2 * s2], cov * esc ** 2, size=int(n_f)) if n_f else np.empty((0, 2))
        df = pd.DataFrame({"V1": np.r_[leg[:, 0], fr[:, 0]], "V2": np.r_[leg[:, 1], fr[:, 1]],
                           "Class": np.r_[np.zeros(len(leg), int), np.ones(len(fr), int)]})

        # --- mismo modelo que el trabajo
        L = df[df.Class == 0]
        m = Modelo(mu1=L.V1.mean(), mu2=L.V2.mean(), sigma1=L.V1.std(ddof=1), sigma2=L.V2.std(ddof=1), c=c)
        cm = clasificar(df, m)
        mt = metricas(cm)
        dentro = cm["tn"] / (cm["tn"] + cm["fp"])
        Lg, _ = m.longitud()
        corr_obs = float(np.corrcoef(L.V1, L.V2)[0, 1])

        st.markdown("**Resultados**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Semiejes a · b", f"{_f(m.a, 3)} · {_f(m.b, 3)}")
        r2.metric("Longitud L · Área", f"{_f(Lg, 2)} · {_f(m.area, 2)}")
        r3.metric("κ máx · κ mín", f"{_f(m.kappa_max, 4)} · {_f(m.kappa_min, 4)}")
        r4.metric("Flujo Φ (Green)", _f(m.flujo_area, 3), f"div F = {_f(m.div_F, 4)}", delta_color="off")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Masa teórica", f"{100 * m.masa:.2f} %".replace(".", ","))
        q2.metric("Legítimas dentro (simulado)", f"{100 * dentro:.2f} %".replace(".", ","),
                  f"{100 * (dentro - m.masa):+.2f} pp".replace(".", ","), delta_color="off")
        q3.metric("Sensibilidad", "—" if not n_f else f"{100 * mt['sensibilidad']:.2f} %".replace(".", ","))
        q4.metric("Precisión", "—" if not cm["tp"] + cm["fp"] else f"{100 * mt['precision']:.2f} %".replace(".", ","))

        g1, g2 = st.columns([1.2, 1])
        with g1:
            lm = min(len(L), 12000)
            fig = G.grafico1(m, L.sample(lm, random_state=1), df[df.Class == 1], simulado=True)
            st.plotly_chart(fig, width="stretch", key="sim_fig")
        with g2:
            st.dataframe(pd.DataFrame({
                "Categoría": ["Legítimas", "Fraudes"],
                "Dentro": [_ent(cm["tn"]), _ent(cm["fn"])],
                "Fuera (alerta)": [_ent(cm["fp"]), _ent(cm["tp"])],
            }), hide_index=True, width="stretch")
            st.caption(f"Parámetros estimados: μ₁ = {_f(m.mu1)}, μ₂ = {_f(m.mu2)}, σ₁ = {_f(m.sigma1)}, σ₂ = {_f(m.sigma2)}. "
                       f"Correlación observada: {_f(corr_obs, 3)}.")
            if abs(rho) > 0.3:
                st.warning("Con |ρ| alto la elipse alineada con los ejes ya no representa bien la nube: el modelo diagonal pierde precisión "
                           "(por eso el trabajo propone, como mejora, la Mahalanobis completa).")
            if mt["total"]:
                st.caption(f"Falsos positivos: {100 * mt['fpr']:.2f} % de las legítimas · F₁ = {100 * mt['f1']:.2f} %".replace(".", ","))
