# Cálculo II · TPAG 2026 · Grupo 2 — App Streamlit

Frontera de normalidad y dinámica multivariable del error en el plano (V₁, V₂) — detección de fraude.

🌐 **App en línea:** [Ver aplicación en Streamlit](https://finalcalculo2-qvy9ybulnqd3qlzxgsypwf.streamlit.app/)

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos
El CSV ([Kaggle *Credit Card Fraud Detection*](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), columnas `V1`, `V2`, `Class`) **no está incluido**.
- Copialo como `data/creditcard.csv` (la app lo detecta sola), o
- subilo desde «Opciones», o
- usá «Parámetros del informe (sin CSV)», que muestra todos los cálculos con los valores publicados.

## Estructura
- `modelo.py`   lógica (misma que el notebook de Colab): E_d, gradiente, Hessiana, r(t), κ(t), L, masa, flujo, clasificación.
- `graficos.py` los 4 gráficos del informe, interactivos (Plotly).
- `app.py`      interfaz Streamlit (6 pestañas).
- `assets/`     informe PDF/LaTeX y gráficos originales.
- `web/`        página de scroll (HTML estático).
