# -*- coding: utf-8 -*-
"""Lógica del modelo diagonal (misma que el notebook de Colab, sin dependencias de Colab).

Modelo: distancia estandarizada con covarianza diagonal (aprox. diagonal de Mahalanobis)
sobre el plano de componentes principales (V1, V2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad

C_DEFAULT = 5.991  # percentil 95 de chi-cuadrado con 2 g.l.

# Parámetros publicados en el informe / resumen_calculos.pdf (Class = 0, N = 284.315)
PARAMS_INFORME = dict(
    mu1=0.00825773748556369, mu2=-0.00627085741580707,
    sigma1=1.92981405165008, sigma2=1.63614605256896,
)
# Matriz de confusión publicada (dataset Kaggle completo, c = 5,991)
CONF_INFORME = dict(tn=272572, fp=11743, fn=258, tp=234)
CORR_INFORME = 0.022487
COV_INFORME = 0.071


# --------------------------------------------------------------------------- datos
def cargar_csv(origen) -> pd.DataFrame:
    """Lee el CSV (ruta o archivo subido) y valida V1, V2, Class (igual que el script original)."""
    df = pd.read_csv(origen, compression="infer")
    df.columns = df.columns.astype(str).str.strip()
    req = ["V1", "V2", "Class"]
    faltan = [x for x in req if x not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas {faltan}. Encontradas: {df.columns.tolist()[:10]}...")
    df = df[req].copy()
    for col in req:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=req).copy()
    df["Class"] = df["Class"].astype(int)
    return df


def parametros_desde_df(df: pd.DataFrame) -> dict:
    """Medias y desvíos muestrales (ddof=1) sobre transacciones legítimas."""
    legit = df[df["Class"] == 0]
    if len(legit) < 2:
        raise ValueError("Se requieren al menos dos filas legítimas.")
    cov = legit[["V1", "V2"]].cov().iloc[0, 1]
    corr = legit[["V1", "V2"]].corr().iloc[0, 1]
    return dict(
        mu1=legit["V1"].mean(), mu2=legit["V2"].mean(),
        sigma1=legit["V1"].std(ddof=1), sigma2=legit["V2"].std(ddof=1),
        cov12=cov, corr12=corr, n_legit=len(legit),
    )


# --------------------------------------------------------------------------- modelo
@dataclass
class Modelo:
    mu1: float
    mu2: float
    sigma1: float
    sigma2: float
    c: float = C_DEFAULT

    # semiejes
    @property
    def a(self): return self.sigma1 * np.sqrt(self.c)
    @property
    def b(self): return self.sigma2 * np.sqrt(self.c)

    # Unidad 2
    def E_d(self, x, y):
        return (x - self.mu1) ** 2 / self.sigma1 ** 2 + (y - self.mu2) ** 2 / self.sigma2 ** 2

    def grad_E(self, x, y):
        return 2 * (x - self.mu1) / self.sigma1 ** 2, 2 * (y - self.mu2) / self.sigma2 ** 2

    @property
    def hessiana(self):
        return np.array([[2 / self.sigma1 ** 2, 0.0], [0.0, 2 / self.sigma2 ** 2]])

    @property
    def det_H(self): return 4 / (self.sigma1 ** 2 * self.sigma2 ** 2)
    @property
    def div_F(self): return 2 / self.sigma1 ** 2 + 2 / self.sigma2 ** 2

    # Unidad 1
    def r(self, t): return self.mu1 + self.a * np.cos(t), self.mu2 + self.b * np.sin(t)
    def v(self, t): return -self.a * np.sin(t), self.b * np.cos(t)
    def acel(self, t): return -self.a * np.cos(t), -self.b * np.sin(t)
    def rapidez(self, t):
        vx, vy = self.v(t)
        return np.hypot(vx, vy)

    def kappa(self, t):
        a, b = self.a, self.b
        return a * b / (a ** 2 * np.sin(t) ** 2 + b ** 2 * np.cos(t) ** 2) ** 1.5

    @property
    def kappa_max(self): return self.a / self.b ** 2
    @property
    def kappa_min(self): return self.b / self.a ** 2

    def longitud(self):
        return quad(self.rapidez, 0, 2 * np.pi)

    @property
    def area(self): return np.pi * self.a * self.b
    @property
    def flujo_area(self): return self.div_F * self.area
    @property
    def masa(self): return 1 - np.exp(-self.c / 2)

    def normal_ext(self, t):
        """Normal exterior unitaria en r(t)."""
        nx, ny = self.b * np.cos(t), self.a * np.sin(t)
        n = np.hypot(nx, ny)
        return nx / n, ny / n

    def flujo_contorno(self):
        def f(t):
            xx, yy = self.r(t)
            f1, f2 = self.grad_E(xx, yy)
            return f1 * self.b * np.cos(t) + f2 * self.a * np.sin(t)
        return quad(f, 0, 2 * np.pi, epsabs=1e-9, epsrel=1e-9)

    def derivada_direccional(self, x, y, ang_rad):
        gx, gy = self.grad_E(x, y)
        return gx * np.cos(ang_rad) + gy * np.sin(ang_rad)

    def resumen(self) -> pd.DataFrame:
        L, _ = self.longitud()
        fc, _ = self.flujo_contorno()
        nombres = ["mu1", "mu2", "sigma1", "sigma2", "c", "a", "b", "kappa_max", "kappa_min",
                   "longitud_arco", "area", "divergencia", "flujo_area", "flujo_contorno", "masa_teorica"]
        valores = [self.mu1, self.mu2, self.sigma1, self.sigma2, self.c, self.a, self.b,
                   self.kappa_max, self.kappa_min, L, self.area, self.div_F, self.flujo_area, fc, self.masa]
        return pd.DataFrame({"Magnitud": nombres, "Valor": valores})


# --------------------------------------------------------------------------- clasificación
def clasificar(df: pd.DataFrame, m: Modelo) -> dict:
    ed = m.E_d(df["V1"].to_numpy(), df["V2"].to_numpy())
    alerta = ed > m.c
    cls = df["Class"].to_numpy()
    return dict(
        tn=int(((cls == 0) & ~alerta).sum()), fp=int(((cls == 0) & alerta).sum()),
        fn=int(((cls == 1) & ~alerta).sum()), tp=int(((cls == 1) & alerta).sum()),
    )


def metricas(cm: dict) -> dict:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    div = lambda n, d: n / d if d else float("nan")
    rec, prec = div(tp, tp + fn), div(tp, tp + fp)
    return dict(
        sensibilidad=rec, especificidad=div(tn, tn + fp), fpr=div(fp, tn + fp), precision=prec,
        f1=div(2 * prec * rec, prec + rec) if tp else 0.0, total=tp + fp + tn + fn,
    )
