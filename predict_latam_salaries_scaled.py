#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis predictivo con gráficas en mejor escala
- Histogramas en log y recortes por cuantiles (1–99%)
- Scatter Real vs Predicho en log–log y versión recortada
- Hexbin para alta densidad
- Residuos con límites por cuantiles
Salidas en ./outputs_scaled/
"""

import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -------------------- utilidades --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def parse_years_code(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if "less than" in s: return 0.5
    if "more than" in s:
        nums = [int(t) for t in s.split() if t.isdigit()]
        return float(nums[0]) + 1 if nums else 51.0
    try: return float(s)
    except: return np.nan

def parse_float(x):
    if pd.isna(x): return np.nan
    try: return float(str(x).strip())
    except: return np.nan

def winsor_limits(arr, lo=1, hi=99):
    qlo, qhi = np.percentile(arr, [lo, hi])
    return qlo, qhi

def clip_by_quantiles(arr, lo=1, hi=99):
    qlo, qhi = winsor_limits(arr, lo, hi)
    return np.clip(arr, qlo, qhi), qlo, qhi

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="latam_devs.xlsx")
    ap.add_argument("--sheet", default="LATAM")
    ap.add_argument("--output_dir", default="outputs_scaled")
    ap.add_argument("--include_currency", action="store_true",
                    help="Incluye 'Currency' como feature si existe")
    args = ap.parse_args()

    out = ensure_dir(args.output_dir)
    charts = ensure_dir(os.path.join(out, "charts"))

    # 1) carga
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    target = "ConvertedCompYearly"
    if target not in df.columns:
        raise ValueError(f"No existe columna objetivo '{target}'.")

    # 2) features
    base_feats = [
        "Age","Country","EdLevel","WorkExp","YearsCode",
        "Employment","EmploymentAddl","DevType","OrgSize","ICorPM",
        "RemoteWork","Industry","MainBranch","JobSat",
        "LanguageHaveWorkedWith","DatabaseHaveWorkedWith","PlatformHaveWorkedWith",
        "WebframeHaveWorkedWith","DevEnvsHaveWorkedWith"
    ]
    if args.include_currency and "Currency" in df.columns:
        base_feats.append("Currency")

    feats = [c for c in base_feats if c in df.columns]
    dfm = df[feats + [target]].copy()

    # 3) limpieza básica
    if "YearsCode" in dfm: dfm["YearsCode"] = dfm["YearsCode"].apply(parse_years_code)
    if "WorkExp"  in dfm: dfm["WorkExp"]  = dfm["WorkExp"].apply(parse_float)

    y = dfm[target]
    X = dfm.drop(columns=[target])

    # 4) tipos
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    # si usas scikit-learn >=1.2 cambia a sparse_output=False
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ])

    # 5) split (quita nulos en y)
    mask = y.notna() & np.isfinite(y) & (y > 0)
    Xc, yc = X.loc[mask], y.loc[mask]
    X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

    # 6) modelos
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=250, max_depth=12, n_jobs=-1, random_state=42)
    }

    results = []
    pipes = {}
    for name, m in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = math.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results.append({"Modelo": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        pipes[name] = pipe
        print(f"[{name}] R2={r2:.4f} MAE={mae:,.0f} RMSE={rmse:,.0f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(out, "model_results_scaled.csv"), index=False)

    best_name = results_df.iloc[0]["Modelo"]
    best_pipe = pipes[best_name]
    y_pred = best_pipe.predict(X_test)

    # 7) export reducido
    used_cols = num_cols + cat_cols + [target]
    dfm[used_cols].to_excel(os.path.join(out, "latam_devs_reduced_scaled.xlsx"), index=False)

    # 8) GRAFICOS MEJORADOS ----------------------------------------
    sal = yc.copy()

    # 8.1 Histograma LOG
    sal_pos = sal[sal > 0]
    plt.figure()
    plt.hist(sal_pos, bins=40)
    plt.xscale("log")
    plt.xlabel("Salario anual (USD, escala log)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de salarios LATAM (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "01a_hist_salarios_log.png"), dpi=150)
    plt.close()

    # 8.2 Histograma recortado 1–99% con estadísticos
    sal_trim = sal_pos.copy()
    q1, q99 = np.percentile(sal_trim, [1, 99])
    sal_trim = sal_trim[(sal_trim >= q1) & (sal_trim <= q99)]
    median = np.median(sal_trim); q25, q75 = np.percentile(sal_trim, [25, 75])

    plt.figure()
    plt.hist(sal_trim, bins=40)
    plt.axvline(median, linestyle="--", label=f"Mediana ≈ {median:,.0f}")
    plt.axvline(q25, linestyle=":",  label=f"Q1 ≈ {q25:,.0f}")
    plt.axvline(q75, linestyle=":",  label=f"Q3 ≈ {q75:,.0f}")
    plt.xlabel("Salario anual (USD) — recorte 1–99%")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de salarios (trim 1–99%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "01b_hist_salarios_trim.png"), dpi=150)
    plt.close()

    # 8.3 Real vs Predicho en LOG–LOG
    yt = y_test.copy()
    yp = pd.Series(y_pred, index=y_test.index)

    # remover no positivos para log
    mask_pos = (yt > 0) & (yp > 0)
    yt_pos, yp_pos = yt[mask_pos], yp[mask_pos]

    plt.figure()
    plt.scatter(yt_pos, yp_pos, alpha=0.5)
    plt.xscale("log"); plt.yscale("log")
    lo = min(yt_pos.min(), yp_pos.min()); hi = max(yt_pos.max(), yp_pos.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Salario real (USD, log)")
    plt.ylabel("Salario predicho (USD, log)")
    plt.title(f"Reales vs. Predichos (log–log) — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "02a_reales_vs_pred_loglog.png"), dpi=150)
    plt.close()

    # 8.4 Real vs Predicho recortado por cuantiles + línea diagonal
    ytr, lo_r, hi_r = clip_by_quantiles(yt.values, 1, 99)
    ypr, lo_p, hi_p = clip_by_quantiles(yp.values, 1, 99)
    plt.figure()
    plt.scatter(ytr, ypr, alpha=0.5)
    lo = float(min(ytr.min(), ypr.min())); hi = float(max(ytr.max(), ypr.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"Real (USD) trim 1–99%")
    plt.ylabel(f"Predicho (USD) trim 1–99%")
    plt.title(f"Reales vs. Predichos (trim) — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "02b_reales_vs_pred_trim.png"), dpi=150)
    plt.close()

    # 8.5 Hexbin (densidad) para evitar sobreplotting
    plt.figure()
    plt.hexbin(yt_pos, yp_pos, gridsize=40)  # sin elegir colores explícitos
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Real (USD, log)")
    plt.ylabel("Predicho (USD, log)")
    plt.title(f"Densidad Real vs. Predicho (hexbin, log–log) — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "02c_reales_vs_pred_hexbin_loglog.png"), dpi=150)
    plt.close()

    # 8.6 Residuos con límites por cuantiles y eje X en log
    resid = yt - yp
    # recorte de residuales para visualizar (1–99% de |resid|)
    ab = np.abs(resid.values)
    rcut = np.percentile(ab, 99)
    resid_clip = np.clip(resid.values, -rcut, rcut)

    plt.figure()
    plt.scatter(yp_pos, resid.loc[mask_pos].clip(-rcut, rcut), alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Salario predicho (USD, log)")
    plt.ylabel("Error (residuo) — trim 1–99%")
    plt.title(f"Análisis de residuos (recorte) — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "03_residuos_trim_logx.png"), dpi=150)
    plt.close()

    # 8.7 Comparación de modelos
    plt.figure()
    plt.bar(results_df["Modelo"], results_df["R2"])
    plt.ylabel("R²")
    plt.title("Comparación de modelos (R²)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "05_comparacion_modelos_R2.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.bar(results_df["Modelo"], results_df["MAE"])
    plt.ylabel("MAE (USD)")
    plt.title("Comparación de modelos (MAE)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "06_comparacion_modelos_MAE.png"), dpi=150)
    plt.close()

    print("\n✅ Listo. Revisa las figuras en:", charts)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
