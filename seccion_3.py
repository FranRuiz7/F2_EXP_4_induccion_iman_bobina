# -*- coding: utf-8 -*-
"""
Guía N.° 4 - Física Experimental II (UNSJ)
Script 03: Estadística por condición, bootstrap exploratorio (B=5000),
           análisis de sensibilidad y figuras finales (Secciones 6.4, 6.6 y 7).

Lee:
    - tabla_ciclos.csv (generado por Script 02)

Genera:
    - cuadro2_resumen.csv (Cuadro 2 completo para el informe)
    - fig_03_integrales_individuales.png (I_in, I_out, medianas, IC bootstrap y cierre C)
    - fig_03_Vpk_vs_invTin.png (diagnóstico V_pk vs 1/T_in e invariancia de |I_in|)
    - fig_03_bootstrap_y_sensibilidad.png (distribuciones bootstrap y presupuesto de sensibilidad)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURACIÓN REPRODUCIBLE (documentar en el informe - Sec. 9)
# =============================================================================
CARPETA = Path(__file__).parent if "__file__" in globals() else Path(".")
ARCHIVO_TABLA = CARPETA / "tabla_ciclos.csv"
ARCHIVO_CUADRO2 = CARPETA / "cuadro2_resumen.csv"

N_ESPIRAS = 300
B_BOOTSTRAP = 5000      # Número de remuestreos exigido por Sec. 6.6
SEED = 2026             # Semilla aleatoria fija para reproducibilidad (Sec. 9)
ORDEN_COND = ["L", "M", "R"]
NOMBRES_COND = {"L": "Lento (L)", "M": "Intermedio (M)", "R": "Rápido (R)"}
COLORES_COND = {"L": "tab:green", "M": "tab:orange", "R": "tab:purple"}

GUARDAR_FIGURAS = True


# =============================================================================
# 2. CARGA DE RESULTADOS INDIVIDUALES (SCRIPT 02)
# =============================================================================
if not ARCHIVO_TABLA.exists():
    raise FileNotFoundError(
        f"No se encontró {ARCHIVO_TABLA.name}. Ejecutá primero 02_procesamiento_ciclos.py."
    )

df = pd.read_csv(ARCHIVO_TABLA)

# Magnitudes auxiliares en unidades físicas convenientes
df["abs_I_in_mVs"]   = df["I_in_Vs"].abs() * 1e3
df["abs_I_out_mVs"]  = df["I_out_Vs"].abs() * 1e3
df["C_mVs"]          = df["C_Vs"] * 1e3
df["abs_dPhi_uWb"]   = df["dPhi_pulso_in_Wb"].abs() * 1e6
df["inv_Tin_1_s"]    = 1.0 / df["T_in_s"]

# Desviaciones absolutas por procesamiento (en mVs) para cada ciclo
df["sens_base_lin_mVs"] = (df["I_in_lin_Vs"] - df["I_in_Vs"]).abs() * 1e3
df["sens_vent2_mVs"]    = (df["I_in_vent2_Vs"] - df["I_in_Vs"]).abs() * 1e3
df["sens_sg1_mVs"]      = (df["I_in_sg1_Vs"] - df["I_in_Vs"]).abs() * 1e3
df["sens_sg2_mVs"]      = (df["I_in_sg2_Vs"] - df["I_in_Vs"]).abs() * 1e3
df["escala_Tub_mVs"]    = df["u_Iin_offset_Vs"] * 1e3


# =============================================================================
# 3. ESTADÍSTICA POR CONDICIÓN Y BOOTSTRAP EXPLORATORIO (B = 5000)
# =============================================================================
rng = np.random.default_rng(SEED)

filas_cuadro2 = []
boot_muestras = {}

for cond in ORDEN_COND:
    sub = df[df["condicion"] == cond].reset_index(drop=True)
    n = len(sub)

    abs_in  = sub["abs_I_in_mVs"].to_numpy()
    abs_out = sub["abs_I_out_mVs"].to_numpy()
    c_vals  = sub["C_mVs"].to_numpy()
    vpk     = sub["V_pk_in_V"].to_numpy()
    tin     = sub["T_in_s"].to_numpy()

    # Estadísticos descriptivos robustos (mediana, rango e IQR)
    med_in   = float(np.median(abs_in))
    rango_in = float(np.max(abs_in) - np.min(abs_in))
    q1_in, q3_in = np.percentile(abs_in, [25, 75])
    iqr_in   = float(q3_in - q1_in)

    med_out   = float(np.median(abs_out))
    rango_out = float(np.max(abs_out) - np.min(abs_out))

    med_c   = float(np.median(c_vals))
    med_vpk = float(np.median(vpk))
    med_tin = float(np.median(tin))

    # Bootstrap emparejado: se remuestrean ÍNDICES DE CICLO COMPLETO (Sec. 6.6)
    idx_boot = rng.integers(0, n, size=(B_BOOTSTRAP, n))
    boot_med_in   = np.median(abs_in[idx_boot], axis=1)
    boot_med_out  = np.median(abs_out[idx_boot], axis=1)
    boot_med_diff = np.median(abs_in[idx_boot] - abs_out[idx_boot], axis=1)

    ic95_in_lo, ic95_in_hi     = np.percentile(boot_med_in, [2.5, 97.5])
    ic95_out_lo, ic95_out_hi   = np.percentile(boot_med_out, [2.5, 97.5])
    ic95_dif_lo, ic95_dif_hi   = np.percentile(boot_med_diff, [2.5, 97.5])
    n_valores_unicos_boot      = len(np.unique(np.round(boot_med_in, 8)))

    boot_muestras[cond] = {
        "boot_in": boot_med_in,
        "boot_out": boot_med_out,
        "boot_diff": boot_med_diff,
        "ic95_in": (float(ic95_in_lo), float(ic95_in_hi)),
        "med_in": med_in,
    }

    # Sensibilidades al procesamiento evaluadas por separado (sin sumar en cuadratura)
    s_base_lin = float(sub["sens_base_lin_mVs"].max())
    s_vent2    = float(sub["sens_vent2_mVs"].max())
    s_sg1      = float(sub["sens_sg1_mVs"].max())
    s_sg2      = float(sub["sens_sg2_mVs"].max())
    s_Tub_med  = float(sub["escala_Tub_mVs"].median())
    s_Tub_max  = float(sub["escala_Tub_mVs"].max())
    max_dVpk_sg1 = float(sub["dVpk_sg1_pct"].abs().max())
    max_dVpk_sg2 = float(sub["dVpk_sg2_pct"].abs().max())

    filas_cuadro2.append({
        "condicion": cond,
        "n_ciclos": n,
        "med_Vpk_V": med_vpk,
        "med_Tin_s": med_tin,
        "med_abs_Iin_mVs": med_in,
        "rango_abs_Iin_mVs": rango_in,
        "iqr_abs_Iin_mVs": iqr_in,
        "ic95_boot_in_lo_mVs": float(ic95_in_lo),
        "ic95_boot_in_hi_mVs": float(ic95_in_hi),
        "boot_valores_unicos": int(n_valores_unicos_boot),
        "med_abs_dPhi_uWb": med_in * 1e3 / N_ESPIRAS,
        "ic95_boot_dPhi_lo_uWb": float(ic95_in_lo) * 1e3 / N_ESPIRAS,
        "ic95_boot_dPhi_hi_uWb": float(ic95_in_hi) * 1e3 / N_ESPIRAS,
        "med_abs_Iout_mVs": med_out,
        "rango_abs_Iout_mVs": rango_out,
        "ic95_boot_out_lo_mVs": float(ic95_out_lo),
        "ic95_boot_out_hi_mVs": float(ic95_out_hi),
        "med_diff_in_out_mVs": float(np.median(abs_in - abs_out)),
        "ic95_boot_diff_lo_mVs": float(ic95_dif_lo),
        "ic95_boot_diff_hi_mVs": float(ic95_dif_hi),
        "med_C_mVs": med_c,
        "med_epsC_pct": float(sub["eps_C"].median() * 100.0),
        "sens_max_base_lin_mVs": s_base_lin,
        "sens_max_vent2_mVs": s_vent2,
        "sens_max_sg1_mVs": s_sg1,
        "sens_max_sg2_mVs": s_sg2,
        "escala_Tub_med_mVs": s_Tub_med,
        "escala_Tub_max_mVs": s_Tub_max,
        "max_dVpk_sg1_pct": max_dVpk_sg1,
        "max_dVpk_sg2_pct": max_dVpk_sg2,
    })

df_c2 = pd.DataFrame(filas_cuadro2)
df_c2.to_csv(ARCHIVO_CUADRO2, index=False)


# =============================================================================
# 4. REPORTE EN CONSOLA (CUADRO 2 Y DATOS CLAVE PARA LA SECCIÓN 8)
# =============================================================================
print("=" * 98)
print("CUADRO 2 - RESUMEN POR CONDICIÓN (Sec. 6.6 y Sec. 7)")
print(f"Bootstrap exploratorio: unidad = ciclo completo | estadístico = mediana | B = {B_BOOTSTRAP} | seed = {SEED}")
print("=" * 98)

vista_c2 = pd.DataFrame({
    "Cond": df_c2["condicion"],
    "Med |I_in| (mVs)": df_c2["med_abs_Iin_mVs"].map(lambda x: f"{x:.3f}"),
    "Rango / IQR (mVs)": [
        f"{r:.3f} / {q:.3f}" for r, q in zip(df_c2["rango_abs_Iin_mVs"], df_c2["iqr_abs_Iin_mVs"])
    ],
    "IC95% boot (mVs)": [
        f"[{a:.3f}, {b:.3f}]" for a, b in zip(df_c2["ic95_boot_in_lo_mVs"], df_c2["ic95_boot_in_hi_mVs"])
    ],
    "Med |dPhi| (uWb)": df_c2["med_abs_dPhi_uWb"].map(lambda x: f"{x:.2f}"),
    "Med |I_out| (mVs)": df_c2["med_abs_Iout_mVs"].map(lambda x: f"{x:.3f}"),
    "Med C (mVs)": df_c2["med_C_mVs"].map(lambda x: f"{x:+.3f}"),
})
print(vista_c2.to_string(index=False))

print("\nSENSIBILIDAD AL PROCESAMIENTO POR CONDICIÓN (componentes separadas, en mVs):")
vista_sens = pd.DataFrame({
    "Cond": df_c2["condicion"],
    "T*u_b med/max": [
        f"{a:.3f} / {b:.3f}" for a, b in zip(df_c2["escala_Tub_med_mVs"], df_c2["escala_Tub_max_mVs"])
    ],
    "Base lineal": df_c2["sens_max_base_lin_mVs"].map(lambda x: f"{x:.4f}"),
    "2.a ventana": df_c2["sens_max_vent2_mVs"].map(lambda x: f"{x:.4f}"),
    "SG1 / SG2 (|I_in|)": [
        f"{a:.4f} / {b:.4f}" for a, b in zip(df_c2["sens_max_sg1_mVs"], df_c2["sens_max_sg2_mVs"])
    ],
    "SG1 / SG2 (V_pk %)": [
        f"{a:.2f}% / {b:.2f}%" for a, b in zip(df_c2["max_dVpk_sg1_pct"], df_c2["max_dVpk_sg2_pct"])
    ],
})
print(vista_sens.to_string(index=False))

print("\nDATOS AUXILIARES PARA PREGUNTAS DEL INFORME (SECCIÓN 8):")
# Pregunta 6.2 / 8.3: comparación meseta vs pulso de inserción
max_diff_mes = float(df["diff_meseta_pulso_Wb"].abs().max()) * 1e3
print(f"  - Máx. |ΔΛ_meseta - (-I_in)| en los 15 ciclos         : {max_diff_mes:.4f} mWb")

# Pregunta 6.3: chequeo A+ + A- = C
max_check_A = float(df["check_A_minus_C_Vs"].abs().max()) * 1e3
print(f"  - Máx. |(A+ + A-) - C| en los 15 ciclos               : {max_check_A:.6f} mVs")

# Pregunta 7: efecto cuantitativo de no corregir el offset
df["err_C_sin_corr_mVs"] = (df["C_sin_corr_Vs"] - df["C_Vs"]).abs() * 1e3
idx_peor_off = df["err_C_sin_corr_mVs"].idxmax()
r_peor = df.loc[idx_peor_off]
print(
    f"  - Mayor efecto del offset en cierre C ({r_peor['ensayo']}): "
    f"b0 = {r_peor['b0_V']*1e3:+.3f} mV -> "
    f"C_sin_corr = {r_peor['C_sin_corr_Vs']*1e3:+.3f} mVs vs C_corr = {r_peor['C_Vs']*1e3:+.3f} mVs"
)

# Comparación emparejada inserción vs extracción (asimetría de cierre)
for _, r in df_c2.iterrows():
    print(
        f"  - {r['condicion']}: mediana(|I_in| - |I_out|) = {r['med_diff_in_out_mVs']:+.3f} mVs "
        f"(IC95% boot emparejado: [{r['ic95_boot_diff_lo_mVs']:+.3f}, {r['ic95_boot_diff_hi_mVs']:+.3f}] mVs)"
    )

print(f"\nArchivo guardado: {ARCHIVO_CUADRO2.name}")
print("=" * 98)


# =============================================================================
# 5. FIGURA 1: INTEGRALES INDIVIDUALES (|I_in|, |I_out|) Y CIERRE C (Sec. 7)
# =============================================================================
fig1, (ax_i, ax_c) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1.0]})

x_pos = np.arange(len(df))
etiquetas_x = df["ensayo"].tolist()

# Panel superior: |I_in| y |I_out| por ciclo individual + mediana e IC95% bootstrap por condición
ax_i.scatter(x_pos - 0.12, df["abs_I_in_mVs"], s=58, marker="o", color="tab:blue",
             zorder=4, label=r"$|I_{in}|$ (inserción, individual)")
ax_i.scatter(x_pos + 0.12, df["abs_I_out_mVs"], s=58, marker="s", facecolors="none",
             edgecolors="tab:red", linewidths=1.6, zorder=4, label=r"$|I_{out}|$ (extracción, individual)")

for k, cond in enumerate(ORDEN_COND):
    x0, x1 = k * 5 - 0.40, k * 5 + 4.40
    rc = df_c2[df_c2["condicion"] == cond].iloc[0]
    col = COLORES_COND[cond]

    # Banda del IC95% bootstrap exploratorio para la mediana de |I_in|
    ax_i.fill_between(
        [x0, x1], rc["ic95_boot_in_lo_mVs"], rc["ic95_boot_in_hi_mVs"],
        color=col, alpha=0.16,
        label="IC 95% bootstrap exploratorio (mediana $|I_{in}|$)" if k == 0 else None
    )
    ax_i.hlines(
        rc["med_abs_Iin_mVs"], x0, x1, colors=col, lw=2.2, ls="-",
        label="Mediana por condición ($|I_{in}|$)" if k == 0 else None
    )
    if k < 2:
        ax_i.axvline(k * 5 + 4.5, color="gray", ls=":", lw=1.0)
        ax_c.axvline(k * 5 + 4.5, color="gray", ls=":", lw=1.0)

ax_i.set_ylabel(r"Magnitud de integral $|I|$ (mVs = mWb)")
ax_i.set_title("Integrales individuales de inserción y extracción por ciclo y resumen por condición")
ax_i.grid(True, axis="y", alpha=0.35)
ax_i.legend(loc="upper left", fontsize=8.5, ncol=2)

# Eje derecho en µWb por espira
ax_i_der = ax_i.twinx()
yi1, yi2 = ax_i.get_ylim()
ax_i_der.set_ylim(yi1 * 1e3 / N_ESPIRAS, yi2 * 1e3 / N_ESPIRAS)
ax_i_der.set_ylabel(rf"$|\Delta\overline{{\Phi}}| = |I|/{N_ESPIRAS}$ ($\mu$Wb)")

# Panel inferior: falta de cierre C por ciclo con barras de escala de offset ±T*u_b
colores_bar = [COLORES_COND[c] for c in df["condicion"]]
ax_c.axhline(0, color="black", lw=0.9, ls="--")
ax_c.errorbar(
    x_pos, df["C_mVs"], yerr=df["u_C_offset_Vs"] * 1e3,
    fmt="o", color="black", ecolor="gray", elinewidth=1.3, capsize=3.5, zorder=4,
    label=r"$C = \int_{ciclo} V_c\,dt$ (barras: escala $\pm T_{ciclo}\,u_b$)"
)
for xi, ci, ec, col in zip(x_pos, df["C_mVs"], df["eps_C"] * 100.0, colores_bar):
    ax_c.scatter(xi, ci, color=col, s=55, zorder=5)
    offset_txt = 0.08 if ci >= 0 else -0.14
    ax_c.text(xi, ci + offset_txt, f"{ec:.1f}%", ha="center", va="center", fontsize=7.5, color="dimgray")

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(etiquetas_x)
ax_c.set_xlabel("Ensayo (etiquetas numéricas sobre los puntos indican $\\epsilon_C$ en %)")
ax_c.set_ylabel(r"Integral de cierre $C$ (mVs)")
ax_c.grid(True, axis="y", alpha=0.35)
ax_c.legend(loc="lower left", fontsize=8.5)

fig1.tight_layout()
if GUARDAR_FIGURAS:
    fig1.savefig(CARPETA / "fig_03_integrales_individuales.png", dpi=300)


# =============================================================================
# 6. FIGURA 2: DIAGNÓSTICO V_pk vs 1/T_in E INVARIANCIA DE |I_in| (Sec. 6.4 y 7)
# =============================================================================
fig2, (ax_vpk, ax_inv) = plt.subplots(1, 2, figsize=(11.5, 4.8))

for cond in ORDEN_COND:
    sub = df[df["condicion"] == cond]
    col = COLORES_COND[cond]
    etq = NOMBRES_COND[cond]

    ax_vpk.scatter(sub["inv_Tin_1_s"], sub["V_pk_in_V"] * 1e3, s=65, color=col,
                   edgecolors="black", linewidths=0.5, label=etq, zorder=4)
    ax_inv.scatter(sub["inv_Tin_1_s"], sub["abs_I_in_mVs"], s=65, color=col,
                   edgecolors="black", linewidths=0.5, label=etq, zorder=4)

# Panel izquierdo: V_pk vs 1/T_in (sin imponer proporcionalidad exacta)
ax_vpk.set_title(r"Diagnóstico amplitud vs. inverso de duración (Sec. 6.4)")
ax_vpk.set_xlabel(r"$1 / T_{in}$ ($\mathrm{s}^{-1}$)")
ax_vpk.set_ylabel(r"Amplitud de pico $V_{pk}$ (mV)")
ax_vpk.grid(True, alpha=0.35)
ax_vpk.legend(loc="upper left", fontsize=9)

# Panel derecho: |I_in| vs 1/T_in con referencia de la mediana global
med_global_in = float(df["abs_I_in_mVs"].median())
ax_inv.axhline(med_global_in, color="black", ls="--", lw=1.1, alpha=0.75,
               label=f"Mediana global = {med_global_in:.2f} mVs")
ax_inv.set_title(r"Invariancia de la integral $|I_{in}|$ frente a la rapidez")
ax_inv.set_xlabel(r"$1 / T_{in}$ ($\mathrm{s}^{-1}$)")
ax_inv.set_ylabel(r"$|I_{in}|$ (mVs = mWb)")
ax_inv.set_ylim(0.85 * med_global_in, 1.15 * med_global_in)
ax_inv.grid(True, alpha=0.35)
ax_inv.legend(loc="lower right", fontsize=8.5)

fig2.tight_layout()
if GUARDAR_FIGURAS:
    fig2.savefig(CARPETA / "fig_03_Vpk_vs_invTin.png", dpi=300)


# =============================================================================
# 7. FIGURA 3: BOOTSTRAP EXPLORATORIO Y COMPARACIÓN DE SENSIBILIDADES (Sec. 6.6)
# =============================================================================
fig3, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(11.8, 4.8))

# Panel izquierdo: distribuciones discretas de la mediana bootstrap (n=5 -> pocos valores únicos)
y_pos = np.arange(len(ORDEN_COND))
for k, cond in enumerate(ORDEN_COND):
    b_vals = boot_muestras[cond]["boot_in"]
    ic_lo, ic_hi = boot_muestras[cond]["ic95_in"]
    med_v = boot_muestras[cond]["med_in"]
    col = COLORES_COND[cond]

    # Puntos individuales de los 5 ciclos de esa condición
    sub_vals = df[df["condicion"] == cond]["abs_I_in_mVs"].to_numpy()
    ax_b.scatter(sub_vals, np.full_like(sub_vals, k + 0.16), color=col, s=38, alpha=0.85,
                 marker="o", label="Ciclos individuales" if k == 0 else None, zorder=4)

    # Barra horizontal del IC 95% bootstrap y marcador de mediana
    ax_b.hlines(k - 0.08, ic_lo, ic_hi, color=col, lw=4.0, alpha=0.55,
                label="IC 95% bootstrap (B=5000)" if k == 0 else None)
    ax_b.scatter([med_v], [k - 0.08], color="black", marker="D", s=45, zorder=5,
                 label="Mediana" if k == 0 else None)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([NOMBRES_COND[c] for c in ORDEN_COND])
ax_b.set_xlabel(r"$|I_{in}|$ (mVs = mWb)")
ax_b.set_title("Ciclos individuales, mediana e IC 95% bootstrap exploratorio")
ax_b.grid(True, axis="x", alpha=0.35)
ax_b.legend(loc="lower left", fontsize=8.5)

# Panel derecho: comparación de escalas de variación (sin sumar en cuadratura)
ancho = 0.16
x_g = np.arange(len(ORDEN_COND))
ax_s.bar(x_g - 2.0 * ancho, df_c2["rango_abs_Iin_mVs"], width=ancho,
         color="tab:blue", label="Rango entre ciclos")
ax_s.bar(x_g - 1.0 * ancho, df_c2["escala_Tub_max_mVs"], width=ancho,
         color="tab:red", label=r"Escala offset máx. ($T u_b$)")
ax_s.bar(x_g + 0.0 * ancho, df_c2["sens_max_vent2_mVs"], width=ancho,
         color="tab:orange", label="2.ª ventana de reposo")
ax_s.bar(x_g + 1.0 * ancho, df_c2["sens_max_base_lin_mVs"], width=ancho,
         color="tab:green", label="Base lineal vs. constante")
ax_s.bar(x_g + 2.0 * ancho, df_c2["sens_max_sg2_mVs"], width=ancho,
         color="tab:purple", label="Savitzky-Golay (15,5 ms)")

ax_s.set_xticks(x_g)
ax_s.set_xticklabels(ORDEN_COND)
ax_s.set_ylabel(r"Magnitud de variación en $|I_{in}|$ (mVs)")
ax_s.set_title("Dispersión entre ciclos vs. sensibilidad al procesamiento")
ax_s.grid(True, axis="y", alpha=0.35)
ax_s.legend(loc="upper right", fontsize=8)

fig3.tight_layout()
if GUARDAR_FIGURAS:
    fig3.savefig(CARPETA / "fig_03_bootstrap_y_sensibilidad.png", dpi=300)

plt.show()