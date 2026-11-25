# -*- coding: utf-8 -*-
"""
Guía N.° 4 - Física Experimental II (UNSJ)
Script 01 (mejorado):
  - Auditoría de datos crudos (Sec. 5.1 y 6.1)
  - Gráfico Parte A: signo al invertir el imán (Sec. 5.2)
  - Detección robusta de los 5 ciclos en L, M y R
  - Exportación automática de 'ventanas_ciclos.csv' para el Script 02
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================
CARPETA_DATOS = Path(__file__).parent if "__file__" in globals() else Path(".")

ARCHIVOS = {
    "A_norte": "A_norte.csv",   # iman norte primero
    "A_sur":   "A_sur.csv",     # iman sur primero
    "L":       "L_raw.csv",     # iman lento
    "M":       "M_raw.csv",     # iman medio
    "R":       "R_raw.csv",     # iman rapido
}

ARCHIVO_SALIDA_VENTANAS = CARPETA_DATOS / "ventanas_ciclos.csv"
GUARDAR_FIGURAS = True

# Parámetros del detector robusto (solo se usan para ubicar ventanas)
VENTANA_ENV_S = 0.020    # 20 ms para envolvente RMS de detección
GAP_FUSION_S  = 0.150    # fusiona lóbulos/rebotes separados por < 150 ms
DURACION_MIN_S = 0.025   # descarta picos espúreos más cortos que 25 ms


# =============================================================================
# 2. CARGA ROBUSTA DE CSV DE PASCO CAPSTONE
# =============================================================================
def _a_float(texto):
    """Convierte texto numérico con coma o punto decimal a float."""
    s = texto.strip()
    if s == "":
        return np.nan
    return float(s.replace(",", "."))


def cargar_csv_pasco(ruta_archivo):
    """
    Carga un CSV de PASCO Capstone detectando automáticamente delimitador,
    separador decimal y filas de encabezado.
    """
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}")

    try:
        lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lineas = ruta.read_text(encoding="latin-1").splitlines()

    delimitador = None
    fila_inicio = None

    for sep in (";", "\t", ","):
        for idx, linea in enumerate(lineas[:15]):
            if not linea.strip():
                continue
            fila = next(csv.reader([linea], delimiter=sep))
            if len(fila) < 2:
                continue
            try:
                val0 = _a_float(fila[0])
                val1 = _a_float(fila[1])
                if np.isfinite(val0) and np.isfinite(val1):
                    delimitador = sep
                    fila_inicio = idx
                    break
            except ValueError:
                continue
        if delimitador is not None:
            break

    if delimitador is None:
        raise ValueError(f"No se pudo interpretar el formato numérico de {ruta.name}.")

    encabezados = [l.strip() for l in lineas[:fila_inicio] if l.strip()]

    t_lista, v_lista = [], []
    n_faltantes = 0
    for fila in csv.reader(lineas[fila_inicio:], delimiter=delimitador):
        if not fila or all(c.strip() == "" for c in fila):
            continue
        if len(fila) < 2:
            n_faltantes += 1
            continue
        try:
            tv = _a_float(fila[0])
            vv = _a_float(fila[1])
            if np.isfinite(tv) and np.isfinite(vv):
                t_lista.append(tv)
                v_lista.append(vv)
            else:
                n_faltantes += 1
        except ValueError:
            n_faltantes += 1

    return np.array(t_lista, dtype=float), np.array(v_lista, dtype=float), encabezados, n_faltantes


# =============================================================================
# 3. ESTIMACIÓN DE BASE/RUIDO EN TRAMOS QUIETOS Y DETECCIÓN ROBUSTA DE PULSOS
# =============================================================================
def estimar_reposo_robusto(t, V, bloque_s=0.20):
    """
    Divide la señal en bloques cortos y toma el 30% con menor desviación estándar
    para estimar el nivel de base (v0) y el ruido (sigma) sin contaminación de pulsos.
    """
    dt = np.median(np.diff(t))
    n_blk = max(20, int(round(bloque_s / dt)))
    n_bloques = len(V) // n_blk

    if n_bloques < 4:
        v0 = np.median(V)
        sigma = 1.4826 * np.median(np.abs(V - v0))
        return v0, sigma

    matriz = V[:n_bloques * n_blk].reshape(n_bloques, n_blk)
    stds = np.std(matriz, axis=1)
    n_quietos = max(2, int(np.ceil(0.30 * n_bloques)))
    idx_quietos = np.argsort(stds)[:n_quietos]
    muestras_quietas = matriz[idx_quietos].ravel()

    v0 = float(np.median(muestras_quietas))
    sigma = float(1.4826 * np.median(np.abs(muestras_quietas - v0)))
    return v0, sigma, idx_quietos, n_blk


def detectar_pulsos_robusto(t, V):
    """
    Localiza pulsos principales usando una envolvente RMS corta (solo para detección,
    sin alterar los datos crudos) y fusionando lóbulos/rebotes cercanos.
    """
    dt = float(np.median(np.diff(t)))
    res_rep = estimar_reposo_robusto(t, V)
    v0, sigma = res_rep[0], res_rep[1]
    vc = V - v0

    # Envolvente RMS en ventana de 20 ms (siempre >= 0, evita cancelación de rebotes)
    w = max(5, int(round(VENTANA_ENV_S / dt)))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    env_rms = np.sqrt(np.convolve(vc ** 2, kernel, mode="same"))

    # Nivel de la envolvente en los tramos quietos
    if len(res_rep) == 4:
        idx_q, n_blk = res_rep[2], res_rep[3]
        env_mat = env_rms[: (len(V) // n_blk) * n_blk].reshape(-1, n_blk)
        env_q = env_mat[idx_q].ravel()
    else:
        env_q = env_rms

    mu_env = float(np.median(env_q))
    sd_env = float(1.4826 * np.median(np.abs(env_q - mu_env)))

    # Umbral por encima del piso de ruido de la envolvente
    umbral_env = max(mu_env + 6.0 * sd_env, 2.8 * mu_env)

    sobre = env_rms > umbral_env
    cambios = np.diff(sobre.astype(int), prepend=0, append=0)
    inicios = np.where(cambios == 1)[0]
    fines = np.where(cambios == -1)[0]

    if len(inicios) == 0:
        return [], v0, sigma

    # Fusionar segmentos separados por menos de GAP_FUSION_S (une lóbulos y rebotes)
    fusionados = [[int(inicios[0]), int(fines[0])]]
    for i0, i1 in zip(inicios[1:], fines[1:]):
        if (t[i0] - t[fusionados[-1][1] - 1]) < GAP_FUSION_S:
            fusionados[-1][1] = int(i1)
        else:
            fusionados.append([int(i0), int(i1)])

    # Filtrar segmentos reales (amplitud y duración mínimas)
    vmax_env = float(np.max(env_rms))
    pulsos = []
    for i0, i1 in fusionados:
        dur = float(t[i1 - 1] - t[i0])
        if dur < DURACION_MIN_S:
            continue
        if np.max(env_rms[i0:i1]) < 0.25 * vmax_env:
            continue

        seg_vc = vc[i0:i1]
        idx_rel = int(np.argmax(np.abs(seg_vc)))
        pico = float(seg_vc[idx_rel])

        # Conteo de puntos del lóbulo principal (> 10% del pico) para chequeo Sec. 5.1
        n_pts_principal = int(np.sum(np.abs(seg_vc) > 0.10 * abs(pico)))

        pulsos.append({
            "i_ini": i0,
            "i_fin": i1 - 1,
            "t_ini": float(t[i0]),
            "t_fin": float(t[i1 - 1]),
            "t_pico": float(t[i0 + idx_rel]),
            "v_pico": pico,
            "signo": "+" if pico > 0 else "-",
            "n_puntos_ventana": int(i1 - i0),
            "n_puntos_pulso": n_pts_principal,
            "duracion": dur,
        })

    return pulsos, v0, sigma


def construir_ventanas_ciclos(condicion, nombre_csv, t, pulsos):
    """
    A partir de los 10 pulsos (5 pares inserción/extracción), calcula automáticamente
    todas las ventanas temporales necesarias para el Script 02.
    """
    if len(pulsos) < 2 or len(pulsos) % 2 != 0:
        print(f"  [ALERTA] {condicion}: se detectaron {len(pulsos)} pulsos (se esperaban 10).")
        return []

    n_ciclos = len(pulsos) // 2
    filas = []

    for k in range(n_ciclos):
        p_in = pulsos[2 * k]
        p_out = pulsos[2 * k + 1]

        prev_end = float(t[0]) if k == 0 else pulsos[2 * k - 1]["t_fin"]
        next_start = float(t[-1]) if k == n_ciclos - 1 else pulsos[2 * k + 2]["t_ini"]

        gap_pre = p_in["t_ini"] - prev_end
        gap_pausa = p_out["t_ini"] - p_in["t_fin"]
        gap_post = next_start - p_out["t_fin"]

        # Pequeño margen de reposo a ambos lados de cada pulso (Sec. 6.3)
        m_pre = min(0.05, 0.20 * max(gap_pre, 0.0))
        m_pausa = min(0.05, 0.20 * max(gap_pausa, 0.0))
        m_post = min(0.05, 0.20 * max(gap_post, 0.0))

        t_in_1 = p_in["t_ini"] - m_pre
        t_in_2 = p_in["t_fin"] + m_pausa
        t_out_1 = p_out["t_ini"] - m_pausa
        t_out_2 = p_out["t_fin"] + m_post

        # Pausa interior limpia entre inserción y extracción
        t_pausa_1 = t_in_2
        t_pausa_2 = t_out_1

        # Ventana de reposo anterior
        t_rep_ini_2 = t_in_1
        if k == 0:
            t_rep_ini_1 = max(float(t[0]) + 0.05, t_rep_ini_2 - 0.80)
        else:
            t_rep_ini_1 = prev_end + min(0.08, 0.25 * max(gap_pre, 0.0))

        # Ventana de reposo posterior
        t_rep_fin_1 = t_out_2
        if k == n_ciclos - 1:
            t_rep_fin_2 = min(float(t[-1]) - 0.05, t_rep_fin_1 + 0.80)
        else:
            t_rep_fin_2 = next_start - min(0.08, 0.25 * max(gap_post, 0.0))

        filas.append({
            "ensayo": f"{condicion}{k + 1:02d}",
            "condicion": condicion,
            "archivo_csv": nombre_csv,
            "t_rep_ini_1": round(t_rep_ini_1, 4),
            "t_rep_ini_2": round(t_rep_ini_2, 4),
            "t_in_1": round(t_in_1, 4),
            "t_in_2": round(t_in_2, 4),
            "t_pausa_1": round(t_pausa_1, 4),
            "t_pausa_2": round(t_pausa_2, 4),
            "t_out_1": round(t_out_1, 4),
            "t_out_2": round(t_out_2, 4),
            "t_rep_fin_1": round(t_rep_fin_1, 4),
            "t_rep_fin_2": round(t_rep_fin_2, 4),
            "t_ciclo_1": round(t_in_1, 4),
            "t_ciclo_2": round(t_out_2, 4),
        })

    return filas


def auditar_serie(nombre, t, V, encabezados, n_faltantes):
    """Imprime la auditoría de la serie (Sec. 5.1 y 6.1)."""
    dt = np.diff(t)
    dt_med = float(np.median(dt))
    fs = 1.0 / dt_med if dt_med > 0 else np.nan
    orden_ok = bool(np.all(dt > 0))
    uniforme_pct = 100.0 * (np.max(dt) - np.min(dt)) / dt_med if dt_med > 0 else np.nan

    vmax_abs, vmin_abs = float(np.max(V)), float(np.min(V))
    n_en_max = int(np.sum(np.isclose(V, vmax_abs, rtol=0, atol=1e-9)))
    n_en_min = int(np.sum(np.isclose(V, vmin_abs, rtol=0, atol=1e-9)))
    posible_sat = (n_en_max >= 5) or (n_en_min >= 5)

    pulsos, v0, sigma = detectar_pulsos_robusto(t, V)

    print("=" * 76)
    print(f"SERIE: {nombre}")
    print(f"  Encabezados            : {encabezados}")
    print(f"  Muestras válidas       : {len(t)}  |  Faltantes/NaN: {n_faltantes}")
    print(f"  Duración total         : {t[-1] - t[0]:.3f} s  (de {t[0]:.3f} a {t[-1]:.3f} s)")
    print(f"  Orden temporal         : {'OK' if orden_ok else 'ERROR'}")
    print(f"  dt mediano / fs        : {dt_med*1e3:.4f} ms  /  {fs:.1f} Hz  (var dt: {uniforme_pct:.3f} %)")
    print(f"  Rango de tensión       : [{vmin_abs:+.5f}, {vmax_abs:+.5f}] V")
    print(f"  Posible saturación     : {'ALERTA' if posible_sat else 'No detectada'}")
    print(f"  Reposo (tramos quietos): base = {v0*1e3:+.3f} mV  |  sigma = {sigma*1e3:.3f} mV")

    if pulsos:
        min_pts = min(p["n_puntos_pulso"] for p in pulsos)
        estado_50 = "OK (>= 50 pts)" if min_pts >= 50 else "ALERTA (< 50 pts)"
        print(f"  Pulsos detectados      : {len(pulsos)}  |  Signo 1.er pulso: '{pulsos[0]['signo']}'")
        print(f"  Puntos pulso más breve : {min_pts} pts (>10% pico) -> {estado_50}")
        for k, p in enumerate(pulsos, start=1):
            print(
                f"    P{k:02d} ({p['signo']}): "
                f"tramo [{p['t_ini']:6.3f}, {p['t_fin']:6.3f}] s  |  "
                f"t_pico = {p['t_pico']:6.3f} s  |  "
                f"V_pico = {p['v_pico']:+.4f} V  |  "
                f"{p['n_puntos_pulso']} pts"
            )
    else:
        print("  Pulsos detectados      : Ninguno.")

    return pulsos


# =============================================================================
# 4. EJECUCIÓN PRINCIPAL: AUDITORÍA Y EXPORTACIÓN DE VENTANAS
# =============================================================================
datos = {}
resumen_pulsos = {}
todas_las_ventanas = []

print("\nINICIANDO AUDITORÍA Y DETECCIÓN ROBUSTA DE CICLOS\n")

for clave, nombre_csv in ARCHIVOS.items():
    ruta = CARPETA_DATOS / nombre_csv
    t, V, enc, n_nan = cargar_csv_pasco(ruta)
    datos[clave] = (t, V)
    pulsos = auditar_serie(clave, t, V, enc, n_nan)
    resumen_pulsos[clave] = pulsos

    if clave in ("L", "M", "R"):
        todas_las_ventanas.extend(
            construir_ventanas_ciclos(clave, nombre_csv, t, pulsos)
        )

# Chequeo cruzado de signos
print("=" * 76)
print("VERIFICACIÓN DE COHERENCIA DE SIGNOS (PRIMER PULSO):")
for clave in ("A_norte", "A_sur", "L", "M", "R"):
    signo = resumen_pulsos[clave][0]["signo"] if resumen_pulsos[clave] else "?"
    print(f"  {clave:8s} -> {signo}")
print("  (Esperado: A_norte y A_sur opuestos; L, M y R igual que A_norte)")
print("=" * 76)

# Exportar ventanas_ciclos.csv para el Script 02
df_ventanas = pd.DataFrame(todas_las_ventanas)
df_ventanas.to_csv(ARCHIVO_SALIDA_VENTANAS, index=False)

print(f"\nARCHIVO EXPORTADO PARA SCRIPT 02: {ARCHIVO_SALIDA_VENTANAS.name}")
print(df_ventanas[[
    "ensayo", "t_rep_ini_1", "t_in_1", "t_in_2", "t_out_1", "t_out_2", "t_rep_fin_2"
]].to_string(index=False))
print("=" * 76)


# =============================================================================
# 5. GRÁFICO PARTE A (NORTE VS SUR)
# =============================================================================
fig_a, axes_a = plt.subplots(2, 1, figsize=(10, 6.2), sharex=False, sharey=True)

for ax, clave, titulo, color in [
    (axes_a[0], "A_norte", "Parte A - Inserción con polo Norte primero", "tab:blue"),
    (axes_a[1], "A_sur",   "Parte A - Inserción con polo Sur primero",   "tab:red"),
]:
    t, V = datos[clave]
    ax.plot(t, V, color=color, lw=1.0, label=clave)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("Tensión medida $V_{med}$ (V)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right")

axes_a[1].set_xlabel("Tiempo (s)")
fig_a.suptitle("Parte A: Comparación de signo al invertir la orientación del imán", fontsize=12)
fig_a.tight_layout()

if GUARDAR_FIGURAS:
    fig_a.savefig(CARPETA_DATOS / "fig_01_parte_A_signos.png", dpi=300)


# =============================================================================
# 6. GRÁFICOS DE VERIFICACIÓN DE VENTANAS PARA L, M Y R
# =============================================================================
config_b = [
    ("L", "Condición L (Lento) - Ventanas automáticas para Script 02",      "tab:green"),
    ("M", "Condición M (Intermedio) - Ventanas automáticas para Script 02", "tab:orange"),
    ("R", "Condición R (Rápido) - Ventanas automáticas para Script 02",     "tab:purple"),
]

for clave, titulo, color in config_b:
    t, V = datos[clave]
    sub = df_ventanas[df_ventanas["condicion"] == clave]

    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.plot(t, V, color=color, lw=0.9, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)

    ymax_plot = np.max(np.abs(V)) * 1.12
    ax.set_ylim(-ymax_plot, ymax_plot)

    for idx_row, row in sub.iterrows():
        # Sombrear ventanas para verificación visual rápida
        ax.axvspan(row["t_in_1"], row["t_in_2"], color="tab:blue", alpha=0.14,
                   label="Inserción" if row["ensayo"].endswith("01") else None)
        ax.axvspan(row["t_pausa_1"], row["t_pausa_2"], color="gray", alpha=0.14,
                   label="Pausa interior" if row["ensayo"].endswith("01") else None)
        ax.axvspan(row["t_out_1"], row["t_out_2"], color="tab:red", alpha=0.14,
                   label="Extracción" if row["ensayo"].endswith("01") else None)

        # Etiqueta del ciclo centrado arriba
        t_centro = 0.5 * (row["t_in_1"] + row["t_out_2"])
        ax.text(
            t_centro, ymax_plot * 0.90, row["ensayo"],
            ha="center", va="top", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85)
        )

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Tensión medida $V_{med}$ (V)")
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.18)
    ax.legend(loc="lower right", ncol=3, fontsize=8.5)
    fig.tight_layout()

    if GUARDAR_FIGURAS:
        fig.savefig(CARPETA_DATOS / f"fig_01_inspeccion_{clave}.png", dpi=250)

plt.show()