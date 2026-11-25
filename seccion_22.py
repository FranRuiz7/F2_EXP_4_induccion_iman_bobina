# -*- coding: utf-8 -*-
"""
Guía N.° 4 - Física Experimental II (UNSJ)
Script 02: Procesamiento completo de los 15 ciclos (Secciones 6.1 a 6.5).

Lee:
    - ventanas_ciclos.csv (generado por Script 01)
    - L_raw.csv, M_raw.csv, R_raw.csv (datos crudos de PASCO Capstone)

Genera:
    - tabla_ciclos.csv (Cuadro 1 + columnas de sensibilidad para Script 03)
    - fig_02_ciclo_ejemplo.png (tensión original, línea de base, Vc, ΔΛ y ΔΦ̄)
    - fig_02_comparacion_LMR_ciclos.png (comparación temporal de 1 ciclo de L, M y R)
    - fig_02_suavizado_sg.png (sensibilidad al filtro Savitzky-Golay en ciclo rápido)
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.signal import savgol_filter


# =============================================================================
# 1. CONFIGURACIÓN GENERAL (parámetros reproducibles)
# =============================================================================
CARPETA = Path(__file__).parent if "__file__" in globals() else Path(".")
ARCHIVO_VENTANAS = CARPETA / "ventanas_ciclos.csv"
ARCHIVO_SALIDA_TABLA = CARPETA / "tabla_ciclos.csv"

N_ESPIRAS = 300                 # Documentado en la bobina ("300 VTAS")

# Criterio reproducible para duración T_in (Sec. 6.4):
# límites de la envolvente del pulso donde |Vc| supera K_UMBRAL * sigma_reposo
K_UMBRAL_DURACION = 5.0

# Parámetros del filtro Savitzky-Golay (Sec. 6.5):
# Dos ventanas temporales comparables entre ensayos (en segundos) y orden polinómico
SG_ORDEN = 3
SG_VENTANAS_S = (0.005, 0.015)  # 5 ms (suave) y 15 ms (más ancha)

# Ensayos elegidos para las figuras de diagnóstico (podés cambiarlos por cualquiera)
ENSAYO_EJEMPLO_CICLO = "M01"
ENSAYO_EJEMPLO_SG    = "R01"
ENSAYOS_COMP_LMR     = ("L01", "M01", "R01")

GUARDAR_FIGURAS = True


# =============================================================================
# 2. CARGA DE DATOS CRUDOS (misma función robusta que Script 01)
# =============================================================================
def _a_float(texto):
    s = texto.strip()
    if s == "":
        return np.nan
    return float(s.replace(",", "."))


def cargar_csv_pasco(ruta_archivo):
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}")

    try:
        lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lineas = ruta.read_text(encoding="latin-1").splitlines()

    delimitador, fila_inicio = None, None
    for sep in (";", "\t", ","):
        for idx, linea in enumerate(lineas[:15]):
            if not linea.strip():
                continue
            fila = next(csv.reader([linea], delimiter=sep))
            if len(fila) < 2:
                continue
            try:
                v0, v1 = _a_float(fila[0]), _a_float(fila[1])
                if np.isfinite(v0) and np.isfinite(v1):
                    delimitador, fila_inicio = sep, idx
                    break
            except ValueError:
                continue
        if delimitador is not None:
            break

    if delimitador is None:
        raise ValueError(f"No se pudo interpretar el formato numérico de {ruta.name}.")

    t_lista, v_lista = [], []
    for fila in csv.reader(lineas[fila_inicio:], delimiter=delimitador):
        if len(fila) < 2:
            continue
        try:
            tv, vv = _a_float(fila[0]), _a_float(fila[1])
            if np.isfinite(tv) and np.isfinite(vv):
                t_lista.append(tv)
                v_lista.append(vv)
        except ValueError:
            continue

    return np.array(t_lista, dtype=float), np.array(v_lista, dtype=float)


# =============================================================================
# 3. FUNCIONES AUXILIARES DE PROCESAMIENTO (SECCIONES 6.1 A 6.5)
# =============================================================================
def mascara_rango(t, t1, t2):
    return (t >= t1) & (t <= t2)


def ajustar_ventana_impar(duracion_s, dt, orden, n_max):
    """Convierte una duración en segundos a ventana impar válida para Savitzky-Golay."""
    n_pts = int(round(duracion_s / dt))
    if n_pts % 2 == 0:
        n_pts += 1
    n_min = orden + 2 if (orden + 2) % 2 == 1 else orden + 3
    n_pts = max(n_pts, n_min)
    if n_pts >= n_max:
        n_pts = n_max - 1 if (n_max - 1) % 2 == 1 else n_max - 2
    return int(n_pts), float(n_pts * dt)


def calcular_duracion_pulso(t_seg, vc_seg, umbral):
    """
    Sec. 6.4: Duración entre el primer y el último cruce de |Vc| > umbral
    dentro de la ventana del pulso (abarca todos los lóbulos, incluido rebote).
    """
    idx = np.where(np.abs(vc_seg) > umbral)[0]
    if len(idx) < 2:
        return np.nan, np.nan, np.nan
    t_a = float(t_seg[idx[0]])
    t_b = float(t_seg[idx[-1]])
    return float(t_b - t_a), t_a, t_b


def calcular_integrales_basicas(t, vc, row):
    """Calcula I_in, I_out, C, eps_C, A+, A- y V_pk para una señal Vc dada."""
    m_in = mascara_rango(t, row["t_in_1"], row["t_in_2"])
    m_out = mascara_rango(t, row["t_out_1"], row["t_out_2"])
    m_cic = mascara_rango(t, row["t_ciclo_1"], row["t_ciclo_2"])

    I_in = float(trapezoid(vc[m_in], t[m_in]))
    I_out = float(trapezoid(vc[m_out], t[m_out]))
    C = float(trapezoid(vc[m_cic], t[m_cic]))
    denom = abs(I_in) + abs(I_out)
    eps_C = float(abs(C) / denom) if denom > 0 else np.nan

    A_pos = float(trapezoid(np.maximum(vc[m_cic], 0.0), t[m_cic]))
    A_neg = float(trapezoid(np.minimum(vc[m_cic], 0.0), t[m_cic]))

    V_pk_in = float(np.max(np.abs(vc[m_in])))
    V_pk_out = float(np.max(np.abs(vc[m_out])))

    return {
        "I_in": I_in,
        "I_out": I_out,
        "C": C,
        "eps_C": eps_C,
        "A_pos": A_pos,
        "A_neg": A_neg,
        "V_pk_in": V_pk_in,
        "V_pk_out": V_pk_out,
    }


def analizar_ciclo(t_full, V_full, row):
    """
    Ejecuta las secciones 6.1 a 6.5 sobre un ciclo individual.
    Devuelve:
      - res: diccionario plano con todas las métricas del ciclo
      - serie: diccionario con arrays temporales para graficar
    """
    # Recortar el tramo completo del ciclo (desde reposo inicial hasta reposo final)
    m_tramo = mascara_rango(t_full, row["t_rep_ini_1"], row["t_rep_fin_2"])
    t = t_full[m_tramo].copy()
    V = V_full[m_tramo].copy()
    dt = float(np.median(np.diff(t)))

    # Máscaras de las regiones nominales
    m_rep_ini = mascara_rango(t, row["t_rep_ini_1"], row["t_rep_ini_2"])
    m_rep_fin = mascara_rango(t, row["t_rep_fin_1"], row["t_rep_fin_2"])
    m_rep_ext = m_rep_ini | m_rep_fin
    m_pausa   = mascara_rango(t, row["t_pausa_1"], row["t_pausa_2"])
    m_in      = mascara_rango(t, row["t_in_1"], row["t_in_2"])
    m_out     = mascara_rango(t, row["t_out_1"], row["t_out_2"])

    # -------------------------------------------------------------------------
    # 6.1 Corrección de línea de base (constante, lineal y 2.ª ventana de reposo)
    # -------------------------------------------------------------------------
    mu_ini = float(np.mean(V[m_rep_ini]))
    mu_fin = float(np.mean(V[m_rep_fin]))
    sigma_ini = float(np.std(V[m_rep_ini], ddof=1))
    sigma_fin = float(np.std(V[m_rep_fin], ddof=1))

    # Modelo principal: offset constante con ambos reposos exteriores
    b0 = float(np.mean(V[m_rep_ext]))
    Vc = V - b0
    sigma_rep = float(np.std(Vc[m_rep_ext], ddof=1))

    # Compatibilidad de la pausa interior con el modelo elegido
    media_pausa_vc = float(np.mean(Vc[m_pausa])) if np.sum(m_pausa) >= 3 else np.nan
    std_pausa_vc   = float(np.std(Vc[m_pausa], ddof=1)) if np.sum(m_pausa) >= 3 else np.nan

    # Estimación de incertidumbre del offset u_b (combina error estándar y deriva ini-fin)
    se_rep = sigma_rep / np.sqrt(np.sum(m_rep_ext))
    diff_ext = mu_fin - mu_ini
    u_b = float(max(se_rep, 0.5 * abs(diff_ext)))

    # Modelo alternativo 1: línea de base lineal b(t) = a + c*(t - t0) sobre reposos exteriores
    t_ref = float(t[0])
    c_lin, a_lin = np.polyfit(t[m_rep_ext] - t_ref, V[m_rep_ext], 1)
    b_lin = a_lin + c_lin * (t - t_ref)
    Vc_lin = V - b_lin

    # Modelo alternativo 2 (2.ª selección razonable de ventanas):
    # recorta el 25 % de los extremos de cada ventana de reposo para alejarse de los pulsos
    d_ini = row["t_rep_ini_2"] - row["t_rep_ini_1"]
    d_fin = row["t_rep_fin_2"] - row["t_rep_fin_1"]
    m_rep_alt = (
        mascara_rango(t, row["t_rep_ini_1"] + 0.25 * d_ini, row["t_rep_ini_2"] - 0.25 * d_ini)
        | mascara_rango(t, row["t_rep_fin_1"] + 0.25 * d_fin, row["t_rep_fin_2"] - 0.25 * d_fin)
    )
    b0_alt = float(np.mean(V[m_rep_alt])) if np.sum(m_rep_alt) >= 5 else b0
    Vc_alt = V - b0_alt

    # -------------------------------------------------------------------------
    # 6.2 Integración trapezoidal acumulativa: ΔΛ(t) = -∫ Vc(t') dt'
    # -------------------------------------------------------------------------
    dLambda = -cumulative_trapezoid(Vc, t, initial=0.0)
    dPhi    = dLambda / N_ESPIRAS

    # Comparación sin corrección de línea de base (útil para Pregunta 7 del informe)
    dLambda_sin_corr = -cumulative_trapezoid(V, t, initial=0.0)

    # Cambio interior por diferencia de mesetas (pausa interior - reposo inicial)
    meseta_ini = float(np.mean(dLambda[m_rep_ini]))
    meseta_int = float(np.mean(dLambda[m_pausa])) if np.sum(m_pausa) >= 3 else np.nan
    meseta_fin = float(np.mean(dLambda[m_rep_fin]))
    dLambda_meseta = float(meseta_int - meseta_ini)
    dPhi_meseta    = float(dLambda_meseta / N_ESPIRAS)

    # -------------------------------------------------------------------------
    # 6.3 Integrales de inserción, extracción, cierre y áreas A+, A-
    # -------------------------------------------------------------------------
    met_crudo   = calcular_integrales_basicas(t, Vc, row)
    met_sin_cor = calcular_integrales_basicas(t, V, row)
    met_lin     = calcular_integrales_basicas(t, Vc_lin, row)
    met_alt     = calcular_integrales_basicas(t, Vc_alt, row)

    # -------------------------------------------------------------------------
    # 6.4 Amplitud y duración de las transiciones
    # -------------------------------------------------------------------------
    umbral_dur = K_UMBRAL_DURACION * sigma_rep
    T_in, t_in_u1, t_in_u2 = calcular_duracion_pulso(t[m_in], Vc[m_in], umbral_dur)
    T_out, _, _            = calcular_duracion_pulso(t[m_out], Vc[m_out], umbral_dur)

    # Escala de incertidumbre por offset sobre ventana de inserción y de ciclo (Sec. 6.6)
    dur_vent_in  = float(row["t_in_2"] - row["t_in_1"])
    dur_vent_cic = float(row["t_ciclo_2"] - row["t_ciclo_1"])
    u_Iin_offset = float(dur_vent_in * u_b)
    u_C_offset   = float(dur_vent_cic * u_b)

    # -------------------------------------------------------------------------
    # 6.5 Sensibilidad al suavizado Savitzky-Golay (dos ventanas)
    # -------------------------------------------------------------------------
    w1_pts, w1_s = ajustar_ventana_impar(SG_VENTANAS_S[0], dt, SG_ORDEN, len(Vc))
    w2_pts, w2_s = ajustar_ventana_impar(SG_VENTANAS_S[1], dt, SG_ORDEN, len(Vc))

    Vc_sg1 = savgol_filter(Vc, window_length=w1_pts, polyorder=SG_ORDEN)
    Vc_sg2 = savgol_filter(Vc, window_length=w2_pts, polyorder=SG_ORDEN)

    dLambda_sg1 = -cumulative_trapezoid(Vc_sg1, t, initial=0.0)
    dLambda_sg2 = -cumulative_trapezoid(Vc_sg2, t, initial=0.0)

    met_sg1 = calcular_integrales_basicas(t, Vc_sg1, row)
    met_sg2 = calcular_integrales_basicas(t, Vc_sg2, row)

    dLambda_meseta_sg1 = float(np.mean(dLambda_sg1[m_pausa]) - np.mean(dLambda_sg1[m_rep_ini]))
    dLambda_meseta_sg2 = float(np.mean(dLambda_sg2[m_pausa]) - np.mean(dLambda_sg2[m_rep_ini]))

    # Variación porcentual de V_pk e I_in con SG respecto al dato sin filtrar
    dVpk_sg1_pct = 100.0 * (met_sg1["V_pk_in"] - met_crudo["V_pk_in"]) / met_crudo["V_pk_in"]
    dVpk_sg2_pct = 100.0 * (met_sg2["V_pk_in"] - met_crudo["V_pk_in"]) / met_crudo["V_pk_in"]
    dIin_sg1_pct = 100.0 * (met_sg1["I_in"] - met_crudo["I_in"]) / abs(met_crudo["I_in"])
    dIin_sg2_pct = 100.0 * (met_sg2["I_in"] - met_crudo["I_in"]) / abs(met_crudo["I_in"])

    res = {
        # Identificación
        "ensayo": row["ensayo"],
        "condicion": row["condicion"],
        "fs_Hz": round(1.0 / dt, 2),
        # Cuadro 1 (referencia: datos sin suavizar, línea de base constante)
        "V_pk_in_V": met_crudo["V_pk_in"],
        "V_pk_out_V": met_crudo["V_pk_out"],
        "T_in_s": T_in,
        "T_out_s": T_out,
        "I_in_Vs": met_crudo["I_in"],
        "I_out_Vs": met_crudo["I_out"],
        "C_Vs": met_crudo["C"],
        "eps_C": met_crudo["eps_C"],
        # Áreas positivas y negativas (Sec. 6.3)
        "A_pos_Vs": met_crudo["A_pos"],
        "A_neg_Vs": met_crudo["A_neg"],
        "check_A_minus_C_Vs": (met_crudo["A_pos"] + met_crudo["A_neg"]) - met_crudo["C"],
        # Mesetas y flujo por espira (Sec. 6.2)
        "dLambda_meseta_Wb": dLambda_meseta,
        "dLambda_pulso_in_Wb": -met_crudo["I_in"],
        "diff_meseta_pulso_Wb": dLambda_meseta - (-met_crudo["I_in"]),
        "dPhi_meseta_Wb": dPhi_meseta,
        "dPhi_pulso_in_Wb": -met_crudo["I_in"] / N_ESPIRAS,
        "meseta_fin_Wb": meseta_fin - meseta_ini,
        # Línea de base y ruido (Sec. 6.1)
        "b0_V": b0,
        "mu_rep_ini_V": mu_ini,
        "mu_rep_fin_V": mu_fin,
        "diff_rep_ext_V": diff_ext,
        "sigma_rep_V": sigma_rep,
        "media_pausa_Vc_V": media_pausa_vc,
        "std_pausa_Vc_V": std_pausa_vc,
        "pendiente_lin_V_s": float(c_lin),
        "u_b_V": u_b,
        "u_Iin_offset_Vs": u_Iin_offset,
        "u_C_offset_Vs": u_C_offset,
        # Sensibilidad: sin corrección / base lineal / 2.ª ventana de reposo
        "I_in_sin_corr_Vs": met_sin_cor["I_in"],
        "C_sin_corr_Vs": met_sin_cor["C"],
        "I_in_lin_Vs": met_lin["I_in"],
        "C_lin_Vs": met_lin["C"],
        "I_in_vent2_Vs": met_alt["I_in"],
        "C_vent2_Vs": met_alt["C"],
        # Sensibilidad Savitzky-Golay (Sec. 6.5)
        "sg1_pts": w1_pts,
        "sg1_s": round(w1_s, 5),
        "V_pk_sg1_V": met_sg1["V_pk_in"],
        "I_in_sg1_Vs": met_sg1["I_in"],
        "C_sg1_Vs": met_sg1["C"],
        "dLambda_meseta_sg1_Wb": dLambda_meseta_sg1,
        "dVpk_sg1_pct": dVpk_sg1_pct,
        "dIin_sg1_pct": dIin_sg1_pct,
        "sg2_pts": w2_pts,
        "sg2_s": round(w2_s, 5),
        "V_pk_sg2_V": met_sg2["V_pk_in"],
        "I_in_sg2_Vs": met_sg2["I_in"],
        "C_sg2_Vs": met_sg2["C"],
        "dLambda_meseta_sg2_Wb": dLambda_meseta_sg2,
        "dVpk_sg2_pct": dVpk_sg2_pct,
        "dIin_sg2_pct": dIin_sg2_pct,
    }

    serie = {
        "t": t,
        "t_rel": t - row["t_in_1"],
        "V": V,
        "b0": b0,
        "b_lin": b_lin,
        "Vc": Vc,
        "Vc_sg1": Vc_sg1,
        "Vc_sg2": Vc_sg2,
        "dLambda": dLambda,
        "dPhi": dPhi,
        "dLambda_sin_corr": dLambda_sin_corr,
        "dLambda_sg1": dLambda_sg1,
        "dLambda_sg2": dLambda_sg2,
        "meseta_ini": meseta_ini,
        "meseta_int": meseta_int,
        "t_in_u1": t_in_u1,
        "t_in_u2": t_in_u2,
        "umbral_dur": umbral_dur,
        "row": row,
        "w1_pts": w1_pts,
        "w1_s": w1_s,
        "w2_pts": w2_pts,
        "w2_s": w2_s,
    }
    return res, serie


# =============================================================================
# 4. EJECUCIÓN SOBRE LOS 15 CICLOS
# =============================================================================
if not ARCHIVO_VENTANAS.exists():
    raise FileNotFoundError(
        f"No se encontró {ARCHIVO_VENTANAS.name}. Ejecutá primero 01_inspeccion_y_parte_A.py."
    )

df_ventanas = pd.read_csv(ARCHIVO_VENTANAS)

# Cachear los CSV crudos para leer cada archivo una sola vez
cache_csv = {}
for nombre_csv in df_ventanas["archivo_csv"].unique():
    cache_csv[nombre_csv] = cargar_csv_pasco(CARPETA / nombre_csv)

resultados = []
series_guardadas = {}

for _, row in df_ventanas.iterrows():
    t_full, V_full = cache_csv[row["archivo_csv"]]
    res, serie = analizar_ciclo(t_full, V_full, row)
    resultados.append(res)
    series_guardadas[row["ensayo"]] = serie

df_res = pd.DataFrame(resultados)
df_res.to_csv(ARCHIVO_SALIDA_TABLA, index=False)


# =============================================================================
# 5. IMPRESIÓN EN CONSOLA (CUADRO 1 Y CHEQUEOS DE SENSIBILIDAD)
# =============================================================================
print("=" * 90)
print(f"PROCESAMIENTO COMPLETADO ({len(df_res)} ciclos) -> Guardado en: {ARCHIVO_SALIDA_TABLA.name}")
print(f"Criterio T_in: envolvente |Vc| > {K_UMBRAL_DURACION:.1f} * sigma_reposo (abarca todos los lóbulos)")
print(f"Savitzky-Golay: orden={SG_ORDEN}, ventanas nominales={SG_VENTANAS_S[0]*1e3:.0f} ms y {SG_VENTANAS_S[1]*1e3:.0f} ms")
print("=" * 90)

# Cuadro 1 compacto (unidades en mV / mWb para lectura rápida + V / Vs guardados en CSV)
cuadro1 = pd.DataFrame({
    "Ensayo":       df_res["ensayo"],
    "V_pk (V)":     df_res["V_pk_in_V"].map(lambda x: f"{x:.4f}"),
    "T_in (s)":     df_res["T_in_s"].map(lambda x: f"{x:.4f}"),
    "I_in (mVs)":   (df_res["I_in_Vs"] * 1e3).map(lambda x: f"{x:+.3f}"),
    "I_out (mVs)":  (df_res["I_out_Vs"] * 1e3).map(lambda x: f"{x:+.3f}"),
    "C (mVs)":      (df_res["C_Vs"] * 1e3).map(lambda x: f"{x:+.3f}"),
    "eps_C (%)":    (df_res["eps_C"] * 100).map(lambda x: f"{x:.2f}"),
    "dPhi (uWb)":   (df_res["dPhi_meseta_Wb"] * 1e6).map(lambda x: f"{x:+.2f}"),
})
print("\nCUADRO 1 - RESULTADOS INDIVIDUALES POR CICLO (SIN SUAVIZAR):")
print(cuadro1.to_string(index=False))

# Resumen rápido de sensibilidad al offset y a Savitzky-Golay
print("\nDIAGNÓSTICO DE LÍNEA DE BASE Y SENSIBILIDAD (valores máximos absolutos):")
max_diff_base = float(np.max(np.abs(df_res["I_in_lin_Vs"] - df_res["I_in_Vs"]))) * 1e3
max_diff_vent = float(np.max(np.abs(df_res["I_in_vent2_Vs"] - df_res["I_in_Vs"]))) * 1e3
max_dVpk_sg1  = float(np.max(np.abs(df_res["dVpk_sg1_pct"])))
max_dVpk_sg2  = float(np.max(np.abs(df_res["dVpk_sg2_pct"])))
max_dIin_sg1  = float(np.max(np.abs(df_res["dIin_sg1_pct"])))
max_dIin_sg2  = float(np.max(np.abs(df_res["dIin_sg2_pct"])))
print(f"  Máx. cambio |I_in| (base lineal vs constante) : {max_diff_base:.4f} mVs")
print(f"  Máx. cambio |I_in| (2.ª ventana de reposo)    : {max_diff_vent:.4f} mVs")
print(f"  SG1 ({df_res['sg1_pts'].iloc[0]} pts = {df_res['sg1_s'].iloc[0]*1e3:.1f} ms): "
      f"máx ΔV_pk = {max_dVpk_sg1:.2f}% | máx ΔI_in = {max_dIin_sg1:.2f}%")
print(f"  SG2 ({df_res['sg2_pts'].iloc[0]} pts = {df_res['sg2_s'].iloc[0]*1e3:.1f} ms): "
      f"máx ΔV_pk = {max_dVpk_sg2:.2f}% | máx ΔI_in = {max_dIin_sg2:.2f}%")
print("=" * 90)


# =============================================================================
# 6. FIGURA 1: CICLO COMPLETO (V_med, LÍNEA DE BASE, V_c, ΔΛ Y ΔΦ̄) (Sec. 6.1, 6.2, 7)
# =============================================================================
if ENSAYO_EJEMPLO_CICLO not in series_guardadas:
    ENSAYO_EJEMPLO_CICLO = df_res["ensayo"].iloc[0]

s = series_guardadas[ENSAYO_EJEMPLO_CICLO]
r = s["row"]

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)

# Panel superior: tensión original, línea de base y tensión corregida
ax1.plot(s["t"], s["V"] * 1e3, color="lightgray", lw=1.0, label=r"$V_{med}(t)$ original")
ax1.plot(s["t"], s["Vc"] * 1e3, color="tab:blue", lw=1.1, label=r"$V_c(t) = V_{med} - \hat{b}_0$")
ax1.axhline(s["b0"] * 1e3, color="tab:red", ls="--", lw=1.2,
            label=rf"Base constante $\hat{{b}}_0 = {s['b0']*1e3:+.2f}$ mV")
ax1.plot(s["t"], s["b_lin"] * 1e3, color="tab:orange", ls=":", lw=1.2, label=r"Base lineal $\tilde{b}(t)$")

# Sombrear ventanas de reposo y pausa interior
ax1.axvspan(r["t_rep_ini_1"], r["t_rep_ini_2"], color="tab:green", alpha=0.10, label="Reposo exterior")
ax1.axvspan(r["t_rep_fin_1"], r["t_rep_fin_2"], color="tab:green", alpha=0.10)
ax1.axvspan(r["t_pausa_1"],   r["t_pausa_2"],   color="gray",      alpha=0.16, label="Pausa interior")

ax1.set_ylabel("Tensión (mV)")
ax1.set_title(f"Ciclo completo {ENSAYO_EJEMPLO_CICLO}: corrección de línea de base e integración acumulativa")
ax1.grid(True, alpha=0.35)
ax1.legend(loc="upper right", fontsize=8.5, ncol=2)

# Panel inferior: ΔΛ(t) en eje izquierdo y ΔΦ̄(t) = ΔΛ/N en eje derecho
ax2.plot(s["t"], s["dLambda"] * 1e3, color="tab:purple", lw=1.6,
         label=r"$\Delta\Lambda(t)$ corregida (sin suavizar)")
ax2.plot(s["t"], s["dLambda_sin_corr"] * 1e3, color="tab:gray", ls="--", lw=1.1, alpha=0.8,
         label=r"$\Delta\Lambda(t)$ sin corregir base")

# Marcar niveles de meseta inicial e interior
ax2.hlines(s["meseta_ini"] * 1e3, r["t_rep_ini_1"], r["t_rep_ini_2"],
           colors="tab:green", lw=2.2, label="Meseta inicial")
ax2.hlines(s["meseta_int"] * 1e3, r["t_pausa_1"], r["t_pausa_2"],
           colors="tab:red", lw=2.2,
           label=rf"Meseta interior ($\Delta\Lambda = {(s['meseta_int']-s['meseta_ini'])*1e3:+.2f}$ mWb)")

ax2.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel(r"Cambio de flujo concatenado $\Delta\Lambda$ (mWb)")
ax2.grid(True, alpha=0.35)
ax2.legend(loc="lower right", fontsize=8.5)

# Eje secundario para flujo medio por espira ΔΦ̄ = ΔΛ / 300 (en µWb)
ax2_der = ax2.twinx()
y1, y2 = ax2.get_ylim()
ax2_der.set_ylim((y1 * 1e-3 / N_ESPIRAS) * 1e6, (y2 * 1e-3 / N_ESPIRAS) * 1e6)
ax2_der.set_ylabel(rf"Flujo medio por espira $\Delta\overline{{\Phi}} = \Delta\Lambda/{N_ESPIRAS}$ ($\mu$Wb)")

fig1.tight_layout()
if GUARDAR_FIGURAS:
    fig1.savefig(CARPETA / "fig_02_ciclo_ejemplo.png", dpi=300)


# =============================================================================
# 7. FIGURA 2: COMPARACIÓN DE UN CICLO DE CADA CONDICIÓN (L, M, R) (Sec. 7)
# =============================================================================
fig2, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)

colores_lmr = {"L": "tab:green", "M": "tab:orange", "R": "tab:purple"}
etiquetas_lmr = {"L": "Lento", "M": "Intermedio", "R": "Rápido"}

for ensayo_id in ENSAYOS_COMP_LMR:
    if ensayo_id not in series_guardadas:
        continue
    sc = series_guardadas[ensayo_id]
    cond = sc["row"]["condicion"]
    col = colores_lmr[cond]
    etq = f"{ensayo_id} ({etiquetas_lmr[cond]})"
    # Alineamos t = 0 al inicio de la ventana de inserción de cada ciclo
    ax_v.plot(sc["t_rel"], sc["Vc"] * 1e3, color=col, lw=1.2, label=etq)
    ax_f.plot(sc["t_rel"], sc["dLambda"] * 1e3, color=col, lw=1.5, label=etq)

ax_v.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
ax_v.set_ylabel(r"Tensión corregida $V_c$ (mV)")
ax_v.set_title("Comparación de un ciclo representativo por condición (L, M y R)")
ax_v.grid(True, alpha=0.35)
ax_v.legend(loc="upper right")

ax_f.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
ax_f.set_xlabel("Tiempo relativo al inicio de la inserción (s)")
ax_f.set_ylabel(r"$\Delta\Lambda$ (mWb)")
ax_f.grid(True, alpha=0.35)
ax_f.legend(loc="lower right")

ax_f_der = ax_f.twinx()
fy1, fy2 = ax_f.get_ylim()
ax_f_der.set_ylim((fy1 * 1e-3 / N_ESPIRAS) * 1e6, (fy2 * 1e-3 / N_ESPIRAS) * 1e6)
ax_f_der.set_ylabel(r"$\Delta\overline{\Phi}$ ($\mu$Wb)")

fig2.tight_layout()
if GUARDAR_FIGURAS:
    fig2.savefig(CARPETA / "fig_02_comparacion_LMR_ciclos.png", dpi=300)


# =============================================================================
# 8. FIGURA 3: SENSIBILIDAD AL SUAVIZADO SAVITZKY-GOLAY (Sec. 6.5 y 7)
# =============================================================================
if ENSAYO_EJEMPLO_SG not in series_guardadas:
    ENSAYO_EJEMPLO_SG = df_res["ensayo"].iloc[-1]

sg = series_guardadas[ENSAYO_EJEMPLO_SG]
rsg = sg["row"]

# Zoom sobre el pulso de inserción (donde el pico rápido es más sensible al filtrado)
m_zoom = mascara_rango(sg["t"], rsg["t_in_1"] - 0.03, rsg["t_in_2"] + 0.03)

fig3, (ax_sg1, ax_sg2) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=False)

ax_sg1.plot(sg["t"][m_zoom], sg["Vc"][m_zoom] * 1e3, color="black", lw=1.1, alpha=0.75,
            label="Sin suavizar (referencia)")
ax_sg1.plot(sg["t"][m_zoom], sg["Vc_sg1"][m_zoom] * 1e3, color="tab:blue", lw=1.4,
            label=f"SG1: {sg['w1_pts']} pts ({sg['w1_s']*1e3:.1f} ms), orden {SG_ORDEN}")
ax_sg1.plot(sg["t"][m_zoom], sg["Vc_sg2"][m_zoom] * 1e3, color="tab:red", ls="--", lw=1.4,
            label=f"SG2: {sg['w2_pts']} pts ({sg['w2_s']*1e3:.1f} ms), orden {SG_ORDEN}")
ax_sg1.axhline(0, color="gray", lw=0.7, ls=":")
ax_sg1.set_title(f"Sensibilidad al suavizado Savitzky-Golay en {ENSAYO_EJEMPLO_SG} (zoom pulso de inserción)")
ax_sg1.set_xlabel("Tiempo (s)")
ax_sg1.set_ylabel(r"$V_c$ (mV)")
ax_sg1.grid(True, alpha=0.35)
ax_sg1.legend(loc="upper right", fontsize=8.5)

# Panel inferior: integral acumulativa del ciclo completo con y sin suavizado
ax_sg2.plot(sg["t"], sg["dLambda"] * 1e3, color="black", lw=1.6, label="Sin suavizar (referencia)")
ax_sg2.plot(sg["t"], sg["dLambda_sg1"] * 1e3, color="tab:blue", ls="--", lw=1.3,
            label=f"SG1 ({sg['w1_s']*1e3:.1f} ms)")
ax_sg2.plot(sg["t"], sg["dLambda_sg2"] * 1e3, color="tab:red", ls=":", lw=1.5,
            label=f"SG2 ({sg['w2_s']*1e3:.1f} ms)")
ax_sg2.axhline(0, color="gray", lw=0.7, ls=":")
ax_sg2.set_title(f"Integral acumulativa del ciclo completo {ENSAYO_EJEMPLO_SG}: crudo vs. Savitzky-Golay")
ax_sg2.set_xlabel("Tiempo (s)")
ax_sg2.set_ylabel(r"$\Delta\Lambda$ (mWb)")
ax_sg2.grid(True, alpha=0.35)
ax_sg2.legend(loc="lower right", fontsize=8.5)

fig3.tight_layout()
if GUARDAR_FIGURAS:
    fig3.savefig(CARPETA / "fig_02_suavizado_sg.png", dpi=300)

plt.show()