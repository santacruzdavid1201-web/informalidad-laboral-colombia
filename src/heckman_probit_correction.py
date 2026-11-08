"""
GEIH 2022–2025 — Base única filtrable por año
Trimestre 2 (Abril, Mayo, Junio)
Merge por ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'] dentro de cada período

Versión v17 (corrección de la muestra de la 1ª etapa y de la definición de ocupado):
  - CLASE, DPTO y FEX_C18 se leen del módulo cgsse (todas las personas) y no
    del módulo de ocupados. Antes, CLASE quedaba en NaN para los no ocupados,
    ZONA_RURAL también, y el dropna() de la 1ª etapa eliminaba a casi todos
    los no ocupados: el probit de selección se estimaba sobre ~366 mil
    personas (casi todas ocupadas) en lugar de la PET (~680 mil).
  - Ocupado = persona presente en el módulo "Ocupados" del DANE (bandera
    OCUPADO), en lugar de P6240 == 1. P6240 es la actividad principal de la
    semana pasada y deja por fuera a quienes trabajaron al menos una hora
    pero reportaron estudiar u oficios del hogar como actividad principal,
    a los ausentes temporales y a los trabajadores familiares sin
    remuneración (~40 mil ocupados en 2022–2025).
  - Se corrige la nota metodológica sobre el IMR: n(1ª etapa) y n(2ª etapa)
    son del mismo orden de magnitud, por lo que no se puede argumentar que
    la incertidumbre del regresor generado sea despreciable.

Versión v16 (SE robustos por clúster de hogar, sin Murphy-Topel):
  - Se retira la corrección analítica de Murphy-Topel: las implementaciones
    probadas (con y sin reestimación de un paso) no producían derivadas
    cruzadas confiables (C salía en cero o inestable).
  - Se reemplaza por errores estándar robustos agrupados (cluster-robust)
    por hogar (DIRECTORIO) en TODOS los fits de la 2ª etapa del Heckman
    (modelo principal + las 3 especificaciones de robustez). Esto corrige
    la correlación intra-hogar, que es un problema real y bien
    identificado en esta muestra.
  - Limitación documentada: los SE clusterizados no incorporan la
    incertidumbre de estimación del IMR generado en la 1ª etapa. Dado que
    n de la 1ª etapa (PET, ~1.13M) es varios órdenes de magnitud mayor
    que n de la 2ª etapa (ocupados), esta fuente de sesgo es
    asintóticamente pequeña (Heckman, 1979; Murphy & Topel, 1985), por lo
    que se reporta como limitación metodológica en vez de intentar una
    corrección analítica adicional que no se pudo validar con el tiempo
    disponible.
  - Se elimina el bootstrap (v14, fallaba con error de threading) y la
    corrección Murphy-Topel (v15, C se anulaba o resultaba inestable).

[El resto del docstring de versiones anteriores se conserva igual]
"""

import os, glob, warnings, sys, datetime, time
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


class Tee:
    """
    Duplica todo lo que se escribe en stdout (cualquier print() del script,
    sin importar en qué función esté) hacia la consola Y hacia un archivo
    de log en disco, en tiempo real.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            if hasattr(s, "encoding") and s.encoding:
                try:
                    data_segura = data.encode(s.encoding, errors="replace").decode(s.encoding)
                except Exception:
                    data_segura = data
                s.write(data_segura)
            else:
                s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR    = os.path.join(_REPO_ROOT, "data", "raw", "MICRO")
OUTPUT_DIR  = os.path.join(_REPO_ROOT, "figures", "micro")
AÑOS     = [2022, 2023, 2024, 2025]
MESES    = ["Abril", "Mayo", "Junio"]
KEY      = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]      # llave a nivel de persona
KEY_HOGAR = ["DIRECTORIO", "SECUENCIA_P"]              # llave a nivel de hogar

# ─── MÓDULOS ─────────────────────────────────────────────────────────────────

MODULOS = {
    "cgsse":          ["caracteristicas generales", "seguridad social"],
    "datos_hogar":    ["datos del hogar", "hogar y la vivienda", "hogar_vivienda"],
    "fuerza_trabajo": ["fuerza de trabajo"],
    "migracion":      ["migracion", "migración"],
    "no_ocupados":    ["no ocupados"],
    "ocupados":       ["ocupados"],
    "otras_formas":   ["otras formas de trabajo"],
    "otros_ingresos": ["otros ingresos e impuestos", "otros ingresos"],
}

MODULOS_NIVEL_HOGAR = {"datos_hogar"}

# ─── VARIABLES A RETENER POR MÓDULO ──────────────────────────────────────────

VARS = {
    "cgsse": [
        "P3271", "P3039", "P3038", "P6080", "P2057", "P3042", "P6040",
        "P6090", "P6083", "P6100", "P6110", "P6050", "P6070",
        "CLASE", "DPTO", "FEX_C18",   # v17: variables de diseño para TODAS las personas
    ],
    "datos_hogar": [
        "P6008",
    ],
    "fuerza_trabajo": [
        "P6240",
    ],
    "migracion": [
        "P3384", "P3373", "P3373S3", "P3383",
    ],
    "no_ocupados":    [],
    "ocupados": [
        "P6430", "P3069", "P3045S1", "P3045S2", "P3045S3", "P3046",
        "P6775", "P6765", "P3065", "P3066", "P3067", "P3067S1", "P3067S2",
        "P3068", "P6450", "P6920", "P6930", "P6940", "P6990", "OFICIO_C8",
        "RAMA2D_R4", "P7130", "INGLABO", "P6800", "P1801S3",
        "PER",
    ],
    "otras_formas": [
        "P3079S1", "P3081S1",
    ],
    "otros_ingresos": [
        "P750S2", "P7500S2", "P7495", "P7505",
    ],
}

# ─── UTILIDADES ──────────────────────────────────────────────────────────────

def normalizar(texto):
    t = texto.lower()
    for a, b in zip("áéíóúñ", "aeioun"):
        t = t.replace(a, b)
    return t

def identificar_modulo(nombre_archivo):
    n = normalizar(nombre_archivo)
    for modulo, fragmentos in MODULOS.items():
        if any(normalizar(f) in n for f in fragmentos):
            return modulo
    return None

def leer_csv(path, vars_req):
    key_upper = [k.upper() for k in KEY]
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                df.columns = [c.strip().upper() for c in df.columns]
                if len(df.columns) <= 1:
                    continue
                if not all(k in df.columns for k in key_upper):
                    continue
                vars_extra = [v.upper() for v in vars_req if v.upper() in df.columns]
                return df[key_upper + vars_extra].copy()
            except Exception:
                continue
    return None

def leer_csv_hogar(path, vars_req):
    key_hogar = [k.upper() for k in KEY_HOGAR]
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                df.columns = [c.strip().upper() for c in df.columns]
                if len(df.columns) <= 1:
                    continue
                if not all(k in df.columns for k in key_hogar):
                    continue
                vars_extra = [v.upper() for v in vars_req if v.upper() in df.columns]
                df = df[key_hogar + vars_extra].drop_duplicates(subset=key_hogar)
                return df.copy()
            except Exception:
                continue
    return None

# ─── LECTURA POR PERÍODO ─────────────────────────────────────────────────────

def leer_periodo(año, mes):
    carpeta = os.path.join(BASE_DIR, str(año), mes)
    if not os.path.isdir(carpeta):
        print(f"  ✗  Carpeta no encontrada: {carpeta}")
        return None

    modulo_paths = {}
    csvs = glob.glob(os.path.join(carpeta, "*.csv")) + \
           glob.glob(os.path.join(carpeta, "*.CSV"))
    for path in csvs:
        modulo = identificar_modulo(os.path.basename(path))
        if modulo:
            modulo_paths[modulo] = path
        else:
            print(f"  ?  No identificado: {os.path.basename(path)}")

    df_periodo = None
    key_upper  = [k.upper() for k in KEY]
    key_hogar  = [k.upper() for k in KEY_HOGAR]

    for modulo, vars_req in VARS.items():
        if modulo in MODULOS_NIVEL_HOGAR:
            continue
        if modulo not in modulo_paths:
            print(f"  ✗  Módulo ausente: {modulo} ({año}/{mes})")
            continue
        df = leer_csv(modulo_paths[modulo], vars_req)
        if df is None:
            print(f"  ✗  No se pudo leer: {modulo_paths[modulo]}")
            continue
        print(f"  ✓  {modulo:<20} {len(df):>7,} filas")
        if modulo == "ocupados":
            # v17: toda persona del módulo Ocupados es ocupada según el DANE
            df["OCUPADO"] = 1
        if df_periodo is None:
            df_periodo = df
        else:
            df_periodo = pd.merge(df_periodo, df, on=key_upper, how="left")

    for modulo in MODULOS_NIVEL_HOGAR:
        vars_req = VARS.get(modulo, [])
        if modulo not in modulo_paths:
            print(f"  ✗  Módulo ausente: {modulo} ({año}/{mes})")
            continue
        df_hogar = leer_csv_hogar(modulo_paths[modulo], vars_req)
        if df_hogar is None:
            print(f"  ✗  No se pudo leer: {modulo_paths[modulo]}")
            continue
        print(f"  ✓  {modulo:<20} {len(df_hogar):>7,} filas (nivel hogar)")
        if df_periodo is None:
            df_periodo = df_hogar
        else:
            df_periodo = pd.merge(df_periodo, df_hogar, on=key_hogar, how="left")

    if df_periodo is None:
        return None

    if "OCUPADO" not in df_periodo.columns:
        raise KeyError(f"No se leyó el módulo de ocupados en {año}/{mes}; "
                       f"no se puede construir la bandera OCUPADO.")
    df_periodo["OCUPADO"] = df_periodo["OCUPADO"].fillna(0).astype(int)

    df_periodo["AÑO"] = año
    df_periodo["MES"] = mes
    return df_periodo

# ─── VERIFICACIÓN DE INTEGRIDAD DE VARIABLES ─────────────────────────────────

def verificar_variables_esperadas(geih):
    print(f"\n── Verificación de variables esperadas ──")
    faltantes = []
    for modulo, vars_req in VARS.items():
        for v in vars_req:
            if v.upper() not in geih.columns:
                faltantes.append((modulo, v))

    if not faltantes:
        print("   ✓  Todas las variables declaradas en VARS están presentes.")
    else:
        print(f"   ⚠ {len(faltantes)} variable(s) declaradas en VARS NO llegaron al DataFrame final:")
        for modulo, v in faltantes:
            print(f"      - {v}  (módulo: {modulo})")
        print("   Revisa el nombre de archivo / módulo correspondiente en MODULOS,")
        print("   o si el módulo es a nivel de hogar y falta agregarlo a MODULOS_NIVEL_HOGAR.")

# ─── CONSTRUCCIÓN DE LA BASE ──────────────────────────────────────────────────

def construir_base():
    periodos = []
    for año in AÑOS:
        print(f"\n── {año} ──────────────────────────────────────────")
        for mes in MESES:
            print(f"  {mes}")
            df = leer_periodo(año, mes)
            if df is not None:
                periodos.append(df)
                print(f"     → {len(df):,} observaciones")

    if not periodos:
        raise ValueError("No se cargó ningún período. Revisa rutas y nombres de archivo.")

    geih = pd.concat(periodos, ignore_index=True)

    if "INGLABO" in geih.columns:
        ing = pd.to_numeric(geih["INGLABO"], errors="coerce")
        geih["LN_INGLABO"] = np.where(ing > 0, np.log(ing), np.nan)

    if "P6040" in geih.columns:
        geih["EDAD"] = pd.to_numeric(geih["P6040"], errors="coerce")

    print(f"\n{'─'*50}")
    print(f"DataFrame final: {len(geih):,} observaciones × {len(geih.columns)} variables")
    print(f"Años disponibles: {sorted(geih['AÑO'].unique())}")

    verificar_variables_esperadas(geih)

    # v17: chequeos de consistencia de la muestra
    pet_n  = int((geih["EDAD"] >= 15).sum()) if "EDAD" in geih.columns else None
    ocu_n  = int(geih["OCUPADO"].sum())
    p6240  = int((geih["P6240"] == 1).sum())
    print(f"\n── Chequeo de muestra (v17) ──")
    print(f"   PET (EDAD >= 15):                 {pet_n:,}")
    print(f"   Ocupados (módulo DANE):           {ocu_n:,}")
    print(f"   P6240 == 1 (definición anterior): {p6240:,}")
    print(f"   Missing en CLASE:                 {geih['CLASE'].isna().sum():,}")
    n_mes_hogar = geih.groupby([k.upper() for k in KEY_HOGAR])["MES"].nunique().max()
    if n_mes_hogar > 1:
        print(f"   ⚠ Hay hogares (DIRECTORIO, SECUENCIA_P) repetidos en más de un mes; "
              f"las variables por hogar y el clúster deberían incluir AÑO y MES en la llave.")

    return geih

# ─── VARIABLE DEPENDIENTE (metodología DANE) ─────────────────────────────────

def construir_formal(df):
    df = df.copy()

    df["OFICIO_C8_2D"] = pd.to_numeric(
        df["OFICIO_C8"].astype(str).str[:2], errors="coerce"
    )
    if "PER" in df.columns:
        df["ANIOS"] = pd.to_numeric(df["PER"], errors="coerce") - 1

    df["FORMAL"] = np.nan
    df.loc[df["P6430"] == 3, "FORMAL"] = np.nan
    df.loc[df["P6430"] == 6, "FORMAL"] = 0
    df.loc[df["RAMA2D_R4"].isin(["84", "99"]), "FORMAL"] = 1
    df.loc[df["P6430"] == 8, "FORMAL"] = 0
    df.loc[df["P6430"] == 2, "FORMAL"] = 1

    m = df["P6430"].isin([1, 7])
    df.loc[m & (df["P3045S1"] == 1), "FORMAL"] = 1
    m2 = m & (df["P3045S1"].isin([2, 9]))
    df.loc[m2 & (df["P3046"] == 1), "FORMAL"] = 1
    df.loc[m2 & (df["P3046"] == 2), "FORMAL"] = 0
    m3 = m2 & (df["P3046"] == 9)
    df.loc[m3 & (df["P3069"] >= 4), "FORMAL"] = 1
    df.loc[m3 & (df["P3069"] <= 3), "FORMAL"] = 0

    ind     = df["P6430"].isin([4, 5])
    sin_neg = ind & (~df["P6765"].isin([7]) | df["P6765"].isna())
    df.loc[sin_neg & (df["P3065"] == 1), "FORMAL"] = 1
    df.loc[sin_neg & (df["P3065"].isin([2, 9])) & (df["P3066"] == 1), "FORMAL"] = 1
    df.loc[sin_neg & (df["P3065"].isin([2, 9])) & (df["P3066"] == 2), "FORMAL"] = 0
    df.loc[sin_neg & (df["P6430"] == 5) & (df["P3065"].isin([2, 9])) & (df["P3066"] == 9) & (df["P3069"] >= 4), "FORMAL"] = 1
    df.loc[sin_neg & (df["P6430"] == 5) & (df["P3065"].isin([2, 9])) & (df["P3066"] == 9) & (df["P3069"] <= 3), "FORMAL"] = 0
    df.loc[sin_neg & (df["P6430"] == 4) & (df["P3065"].isin([2, 9])) & (df["P3066"] == 9) & (df["OFICIO_C8_2D"] <= 20), "FORMAL"] = 1
    df.loc[sin_neg & (df["P6430"] == 4) & (df["P3065"].isin([2, 9])) & (df["P3066"] == 9) & (df["OFICIO_C8_2D"] >= 21), "FORMAL"] = 0

    con_neg = ind & (df["P6765"] == 7)
    rm = con_neg & (df["P3067"] == 1) & (df["P3067S1"] == 1)
    if "ANIOS" in df.columns:
        df.loc[rm & (df["P3067S2"] >= df["ANIOS"]), "FORMAL"] = 1
        df.loc[rm & (df["P3067S2"] < df["ANIOS"]),  "FORMAL"] = 0

    rm2 = con_neg & (df["P3067"] == 1) & (df["P3067S1"] == 2)
    df.loc[rm2 & (df["P6775"] == 1), "FORMAL"] = 1
    df.loc[rm2 & (df["P6775"] == 3) & (df["OFICIO_C8_2D"] <= 20), "FORMAL"] = 1
    df.loc[rm2 & (df["P6775"] == 3) & (df["OFICIO_C8_2D"] >= 21), "FORMAL"] = 0
    df.loc[rm2 & (df["P6775"] == 2), "FORMAL"] = 0
    df.loc[rm2 & (df["P6430"] == 4) & (df["P6775"] == 9) & (df["OFICIO_C8_2D"] <= 20), "FORMAL"] = 1
    df.loc[rm2 & (df["P6430"] == 4) & (df["P6775"] == 9) & (df["OFICIO_C8_2D"] >= 21), "FORMAL"] = 0
    df.loc[rm2 & (df["P6430"] == 5) & (df["P6775"] == 9) & (df["P3069"] >= 4), "FORMAL"] = 1
    df.loc[rm2 & (df["P6430"] == 5) & (df["P6775"] == 9) & (df["P3069"] <= 3), "FORMAL"] = 0

    sin_rm = con_neg & (df["P3067"] == 2)
    df.loc[sin_rm & (df["P6775"] == 1) & (df["P3068"] == 1), "FORMAL"] = 1
    df.loc[sin_rm & (df["P6775"] == 1) & (df["P3068"] == 2), "FORMAL"] = 0
    df.loc[sin_rm & (df["P6775"] == 3) & (df["OFICIO_C8_2D"] <= 20), "FORMAL"] = 1
    df.loc[sin_rm & (df["P6775"] == 3) & (df["OFICIO_C8_2D"] >= 21), "FORMAL"] = 0
    df.loc[sin_rm & (df["P6775"] == 1) & (df["P3068"].isin([9, 3])), "FORMAL"] = 0
    df.loc[sin_rm & (df["P6775"] == 2), "FORMAL"] = 0
    df.loc[sin_rm & (df["P6430"] == 5) & (df["P6775"] == 9) & (df["P3069"] >= 4), "FORMAL"] = 1
    df.loc[sin_rm & (df["P6430"] == 5) & (df["P6775"] == 9) & (df["P3069"] <= 3), "FORMAL"] = 0
    df.loc[sin_rm & (df["P6430"] == 4) & (df["P6775"] == 9) & (df["OFICIO_C8_2D"] <= 20), "FORMAL"] = 1
    df.loc[sin_rm & (df["P6430"] == 4) & (df["P6775"] == 9) & (df["OFICIO_C8_2D"] >= 21), "FORMAL"] = 0

    df["SALUD"] = 0
    asal = df["P6430"].isin([1, 3, 7])
    df.loc[asal & (df["P6100"].isin([1, 2])) & (df["P6110"].isin([1, 2, 4])), "SALUD"] = 1
    df.loc[asal & (df["P6100"] == 9) & (df["P6450"] == 2), "SALUD"] = 1
    df.loc[asal & (df["P6110"] == 9) & (df["P6450"] == 2), "SALUD"] = 1

    df["PENSION"] = 0
    df.loc[asal & (df["P6920"] == 3), "PENSION"] = 1
    df.loc[asal & (df["P6920"] == 1) & (df["P6930"].isin([1, 2, 3])) & (df["P6940"].isin([1, 3])), "PENSION"] = 1

    df["FORMAL"] = 0
    df.loc[df["P6430"] == 2, "FORMAL"] = 1
    df.loc[df["P6430"].isin([1, 2, 3, 7]) & (df["SALUD"] == 1) & (df["PENSION"] == 1), "FORMAL"] = 1
    df.loc[(df["P6430"] == 4) & (df["RAMA2D_R4"].isin(["84", "99"])), "FORMAL"] = 1

    df["INFORMAL"] = 1 - df["FORMAL"]
    return df

# ─── DETERMINANTES ESTRUCTURALES DEL PUESTO DE TRABAJO Y DEL HOGAR ──────────

def construir_informalidad_rama(df):
    df = df.copy()
    RAMAS_FORMAL_REGLA = ["84", "99"]

    base_validos = df[~df["RAMA2D_R4"].isin(RAMAS_FORMAL_REGLA)].dropna(subset=["RAMA2D_R4", "INFORMAL"])

    suma_rama = base_validos.groupby("RAMA2D_R4")["INFORMAL"].sum()
    n_rama    = base_validos.groupby("RAMA2D_R4")["INFORMAL"].count()
    tasa_nacional = base_validos["INFORMAL"].mean()

    def tasa_loo(row):
        rama = row["RAMA2D_R4"]
        if pd.isna(rama) or rama in RAMAS_FORMAL_REGLA or rama not in n_rama.index:
            return tasa_nacional
        n = n_rama.loc[rama]
        if n <= 1:
            return tasa_nacional
        return (suma_rama.loc[rama] - row["INFORMAL"]) / (n - 1)

    df["INFORMALIDAD_RAMA"] = df.apply(tasa_loo, axis=1)
    return df


def construir_ocupacion_posicion_zona_migracion(df):
    df = df.copy()

    oficio_1d = pd.to_numeric(df["OFICIO_C8"].astype(str).str[:1], errors="coerce")
    df["OCUPACION_CALIF"] = oficio_1d.isin([1, 2, 3]).astype(int)
    df.loc[oficio_1d.isna(), "OCUPACION_CALIF"] = np.nan

    mapa_posicion = {
        2: "POSICION_GOBIERNO",
        3: "POSICION_DOMESTICO",
        4: "POSICION_CUENTA_PROPIA",
        5: "POSICION_EMPLEADOR",
        6: "POSICION_FAMILIAR_SR",
        7: "POSICION_JORNAL",
        8: "POSICION_OTRO_SR",
    }
    for cod, nombre in mapa_posicion.items():
        df[nombre] = (df["P6430"] == cod).astype(int)
        df.loc[df["P6430"].isna(), nombre] = np.nan

    df["ZONA_RURAL"] = df["CLASE"].map({1: 0, 2: 1})

    df["MIGRANTE_RECIENTE"] = df["P3384"].map({1: np.nan, 2: 0, 3: 1, 4: 1})

    return df


def construir_tasa_dependencia(df):
    df = df.copy()

    if "P6008" not in df.columns:
        raise KeyError(
            "P6008 no está en el DataFrame. Verifica que el módulo "
            "'datos_hogar' se haya leído correctamente (revisa la salida "
            "de 'verificar_variables_esperadas' impresa por construir_base) "
            "y que esté incluido en MODULOS_NIVEL_HOGAR."
        )

    hogar_key = [k.upper() for k in KEY_HOGAR]

    df["_OCUPADO_TMP"] = df["OCUPADO"].astype(int)
    ocupados_hogar = df.groupby(hogar_key)["_OCUPADO_TMP"].transform("sum")
    personas_hogar = pd.to_numeric(df["P6008"], errors="coerce")

    df["TASA_DEPENDENCIA"] = np.where(
        personas_hogar > 0,
        ocupados_hogar / personas_hogar,
        np.nan
    )
    df.drop(columns=["_OCUPADO_TMP"], inplace=True)
    return df

def construir_num_menores_5(df):
    df = df.copy()
    hogar_key = [k.upper() for k in KEY_HOGAR]
    df["_ES_MENOR_5"] = (pd.to_numeric(df["EDAD"], errors="coerce") < 5).astype(int)
    df["NUM_MENORES_5"] = df.groupby(hogar_key)["_ES_MENOR_5"].transform("sum")
    df.drop(columns=["_ES_MENOR_5"], inplace=True)
    return df

def construir_tiene_ing_no_lab(df):
    df = df.copy()
    hogar_key = [k.upper() for k in KEY_HOGAR]
    p7495 = pd.to_numeric(df["P7495"], errors="coerce")
    p7505 = pd.to_numeric(df["P7505"], errors="coerce")
    df["_TIENE_ING_PERS"] = ((p7495 == 1) | (p7505 == 1)).astype(int)
    df["TIENE_ING_NO_LAB"] = df.groupby(hogar_key)["_TIENE_ING_PERS"].transform("max")
    df.drop(columns=["_TIENE_ING_PERS"], inplace=True)
    return df

# ─── RECODIFICACIONES ────────────────────────────────────────────────────────

def recodificar(df):
    df = df.copy()

    df["MUJER"]        = (df["P3271"] == 2).astype(int)

    pension_val = pd.to_numeric(df["P7500S2"], errors="coerce").fillna(0)
    df["RECIBE_PENSION_HOGAR"] = (pension_val == 1).astype(int)
    df["TRANS"] = (
        ((df["P3271"] == 1) & (df["P3039"] == 2)) |
        ((df["P3271"] == 2) & (df["P3039"] == 1)) |
        df["P3039"].isin([3, 4])
    ).astype(int)

    hetero_hombre      = (df["P3271"] == 1) & (df["P3038"] == 2)
    hetero_mujer       = (df["P3271"] == 2) & (df["P3038"] == 1)
    df["NO_HETERO"]    = (~(hetero_hombre | hetero_mujer) & df["P3038"].notna()).astype(int)

    df["ETNIA"]        = df["P6080"].isin([1, 2, 3, 4, 5]).astype(int)
    df["CAMPESINO"]    = df["P2057"].map({1: 1, 2: 0})
    df["SUBSIDIO_GOB"] = (df["P750S2"] == 1).astype(int)
    df["DESEA_CAMBIO"] = df["P7130"].map({1: 1, 2: 0})

    edu = pd.to_numeric(df["P3042"], errors="coerce")
    df["EDUCACION_CAT"] = None
    df.loc[edu.isin([1, 2, 3]), "EDUCACION_CAT"] = "1_Primaria_Menos"
    df.loc[edu.isin([4, 5]), "EDUCACION_CAT"] = "2_Secundaria"
    df.loc[edu >= 6, "EDUCACION_CAT"] = "3_Superior"

    horas = pd.to_numeric(df["P6800"], errors="coerce")
    df["HORAS_CAT"] = None
    df.loc[horas < 35, "HORAS_CAT"] = "1_Parcial_Menos35"
    df.loc[horas.between(35, 48), "HORAS_CAT"] = "2_Completa_35a48"
    df.loc[horas > 48, "HORAS_CAT"] = "3_Sobretrabajo_Mas48"

    df["CUIDADO_NINOS"] = df["P3079S1"].map({1: 1, 2: 0})
    df["MUJER_CUIDADO"] = df["MUJER"] * df["CUIDADO_NINOS"].fillna(0)
    df["CUIDADO_DEPENDIENTES"] = df["P3081S1"].map({1: 1, 2: 0})

    df["EDAD_C"]  = df["EDAD"] - df["EDAD"].mean()
    df["EDAD2_C"] = df["EDAD_C"] ** 2

    df["EN_PAREJA"] = df["P6070"].isin([1, 2, 3]).astype(int)

    df["NACIDO_EXTRANJERO"] = (df["P3373"] == 3).astype(int)

    rama = pd.to_numeric(df["RAMA2D_R4"], errors="coerce")
    conditions = [
        rama.between(1, 3),   rama.between(5, 9),   rama.between(10, 33),
        rama == 35,           rama.between(36, 39), rama.between(41, 43),
        rama.between(45, 47), rama.between(49, 53), rama.between(55, 56),
        rama.between(58, 63), rama.between(64, 66), rama == 68,
        rama.between(69, 75), rama.between(77, 82), rama == 85,
        rama.between(86, 88), rama.between(90, 93), rama.between(94, 96),
        rama.between(97, 98)
    ]
    labels = ["A_Agro","B_Minas","C_Manuf","D_Energ","E_Agua","F_Constr",
              "G_Comerc","H_Transp","I_Aloj","J_InfoCom","K_Finan","L_Inmob",
              "M_Prof","N_Admin","P_Educ","Q_Salud","R_Entret",
              "S_OtrosServ","T_Hogar"]
    df["RAMA_SECCION"] = np.select(conditions, labels, default="Otro")

    df = construir_ocupacion_posicion_zona_migracion(df)

    return df

# ─── MODELO PROBIT ───────────────────────────────────────────────────────────

FORMULA = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
             CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO + 
             C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
             C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) + 
             EDAD_C + EDAD2_C +
             OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
             EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
             C(RAMA_SECCION, Treatment(reference='G_Comerc'))"""

VARS_MODELO = ["MUJER", "MUJER_CUIDADO", "TRANS", "NO_HETERO", "ETNIA",
               "CAMPESINO", "SUBSIDIO_GOB", "DESEA_CAMBIO", "EDUCACION_CAT",
               "HORAS_CAT", "EDAD_C", "EDAD2_C",
               "OCUPACION_CALIF", "ZONA_RURAL", "MIGRANTE_RECIENTE", "TASA_DEPENDENCIA",
               "EN_PAREJA", "TIENE_ING_NO_LAB", "NACIDO_EXTRANJERO",
               "RAMA_SECCION"]

# ─── DIAGNÓSTICO ─────────────────────────────────────────────────────────────

def diagnostico(probit, ocupados):
    datos = ocupados[VARS_MODELO + ["INFORMAL"]].dropna()
    X = datos[VARS_MODELO]
    y = datos["INFORMAL"]

    exog = probit.model.exog
    exog_names = probit.model.exog_names
    vif = pd.DataFrame({
        "Variable": exog_names,
        "VIF": [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
    })
    vif = vif[vif["Variable"] != "Intercept"].copy()
    print("\n── VIF ──")
    print(vif.sort_values("VIF", ascending=False).to_string(index=False))

    y_pred_prob = probit.predict(X)

    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    ix_optimo = np.argmax(youden_j)
    umbral_optimo = thresholds[ix_optimo]
    recall_optimo = tpr[ix_optimo]
    especificidad_optimo = 1 - fpr[ix_optimo]

    print(f"\n── Umbral óptimo (índice de Youden, vía curva ROC) ──")
    print(f"   Umbral óptimo = {umbral_optimo:.4f}  (vs. 0.5 fijo usado antes)")
    print(f"   Índice de Youden (J = Sensibilidad + Especificidad - 1) = {youden_j[ix_optimo]:.4f}")
    print(f"   Recall / Sensibilidad en ese umbral = {recall_optimo:.4f} ({recall_optimo:.1%})")
    print(f"   Especificidad en ese umbral         = {especificidad_optimo:.4f} ({especificidad_optimo:.1%})")

    y_pred_bin  = (y_pred_prob >= umbral_optimo).astype(int)
    residuos    = y - y_pred_prob

    deciles = pd.qcut(y_pred_prob, 10, labels=False, duplicates="drop")
    n_grupos = deciles.nunique()
    hl_data = pd.DataFrame({"y": y.values, "pred": y_pred_prob.values, "decil": deciles})
    hl_obs  = hl_data.groupby("decil")["y"].sum()
    hl_esp  = hl_data.groupby("decil")["pred"].sum()
    hl_n    = hl_data.groupby("decil")["y"].count()
    hl_stat = (((hl_obs - hl_esp)**2) / (hl_esp * (1 - hl_esp/hl_n))).sum()
    hl_df   = max(n_grupos - 2, 1)
    hl_pval = 1 - stats.chi2.cdf(hl_stat, df=hl_df)
    print(f"\n── Hosmer-Lemeshow ──")
    if n_grupos < 10:
        print(f"   Aviso: probabilidades predichas con muchos empates en la cola "
              f"(posible separación casi perfecta en una subpoblación); "
              f"se usaron {n_grupos} grupos en vez de 10 (gl={hl_df}).")
    print(f"   Chi² = {hl_stat:.3f}  |  p-value = {hl_pval:.4f}")
    print(f"   Nota: con N={len(datos):,} este test casi siempre rechaza H0;")
    print(f"   priorizar AUC y % clasificación correcta para evaluar el ajuste.")

    print(f"\n── ROC ──")
    print(f"   AUC = {roc_auc:.4f}")

    cm = confusion_matrix(y, y_pred_bin)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    recall_cm = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    especificidad_cm = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision_cm = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    print(f"\n── Matriz de confusión (umbral = {umbral_optimo:.4f}, óptimo de Youden) ──")
    print(f"   Correctamente clasificados: {(tn+tp)/cm.sum():.1%}")
    print(f"   Recall / Sensibilidad      : {recall_cm:.4f} ({recall_cm:.1%})")
    print(f"   Especificidad              : {especificidad_cm:.4f} ({especificidad_cm:.1%})")
    print(f"   Precisión                  : {precision_cm:.4f} ({precision_cm:.1%})")

    print(f"\n── Estabilidad por año ──")
    coefs_año = {}
    rhs_formula = FORMULA.split("~")[1].strip()
    for año in AÑOS:
        sub = ocupados[ocupados["AÑO"] == año][VARS_MODELO + ["INFORMAL"]].dropna()
        if len(sub) < 1000:
            continue
        m = smf.probit("INFORMAL ~ " + rhs_formula, data=sub).fit(disp=0)
        coefs_año[año] = m.params
        print(f"   {año}: {len(sub):,} obs | Pseudo R² = {m.prsquared:.4f}")

    carpeta_graficas = os.path.join(OUTPUT_DIR, "graficas")
    os.makedirs(carpeta_graficas, exist_ok=True)
    rutas_individuales = []

    alto_vif = max(6, len(vif) * 0.28)
    fig1, ax1 = plt.subplots(figsize=(9, alto_vif))
    colores = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71" for v in vif["VIF"]]
    ax1.barh(vif["Variable"], vif["VIF"], color=colores)
    ax1.axvline(5,  color="#f39c12", linestyle="--", linewidth=1, label="VIF=5")
    ax1.axvline(10, color="#e74c3c", linestyle="--", linewidth=1, label="VIF=10")
    ax1.set_title("Factor de Inflación de Varianza (VIF)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("VIF")
    ax1.tick_params(axis="y", labelsize=8)
    fig1.tight_layout()
    ruta1 = os.path.join(carpeta_graficas, "01_vif.png")
    fig1.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    rutas_individuales.append(ruta1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(y_pred_prob[y==0], bins=50, alpha=0.6, label="Formal",   color="#3498db")
    ax2.hist(y_pred_prob[y==1], bins=50, alpha=0.6, label="Informal", color="#e74c3c")
    ax2.set_title("Distribución de probabilidades predichas")
    ax2.set_xlabel("P(Informal)")
    ax2.legend()
    fig2.tight_layout()
    ruta2 = os.path.join(carpeta_graficas, "02_distribucion_probabilidades.png")
    fig2.savefig(ruta2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    rutas_individuales.append(ruta2)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(fpr, tpr, color="#8e44ad", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax3.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC=0.5)")
    ax3.scatter([fpr[ix_optimo]], [tpr[ix_optimo]], color="#e74c3c", s=80, zorder=5,
                label=f"Umbral óptimo = {umbral_optimo:.3f}\n(Recall={recall_optimo:.1%}, Espec.={especificidad_optimo:.1%})")
    ax3.set_title("Curva ROC")
    ax3.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
    ax3.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax3.legend(fontsize=8, loc="lower right")
    fig3.tight_layout()
    ruta3 = os.path.join(carpeta_graficas, "03_curva_roc.png")
    fig3.savefig(ruta3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    rutas_individuales.append(ruta3)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.bar(range(len(hl_obs)), hl_obs.values, alpha=0.7, label="Observado", color="#2ecc71")
    ax4.bar(range(len(hl_esp)), hl_esp.values, alpha=0.7, label="Esperado",  color="#e67e22", width=0.4)
    ax4.set_title(f"Hosmer-Lemeshow ({n_grupos} grupos)\nChi²={hl_stat:.2f}  p={hl_pval:.4f}")
    ax4.set_xlabel("Grupo (decil de riesgo)")
    ax4.legend()
    fig4.tight_layout()
    ruta4 = os.path.join(carpeta_graficas, "04_hosmer_lemeshow.png")
    fig4.savefig(ruta4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    rutas_individuales.append(ruta4)

    fig5, ax5 = plt.subplots(figsize=(8, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=["Formal", "Informal"]).plot(ax=ax5, colorbar=False)
    ax5.set_title(
        f"Matriz de confusión (umbral óptimo = {umbral_optimo:.3f})\n"
        f"Recall = {recall_cm:.1%}   |   Especificidad = {especificidad_cm:.1%}   |   "
        f"Correctos = {(tn+tp)/cm.sum():.1%}",
        fontsize=10
    )
    fig5.tight_layout()
    ruta5 = os.path.join(carpeta_graficas, "05_matriz_confusion.png")
    fig5.savefig(ruta5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    rutas_individuales.append(ruta5)

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    ax6.scatter(y_pred_prob[::50], residuos[::50], alpha=0.3, s=5, color="#7f8c8d")
    ax6.axhline(0, color="red", linestyle="--", linewidth=1)
    ax6.set_title("Residuos vs probabilidades predichas")
    ax6.set_xlabel("P(Informal) predicha")
    ax6.set_ylabel("Residuo")
    fig6.tight_layout()
    ruta6 = os.path.join(carpeta_graficas, "06_residuos_vs_prob.png")
    fig6.savefig(ruta6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    rutas_individuales.append(ruta6)

    fig7, ax7 = plt.subplots(figsize=(12, 6))
    if coefs_año:
        coef_df = pd.DataFrame(coefs_año).T.drop(columns=["Intercept"], errors="ignore")
        for col in coef_df.columns:
            ax7.plot(coef_df.index, coef_df[col], marker="o", label=col)
        ax7.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax7.set_title("Estabilidad de coeficientes por año")
        ax7.set_xlabel("Año")
        ax7.set_ylabel("Coeficiente")
        ax7.legend(fontsize=7, ncol=4, loc="upper right")
    fig7.tight_layout()
    ruta7 = os.path.join(carpeta_graficas, "07_estabilidad_por_anio.png")
    fig7.savefig(ruta7, dpi=150, bbox_inches="tight")
    plt.close(fig7)
    rutas_individuales.append(ruta7)

    print(f"\n── Gráficas individuales guardadas en: {carpeta_graficas} ──")
    for r in rutas_individuales:
        print(f"   - {r}")

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(vif["Variable"], vif["VIF"], color=colores)
    ax1.axvline(5,  color="#f39c12", linestyle="--", linewidth=1, label="VIF=5")
    ax1.axvline(10, color="#e74c3c", linestyle="--", linewidth=1, label="VIF=10")
    ax1.set_title("Factor de Inflación de Varianza (VIF)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("VIF")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_pred_prob[y==0], bins=50, alpha=0.6, label="Formal",   color="#3498db")
    ax2.hist(y_pred_prob[y==1], bins=50, alpha=0.6, label="Informal", color="#e74c3c")
    ax2.set_title("Distribución de probabilidades predichas")
    ax2.set_xlabel("P(Informal)")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(fpr, tpr, color="#8e44ad", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax3.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC=0.5)")
    ax3.scatter([fpr[ix_optimo]], [tpr[ix_optimo]], color="#e74c3c", s=60, zorder=5,
                label=f"Umbral óptimo = {umbral_optimo:.3f}")
    ax3.set_title("Curva ROC")
    ax3.set_xlabel("Tasa de falsos positivos")
    ax3.set_ylabel("Tasa de verdaderos positivos")
    ax3.legend(fontsize=7, loc="lower right")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(len(hl_obs)), hl_obs.values, alpha=0.7, label="Observado", color="#2ecc71")
    ax4.bar(range(len(hl_esp)), hl_esp.values, alpha=0.7, label="Esperado",  color="#e67e22", width=0.4)
    ax4.set_title(f"Hosmer-Lemeshow ({n_grupos} grupos)\nChi²={hl_stat:.2f}  p={hl_pval:.4f}")
    ax4.set_xlabel("Grupo (decil de riesgo)")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Formal", "Informal"]).plot(ax=ax5, colorbar=False)
    ax5.set_title(
        f"Matriz de confusión (umbral={umbral_optimo:.3f})\n"
        f"Recall={recall_cm:.1%}  Espec.={especificidad_cm:.1%}",
        fontsize=9
    )

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(y_pred_prob[::50], residuos[::50], alpha=0.3, s=5, color="#7f8c8d")
    ax6.axhline(0, color="red", linestyle="--", linewidth=1)
    ax6.set_title("Residuos vs probabilidades predichas")
    ax6.set_xlabel("P(Informal) predicha")
    ax6.set_ylabel("Residuo")

    ax7 = fig.add_subplot(gs[2, :])
    if coefs_año:
        for col in coef_df.columns:
            ax7.plot(coef_df.index, coef_df[col], marker="o", label=col)
        ax7.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax7.set_title("Estabilidad de coeficientes por año")
        ax7.set_xlabel("Año")
        ax7.set_ylabel("Coeficiente")
        ax7.legend(fontsize=7, ncol=4, loc="upper right")

    plt.suptitle("Diagnóstico del modelo Probit — Informalidad laboral GEIH 2022–2025",
                 fontsize=13, fontweight="bold", y=1.01)

    ruta_grafico = os.path.join(OUTPUT_DIR, "diagnostico_probit.png")
    plt.savefig(ruta_grafico, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPanel combinado guardado como: {ruta_grafico}")

# ─── TABLAS DE ESTADÍSTICA DESCRIPTIVA (MICRO) ───────────────────────────────

def generar_estadisticas_descriptivas(geih, ocupados_heck):
    print("\n" + "=" * 80)
    print("── GENERANDO TABLAS DE ESTADÍSTICA DESCRIPTIVA (MICRO) ──")
    print("=" * 80)

    df_oc = ocupados_heck.copy()
    if "EDAD" not in df_oc.columns and "P6040" in df_oc.columns:
        df_oc["EDAD"] = pd.to_numeric(df_oc["P6040"], errors="coerce")
    if "HORAS_SEMANA" not in df_oc.columns and "P6800" in df_oc.columns:
        df_oc["HORAS_SEMANA"] = pd.to_numeric(df_oc["P6800"], errors="coerce")

    df_oc["EDU_PRIMARIA"]   = (df_oc["EDUCACION_CAT"] == "1_Primaria_Menos").astype(float)
    df_oc["EDU_SECUNDARIA"] = (df_oc["EDUCACION_CAT"] == "2_Secundaria").astype(float)
    df_oc["EDU_SUPERIOR"]   = (df_oc["EDUCACION_CAT"] == "3_Superior").astype(float)

    df_oc["HORAS_PARCIAL"]  = (df_oc["HORAS_CAT"] == "1_Parcial_Menos35").astype(float)
    df_oc["HORAS_COMPLETA"] = (df_oc["HORAS_CAT"] == "2_Completa_35a48").astype(float)
    df_oc["HORAS_SOBRE"]    = (df_oc["HORAS_CAT"] == "3_Sobretrabajo_Mas48").astype(float)

    vars_ocupados = [
        ("INFORMAL",            "Trabajador informal (1 = Sí)",                             True),
        ("MUJER",               "Mujer (1 = Sí)",                                           True),
        ("EDAD",                "Edad (años)",                                              False),
        ("CUIDADO_NINOS",       "Cuidado de niños < 5 años en el hogar (1 = Sí)",            True),
        ("MUJER_CUIDADO",       "Interacción Mujer × Cuidado niños (1 = Sí)",               True),
        ("TRANS",               "Persona trans / no binaria (1 = Sí)",                      True),
        ("NO_HETERO",           "Orientación sexual diversa / no hetero (1 = Sí)",          True),
        ("ETNIA",               "Pertenencia étnica minoritaria (1 = Sí)",                  True),
        ("CAMPESINO",           "Autorreconocimiento campesino (1 = Sí)",                   True),
        ("EN_PAREJA",           "En unión libre o casado/a (1 = Sí)",                       True),
        ("EDU_PRIMARIA",        "Nivel educativo: Primaria o menos (1 = Sí)",               True),
        ("EDU_SECUNDARIA",      "Nivel educativo: Secundaria / Media (1 = Sí)",             True),
        ("EDU_SUPERIOR",        "Nivel educativo: Superior / Universitaria (1 = Sí)",       True),
        ("HORAS_SEMANA",        "Horas habituales trabajadas por semana",                   False),
        ("HORAS_PARCIAL",       "Jornada parcial (< 35 horas/sem) (1 = Sí)",                True),
        ("HORAS_COMPLETA",      "Jornada completa (35 a 48 horas/sem) (1 = Sí)",            True),
        ("HORAS_SOBRE",         "Sobretrabajo (> 48 horas/sem) (1 = Sí)",                   True),
        ("DESEA_CAMBIO",        "Desea cambiar de trabajo (1 = Sí)",                        True),
        ("OCUPACION_CALIF",     "Ocupación alta calificación (CIUO 1-3) (1 = Sí)",          True),
        ("ZONA_RURAL",          "Zona rural / Resto (1 = Sí, 0 = Cabecera)",                True),
        ("MIGRANTE_RECIENTE",   "Migrante reciente (residencia diferente hace 1 año)",      True),
        ("NACIDO_EXTRANJERO",   "Nacido en el extranjero (1 = Sí)",                         True),
        ("TASA_DEPENDENCIA",    "Tasa de dependencia del hogar (ocupados/personas)",        False),
        ("SUBSIDIO_GOB",        "Recibe subsidio o ayuda en dinero del gob. (1 = Sí)",      True),
        ("TIENE_ING_NO_LAB",    "Hogar cuenta con ingresos no laborales (1 = Sí)",          True),
        ("IMR",                 "Inverse Mills Ratio (IMR / Heckman)",                      False),
    ]

    if "INFORMALIDAD_RAMA" in df_oc.columns:
        vars_ocupados.insert(20, ("INFORMALIDAD_RAMA", "Tasa informalidad sectorial de la rama (loo)", False))

    filas_oc = []
    for col, etiqueta, es_dum in vars_ocupados:
        if col not in df_oc.columns:
            continue
        s_tot = pd.to_numeric(df_oc[col], errors="coerce").dropna()
        s_for = pd.to_numeric(df_oc.loc[df_oc["INFORMAL"] == 0, col], errors="coerce").dropna()
        s_inf = pd.to_numeric(df_oc.loc[df_oc["INFORMAL"] == 1, col], errors="coerce").dropna()

        n_tot = len(s_tot)
        media_tot = s_tot.mean() if n_tot > 0 else np.nan
        std_tot   = s_tot.std()  if n_tot > 0 else np.nan
        min_tot   = s_tot.min()  if n_tot > 0 else np.nan
        max_tot   = s_tot.max()  if n_tot > 0 else np.nan

        media_for = s_for.mean() if len(s_for) > 0 else np.nan
        std_for   = s_for.std()  if len(s_for) > 0 else np.nan

        media_inf = s_inf.mean() if len(s_inf) > 0 else np.nan
        std_inf   = s_inf.std()  if len(s_inf) > 0 else np.nan

        dif_media = media_inf - media_for
        p_val = np.nan
        if len(s_for) > 1 and len(s_inf) > 1 and not (s_for.std() == 0 and s_inf.std() == 0):
            try:
                t_stat, p_val = stats.ttest_ind(s_inf, s_for, equal_var=False)
            except Exception:
                p_val = np.nan

        stars = ("***" if p_val < 0.01 else
                 "**"  if p_val < 0.05 else
                 "*"   if p_val < 0.10 else "")

        filas_oc.append({
            "variable": col,
            "etiqueta": etiqueta,
            "tipo": "Dummy" if es_dum else "Continua",
            "n_obs": n_tot,
            "total_media": media_tot,
            "total_std": std_tot,
            "total_min": min_tot,
            "total_max": max_tot,
            "formal_media": media_for,
            "formal_std": std_for,
            "informal_media": media_inf,
            "informal_std": std_inf,
            "diferencia": dif_media,
            "p_valor": p_val,
            "estrellas": stars
        })

    tabla_oc_df = pd.DataFrame(filas_oc)

    pet = geih[geih["EDAD"] >= 15].copy()
    if "EDAD" not in pet.columns and "P6040" in pet.columns:
        pet["EDAD"] = pd.to_numeric(pet["P6040"], errors="coerce")
    pet = recodificar(pet)
    if "PARTICIPA" not in pet.columns:
        pet["PARTICIPA"] = pet["OCUPADO"].astype(int)

    pet["EDU_PRIMARIA"]   = (pet["EDUCACION_CAT"] == "1_Primaria_Menos").astype(float)
    pet["EDU_SECUNDARIA"] = (pet["EDUCACION_CAT"] == "2_Secundaria").astype(float)
    pet["EDU_SUPERIOR"]   = (pet["EDUCACION_CAT"] == "3_Superior").astype(float)

    vars_pet = [
        ("PARTICIPA",             "Participa en el mercado laboral / Ocupado (1 = Sí)",      True),
        ("CUIDADO_DEPENDIENTES",  "Cuidado dependientes [Variable Exclusión] (1 = Sí)",      True),
        ("MUJER",                 "Mujer (1 = Sí)",                                          True),
        ("EDAD",                  "Edad (años)",                                             False),
        ("ETNIA",                 "Pertenencia étnica minoritaria (1 = Sí)",                 True),
        ("CAMPESINO",             "Autorreconocimiento campesino (1 = Sí)",                  True),
        ("SUBSIDIO_GOB",          "Recibe subsidio o ayuda en dinero del gob. (1 = Sí)",     True),
        ("EN_PAREJA",             "En unión libre o casado/a (1 = Sí)",                      True),
        ("EDU_PRIMARIA",          "Nivel educativo: Primaria o menos (1 = Sí)",              True),
        ("EDU_SECUNDARIA",        "Nivel educativo: Secundaria / Media (1 = Sí)",            True),
        ("EDU_SUPERIOR",          "Nivel educativo: Superior / Universitaria (1 = Sí)",      True),
        ("ZONA_RURAL",            "Zona rural / Resto (1 = Sí, 0 = Cabecera)",               True),
        ("MIGRANTE_RECIENTE",     "Migrante reciente (residencia diferente hace 1 año)",     True),
        ("NACIDO_EXTRANJERO",     "Nacido en el extranjero (1 = Sí)",                         True),
    ]

    filas_pet = []
    for col, etiqueta, es_dum in vars_pet:
        if col not in pet.columns:
            continue
        s_tot = pd.to_numeric(pet[col], errors="coerce").dropna()
        s_par = pd.to_numeric(pet.loc[pet["PARTICIPA"] == 1, col], errors="coerce").dropna()
        s_nop = pd.to_numeric(pet.loc[pet["PARTICIPA"] == 0, col], errors="coerce").dropna()

        n_tot = len(s_tot)
        media_tot = s_tot.mean() if n_tot > 0 else np.nan
        std_tot   = s_tot.std()  if n_tot > 0 else np.nan
        min_tot   = s_tot.min()  if n_tot > 0 else np.nan
        max_tot   = s_tot.max()  if n_tot > 0 else np.nan

        media_par = s_par.mean() if len(s_par) > 0 else np.nan
        std_par   = s_par.std()  if len(s_par) > 0 else np.nan

        media_nop = s_nop.mean() if len(s_nop) > 0 else np.nan
        std_nop   = s_nop.std()  if len(s_nop) > 0 else np.nan

        dif_media = media_par - media_nop
        p_val = np.nan
        if len(s_par) > 1 and len(s_nop) > 1 and not (s_par.std() == 0 and s_nop.std() == 0):
            try:
                t_stat, p_val = stats.ttest_ind(s_par, s_nop, equal_var=False)
            except Exception:
                p_val = np.nan

        stars = ("***" if p_val < 0.01 else
                 "**"  if p_val < 0.05 else
                 "*"   if p_val < 0.10 else "")

        filas_pet.append({
            "variable": col,
            "etiqueta": etiqueta,
            "tipo": "Dummy" if es_dum else "Continua",
            "n_obs": n_tot,
            "total_media": media_tot,
            "total_std": std_tot,
            "total_min": min_tot,
            "total_max": max_tot,
            "participa_media": media_par,
            "participa_std": std_par,
            "noparticipa_media": media_nop,
            "noparticipa_std": std_nop,
            "diferencia": dif_media,
            "p_valor": p_val,
            "estrellas": stars
        })

    tabla_pet_df = pd.DataFrame(filas_pet)

    ruta_csv_oc  = os.path.join(OUTPUT_DIR, "tabla_descriptiva_ocupados.csv")
    ruta_csv_pet = os.path.join(OUTPUT_DIR, "tabla_descriptiva_pet_seleccion.csv")
    tabla_oc_df.to_csv(ruta_csv_oc, index=False, encoding="utf-8")
    tabla_pet_df.to_csv(ruta_csv_pet, index=False, encoding="utf-8")

    def _formato_num(val, es_dummy=False):
        if pd.isna(val): return "     -    "
        if es_dummy:
            return f"{val*100:6.2f}%"
        return f"{val:8.3f}"

    lineas_txt = []
    lineas_txt.append("=" * 115)
    lineas_txt.append("TABLA 1: ESTADÍSTICA DESCRIPTIVA — MUESTRA DE OCUPADOS (PROBIT Y 2ª ETAPA HECKMAN)")
    lineas_txt.append("GEIH Colombia (2022–2025, Trimestre 2: Abril–Junio)")
    n_for = (df_oc['INFORMAL'] == 0).sum()
    pct_for = (df_oc['INFORMAL'] == 0).mean()
    n_inf = (df_oc['INFORMAL'] == 1).sum()
    pct_inf = (df_oc['INFORMAL'] == 1).mean()
    lineas_txt.append(f"Total Ocupados: {len(df_oc):,}  |  Formales: {n_for:,} ({pct_for:.1%})  |  Informales: {n_inf:,} ({pct_inf:.1%})")
    lineas_txt.append("=" * 115)
    lineas_txt.append(
        f"{'Variable / Indicador':<44} | {'Total Media':^12} {'(Std)':^10} | "
        f"{'Formal Media':^12} | {'Informal Media':^14} | {'Dif (Inf-Form)':^14}"
    )
    lineas_txt.append("-" * 115)

    for r in filas_oc:
        es_d = (r["tipo"] == "Dummy")
        v_nombre = r["etiqueta"][:43]
        tot_m = _formato_num(r["total_media"], es_d)
        tot_s = f"({r['total_std']:.3f})" if not pd.isna(r["total_std"]) else "   -   "
        for_m = _formato_num(r["formal_media"], es_d)
        inf_m = _formato_num(r["informal_media"], es_d)

        if pd.isna(r["diferencia"]):
            dif_s = "      -     "
        else:
            if es_d:
                dif_s = f"{r['diferencia']*100:+6.2f} pp {r['estrellas']:<3}"
            else:
                dif_s = f"{r['diferencia']:+8.3f} {r['estrellas']:<3}"

        lineas_txt.append(
            f"{v_nombre:<44} | {tot_m:^12} {tot_s:^10} | {for_m:^12} | {inf_m:^14} | {dif_s:^14}"
        )

    lineas_txt.append("-" * 115)
    lineas_txt.append("Notas: En variables binarias (Dummy) se reporta el porcentaje (%); en continuas se reporta la media y (desv. estándar).")
    lineas_txt.append("Diferencia (Informal - Formal) evaluada con prueba t de Welch para dos muestras independientes.")
    lineas_txt.append("Nivel de significancia: *** p < 0.01, ** p < 0.05, * p < 0.10. pp = puntos porcentuales.")
    lineas_txt.append("\n\n")

    lineas_txt.append("=" * 115)
    lineas_txt.append("TABLA 2: ESTADÍSTICA DESCRIPTIVA — POBLACIÓN EN EDAD DE TRABAJAR (PET >= 15 AÑOS, 1ª ETAPA HECKMAN)")
    lineas_txt.append("GEIH Colombia (2022–2025, Trimestre 2: Abril–Junio)")
    n_par = (pet['PARTICIPA'] == 1).sum()
    pct_par = (pet['PARTICIPA'] == 1).mean()
    n_nop = (pet['PARTICIPA'] == 0).sum()
    pct_nop = (pet['PARTICIPA'] == 0).mean()
    lineas_txt.append(f"Total PET: {len(pet):,}  |  Participan: {n_par:,} ({pct_par:.1%})  |  No participan: {n_nop:,} ({pct_nop:.1%})")
    lineas_txt.append("=" * 115)
    lineas_txt.append(
        f"{'Variable / Indicador':<44} | {'Total Media':^12} {'(Std)':^10} | "
        f"{'Participa Media':^15} | {'No Part. Media':^14} | {'Dif (Part-NoPart)':^16}"
    )
    lineas_txt.append("-" * 115)

    for r in filas_pet:
        es_d = (r["tipo"] == "Dummy")
        v_nombre = r["etiqueta"][:43]
        tot_m = _formato_num(r["total_media"], es_d)
        tot_s = f"({r['total_std']:.3f})" if not pd.isna(r["total_std"]) else "   -   "
        par_m = _formato_num(r["participa_media"], es_d)
        nop_m = _formato_num(r["noparticipa_media"], es_d)

        if pd.isna(r["diferencia"]):
            dif_s = "       -      "
        else:
            if es_d:
                dif_s = f"{r['diferencia']*100:+6.2f} pp {r['estrellas']:<3}"
            else:
                dif_s = f"{r['diferencia']:+8.3f} {r['estrellas']:<3}"

        lineas_txt.append(
            f"{v_nombre:<44} | {tot_m:^12} {tot_s:^10} | {par_m:^15} | {nop_m:^14} | {dif_s:^16}"
        )

    lineas_txt.append("-" * 115)
    lineas_txt.append("Notas: En variables binarias (Dummy) se reporta el porcentaje (%); en continuas se reporta la media y (desv. estándar).")
    lineas_txt.append("CUIDADO_DEPENDIENTES es la variable de exclusión exógena para la ecuación de participación laboral.")
    lineas_txt.append("Diferencia (Participa - No Participa) evaluada con prueba t de Welch para dos muestras independientes.")
    lineas_txt.append("Nivel de significancia: *** p < 0.01, ** p < 0.05, * p < 0.10. pp = puntos porcentuales.")
    lineas_txt.append("=" * 115 + "\n")

    reporte_str = "\n".join(lineas_txt)
    print("\n" + reporte_str)

    ruta_txt = os.path.join(OUTPUT_DIR, "tablas_estadistica_descriptiva.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(reporte_str)

    print(f"✓ Tabla descriptiva de Ocupados guardada en: {ruta_csv_oc}")
    print(f"✓ Tabla descriptiva de PET (selección) guardada en: {ruta_csv_pet}")
    print(f"✓ Reporte consolidado en texto guardado en: {ruta_txt}\n")

    return tabla_oc_df, tabla_pet_df

# ─── GUARDADO DE RESULTADOS EN TEXTO ─────────────────────────────────────────

def guardar_resultados_txt(probit, mfx, probit_sel, probit_heck, mfx_heck):
    ruta_txt = os.path.join(OUTPUT_DIR, "resultados_probit.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("RESULTADOS — PROBIT INFORMALIDAD GEIH 2022–2025\n")
        f.write("=" * 70 + "\n\n")

        f.write("── PROBIT SIMPLE (sin corrección de selección) ──\n\n")
        f.write(str(probit.summary()))
        f.write("\n\n")
        f.write(str(mfx.summary()))
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("── PRIMERA ETAPA (selección / participación laboral) ──\n\n")
        f.write(str(probit_sel.summary()))
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("── SEGUNDA ETAPA (informalidad corregida por Heckman) ──\n")
        f.write("   NOTA METODOLÓGICA: los errores estándar reportados abajo son\n")
        f.write("   robustos, agrupados (cluster-robust) por hogar (DIRECTORIO),\n")
        f.write("   lo que corrige la correlación intra-hogar en la muestra de\n")
        f.write("   ocupados. No incorporan la incertidumbre de estimación del\n")
        f.write("   IMR (regresor generado en la 1ra etapa), por lo que pueden\n")
        f.write("   subestimar los errores estándar de la 2da etapa (Murphy &\n")
        f.write("   Topel, 1985). Se reporta como limitación metodológica.\n\n")
        f.write(str(probit_heck.summary()))
        f.write("\n\n")
        f.write(str(mfx_heck.summary()))
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("── OTROS ARCHIVOS GENERADOS EN ESTA EJECUCIÓN ──\n\n")
        f.write("Tablas de estadística descriptiva:\n")
        f.write(f"   - Ocupados (CSV): {os.path.join(OUTPUT_DIR, 'tabla_descriptiva_ocupados.csv')}\n")
        f.write(f"   - PET / Selección (CSV): {os.path.join(OUTPUT_DIR, 'tabla_descriptiva_pet_seleccion.csv')}\n")
        f.write(f"   - Reporte consolidado (TXT): {os.path.join(OUTPUT_DIR, 'tablas_estadistica_descriptiva.txt')}\n\n")
        f.write(f"Panel combinado de diagnóstico: {os.path.join(OUTPUT_DIR, 'diagnostico_probit.png')}\n")
        f.write(f"Gráficas individuales (carpeta): {os.path.join(OUTPUT_DIR, 'graficas')}\n")
        f.write("   01_vif.png\n")
        f.write("   02_distribucion_probabilidades.png\n")
        f.write("   03_curva_roc.png\n")
        f.write("   04_hosmer_lemeshow.png\n")
        f.write("   05_matriz_confusion.png\n")
        f.write("   06_residuos_vs_prob.png\n")
        f.write("   07_estabilidad_por_anio.png\n")
        f.write(f"Reporte de robustez (con SE cluster-robust): {os.path.join(OUTPUT_DIR, 'reporte_robustez.txt')}\n")
        f.write("\nEl log completo de TODO lo impreso durante esta ejecución\n")
        f.write("(lectura por año/mes, verificación de variables, VIF, Hosmer-\n")
        f.write("Lemeshow, ROC, diagnóstico IMR, etc.) está en un archivo separado\n")
        f.write("con nombre 'log_completo_ejecucion_<timestamp>.txt' en la misma\n")
        f.write(f"carpeta: {OUTPUT_DIR}\n")

    print(f"Resultados guardados como: {ruta_txt}")

# ─── CORRECCIÓN DE HECKMAN ───────────────────────────────────────────────────

_FORMULA_SEL = ("PARTICIPA ~ CUIDADO_DEPENDIENTES + ETNIA + "
                "CAMPESINO + SUBSIDIO_GOB + C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) + "
                "EDAD_C + EDAD2_C + EN_PAREJA + "
                "ZONA_RURAL + MIGRANTE_RECIENTE + NACIDO_EXTRANJERO")

_VARS_SEL = ["CUIDADO_DEPENDIENTES", "ETNIA", "CAMPESINO",
             "SUBSIDIO_GOB", "EDUCACION_CAT", "EDAD_C", "EDAD2_C",
             "EN_PAREJA", "ZONA_RURAL", "MIGRANTE_RECIENTE", "NACIDO_EXTRANJERO"]

_FORMULA_PRINCIPAL = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
                     CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO +
                     C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
                     C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) +
                     EDAD_C + EDAD2_C +
                     OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
                     EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
                     C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

_FORMULA_M2 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO +
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) +
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

_FORMULA_M3 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + DESEA_CAMBIO +
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) +
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

_FORMULA_M4 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + DESEA_CAMBIO +
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) +
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

_NOMBRES_MODELOS = ["Principal", "M2_SinEnPar", "M3_SinSubsidio", "M4_Combinado"]
_FORMULAS_2DA   = [_FORMULA_PRINCIPAL, _FORMULA_M2, _FORMULA_M3, _FORMULA_M4]


def _calcular_imr(probit_sel_fit, pet_df, vars_sel):
    from scipy.stats import norm
    pet_validos = pet_df[vars_sel].dropna()
    if len(pet_validos) == 0:
        return pd.Series(index=pet_df.index, dtype=float)

    try:
        xb = probit_sel_fit.predict(pet_validos, which="linear")
    except TypeError:
        xb = probit_sel_fit.predict(pet_validos, linear=True)

    if hasattr(xb, 'values'):
        xb = xb.values.astype(float)
    else:
        xb = np.array(xb).astype(float)

    pdf = norm.pdf(xb)
    cdf = norm.cdf(xb)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    imr = (pdf / cdf).astype(float)

    return pd.Series(imr, index=pet_validos.index)


def estimar_heckman(geih, ocupados):
    pet = geih[geih["EDAD"] >= 15].copy()
    pet = recodificar(pet)
    pet["PARTICIPA"] = pet["OCUPADO"].astype(int)   # v17: ocupado según módulo DANE

    print(f"\n── Chequeo de balance de variables de exclusión por género ──")
    print("   Media de CUIDADO_DEPENDIENTES por género:")
    print(pet.groupby("MUJER")["CUIDADO_DEPENDIENTES"].mean().to_string())

    corr_ninos = pet[["CUIDADO_DEPENDIENTES", "MUJER"]].dropna().corr().iloc[0, 1]
    print(f"\n   Correlación CUIDADO_DEPENDIENTES vs MUJER: {corr_ninos:+.3f}")
    print(f"   (Se usará CUIDADO_DEPENDIENTES como exclusión exógena en la 1ª etapa)")

    n_nan_dep = pet["CUIDADO_DEPENDIENTES"].isna().sum()
    print(f"\n   Missing en CUIDADO_DEPENDIENTES: {n_nan_dep:,} de {len(pet):,}")

    pet_clean = pet[_VARS_SEL + ["PARTICIPA"]].dropna()

    print(f"\n── Primera etapa (selección) ──")
    print(f"   PET total (EDAD >= 15):           {len(pet):,}")
    print(f"   PET usada en el probit (sin NaN): {len(pet_clean):,}")
    print(f"   Proporción de ocupados en la PET: {pet_clean['PARTICIPA'].mean():.1%}")

    probit_sel = smf.probit(_FORMULA_SEL, data=pet_clean).fit()
    print(probit_sel.summary())

    imr_all = _calcular_imr(probit_sel, pet, _VARS_SEL)

    print(f"\n   IMR — primeros 5 valores: {imr_all.head().values}")
    print(f"   IMR — media: {imr_all.mean():.4f}  |  std: {imr_all.std():.4f}")

    ocupados_heck = ocupados.copy()
    ocupados_heck["IMR"] = imr_all.reindex(ocupados_heck.index)

    n_sin_imr = ocupados_heck["IMR"].isna().sum()
    if n_sin_imr > 0:
        print(f"   Aviso: {n_sin_imr:,} ocupados sin IMR calculable "
              f"(missing en _VARS_SEL) — se excluirán del probit de 2ª etapa")

    print(f"\n── Diagnóstico: correlación IMR vs covariables clave ──")
    diag_corr = ocupados_heck[["IMR", "MUJER", "ETNIA", "CAMPESINO", "SUBSIDIO_GOB"]].dropna()
    corr_imr  = diag_corr.corr()["IMR"].drop("IMR")
    for var, val in corr_imr.items():
        bandera = "⚠ ALTA (posible identificación débil)" if abs(val) > 0.6 else "OK"
        print(f"   IMR vs {var:<13}: {val:+.3f}   {bandera}")

    # ── Segunda etapa: SE ROBUSTOS AGRUPADOS POR HOGAR (DIRECTORIO) ─────
    # Corrige la correlación intra-hogar. No corrige la incertidumbre del
    # IMR generado en la 1ra etapa (ver nota metodológica en el TXT de
    # resultados), pero es el ajuste estándar y defendible dado el tiempo.
    print(f"\n── Segunda etapa (informalidad corregida, SE cluster-robust por hogar) ──")
    ocupados_heck_valid = ocupados_heck.dropna(subset=["IMR"])
    probit_heck = smf.probit(_FORMULA_PRINCIPAL, data=ocupados_heck_valid).fit(
        cov_type="cluster",
        cov_kwds={"groups": ocupados_heck_valid["DIRECTORIO"]},
    )
    print(probit_heck.summary())

    mfx_heck = probit_heck.get_margeff(at="overall")
    print(mfx_heck.summary())

    return probit_sel, probit_heck, ocupados_heck, mfx_heck


def correr_analisis_robustez(ocupados_heck):
    print("\n" + "="*50)
    print("── CORRIENDO ANÁLISIS DE ROBUSTEZ Y ESPECIFICACIÓN (SE cluster-robust por hogar) ──")
    print("="*50)

    ocupados_heck_valid = ocupados_heck.dropna(subset=["IMR"])
    groups = ocupados_heck_valid["DIRECTORIO"]

    print("Estimando Modelo 1 (Base, == modelo principal)...")
    m1 = smf.probit(_FORMULA_PRINCIPAL, data=ocupados_heck_valid).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}
    )
    print("Estimando Modelo 2 (Robustez 1: sin EN_PAREJA en 2ª etapa)...")
    m2 = smf.probit(_FORMULA_M2, data=ocupados_heck_valid).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}
    )
    print("Estimando Modelo 3 (Robustez 2: sin SUBSIDIO_GOB en 2ª etapa)...")
    m3 = smf.probit(_FORMULA_M3, data=ocupados_heck_valid).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}
    )
    print("Estimando Modelo 4 (Robustez combinada)...")
    m4 = smf.probit(_FORMULA_M4, data=ocupados_heck_valid).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}
    )

    modelos = [m1, m2, m3, m4]
    variables_reportar = ["MUJER", "MUJER_CUIDADO", "SUBSIDIO_GOB",
                          "TIENE_ING_NO_LAB", "EN_PAREJA", "IMR"]

    lineas_tabla = []
    lineas_tabla.append(
        f"{'Variable':<22} | {'M1 (Base)':^18} | {'M2 (Rob 1 Hogar)':^18} "
        f"| {'M3 (Rob 2 Subs)':^18} | {'M4 (Combinado)':^18}"
    )
    lineas_tabla.append("-" * 105)

    for var in variables_reportar:
        coefs_str, pvals_str = [], []
        for m in modelos:
            if var in m.params:
                coef  = m.params[var]
                pval  = m.pvalues[var]
                se    = m.bse[var]
                stars = ("***" if pval < 0.01 else
                         "**"  if pval < 0.05 else
                         "*"   if pval < 0.1  else "")
                coefs_str.append(f"{coef:+.4f}{stars}")
                pvals_str.append(f"({se:.4f})")
            else:
                coefs_str.append("Excluida")
                pvals_str.append("")
        lineas_tabla.append(
            f"{var:<22} | {coefs_str[0]:^18} | {coefs_str[1]:^18} "
            f"| {coefs_str[2]:^18} | {coefs_str[3]:^18}"
        )
        lineas_tabla.append(
            f"{'':<22} | {pvals_str[0]:^18} | {pvals_str[1]:^18} "
            f"| {pvals_str[2]:^18} | {pvals_str[3]:^18}"
        )

    lineas_tabla.append("-" * 105)
    obs_str = [f"{m.nobs:,}"        for m in modelos]
    pr2_str = [f"{m.prsquared:.4f}" for m in modelos]
    ll_str  = [f"{m.llf:,.1f}"      for m in modelos]
    lineas_tabla.append(
        f"{'Observaciones':<22} | {obs_str[0]:^18} | {obs_str[1]:^18} "
        f"| {obs_str[2]:^18} | {obs_str[3]:^18}"
    )
    lineas_tabla.append(
        f"{'Pseudo R-squared':<22} | {pr2_str[0]:^18} | {pr2_str[1]:^18} "
        f"| {pr2_str[2]:^18} | {pr2_str[3]:^18}"
    )
    lineas_tabla.append(
        f"{'Log-Likelihood':<22} | {ll_str[0]:^18} | {ll_str[1]:^18} "
        f"| {ll_str[2]:^18} | {ll_str[3]:^18}"
    )

    reporte_str = "\n".join(lineas_tabla)
    print("\n" + reporte_str + "\n")

    ruta_rob = os.path.join(OUTPUT_DIR, "reporte_robustez.txt")
    with open(ruta_rob, "w", encoding="utf-8") as f:
        f.write("REPORTE DE ROBUSTEZ Y ESPECIFICACIÓN - HECKMAN INFORMALIDAD\n")
        f.write("="*70 + "\n\n")
        f.write("Notas: SE robustos agrupados (cluster-robust) por hogar (DIRECTORIO).\n")
        f.write("No incorporan la incertidumbre de estimación del IMR generado en\n")
        f.write("la 1ra etapa, por lo que pueden subestimar los errores estándar\n")
        f.write("(Murphy & Topel, 1985). Se reporta como limitación metodológica.\n")
        f.write("Significancia: *** p<0.01, ** p<0.05, * p<0.1\n\n")
        f.write(reporte_str)
        f.write("\n\n" + "="*70 + "\n")
    print(f"Reporte de robustez guardado en: {ruta_rob}")

    return modelos

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_log = os.path.join(OUTPUT_DIR, f"log_completo_ejecucion_{timestamp}.txt")

    stdout_original = sys.stdout
    log_file = open(ruta_log, "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(stdout_original, log_file)

    try:
        print(f"── Log completo de ejecución iniciado: {timestamp} ──")
        print(f"   Se guardará en: {ruta_log}\n")

        geih = construir_base()

        geih = construir_tasa_dependencia(geih)

        geih = construir_num_menores_5(geih)

        geih = construir_tiene_ing_no_lab(geih)

        ocupados = geih[(geih["OCUPADO"] == 1) & (geih["EDAD"] >= 15)].copy()
        print(f"\nOcupados: {len(ocupados):,}")

        ocupados = construir_formal(ocupados)
        print(f"Tasa de informalidad: {ocupados['INFORMAL'].mean():.1%}")

        ocupados = construir_informalidad_rama(ocupados)

        ocupados = recodificar(ocupados)

        probit = smf.probit(FORMULA, data=ocupados).fit()
        print(probit.summary())

        mfx = probit.get_margeff(at='overall')
        print(mfx.summary())

        probit_sel, probit_heck, ocupados_heck, mfx_heck = estimar_heckman(geih, ocupados)

        generar_estadisticas_descriptivas(geih, ocupados_heck)

        modelos_rob = correr_analisis_robustez(ocupados_heck)

        diagnostico(probit, ocupados)

        guardar_resultados_txt(probit, mfx, probit_sel, probit_heck, mfx_heck)

        print(f"\n── Ejecución completada sin errores ──")

    except Exception:
        import traceback
        print("\n── ERROR: la ejecución se detuvo por una excepción ──")
        traceback.print_exc(file=sys.stdout)
        raise

    finally:
        sys.stdout = stdout_original
        log_file.close()
        print(f"\nLog completo de la ejecución guardado en: {ruta_log}")