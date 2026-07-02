"""
GEIH 2022–2025 — Base única filtrable por año
Trimestre 2 (Abril, Mayo, Junio)
Merge por ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'] dentro de cada período

Versión corregida (v4 — SUBSIDIO_GOB incorporada a la ecuación de selección):
  - EDAD se centra antes de elevar al cuadrado (reduce VIF mecánico)
  - IMR de Heckman se calcula una sola vez, sin duplicar predict()
  - Variable de exclusión de Heckman: CUIDADO_DEPENDIENTES (cuidado de
    adultos mayores/enfermos/discapacidad en el propio hogar), en lugar de
    CUIDADO_NINOS. Esto se hizo porque CUIDADO_NINOS está fuertemente
    correlacionada con MUJER, lo que generaba alta correlación IMR-MUJER
    (r=0.56) y dificultaba separar selección de efecto condicional.
  - Al liberar CUIDADO_NINOS de la 1ª etapa, MUJER_CUIDADO (interacción
    MUJER × CUIDADO_NINOS) puede reincorporarse a la ecuación de resultado
    de la 2ª etapa sin violar el supuesto de exclusión.
  - SUBSIDIO_GOB se agrega también a la ecuación de selección (1ª etapa):
    existe para toda la PET y la teoría predice un efecto-ingreso sobre la
    decisión de participar, además del efecto sobre informalidad
    condicional ya documentado. Se aplica el mismo criterio usado para
    MUJER, ETNIA, CAMPESINO y EDUCACION (variables presentes en ambas
    ecuaciones porque existen para toda la PET y la teoría predice un
    canal de selección).
  - Diagnóstico de balance de variables de exclusión por género, y de
    colinealidad IMR vs covariables clave
  - predict(which="linear") en vez del argumento deprecado linear=True

Versión v5 (LN_INGLABO retirada del lado derecho del probit):
  - LN_INGLABO se sigue calculando y queda disponible en el DataFrame
    completo y en `ocupados` (para análisis bivariado / diferenciación
    cartográfica por departamento), pero se retira de las fórmulas del
    probit (modelo simple y 2ª etapa de Heckman).
  - Motivo: relación de causalidad inversa más plausible que la directa.
    La condición de informalidad determina en buena medida el salario
    (acceso a prestaciones, negociación, formalidad contractual), no al
    revés. Mantenerla como regresor de INFORMAL genera endogeneidad
    (no solo colinealidad), lo que sesga el resto de los coeficientes de
    forma impredecible. Se prioriza corregir la endogeneidad sobre
    "ganar" Pseudo R².
  - HORAS_SEMANA se mantiene en el modelo. Su VIF alto (19.07) en la
    versión anterior probablemente se debía en buena parte a su relación
    mecánica con LN_INGLABO (ingreso ≈ horas × tarifa); al retirar
    LN_INGLABO del lado derecho, se espera que el VIF de HORAS_SEMANA baje
    sustancialmente. Verificar en el VIF de esta corrida; si sigue alto,
    evaluar su exclusión en una versión posterior.

Versión v6 (enriquecimiento de determinantes estructurales):
  - Se incorporan determinantes del puesto de trabajo y del entorno del
    hogar, ausentes hasta v5 (que solo cubría perfil demográfico/oferta
    laboral): INFORMALIDAD_RAMA, OCUPACION_CALIF, dummies de POSICION,
    ZONA_RURAL, MIGRANTE_RECIENTE, TASA_DEPENDENCIA.
  - INFORMALIDAD_RAMA: tasa de informalidad por rama (RAMA2D_R4, pool
    2022-2025), calculada leave-one-out para evitar que la informalidad
    de cada persona contamine mecánicamente la tasa de su propio sector.
    Las ramas 84 y 99 (definidas como FORMAL=1 por regla administrativa
    dentro de construir_formal, no por criterios de afiliación) se
    excluyen del cálculo de tasas y reciben la tasa promedio nacional,
    para evitar circularidad con la variable dependiente.
  - OCUPACION_CALIF: dummy de calificación ocupacional a partir del gran
    grupo CIUO-08 (1er dígito de OFICIO_C8).
  - POSICION_*: dummies de posición ocupacional (P6430), base = asalariado
    privado.
  - ZONA_RURAL: dummy desde CLASE (1=Cabecera, 2=Resto).
  - MIGRANTE_RECIENTE: dummy desde P3384 (lugar de residencia hace 12
    meses); código 1 ("no había nacido") tratado como missing.
  - TASA_DEPENDENCIA: razón ocupados/total de personas del hogar,
    agregada por hogar (DIRECTORIO+SECUENCIA_P) a partir de P6008 y
    P6240. Pendiente de extender con interacción JEFE_HOGAR ×
    TASA_DEPENDENCIA cuando se confirme disponibilidad de P6051.
  - P3069 (n° empleados) se descarta como determinante: entra
    directamente en la regla de construcción de FORMAL para cuenta
    propia/independientes, por lo que su inclusión como regresor sería
    definicional, no causal.

Versión v7 (corrección de lectura del módulo "Datos del hogar y la
vivienda" — fix de KeyError: 'P6008'):
  - DIAGNÓSTICO DEL ERROR ORIGINAL: el módulo "Datos del hogar y la
    vivienda" SÍ existe y SÍ contiene P6008 (confirmado por inspección
    directa de columnas: PERIODO, MES, PER, DIRECTORIO, SECUENCIA_P,
    HOGAR, REGIS, AREA, CLASE, FEX_C18, DPTO, ... P6008). El problema no
    era que la variable no existiera, sino que ese archivo es a nivel de
    HOGAR (su llave natural es solo DIRECTORIO + SECUENCIA_P) y NO trae
    la columna ORDEN, porque la información no varía por persona dentro
    del hogar. La función leer_csv() original exigía las tres columnas
    de KEY (DIRECTORIO, SECUENCIA_P, ORDEN) para aceptar cualquier
    archivo, así que el módulo de hogar fallaba la condición
    `all(k in df.columns for k in key_upper)` en las 9 combinaciones de
    encoding/separador probadas, devolvía None, y se descartaba en
    silencio (mensaje "✗ No se pudo leer"). Como el resto del pipeline no
    lanza excepción en ese punto, el error solo aparecía más adelante,
    al intentar usar P6008 en construir_tasa_dependencia().
  - FIX: se añade leer_csv_hogar(), una función de lectura que solo exige
    DIRECTORIO + SECUENCIA_P (sin ORDEN) para los módulos a nivel de
    hogar. leer_periodo() ahora trata "datos_hogar" como un caso aparte:
    se lee con leer_csv_hogar() y se mergea por la llave reducida
    (DIRECTORIO + SECUENCIA_P), de modo que P6008 se propaga
    correctamente a todas las personas del hogar.
  - Se añade además una verificación explícita y temprana (en
    construir_base) de que todas las columnas declaradas en VARS hayan
    llegado efectivamente al DataFrame final, para que cualquier
    variable perdida en el merge produzca una advertencia clara en
    consola en el momento de construir la base, en lugar de un
    KeyError varios pasos después en una función distinta.

Versión v8 (dos fixes: Hosmer-Lemeshow con empates en la cola, y retiro
de POSICION_* del lado derecho del probit por cuasi-separación):
  - FIX 1 — Hosmer-Lemeshow: pd.qcut(y_pred_prob, 10, ...) fallaba con
    "ValueError: Bin edges must be unique" porque una fracción grande de
    observaciones tiene P(informal) ≈ 1 de forma casi exacta (ver FIX 2
    para la causa). Se agrega duplicates="drop", que colapsa los bordes
    repetidos en menos de 10 grupos efectivos; los grados de libertad del
    test (n_grupos - 2) se recalculan dinámicamente en vez de asumir 8
    fijos, y el gráfico G4 usa range(len(hl_obs)) en vez de range(10).
  - FIX 2 — POSICION_* retirada de FORMULA, VARS_MODELO y formula_2da
    (probit simple y 2ª etapa de Heckman). Diagnóstico: la corrida con
    POSICION_* incluida arrojó "Possibly complete quasi-separation: A
    fraction 0.50 of observations can be perfectly predicted", con
    coeficientes y errores estándar absurdos (cientos a billones) para
    POSICION_GOBIERNO, CUENTA_PROPIA, EMPLEADOR, FAMILIAR_SR y OTRO_SR, y
    "converged: False". Causa raíz: construir_formal() usa P6430
    (posición ocupacional) como insumo DIRECTO de varias ramas de la
    regla de construcción de FORMAL (ej. P6430==2 -> FORMAL=1 siempre;
    cuenta propia/empleador resueltos casi deterministicamente vía
    registro mercantil, contabilidad y rama 84/99). Esto hace que las
    dummies POSICION_* sean, para esas categorías, componentes casi
    definicionales de la variable dependiente, no determinantes causales
    independientes — el mismo problema ya identificado para LN_INGLABO
    (v5) y P3069 (v6), aplicado ahora a POSICION_*.
    Las dummies POSICION_* y construir_ocupacion_posicion_zona_migracion()
    NO se eliminan del DataFrame: siguen disponibles para análisis
    descriptivo o el dashboard de Power BI, solo se retiran de las
    fórmulas de regresión.
"""

import os, glob, warnings, sys, datetime
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

    Esto resuelve el requerimiento de que el .txt de resultados no solo
    contenga los 4 modelos finales (probit simple, 1ª etapa, 2ª etapa de
    Heckman y sus efectos marginales), sino TODO lo que el script imprime
    al ejecutarse línea por línea: progreso de lectura por año/mes,
    verificación de variables esperadas, conteo de ocupados, tasa de
    informalidad, chequeo de balance de exclusión, diagnóstico IMR, VIF,
    Hosmer-Lemeshow, ROC, matriz de confusión, estabilidad por año, etc.
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

BASE_DIR    = r"C:\Users\david\OneDrive\Desktop\TODO\Investigación\Forma micro\Bases de datos"
OUTPUT_DIR  = r"C:\Users\david\OneDrive\Desktop\TODO\Investigación\Forma micro\resultados micro"
AÑOS     = [2022, 2023, 2024, 2025]
MESES    = ["Abril", "Mayo", "Junio"]
KEY      = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]      # llave a nivel de persona
KEY_HOGAR = ["DIRECTORIO", "SECUENCIA_P"]              # llave a nivel de hogar

# ─── MÓDULOS ─────────────────────────────────────────────────────────────────
# "datos_hogar" es el único módulo a nivel de HOGAR (sin ORDEN). Todos los
# demás son a nivel de PERSONA. Esta distinción se usa en leer_periodo().

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
        "P3271",   # Sexo al nacer
        "P3039",   # Hombre/mujer trans
        "P3038",   # Orientación sexual
        "P6080",   # Pertenencia étnica
        "P2057",   # Campesino
        "P3042",   # Nivel educativo
        "P6040",   # Edad
        "P6090",   # Salud subsidiada
        "P6083",   # Madre en el hogar
        "P6100",   # Afiliación salud
        "P6110",   # Tipo afiliación salud
        "P6050",   # Parentesco con jefe de hogar
        "P6070",   # Estado civil
    ],
    "datos_hogar": [
        "P6008",   # Total de personas en el hogar (nivel hogar, sin ORDEN)
    ],
    "fuerza_trabajo": [
        "P6240",   # Actividad principal semana pasada
    ],
    "migracion": [
        "P3384",   # Dónde vivía hace 12 meses (1=no había nacido, 2=este municipio, 3=otro municipio, 4=otro país)
        "P3373",   # ¿Siempre ha vivido en este municipio?
        "P3373S3", # País de nacimiento
        "P3383",   # ¿Hace 1 año vivía en este municipio?
    ],
    "no_ocupados":    [],
    "ocupados": [
        "P6430",   # Posición ocupacional
        "CLASE",   # Zona: 1=Cabecera, 2=Resto (rural)
        "P3069",   # N° empleados
        "P3045S1", # Cámara de comercio
        "P3045S2", # RUT
        "P3045S3", # Personería jurídica
        "P3046",   # Contabilidad (≥2022)
        "P6775",   # Contabilidad (versión anterior)
        "P6765",   # ¿Tiene negocio?
        "P3065",   # Registro mercantil
        "P3066",   # Tipo registro
        "P3067",   # Registro mercantil vigente
        "P3067S1", # Tipo registro mercantil
        "P3067S2", # Año registro
        "P3068",   # Separación gastos hogar-negocio
        "P6450",   # Cotizante/beneficiario
        "P6920",   # Cotiza pensiones
        "P6930",   # Tipo fondo pensión
        "P6940",   # Estado cotización
        "P6990",   # Afiliado ARP
        "OFICIO_C8",   # Código ocupación
        "RAMA2D_R4",   # Rama actividad económica
        "P7130",   # Desea cambiar trabajo
        "INGLABO", # Ingresos laborales
        "P6800",   # Horas trabajadas a la semana
        "P1801S3", # Trabajadores sin remuneración
        "DPTO",    # Departamento
        "FEX_C18", # Factor de expansión
        "PER",     # Período
    ],
    "otras_formas": [
        "P3079S1", # Cuidado de niños menores de 5 años en el propio hogar
        "P3081S1", # Cuidado de personas enfermas/discapacidad/adultos mayores en el propio hogar
    ],
    "otros_ingresos": [
        "P750S2",   # Ayuda en dinero del gobierno
        "P7500S2",  # Recibe jubilación o pensión
        "P7495",    # Ingreso jubilación en el último mes
        "P7505",    # Dinero por ayudas en el último mes
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
    """
    Lectura de módulos a NIVEL DE PERSONA: exige las tres columnas de KEY
    (DIRECTORIO, SECUENCIA_P, ORDEN). No usar para módulos a nivel de
    hogar (ver leer_csv_hogar).
    """
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
    """
    Lectura de módulos a NIVEL DE HOGAR (ej. "Datos del hogar y la
    vivienda"): solo exige DIRECTORIO + SECUENCIA_P, SIN ORDEN, porque la
    información de estos módulos no varía por persona dentro del hogar.

    Este es el fix del KeyError: 'P6008' — leer_csv() exigía ORDEN, que
    este archivo no tiene, así que el módulo se descartaba en silencio.

    Se deduplica por la llave de hogar como salvaguarda, en caso de que
    el archivo fuente tenga alguna fila repetida por hogar.
    """
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

    # ── Módulos a nivel de PERSONA (llave completa DIRECTORIO+SECUENCIA_P+ORDEN) ──
    for modulo, vars_req in VARS.items():
        if modulo in MODULOS_NIVEL_HOGAR:
            continue  # se procesan aparte, más abajo, con llave de hogar
        if modulo not in modulo_paths:
            print(f"  ✗  Módulo ausente: {modulo} ({año}/{mes})")
            continue
        df = leer_csv(modulo_paths[modulo], vars_req)
        if df is None:
            print(f"  ✗  No se pudo leer: {modulo_paths[modulo]}")
            continue
        print(f"  ✓  {modulo:<20} {len(df):>7,} filas")
        if df_periodo is None:
            df_periodo = df
        else:
            df_periodo = pd.merge(df_periodo, df, on=key_upper, how="left")

    # ── Módulos a nivel de HOGAR (llave reducida DIRECTORIO+SECUENCIA_P) ──
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

    df["_OCUPADO_TMP"] = (df["P6240"] == 1).astype(int)
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
    
    # Paso 1 — Construir la variable de exclusión:
    # Se imputan los NaN/missing a 0 (asumiendo "no responde" ≈ "no recibe pensión")
    pension_val = pd.to_numeric(df["P7500S2"], errors="coerce").fillna(0)
    df["RECIBE_PENSION_HOGAR"] = (pension_val == 1).astype(int)
    df["TRANS"] = (
        ((df["P3271"] == 1) & (df["P3039"] == 2)) |  # nació hombre, se reconoce mujer
        ((df["P3271"] == 2) & (df["P3039"] == 1)) |  # nació mujer, se reconoce hombre
        df["P3039"].isin([3, 4])                      # no binario / otra forma
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
               "RAMA_SECCION"]  # <- FIX v11: faltaba esta variable (estaba en FORMULA
                                  #    pero no aquí), causaba NameError en diagnostico()
                                  #    al intentar predict() sin esta columna en X.




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

    # ── FIX v12: umbral óptimo (índice de Youden) en vez de 0.5 fijo ──
    # La curva ROC traza, para cada umbral posible, la Tasa de Verdaderos
    # Positivos (TPR = Recall = Sensibilidad) contra la Tasa de Falsos
    # Positivos (FPR = 1 - Especificidad). El índice de Youden
    # (J = TPR - FPR = Sensibilidad + Especificidad - 1) se maximiza en el
    # punto de la curva ROC más alejado de la diagonal de "no discriminación"
    # — ese es el umbral que mejor balancea ambos tipos de error, en vez de
    # asumir arbitrariamente que 0.5 es el punto de corte adecuado (lo cual
    # rara vez es óptimo cuando, como aquí, las clases no están balanceadas:
    # 62% informal vs 38% formal).
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    ix_optimo = np.argmax(youden_j)
    umbral_optimo = thresholds[ix_optimo]
    recall_optimo = tpr[ix_optimo]          # Sensibilidad = TPR en el umbral óptimo
    especificidad_optimo = 1 - fpr[ix_optimo]  # Especificidad = 1 - FPR en el umbral óptimo

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

    # ── Matriz de confusión y métricas, calculadas en el umbral óptimo ──
    cm = confusion_matrix(y, y_pred_bin)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    recall_cm = tp / (tp + fn) if (tp + fn) > 0 else np.nan          # Recall = Sensibilidad
    especificidad_cm = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # Especificidad
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

    # ── FIX v11: carpeta dedicada para las gráficas individuales ──
    # Cada gráfica del panel se guarda también como archivo independiente
    # en OUTPUT_DIR/graficas, además del panel combinado de 7 subplots que
    # ya se guardaba antes. Esto resuelve el requerimiento de tener cada
    # gráfica por separado para poder usarlas individualmente en la tesis.
    carpeta_graficas = os.path.join(OUTPUT_DIR, "graficas")
    os.makedirs(carpeta_graficas, exist_ok=True)
    rutas_individuales = []

    # G1 — VIF
    # FIX v12: alto dinámico según n° de variables (con 19 dummies de
    # RAMA_SECCION + el resto de regresores, una altura fija de 6" las
    # amontona). 0.28" por barra, con un mínimo de 6" para modelos chicos.
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

    # G2 — Distribución de probabilidades predichas
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

    # G3 — Curva ROC (con el punto óptimo de Youden marcado)
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

    # G4 — Hosmer-Lemeshow
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

    # G5 — Matriz de confusión (umbral óptimo de Youden, con Recall y Especificidad)
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

    # G6 — Residuos vs probabilidades predichas
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

    # G7 — Estabilidad de coeficientes por año
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

    # ── Panel combinado (se conserva igual que antes, para vista general) ──
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

        f.write("── SEGUNDA ETAPA (informalidad corregida por Heckman) ──\n\n")
        f.write(str(probit_heck.summary()))
        f.write("\n\n")
        f.write(str(mfx_heck.summary()))
        f.write("\n\n" + "=" * 70 + "\n\n")

        # FIX v11: nota de referencia cruzada hacia los demás archivos de
        # salida (gráficas individuales y log completo de la ejecución),
        # para que este .txt funcione como índice rápido de todo lo
        # generado en OUTPUT_DIR, sin tener que adivinar nombres de archivo.
        f.write("── OTROS ARCHIVOS GENERADOS EN ESTA EJECUCIÓN ──\n\n")
        f.write(f"Panel combinado de diagnóstico: {os.path.join(OUTPUT_DIR, 'diagnostico_probit.png')}\n")
        f.write(f"Gráficas individuales (carpeta): {os.path.join(OUTPUT_DIR, 'graficas')}\n")
        f.write("   01_vif.png\n")
        f.write("   02_distribucion_probabilidades.png\n")
        f.write("   03_curva_roc.png\n")
        f.write("   04_hosmer_lemeshow.png\n")
        f.write("   05_matriz_confusion.png\n")
        f.write("   06_residuos_vs_prob.png\n")
        f.write("   07_estabilidad_por_anio.png\n")
        f.write("\nEl log completo de TODO lo impreso durante esta ejecución\n")
        f.write("(lectura por año/mes, verificación de variables, VIF, Hosmer-\n")
        f.write("Lemeshow, ROC, diagnóstico IMR, etc.) está en un archivo separado\n")
        f.write("con nombre 'log_completo_ejecucion_<timestamp>.txt' en la misma\n")
        f.write(f"carpeta: {OUTPUT_DIR}\n")

    print(f"Resultados guardados como: {ruta_txt}")

# ─── CORRECCIÓN DE HECKMAN ───────────────────────────────────────────────────

def estimar_heckman(geih, ocupados):
    from scipy.stats import norm

    pet = geih[geih["EDAD"] >= 15].copy()
    pet = recodificar(pet)
    pet["PARTICIPA"] = (pet["P6240"] == 1).astype(int)

    print(f"\n── Chequeo de balance de variables de exclusión por género ──")
    print("   Media de CUIDADO_DEPENDIENTES por género:")
    print(pet.groupby("MUJER")["CUIDADO_DEPENDIENTES"].mean().to_string())

    corr_ninos = pet[["CUIDADO_DEPENDIENTES", "MUJER"]].dropna().corr().iloc[0, 1]
    print(f"\n   Correlación CUIDADO_DEPENDIENTES vs MUJER: {corr_ninos:+.3f}")
    print(f"   (Se usará CUIDADO_DEPENDIENTES como exclusión exógena en la 1ª etapa)")

    n_nan_dep = pet["CUIDADO_DEPENDIENTES"].isna().sum()
    print(f"\n   Missing en CUIDADO_DEPENDIENTES: {n_nan_dep:,} de {len(pet):,}")

    VARS_SEL = ["CUIDADO_DEPENDIENTES", "ETNIA", "CAMPESINO",
                "SUBSIDIO_GOB", "EDUCACION_CAT", "EDAD_C", "EDAD2_C",
                "EN_PAREJA",
                "ZONA_RURAL", "MIGRANTE_RECIENTE", "NACIDO_EXTRANJERO"]
    pet_clean = pet[VARS_SEL + ["PARTICIPA"]].dropna()

    print(f"\n── Primera etapa (selección) ──")
    print(f"   Población en edad de trabajar: {len(pet_clean):,}")
    print(f"   Tasa de participación: {pet_clean['PARTICIPA'].mean():.1%}")

    formula_sel = ("PARTICIPA ~ CUIDADO_DEPENDIENTES + ETNIA + "
                    "CAMPESINO + SUBSIDIO_GOB + C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) + "
                    "EDAD_C + EDAD2_C + EN_PAREJA + "
                    "ZONA_RURAL + MIGRANTE_RECIENTE + NACIDO_EXTRANJERO")
    probit_sel = smf.probit(formula_sel, data=pet_clean).fit()
    print(probit_sel.summary())

    pet_validos = pet[VARS_SEL].dropna()
    try:
        xb_all = probit_sel.predict(pet_validos, which="linear")
    except TypeError:
        xb_all = probit_sel.predict(pet_validos, linear=True)

    imr_all = pd.Series(
        norm.pdf(xb_all) / norm.cdf(xb_all),
        index=pet_validos.index
    )

    print(f"\n   IMR — primeros 5 valores: {imr_all.head().values}")
    print(f"   IMR — media: {imr_all.mean():.4f}  |  std: {imr_all.std():.4f}")

    ocupados_heck = ocupados.copy()
    ocupados_heck["IMR"] = imr_all.reindex(ocupados_heck.index)

    n_sin_imr = ocupados_heck["IMR"].isna().sum()
    if n_sin_imr > 0:
        print(f"   Aviso: {n_sin_imr:,} ocupados sin IMR calculable "
              f"(missing en VARS_SEL) — se excluirán del probit de 2ª etapa")

    print(f"\n── Diagnóstico: correlación IMR vs covariables clave ──")
    diag_corr = ocupados_heck[["IMR", "MUJER", "ETNIA", "CAMPESINO", "SUBSIDIO_GOB"]].dropna()
    corr_imr  = diag_corr.corr()["IMR"].drop("IMR")
    for var, val in corr_imr.items():
        bandera = "⚠ ALTA (posible identificación débil)" if abs(val) > 0.6 else "OK"
        print(f"   IMR vs {var:<13}: {val:+.3f}   {bandera}")

    print(f"\n── Segunda etapa (informalidad corregida) ──")
    formula_2da = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
                     CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO + 
                     C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
                     C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) + 
                     EDAD_C + EDAD2_C +
                     OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
                     EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
                     C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

    probit_heck = smf.probit(formula_2da, data=ocupados_heck).fit()
    print(probit_heck.summary())

    mfx_heck = probit_heck.get_margeff(at="overall")
    print(mfx_heck.summary())

    return probit_sel, probit_heck, ocupados_heck, mfx_heck

def correr_analisis_robustez(geih, ocupados):
    from scipy.stats import norm
    import statsmodels.formula.api as smf

    print("\n" + "="*50)
    print("── CORRIENDO ANÁLISIS DE ROBUSTEZ Y ESPECIFICACIÓN ──")
    print("="*50)

    # 1. Primera etapa (es la misma para todos)
    pet = geih[geih["EDAD"] >= 15].copy()
    pet = recodificar(pet)
    pet["PARTICIPA"] = (pet["P6240"] == 1).astype(int)

    VARS_SEL = ["CUIDADO_DEPENDIENTES", "ETNIA", "CAMPESINO",
                "SUBSIDIO_GOB", "EDUCACION_CAT", "EDAD_C", "EDAD2_C",
                "EN_PAREJA",
                "ZONA_RURAL", "MIGRANTE_RECIENTE", "NACIDO_EXTRANJERO"]
    pet_clean = pet[VARS_SEL + ["PARTICIPA"]].dropna()

    formula_sel = ("PARTICIPA ~ CUIDADO_DEPENDIENTES + ETNIA + "
                    "CAMPESINO + SUBSIDIO_GOB + C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) + "
                    "EDAD_C + EDAD2_C + EN_PAREJA + "
                    "ZONA_RURAL + MIGRANTE_RECIENTE + NACIDO_EXTRANJERO")
    probit_sel = smf.probit(formula_sel, data=pet_clean).fit(disp=0)

    # Calculamos IMR
    pet_validos = pet[VARS_SEL].dropna()
    try:
        xb_all = probit_sel.predict(pet_validos, which="linear")
    except TypeError:
        xb_all = probit_sel.predict(pet_validos, linear=True)
    imr_all = pd.Series(norm.pdf(xb_all) / norm.cdf(xb_all), index=pet_validos.index)

    ocupados_heck = ocupados.copy()
    ocupados_heck["IMR"] = imr_all.reindex(ocupados_heck.index)
    ocupados_heck = ocupados_heck.dropna(subset=["IMR"])

    # Fórmulas de la segunda etapa para los modelos de robustez
    # Base (Modelo 1)
    f_m1 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO + 
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) +  
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

    # Robustez 1: Exclusión de variables del hogar (EN_PAREJA se retira de la 2ª etapa)
    f_m2 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + SUBSIDIO_GOB + DESEA_CAMBIO + 
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) + 
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

    # Robustez 2: Exclusión de SUBSIDIO_GOB (se retira de la 2ª etapa)
    f_m3 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + DESEA_CAMBIO + 
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) + 
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              EN_PAREJA + TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

    # Robustez 1+2 Combinado (Exclusiones familiares + Subsidio)
    f_m4 = """INFORMAL ~ MUJER + MUJER_CUIDADO + TRANS + NO_HETERO + ETNIA +
              CAMPESINO + DESEA_CAMBIO + 
              C(EDUCACION_CAT, Treatment(reference='1_Primaria_Menos')) +
              C(HORAS_CAT, Treatment(reference='2_Completa_35a48')) + 
              EDAD_C + EDAD2_C +
              OCUPACION_CALIF + ZONA_RURAL + MIGRANTE_RECIENTE + TASA_DEPENDENCIA +
              TIENE_ING_NO_LAB + NACIDO_EXTRANJERO +
              C(RAMA_SECCION, Treatment(reference='G_Comerc')) + IMR"""

    print("Estimando Modelo 1 (Base)...")
    m1 = smf.probit(f_m1, data=ocupados_heck).fit(disp=0)
    print("Estimando Modelo 2 (Robustez 1: Exclusiones Hogar)...")
    m2 = smf.probit(f_m2, data=ocupados_heck).fit(disp=0)
    print("Estimando Modelo 3 (Robustez 2: Exclusión Subsidio)...")
    m3 = smf.probit(f_m3, data=ocupados_heck).fit(disp=0)
    print("Estimando Modelo 4 (Robustez Combinada)...")
    m4 = smf.probit(f_m4, data=ocupados_heck).fit(disp=0)

    modelos = [m1, m2, m3, m4]
    variables_reportar = ["MUJER", "MUJER_CUIDADO", "SUBSIDIO_GOB", "TIENE_ING_NO_LAB", "EN_PAREJA", "IMR"]

    lineas_tabla = []
    lineas_tabla.append(f"{'Variable':<22} | {'M1 (Base)':^18} | {'M2 (Rob 1 Hogar)':^18} | {'M3 (Rob 2 Subs)':^18} | {'M4 (Combinado)':^18}")
    lineas_tabla.append("-" * 105)

    for var in variables_reportar:
        coefs_str = []
        pvals_str = []
        for m in modelos:
            if var in m.params:
                coef = m.params[var]
                pval = m.pvalues[var]
                se = m.bse[var]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                coefs_str.append(f"{coef:+.4f}{stars}")
                pvals_str.append(f"({se:.4f})")
            else:
                coefs_str.append("Excluida")
                pvals_str.append("")
        lineas_tabla.append(f"{var:<22} | {coefs_str[0]:^18} | {coefs_str[1]:^18} | {coefs_str[2]:^18} | {coefs_str[3]:^18}")
        lineas_tabla.append(f"{'':<22} | {pvals_str[0]:^18} | {pvals_str[1]:^18} | {pvals_str[2]:^18} | {pvals_str[3]:^18}")

    lineas_tabla.append("-" * 105)
    obs_str = [f"{m.nobs:,}" for m in modelos]
    pr2_str = [f"{m.prsquared:.4f}" for m in modelos]
    ll_str = [f"{m.llf:,.1f}" for m in modelos]
    lineas_tabla.append(f"{'Observaciones':<22} | {obs_str[0]:^18} | {obs_str[1]:^18} | {obs_str[2]:^18} | {obs_str[3]:^18}")
    lineas_tabla.append(f"{'Pseudo R-squared':<22} | {pr2_str[0]:^18} | {pr2_str[1]:^18} | {pr2_str[2]:^18} | {pr2_str[3]:^18}")
    lineas_tabla.append(f"{'Log-Likelihood':<22} | {ll_str[0]:^18} | {ll_str[1]:^18} | {ll_str[2]:^18} | {ll_str[3]:^18}")

    reporte_str = "\n".join(lineas_tabla)
    print("\n" + reporte_str + "\n")

    ruta_rob = os.path.join(OUTPUT_DIR, "reporte_robustez.txt")
    with open(ruta_rob, "w", encoding="utf-8") as f:
        f.write("REPORTE DE ROBUSTEZ Y ESPECIFICACIÓN - HECKMAN INFORMALIDAD\n")
        f.write("="*70 + "\n\n")
        f.write("Notas: Coeficientes reportados con errores estándar en paréntesis.\n")
        f.write("Significancia: *** p<0.01, ** p<0.05, * p<0.1\n\n")
        f.write(reporte_str)
        f.write("\n\n" + "="*70 + "\n")
    print(f"Reporte de robustez guardado en: {ruta_rob}")

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── FIX v11: log completo de TODA la ejecución ──
    # Antes, el .txt final (guardar_resultados_txt) solo contenía los 4
    # modelos (probit simple + mfx, 1ª etapa, 2ª etapa + mfx_heck), porque
    # esa función arma el texto manualmente a partir de los objetos
    # `.summary()`. Todo lo demás que el script imprime por consola al
    # ejecutarse línea por línea (progreso de lectura por año/mes,
    # verificación de variables esperadas, conteo de ocupados, tasa de
    # informalidad, chequeo de balance de exclusión, diagnóstico IMR, VIF,
    # Hosmer-Lemeshow, ROC, matriz de confusión, estabilidad por año, rutas
    # de las gráficas guardadas, etc.) NO quedaba guardado en ningún lado.
    #
    # La clase Tee (definida al inicio del script) duplica cada print()
    # hacia la consola (para que sigas viendo el progreso en vivo) Y hacia
    # este archivo de log, en tiempo real, sin importar en qué función del
    # script ocurra el print(). Se abre en modo "w" con buffering de línea
    # (buffering=1) para que el archivo se vaya escribiendo a medida que
    # avanza la ejecución, no solo al final.
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

        ocupados = geih[geih["P6240"] == 1].copy()
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

        # Correr análisis de robustez y guardar reporte
        correr_analisis_robustez(geih, ocupados)

        diagnostico(probit, ocupados)

        # Este sigue generando el .txt estructurado solo con los 4 modelos
        # (más fácil de citar/copiar en la tesis que el log completo).
        guardar_resultados_txt(probit, mfx, probit_sel, probit_heck, mfx_heck)

        print(f"\n── Ejecución completada sin errores ──")

    except Exception:
        # Si algo falla a mitad de camino, el traceback completo también
        # queda registrado en el log (antes se perdía, porque solo iba a
        # la consola de la terminal/IDE).
        import traceback
        print("\n── ERROR: la ejecución se detuvo por una excepción ──")
        traceback.print_exc(file=sys.stdout)
        raise

    finally:
        # Se restaura stdout y se cierra el archivo SIEMPRE, haya habido
        # error o no, para no dejar el log a medio escribir ni el archivo
        # bloqueado.
        sys.stdout = stdout_original
        log_file.close()
        print(f"\nLog completo de la ejecución guardado en: {ruta_log}")