"""
Modelo de panel — Informalidad macro-territorial (IDC 2025)
Departamento × Año IDC

"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
import statsmodels.api as sm
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects, compare
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import chi2, f as f_dist

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

RUTA_BASE      = r"C:\Users\david\OneDrive\Desktop\TODO\Investigación\Forma macro"
RUTA_DATOS     = os.path.join(RUTA_BASE, "Base_IDC_web_2025-1.xlsx")
CARPETA_SALIDA = os.path.join(RUTA_BASE, "Resultados macro")

# Variables del modelo final de panel (v13).
# v13: especificación sin tejido_empresarial_pca.
# Incluye: capital humano (capital_humano_digital_pca), infraestructura vial
# (infraestructura_territorial_pca), mercado laboral (laboral_pca),
# innovación (innovacion_pca).
VARS_MODELO = ["capital_humano_digital_pca",
               "infraestructura_territorial_pca", "laboral_pca",
               "innovacion_pca"]

# ── Definición de cada bloque PCA: columnas + cuáles son "inversas" ──
# (inversas = mayor valor original significa PEOR desempeño; se multiplican
#  por -1 antes de estandarizar para que el sentido del índice sea uniforme)

# NOTA METODOLÓGICA (v6): institucional_productivo_pca agrupa autonomía
# fiscal (Pilar 1 - Gestión fiscal), profundización financiera (Pilar 10),
# densidad empresarial y participación de medianas/grandes empresas
# (Pilar 8 - Dinámica empresarial, mismo sub-pilar NEG-2) y tamaño del
# mercado interno (Pilar 11). Es la dimensión "institucional-productiva"
# del territorio: capacidad fiscal, financiera y empresarial, separada de
# la dimensión de capital humano/digital (ver siguiente bloque) que en v4
# estaba forzadamente combinada con esta por colinealidad within.
PCA_INSTITUCIONAL_PRODUCTIVO = {
    "nombre": "Capacidad institucional-productiva (excluido del modelo desde v9 — solo diagnóstico)",
    "cols":   ["INS-2-1", "FIN-1-4", "NEG-2-2", "NEG-2-3", "TAM-1-1"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v6): capital_humano_digital_pca agrupa cobertura y
# calidad educativa (media, Saber 11, superior) con infraestructura y
# capacidades TIC. Mantiene la lógica de v4 (educación + TIC combinados
# por colinealidad empírica, r=0.94 en el pool completo) pero ahora aislado
# de las variables institucionales-productivas, que tienen una lógica
# teórica distinta (capacidad fiscal/financiera/empresarial vs. desarrollo
# de capital humano y adopción tecnológica).
PCA_CAPITAL_HUMANO_DIGITAL = {
    "nombre": "Capital humano y transformación digital",
    "cols":   ["EDU-1-4", "EDU-2-1", "EDS-1-1",
               "TIC-1-1", "TIC-1-2", "TIC-1-4",
               "TIC-2-1", "TIC-2-2", "TIC-2-3"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v6): infraestructura_territorial_pca reemplaza a
# cobertura_pca (v3-v5). VARIANTE A (la que entra a VARS_MODELO): solo red
# vial (INF-2-2, INF-2-5), idéntica en composición a la antigua
# cobertura_pca. Se mantiene el método "ponderado" (promedio por carga²)
# heredado de cobertura_pca por consistencia histórica del diagnóstico.
PCA_INFRAESTRUCTURA_TERRITORIAL_A = {
    "nombre": "Infraestructura territorial — Variante A (solo vial)",
    "cols":    ["INF-2-2", "INF-2-5"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v6): VARIANTE B (solo diagnóstico, no entra al modelo
# hasta decisión basada en KMO/Bartlett comparado con la Variante A): vial
# + servicios públicos (acueducto, gas, energía eléctrica, alcantarillado;
# costo de energía como inversa, ya que mayor costo = peor desempeño).
# Conectividad aérea (INF-3-3/3-4) y costos de transporte (INF-3-1/3-2)
# se excluyen por decisión del usuario: conectividad queda fuera del
# esquema de infraestructura territorial en esta versión.
PCA_INFRAESTRUCTURA_TERRITORIAL_B = {
    "nombre": "Infraestructura territorial — Variante B (vial + servicios públicos, solo diagnóstico)",
    "cols":    ["INF-2-2", "INF-2-5",
                "INF-1-1", "INF-1-2", "INF-1-3", "INF-1-5", "INF-1-4"],
    "inversas": ["INF-1-4"],
}

# NOTA METODOLÓGICA (v7): VARIANTE C — "infraestructura_amplia_pca". Pilar 3
# teórico (infraestructura física y conectividad) en su versión más amplia
# posible con el diccionario IDC: Variante B completa + vial restante
# (INF-2-1, INF-2-3, INF-2-4, INF-2-6, que ni A ni B usaban) + conectividad
# y costos de transporte (INF-3-1 a INF-3-4). Esta variante NO tenía "hogar
# conceptual" en v6 (nota explícita del usuario sobre conectividad aérea
# quedando fuera); con el pilar teórico de hoy sí lo tiene. Se diagnostica
# para comparar contra A y B antes de decidir si alguna reemplaza a la otra.
# INF-3-1 e INF-3-2 (costos de transporte) son inversas: mayor costo = peor.
PCA_INFRAESTRUCTURA_AMPLIA = {
    "nombre": "Infraestructura territorial — Variante C (vial completo + servicios + conectividad, candidato v7, solo diagnóstico)",
    "cols":    ["INF-2-1", "INF-2-2", "INF-2-3", "INF-2-4", "INF-2-5", "INF-2-6",
                "INF-1-1", "INF-1-2", "INF-1-3", "INF-1-5", "INF-1-4",
                "INF-3-1", "INF-3-2", "INF-3-3", "INF-3-4"],
    "inversas": ["INF-1-4", "INF-3-1", "INF-3-2"],
}

# NOTA METODOLÓGICA (v7/v9/v9.1): institucional_puro_pca. Pilar 1 teórico (capacidad
# institucional y fiscal) AISLADO de las variables de mercado que en v8 estaban
# en institucional_productivo_pca (FIN-1-4, NEG-2-2, NEG-2-3, TAM-1-1).
# Promovido a VARS_MODELO en v9. Reajustado en v9.1: reducido a un núcleo de 5
# variables (INS-1-1, INS-1-3, INS-2-1, INS-2-3, INS-3-3), logrando KMO=0.77
# y varianza explicada=60.5% (antes KMO=0.65 y 34.4% con 9 variables).
PCA_INSTITUCIONAL_PURO = {
    "nombre": "Capacidad institucional y fiscal (EN EL MODELO desde v9, recalculado con 5 variables)",
    "cols":    ["INS-1-1", "INS-1-3",
                "INS-2-1", "INS-2-3",
                "INS-3-3"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): tic_puro_pca. Pilar 4 teórico (adopción
# tecnológica) AISLADO de educación, para probar si la fusión v4-v6
# (capital_humano_digital_pca, justificada por r=0.94 within en el panel de
# entonces) sigue siendo necesaria con el panel completo 2019-2025. Mismas
# columnas TIC que ya están en capital_humano_digital_pca, pero sin EDU/EDS.
# TIC-1-3 (hogares con computador) se mantiene fuera, igual que en v6, por
# instrucción explícita del usuario.
PCA_TIC_PURO = {
    "nombre": "Adopción tecnológica (TIC) — aislado de educación (candidato v7, solo diagnóstico)",
    "cols":    ["TIC-1-1", "TIC-1-2", "TIC-1-4",
                "TIC-2-1", "TIC-2-2", "TIC-2-3"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): educacion_basica_pca. Pilar 5 teórico (capital
# humano), sub-bloque de educación básica y media. EDU-1-4 y EDU-2-1 ya
# están en capital_humano_digital_pca (bloque actual del modelo); aquí se
# amplía a EDU-1 y EDU-2 completos para ver si el bloque ampliado sostiene
# mejor validez factorial que las dos variables sueltas que se usan hoy.
# Inversas: EDU-1-5 (deserción escolar) y EDU-2-4 (relación estudiantes-
# docentes, donde más estudiantes por docente = peor calidad).
PCA_EDUCACION_BASICA = {
    "nombre": "Educación básica y media — cobertura y calidad (candidato v7, solo diagnóstico)",
    "cols":    ["EDU-1-1", "EDU-1-2", "EDU-1-3", "EDU-1-4", "EDU-1-5",
                "EDU-2-1", "EDU-2-2", "EDU-2-3", "EDU-2-4"],
    "inversas": ["EDU-1-5", "EDU-2-4"],
}

# NOTA METODOLÓGICA (v7): educacion_superior_pca. Pilar 5 teórico, sub-
# bloque de educación superior y formación para el trabajo. EDS-1-1 ya está
# en capital_humano_digital_pca; aquí se amplía a EDS-1, EDS-2 y EDS-3
# completos. EDS-2-4 (dominio de inglés) se construye también como variable
# única aparte (ver ds_ingles_pca) porque mide inserción a mercados
# laborales globalizados, un mecanismo distinto a calidad educativa general
# — se evalúa primero dentro del bloque para ver si correlaciona bien; si
# no, se deja como variable única.
PCA_EDUCACION_SUPERIOR = {
    "nombre": "Educación superior y formación para el trabajo (candidato v7, solo diagnóstico)",
    "cols":    ["EDS-1-1", "EDS-1-2", "EDS-1-3",
                "EDS-2-1", "EDS-2-2", "EDS-2-3", "EDS-2-4",
                "EDS-3-1", "EDS-3-2"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): ds_ingles_pca. Variable única estandarizada
# (mismo tratamiento que tramites_pca en v7 anterior): dominio de inglés
# mide específicamente inserción a mercados laborales formales
# globalizados/exportadores, un mecanismo más estrecho que "calidad de
# educación superior" en general. Se deja fuera de educacion_superior_pca
# y se reporta aparte para comparar variación within de forma aislada.
PCA_DS_INGLES = {
    "nombre": "Dominio de inglés — variable única estandarizada (candidato v7, solo diagnóstico)",
    "cols":    ["EDS-2-4"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): salud_resultados_pca. Pilar 5 teórico, sub-bloque
# de RESULTADOS en salud (distinto de salud_cobertura_pca de la versión v7
# anterior, que mide ACCESO a programas preventivos). Mecanismo teórico
# distinto: cobertura mide la oferta/acceso al sistema; resultados mide el
# desenlace de salud poblacional, que afecta productividad laboral directa
# (Bloom & Canning; capital humano vía salud). Inversas: mortalidad infantil
# y materna (mayor valor = peor desempeño); expectativa de vida no se
# invierte (mayor valor = mejor desempeño, ya en la dirección correcta).
PCA_SALUD_RESULTADOS = {
    "nombre": "Salud — resultados poblacionales (candidato v7, solo diagnóstico)",
    "cols":    ["SAL-2-1", "SAL-2-2", "SAL-2-3"],
    "inversas": ["SAL-2-1", "SAL-2-2"],
}

# NOTA METODOLÓGICA (v7): salud_calidad_pca. Pilar 5 teórico, sub-bloque de
# CALIDAD/capacidad instalada del sistema de salud. Construido por separado
# de salud_oferta_pca (versión v7 anterior, que solo usaba SAL-3-2/3-3/3-4)
# para incluir también SAL-3-1 (comunidad de la salud), no usado antes. Se
# mantiene la distinción ya hecha: oferta instalada es un mecanismo más
# débil/indirecto que cobertura preventiva o resultados de salud.
PCA_SALUD_CALIDAD = {
    "nombre": "Salud — calidad y capacidad instalada (candidato v7, solo diagnóstico, amplía a salud_oferta_pca con SAL-3-1)",
    "cols":    ["SAL-3-1", "SAL-3-2", "SAL-3-3", "SAL-3-4"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): laboral_ampliado_pca. Pilar 6 teórico (eficiencia
# y dinamismo del mercado), versión ampliada del laboral_pca actual (que
# solo usa LAB-1-1 y LAB-1-2). Se agregan LAB-1-4 (subocupación) y LAB-1-5
# (empleo vulnerable), ambas inversas. ADVERTENCIA DE CIRCULARIDAD: LAB-1-3
# (formalidad laboral) es la base de la variable dependiente
# (informalidad = 100 - LAB-1-3) y NUNCA debe incluirse aquí ni en ningún
# otro bloque que se use como regresor — se omite explícitamente.
PCA_LABORAL_AMPLIADO = {
    "nombre": "Mercado laboral ampliado — excl. LAB-1-3 por circularidad con la variable dependiente (candidato v7, solo diagnóstico)",
    "cols":    ["LAB-1-1", "LAB-1-2", "LAB-1-4", "LAB-1-5"],
    "inversas": ["LAB-1-2", "LAB-1-4", "LAB-1-5"],
}

# NOTA METODOLÓGICA (v7): financiero_pca. Pilar 6 teórico, sub-bloque de
# sistema financiero. El bloque actual del modelo (institucional_productivo_pca)
# solo usa FIN-1-4 (profundización financiera); aquí se amplía a FIN-1
# completo (cobertura de establecimientos, cuentas de ahorro, seguros,
# profundización) para ver si el bloque ampliado sostiene mejor validez
# factorial que la variable suelta que se usa hoy.
PCA_FINANCIERO = {
    "nombre": "Sistema financiero — cobertura y profundización (candidato v7, solo diagnóstico, amplía FIN-1-4 ya usado en institucional_productivo_pca)",
    "cols":    ["FIN-1-1", "FIN-1-2", "FIN-1-3", "FIN-1-4"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): mercado_pca. Pilar 6 teórico, sub-bloque de
# tamaño de mercado. El bloque actual solo usa TAM-1-1 (tamaño de mercado
# interno); aquí se amplía a TAM-1 + TAM-2 completos (mercado externo,
# apertura comercial). v6 había descartado explícitamente TAM-2-1/TAM-2-2
# por riesgo de reproducir colinealidad within con TAM-1-1 — ese riesgo
# sigue vigente, este bloque existe para MEDIRLO, no para asumir que no
# aplica.
PCA_MERCADO = {
    "nombre": "Tamaño de mercado — interno y externo (candidato v7, solo diagnóstico, amplía TAM-1-1 ya usado en institucional_productivo_pca; v6 había descartado TAM-2 por riesgo de colinealidad within con TAM-1-1, aquí se mide explícitamente)",
    "cols":    ["TAM-1-1", "TAM-2-1", "TAM-2-2"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v7): sofisticacion_pca. Pilar 7 teórico (innovación y
# sofisticación productiva), sub-bloque de sofisticación exportadora. Mide
# diversificación de mercados de destino y de canasta exportadora — ambas
# inversas (mayor concentración/menor diversificación = peor desempeño en
# el sentido del IDC, por lo que valores altos de "diversificación" deberían
# ser el sentido deseado; se documentan como inversas según el diccionario
# original del usuario, que las marca con "Sí").
PCA_SOFISTICACION = {
    "nombre": "Sofisticación y diversificación exportadora (candidato v7, sin precedente en v3-v6, solo diagnóstico)",
    "cols":    ["SOF-1-1", "SOF-1-2"],
    "inversas": ["SOF-1-1", "SOF-1-2"],
}

# NOTA METODOLÓGICA (v7/v8): innovacion_pca. Pilar 7 teórico, sub-bloque de
# innovación: investigación (INN-1) + registros de propiedad industrial
# (INN-2). Promovido a VARS_MODELO en v8 tras superar el filtro de
# KMO/within y mostrar signo estable y significativo en FE y RE. Sin
# solapamiento conceptual con los 4 regresores previos.
# Se construye con los mismos insumos que en v7; el cambio es solo que ahora
# entra a la regresión.
PCA_INNOVACION = {
    "nombre": "Innovación — investigación y propiedad industrial (EN EL MODELO desde v8)",
    "cols":    ["INN-1-1", "INN-1-2", "INN-1-3", "INN-1-4",
                "INN-2-1", "INN-2-2", "INN-2-3", "INN-2-4"],
    "inversas": [],
}

# NOTA METODOLÓGICA (v10): tejido_empresarial_pca representa la estructura
# y dinámica del tejido empresarial regional (densidad de empresas y peso de medianas/grandes).
PCA_TEJIDO_EMPRESARIAL = {
    "nombre": "Tejido empresarial (densidad y tamaño de empresas)",
    "cols":    ["NEG-2-2", "NEG-2-3"],
    "inversas": [],
}

PCA_LABORAL = {
    "nombre": "Mercado laboral (excl. variables redundantes con informalidad)",
    "cols":    ["LAB-1-1", "LAB-1-2"],
    "inversas": ["LAB-1-2"],  # tasa de desempleo
}

# NOTA METODOLÓGICA (v7): negocios_estructura_pca. Pilar 6 teórico
# (eficiencia y dinamismo del mercado), variables del Pilar 8 (Entorno para
# los negocios) NO usadas en institucional_productivo_pca ni en
# tramites_pca: concentración sectorial (NEG-1-2 secundario, NEG-1-3
# terciario — estructura de la economía departamental entre sectores) y
# tasa de registro empresarial (NEG-2-1 — dinamismo de formación de nuevas
# empresas, distinto del "Dinámica empresarial" de NEG-2-2/2-3 que ya está
# en el bloque actual del modelo). Son tres mecanismos teóricamente
# distintos (estructura sectorial vs. dinamismo de entrada de empresas)
# unidos aquí en un solo bloque diagnóstico SOLO para medir si el KMO los
# sostiene juntos; si no, se dividen en concentracion_sectorial_pca y
# registro_empresarial_pca por separado, mismo patrón que seguridad_pca en
# v2-v3. Inversas: NEG-1-2 y NEG-1-3 marcadas como "Sí" en el diccionario
# del usuario (mayor concentración en un sector = peor desempeño/menor
# diversificación productiva).
PCA_NEGOCIOS_ESTRUCTURA = {
    "nombre": "Estructura y dinamismo empresarial — concentración sectorial y registro (candidato v7, solo diagnóstico)",
    "cols":    ["NEG-1-2", "NEG-1-3", "NEG-2-1"],
    "inversas": ["NEG-1-2", "NEG-1-3"],
}

# NOTA METODOLÓGICA (v4.1): ambiental_pca excluido del modelo definitivamente.
# KMO global=0.36 y varianza CP1=27% indican constructo no válido; coeficiente
# ns en FE (β=0.28, p=0.39) y RE (β=0.37, p=0.27). Se conserva solo para
# diagnóstico PCA en el reporte. Inversas: AMB-1-1 (deforestación), AMB-1-2
# (eficiencia energética), AMB-1-3 (eficiencia en uso del agua).
PCA_AMBIENTAL = {
    "nombre": "Sostenibilidad ambiental (excluido del modelo — solo diagnóstico PCA)",
    "cols":    ["AMB-1-1", "AMB-1-2", "AMB-1-3",
                "AMB-2-1", "AMB-2-2"],
    "inversas": ["AMB-1-1", "AMB-1-2", "AMB-1-3"],
}

# ── Bloques excluidos del modelo (se construyen y reportan, pero no entran
#    a la regresión). Se conservan para el reporte diagnóstico PCA. ──

# criminalidad_pca: excluido en v4. KMO=0.65 aceptable, varianza CP1=60%,
# pero el coeficiente cambió de signo entre Pooled (+1.51**) y FE/RE (~-0.07
# a -0.15, ns) — señal de que el efecto between (departamentos más violentos
# son más informales) es espurio respecto al efecto within (variación anual
# en criminalidad no predice cambios en informalidad dentro del depto).
PCA_CRIMINALIDAD = {
    "nombre": "Criminalidad (excluido del modelo — solo diagnóstico PCA)",
    "cols":    ["INS-4-1", "INS-4-2", "INS-4-3"],
    "inversas": ["INS-4-1", "INS-4-2", "INS-4-3"],
}

# justicia_pca: excluido en v4. KMO=0.47 < 0.50 (umbral mínimo), varianza
# CP1=44%, carga negativa en INS-4-6 — el CP1 no captura una dimensión
# coherente. Coeficiente ns en todos los modelos (p>0.83 en RE).
PCA_JUSTICIA = {
    "nombre": "Eficiencia de la justicia (excluido del modelo — solo diagnóstico PCA)",
    "cols":    ["INS-4-4", "INS-4-5", "INS-4-6"],
    "inversas": [],
}

# ── CANDIDATOS v7 (heredados de la versión anterior) — exploración de
#    determinantes adicionales. Todos se construyen y diagnostican pero NO
#    entran a VARS_MODELO todavía. ──

# salud_cobertura_pca: cobertura de programas preventivos (vacunación,
# controles prenatales). Mecanismo teórico (Levy 2008; Perry et al. 2007):
# alta cobertura de salud pública gratuita reduce el incentivo a buscar
# empleo formal solo para acceder a seguridad social en salud — sustitución
# entre régimen subsidiado y seguro social ligado al empleo. Es una
# aproximación indirecta (no mide directamente afiliación a régimen
# subsidiado/contributivo, que no está disponible en este IDC).
PCA_SALUD_COBERTURA = {
    "nombre": "Salud — cobertura de programas preventivos (candidato v7, solo diagnóstico)",
    "cols":    ["SAL-1-1", "SAL-1-2", "SAL-1-3"],
    "inversas": [],
}

# salud_oferta_pca: oferta médica instalada (médicos, camas). Mecanismo
# teórico más débil e indirecto que cobertura: mide capacidad instalada del
# sistema de salud, que probablemente correlaciona con desarrollo regional
# general ya capturado por institucional_productivo_pca y
# capital_humano_digital_pca, más que con el mecanismo de sustitución
# seguro-social-vs-informalidad. Se construye por separado de cobertura
# (no se mezclan: acceso preventivo y oferta instalada son constructos
# distintos).
# [NOTA v7: ver también salud_calidad_pca arriba, que amplía este mismo
# bloque con SAL-3-1. Se mantienen ambos para comparar si SAL-3-1 aporta
# o diluye el KMO del bloque original.]
PCA_SALUD_OFERTA = {
    "nombre": "Salud — oferta médica instalada (candidato v7, solo diagnóstico)",
    "cols":    ["SAL-3-2", "SAL-3-3", "SAL-3-4"],
    "inversas": [],
}

# tramites_pca: variable única (NEG-1-1, facilitación de trámites). No hay
# con qué construir un PCA de más de una variable — se estandariza sola
# (z-score) y se reporta con el mismo formato de diagnóstico para
# comparabilidad de variación within frente a los demás índices. Mecanismo
# teórico (De Soto; Djankov et al., "Doing Business"): el costo/fricción de
# cumplir con la formalidad regulatoria es uno de los determinantes más
# citados de la decisión de informalidad — distinto de la *capacidad*
# institucional que ya mide institucional_productivo_pca.
PCA_TRAMITES = {
    "nombre": "Trámites empresariales — variable única estandarizada (candidato v7, solo diagnóstico)",
    "cols":    ["NEG-1-1"],
    "inversas": [],
}

# ── Registro completo para construcción de índices ──
# Los bloques excluidos se construyen igual (para el reporte PCA) pero
# no aparecen en VARS_MODELO y no entran a la regresión.
# institucional_puro_pca (v9), infraestructura_territorial_pca (Variante A)
# e innovacion_pca (v8) SÍ entran al modelo; todas las demás variantes/bloques
# son solo diagnóstico.
TODOS_LOS_BLOQUES = [
    ("institucional_puro_pca",             PCA_INSTITUCIONAL_PURO),
    ("capital_humano_digital_pca",        PCA_CAPITAL_HUMANO_DIGITAL),
    ("infraestructura_territorial_pca",   PCA_INFRAESTRUCTURA_TERRITORIAL_A),
    ("laboral_pca",                       PCA_LABORAL),
    ("innovacion_pca",                    PCA_INNOVACION),
    ("tejido_empresarial_pca",            PCA_TEJIDO_EMPRESARIAL),
    # Excluidos del modelo — solo diagnóstico PCA (heredados v4-v6):
    ("infraestructura_territorial_serv_pca", PCA_INFRAESTRUCTURA_TERRITORIAL_B),
    ("ambiental_pca",              PCA_AMBIENTAL),
    ("criminalidad_pca",           PCA_CRIMINALIDAD),
    ("justicia_pca",               PCA_JUSTICIA),
    # Candidatos v7 heredados — exploración de determinantes adicionales:
    ("salud_cobertura_pca",        PCA_SALUD_COBERTURA),
    ("salud_oferta_pca",           PCA_SALUD_OFERTA),
    ("tramites_pca",               PCA_TRAMITES),
    # ── NUEVOS v7 — mapeo a los 7 pilares teóricos (todos solo diagnóstico) ──
    # Pilar 1: institucional-productivo (v8 y anteriores) — ahora solo diagnóstico (v9)
    ("institucional_productivo_pca", PCA_INSTITUCIONAL_PRODUCTIVO),
    # Pilar 3: infraestructura amplia vs. Variantes A/B existentes
    ("infraestructura_amplia_pca", PCA_INFRAESTRUCTURA_AMPLIA),
    # Pilar 4: TIC puro vs. fusión actual (capital_humano_digital_pca)
    ("tic_puro_pca",               PCA_TIC_PURO),
    # Pilar 5: capital humano — sub-bloques separados (NO fusionados)
    ("educacion_basica_pca",       PCA_EDUCACION_BASICA),
    ("educacion_superior_pca",     PCA_EDUCACION_SUPERIOR),
    ("ds_ingles_pca",              PCA_DS_INGLES),
    ("salud_resultados_pca",       PCA_SALUD_RESULTADOS),
    ("salud_calidad_pca",          PCA_SALUD_CALIDAD),
    # Pilar 6: eficiencia y dinamismo del mercado — ampliaciones
    ("laboral_ampliado_pca",       PCA_LABORAL_AMPLIADO),
    ("financiero_pca",             PCA_FINANCIERO),
    ("mercado_pca",                PCA_MERCADO),
    ("negocios_estructura_pca",    PCA_NEGOCIOS_ESTRUCTURA),
    # Pilar 7: sofisticación — sin precedente en v3-v6 (innovacion_pca ya
    # subió a VARS_MODELO en v8, sofisticacion_pca sigue en diagnóstico)
    ("sofisticacion_pca",          PCA_SOFISTICACION),
]

COLS_NUMERICAS = ["LAB-1-3"] + sorted(set(
    col for _, bloque in TODOS_LOS_BLOQUES for col in bloque["cols"]
))


# ─── UTILIDAD: acumulador de texto para el reporte ───────────────────────────

class Reporte:
    """Acumula todo lo que se imprime por consola para volcarlo a un .txt
    al final, sin tener que duplicar cada print."""
    def __init__(self):
        self.lineas = []

    def log(self, *args, sep=" ", end="\n"):
        texto = sep.join(str(a) for a in args)
        import sys
        enc = sys.stdout.encoding or "utf-8"
        try:
            texto_seguro = texto.encode(enc, errors="replace").decode(enc)
        except Exception:
            texto_seguro = texto
        print(texto_seguro, end=end)
        self.lineas.append(texto)

    def guardar(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lineas))


# ─── LECTURA DE DATOS ─────────────────────────────────────────────────────────

def _detectar_skiprows(ruta, hoja):
    """
    Lee las primeras 5 filas del Excel sin saltarse nada y devuelve el número
    de filas a saltar para que la primera fila leída sea la de encabezados.

    Busca la fila cuyo primer valor no numérico sea 'Departamento' o similar,
    o cuyas celdas contengan los códigos IDC ('INS', 'EDU', 'LAB', etc.).
    Si no lo encuentra, devuelve 0 (sin skiprows).
    """
    preview = pd.read_excel(ruta, sheet_name=hoja, header=None, nrows=8)
    for i, row in preview.iterrows():
        vals = [str(v).strip() for v in row.values]
        # La fila de encabezado tiene celdas con texto IDC o 'Departamento'
        texto = " ".join(vals).upper()
        if "DEPARTAMENTO" in texto or "LAB-1" in texto or "INS-2" in texto:
            return i  # saltar las filas anteriores a esta
    return 0  # fallback: leer desde el principio


def _leer_excel_idc(ruta, cols_numericas, rep, etiqueta=""):
    """
    Lee un Excel IDC y devuelve un DataFrame limpio con las columnas numéricas
    casteadas.

    Detecta automáticamente cuántas filas saltar (skiprows) inspeccionando
    las primeras filas del archivo, para tolerar diferencias de estructura
    entre versiones del IDC.

    etiqueta : texto para los mensajes de log.
    """
    hoja = "Dato_original"
    skiprows = _detectar_skiprows(ruta, hoja)
    rep.log(f"   [{etiqueta}] skiprows detectado: {skiprows}")

    df = pd.read_excel(ruta, sheet_name=hoja, skiprows=skiprows, header=0)
    rep.log(f"   [{etiqueta}] Primeras columnas: {list(df.columns[:6])}")

    # ── Detectar columna de año ──
    col_anio = None
    for c in df.columns:
        c_norm = str(c).strip().lower().replace("á", "a").replace("ñ", "n")
        if c_norm in ("ano", "year", "ano idc") or "ano idc" in c_norm or "year idc" in c_norm:
            col_anio = c
            break
    if col_anio is None:
        raise ValueError(
            f"[{etiqueta}] No se encontró columna de año en {ruta}.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )
    rep.log(f"   [{etiqueta}] Columna de año detectada: '{col_anio}'")
    if col_anio != "Año IDC":
        df = df.rename(columns={col_anio: "Año IDC"})

    # ── Detectar columna de departamento ──
    col_depto = None
    for c in df.columns:
        c_norm = str(c).strip().lower()
        if "departamento" in c_norm or "depto" in c_norm:
            col_depto = c
            break
    if col_depto is not None and col_depto != "Departamento":
        df = df.rename(columns={col_depto: "Departamento"})

    # ── Castear columnas numéricas ──
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            rep.log(f"   [!] [{etiqueta}] Columna no encontrada: {col}")

    return df


def cargar_datos(rep):
    """
    Carga el panel IDC 2019-2025 (Base_IDC_web_2025-1.xlsx).

    NOTA (v5): el año 2018 (IDC 1.0) se excluyó deliberadamente. El IDC 2018
    y el IDC 2019-2025 (IDC 2.0, vigente desde 2020) no son metodológicamente
    comparables: distinta estructura de factores/pilares (3/10 vs 4/13),
    distinta cobertura departamental (27 vs 33), distinta ponderación
    (por etapa de desarrollo vs promedio simple de pilares), e insumos
    recalculados en variables que entran directamente al modelo (red vial,
    parámetros educativos, TIC, entorno de negocios, mercado laboral,
    sistema financiero). La fuente metodológica declara explícitamente que
    2018 "no es comparable" con las versiones posteriores. Mezclar ambas
    arquitecturas de índice en un mismo PCA y panel de efectos fijos
    introduciría un quiebre estructural espurio justo en la dimensión
    within que identifica al estimador FE.
    """
    cols_num = COLS_NUMERICAS

    df = _leer_excel_idc(RUTA_DATOS, cols_num, rep, etiqueta="2019-2025")

    df = df.sort_values(["Departamento", "Año IDC"]).reset_index(drop=True)

    # ── Variable dependiente ──
    df["informalidad"] = 100 - df["LAB-1-3"]

    # ── Reporte de cobertura ──
    rep.log(f"\nPanel final — filas totales : {len(df):,}")
    rep.log(f"Departamentos               : {df['Departamento'].nunique()}")
    rep.log(f"Años IDC                    : {sorted(df['Año IDC'].unique())}")
    rep.log(f"Obs. con informalidad válida: {df['informalidad'].notna().sum():,}")

    # Cobertura por variable clave
    rep.log("\nCobertura por variable del modelo (obs. no-NaN):")
    for col in ["INS-2-1", "FIN-1-4", "NEG-2-2", "EDU-1-4", "TIC-1-1",
                "INF-2-2", "LAB-1-1", "LAB-1-2", "LAB-1-3",
                "INN-1-1", "INN-1-2", "INN-2-1", "INN-2-2"]:
        if col in df.columns:
            rep.log(f"  {col:<12}: {df[col].notna().sum():>3} / {len(df)}")

    # Cobertura de candidatos v7 (heredados: salud, trámites)
    rep.log("\nCobertura de candidatos v7 heredados (obs. no-NaN):")
    for col in ["SAL-1-1", "SAL-1-2", "SAL-1-3",
                "SAL-3-2", "SAL-3-3", "SAL-3-4", "NEG-1-1"]:
        if col in df.columns:
            rep.log(f"  {col:<12}: {df[col].notna().sum():>3} / {len(df)}")

    # Cobertura de variables nuevas v7 (mapeo a 7 pilares teóricos)
    rep.log("\nCobertura de variables nuevas v7 — 7 pilares teóricos (obs. no-NaN):")
    cols_nuevas_v7 = sorted(set(
        col
        for nombre in ("institucional_puro_pca", "infraestructura_amplia_pca",
                       "tic_puro_pca", "educacion_basica_pca",
                       "educacion_superior_pca", "ds_ingles_pca",
                       "salud_resultados_pca", "salud_calidad_pca",
                       "laboral_ampliado_pca", "financiero_pca", "mercado_pca",
                       "negocios_estructura_pca",
                       "sofisticacion_pca", "innovacion_pca")
        for col in dict(TODOS_LOS_BLOQUES)[nombre]["cols"]
    ))
    for col in cols_nuevas_v7:
        if col in df.columns:
            rep.log(f"  {col:<12}: {df[col].notna().sum():>3} / {len(df)}")

    # Advertir departamentos con menos de 7 períodos (2019-2025)
    n_anios_esperados = df["Año IDC"].nunique()
    conteo = df.groupby("Departamento")["Año IDC"].count()
    incompletos = conteo[conteo < n_anios_esperados]
    if not incompletos.empty:
        rep.log(f"\n   [!] Departamentos con menos de {n_anios_esperados} períodos:")
        rep.log(incompletos.to_string())

    return df


# ─── CONSTRUCCIÓN GENÉRICA DE PCA (reutilizable para todos los bloques) ──────

def construir_pca(df, nombre_var, bloque, rep, metodo="pca"):
    """
    Construye un índice a partir de un bloque de columnas IDC.

    nombre_var : nombre de la columna resultante (ej. 'laboral_pca')
    bloque     : dict con 'nombre', 'cols', 'inversas'
    metodo     : 'pca' (componente principal) o 'ponderado' (promedio
                 ponderado por carga al cuadrado, como cobertura_pca)

    DISEÑO DE SIGNO ESTABLE
    ───────────────────────
    El PCA se AJUSTA una sola vez sobre el pool completo de observaciones
    válidas (todas las filas sin NaN en las columnas del bloque). Luego se
    PROYECTAN todas las observaciones sobre ese componente fijo.

    Ancla de signo: después de ajustar, se verifica que la suma de cargas
    sea positiva. Si es negativa, se multiplican cargas y scores por -1,
    de modo que "mayor valor del índice = mejor desempeño" sea siempre
    consistente con la orientación semántica del bloque.

    Devuelve df actualizado y un dict con diagnóstico (KMO, Bartlett,
    varianza explicada, cargas, correlaciones internas).
    """
    cols = bloque["cols"]
    cols_presentes = [c for c in cols if c in df.columns]
    if len(cols_presentes) < len(cols):
        faltantes = set(cols) - set(cols_presentes)
        rep.log(f"   ⚠ Columnas faltantes para {nombre_var}: {faltantes}")
    cols = cols_presentes

    # ── CASO ESPECIAL: una sola columna (ej. NEG-1-1, EDS-2-4, sin par
    # para PCA) ──
    # No hay matriz multivariada con la que construir un componente principal
    # real ni con la que calcular KMO/Bartlett (ambos requieren ≥2 variables
    # correlacionadas entre sí). El índice resultante es simplemente el
    # z-score de la variable, reportado bajo el mismo formato para que sea
    # comparable en variación within con los demás índices del modelo.
    if len(cols) == 1:
        X_raw = df[cols].dropna()
        X = X_raw.copy()
        for col in bloque["inversas"]:
            if col in X.columns:
                X[col] = -X[col]
        scaler = StandardScaler()
        valores = scaler.fit_transform(X).flatten()
        df.loc[X_raw.index, nombre_var] = valores

        diag = {
            "nombre": bloque["nombre"], "cols": cols, "n_obs": len(X_raw),
            "kmo_total": np.nan, "kmo_por_var": pd.Series(dtype=float),
            "bartlett_stat": np.nan, "bartlett_p": np.nan,
            "varianza_explicada": 1.0,
            "cargas": pd.Series([1.0], index=cols),
            "corr_interna": df[cols].corr().round(3),
        }

        rep.log(f"\n### {nombre_var} · {bloque['nombre']} ###")
        rep.log("Método: variable única estandarizada (z-score) — NO es PCA real,")
        rep.log("        no hay matriz multivariada para componente principal ni KMO/Bartlett.")
        rep.log(f"Columnas: {cols}")
        if bloque["inversas"]:
            rep.log(f"Invertidas (×-1 antes de estandarizar): {bloque['inversas']}")
        rep.log(f"N observaciones válidas: {diag['n_obs']:,}")
        rep.log("(KMO/Bartlett no aplican: se requieren >=2 variables)")

        return df, diag

    # ── Preparar matriz: invertir columnas "inversas" antes de estandarizar ──
    X_raw = df[cols].dropna()
    X = X_raw.copy()
    for col in bloque["inversas"]:
        if col in X.columns:
            X[col] = -X[col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Ajustar PCA UNA VEZ sobre el pool completo ──
    pca_aux = PCA(n_components=1)
    pca_aux.fit(X_scaled)
    cargas = pca_aux.components_[0].copy()

    # ── Ancla de signo: suma de cargas positiva ──
    if cargas.sum() < 0:
        cargas = -cargas
        rep.log(f"   [signo] {nombre_var}: cargas invertidas para que suma > 0")

    # ── Proyectar todas las obs. sobre el componente fijo ──
    if metodo == "pca":
        # Proyección manual con las cargas ancladas (en vez de transform())
        valores = X_scaled @ cargas
    else:  # "ponderado"
        pesos = cargas**2 / (cargas**2).sum()
        valores = X_scaled @ pesos

    df.loc[X_raw.index, nombre_var] = valores

    # ── Diagnóstico KMO + Bartlett ──
    diag = {"nombre": bloque["nombre"], "cols": cols, "n_obs": len(X_raw)}
    if len(cols) >= 2:
        try:
            kmo_por_var, kmo_total = calculate_kmo(X)
            bartlett_stat, bartlett_p = calculate_bartlett_sphericity(X)
            diag["kmo_total"]     = float(kmo_total)
            diag["kmo_por_var"]   = pd.Series(kmo_por_var, index=cols)
            diag["bartlett_stat"] = bartlett_stat
            diag["bartlett_p"]    = bartlett_p
        except Exception as e:
            rep.log(f"   ⚠ No se pudo calcular KMO/Bartlett para {nombre_var}: {e}")
            diag["kmo_total"]     = np.nan
            diag["kmo_por_var"]   = pd.Series(dtype=float)
            diag["bartlett_stat"] = np.nan
            diag["bartlett_p"]    = np.nan
    else:
        diag["kmo_total"]     = np.nan
        diag["kmo_por_var"]   = pd.Series(dtype=float)
        diag["bartlett_stat"] = np.nan
        diag["bartlett_p"]    = np.nan

    diag["varianza_explicada"] = pca_aux.explained_variance_ratio_[0]
    diag["cargas"]             = pd.Series(cargas, index=cols)
    diag["corr_interna"]       = df[cols].corr().round(3)

    # ── Log al reporte ──
    rep.log(f"\n### {nombre_var} · {bloque['nombre']} ###")
    rep.log(f"Método: {'PCA puro (cargas fijas en pool, signo anclado)' if metodo == 'pca' else 'Promedio ponderado por carga²'}")
    rep.log(f"Columnas: {cols}")
    if bloque["inversas"]:
        rep.log(f"Invertidas (×-1 antes de estandarizar): {bloque['inversas']}")
    rep.log(f"N observaciones válidas: {diag['n_obs']:,}")
    rep.log(f"Varianza explicada (CP1): {diag['varianza_explicada']:.4f}")
    if not np.isnan(diag["kmo_total"]):
        rep.log(f"KMO global: {diag['kmo_total']:.4f}")
        rep.log("KMO por variable:")
        rep.log(diag["kmo_por_var"].round(4).to_string())
        rep.log(f"Bartlett: chi2={diag['bartlett_stat']:.4f}  p={diag['bartlett_p']:.4g}")
    rep.log("Cargas factoriales (ancladas):")
    rep.log(diag["cargas"].to_string())
    rep.log("Correlaciones internas:")
    rep.log(diag["corr_interna"].to_string())

    return df, diag


# ─── COMPARACIÓN GENÉRICA ENTRE VARIANTES DE UN MISMO PILAR ──────────────────

def comparar_variantes(diagnosticos_pca, nombre_actual, nombre_candidato,
                        etiqueta_actual, etiqueta_candidato, rep,
                        titulo="COMPARACIÓN DE VARIANTES"):
    """
    Versión genérica de comparar_variantes_infraestructura (v6), reutilizada
    para cualquier par (bloque actual del modelo, bloque candidato nuevo).

    Imprime tabla comparativa de KMO/Bartlett/varianza explicada/N variables/
    N obs., señala con qué variables nuevas aparece el candidato, y aplica la
    misma regla de decisión sugerida (no automática) que v6 usó para
    infraestructura A vs B:

      - Preferir el candidato si su KMO global mejora, las variables nuevas
        tienen KMO≥0.50, y la varianza explicada no cae más de un 20%
        respecto al bloque actual.
      - Mantener el bloque actual en caso contrario.

    No decide nada por sí sola: solo deja la comparación lista en el reporte.
    """
    diag_actual = diagnosticos_pca.get(nombre_actual)
    diag_cand   = diagnosticos_pca.get(nombre_candidato)
    if diag_actual is None or diag_cand is None:
        return

    rep.log(f"\n{'='*70}")
    rep.log(f"{titulo}: {etiqueta_actual} (actual) vs {etiqueta_candidato} (candidato)")
    rep.log(f"{'='*70}")
    rep.log(f"{'':<30} {etiqueta_actual:>22} {etiqueta_candidato:>30}")
    rep.log(f"{'-'*30} {'-'*22} {'-'*30}")
    rep.log(f"{'KMO global':<30} {diag_actual['kmo_total']:>22.4f} {diag_cand['kmo_total']:>30.4f}")
    rep.log(f"{'Bartlett p-value':<30} {diag_actual['bartlett_p']:>22.4g} {diag_cand['bartlett_p']:>30.4g}")
    rep.log(f"{'Varianza explicada (CP1)':<30} {diag_actual['varianza_explicada']:>22.4f} {diag_cand['varianza_explicada']:>30.4f}")
    rep.log(f"{'N variables':<30} {len(diag_actual['cols']):>22d} {len(diag_cand['cols']):>30d}")
    rep.log(f"{'N obs. válidas':<30} {diag_actual['n_obs']:>22,} {diag_cand['n_obs']:>30,}")

    cols_nuevas = set(diag_cand["cols"]) - set(diag_actual["cols"])
    if cols_nuevas and not diag_cand["kmo_por_var"].empty:
        rep.log(f"\nKMO por variable — {etiqueta_candidato} (variables nuevas resaltadas):")
        for col, val in diag_cand["kmo_por_var"].items():
            marca = "  <- variable nueva" if col in cols_nuevas else ""
            bandera = "  [!] KMO < 0.50" if val < 0.50 else ""
            rep.log(f"  {col:<12}: {val:.4f}{marca}{bandera}")

    rep.log(f"\n{'-'*60}")
    rep.log("RECOMENDACIÓN (basada en regla de decisión, NO aplicada automáticamente):")
    rep.log(f"{'-'*60}")
    kmo_nuevas_ok = (diag_cand["kmo_por_var"].empty or
                      all(diag_cand["kmo_por_var"].get(c, 0) >= 0.50 for c in cols_nuevas))
    kmo_cand_valido = not np.isnan(diag_cand["kmo_total"])
    if (kmo_cand_valido and diag_cand["kmo_total"] > diag_actual["kmo_total"] and kmo_nuevas_ok
            and diag_cand["varianza_explicada"] >= diag_actual["varianza_explicada"] * 0.8):
        rep.log(f"  -> {etiqueta_candidato} parece preferible: KMO global mejora, las variables")
        rep.log("    nuevas tienen KMO>=0.50, y la varianza explicada no cae más de un 20%")
        rep.log(f"    respecto a {etiqueta_actual}. Considerar promover a VARS_MODELO.")
    else:
        razon = []
        if not kmo_cand_valido or diag_cand["kmo_total"] <= diag_actual["kmo_total"]:
            razon.append("el KMO global no mejora (o el candidato no tiene KMO válido)")
        if not kmo_nuevas_ok:
            razon.append("alguna variable nueva tiene KMO<0.50")
        if diag_cand["varianza_explicada"] < diag_actual["varianza_explicada"] * 0.8:
            razon.append("la varianza explicada cae más de un 20%")
        rep.log(f"  -> Mantener {etiqueta_actual} en el modelo: {', '.join(razon)}.")


def comparar_variantes_infraestructura(diagnosticos_pca, rep):
    """
    Wrapper específico (compatibilidad v6): compara la Variante A (vial,
    en VARS_MODELO) contra la Variante B (vial+servicios, diagnóstico) y,
    en v7, también contra la Variante C (infraestructura amplia, Pilar 3
    teórico completo).
    """
    comparar_variantes(
        diagnosticos_pca,
        "infraestructura_territorial_pca", "infraestructura_territorial_serv_pca",
        "Variante A (vial)", "Variante B (vial+servicios)", rep,
        titulo="COMPARACIÓN — Infraestructura territorial: A vs B",
    )
    comparar_variantes(
        diagnosticos_pca,
        "infraestructura_territorial_pca", "infraestructura_amplia_pca",
        "Variante A (vial)", "Variante C (Pilar 3 completo)", rep,
        titulo="COMPARACIÓN — Infraestructura territorial: A vs C (v7, pilar teórico amplio)",
    )


def comparar_pilares_v7(diagnosticos_pca, rep):
    """
    Corre todas las comparaciones nuevas de v7: cada bloque actual del
    modelo contra su candidato teórico equivalente. Misma lógica que
    comparar_variantes_infraestructura, generalizada a los 7 pilares.
    """
    rep.log(f"\n{'#'*70}")
    rep.log("# COMPARACIONES v7 — bloques actuales del modelo vs. pilares teóricos")
    rep.log(f"{'#'*70}")

    # Pilar 1: institucional puro (actual v9.1) vs. institucional-productivo (anterior v8)
    comparar_variantes(
        diagnosticos_pca,
        "institucional_puro_pca", "institucional_productivo_pca",
        "Actual (INS puro 5 vars, v9.1)", "Anterior (institucional+mercado, v8)", rep,
        titulo="COMPARACIÓN — Pilar 1: institucional puro (actual v9.1) vs institucional-productivo (anterior v8)",
    )

    # Pilar 4: TIC puro vs. capital_humano_digital_pca (actual, fusión con educación)
    comparar_variantes(
        diagnosticos_pca,
        "capital_humano_digital_pca", "tic_puro_pca",
        "Actual (TIC+educación fusionados)", "Pilar 4 puro (solo TIC)", rep,
        titulo="COMPARACIÓN — Pilar 4: capital humano+digital (actual) vs TIC puro (candidato)",
    )

    # Pilar 6: laboral ampliado vs. laboral_pca (actual)
    comparar_variantes(
        diagnosticos_pca,
        "laboral_pca", "laboral_ampliado_pca",
        "Actual (LAB-1-1, LAB-1-2)", "Pilar 6 ampliado (+subocupación+vulnerable)", rep,
        titulo="COMPARACIÓN — Pilar 6: laboral (actual) vs laboral ampliado (candidato)",
    )

    # Pilar 6: financiero ampliado vs. FIN-1-4 dentro de institucional_productivo_pca
    diag_fin = diagnosticos_pca.get("financiero_pca")
    if diag_fin is not None and not np.isnan(diag_fin["kmo_total"]):
        rep.log(f"\n{'-'*60}")
        rep.log("NOTA — Pilar 6: sistema financiero ampliado (financiero_pca)")
        rep.log(f"{'-'*60}")
        rep.log(f"  KMO global: {diag_fin['kmo_total']:.4f}  "
                f"Varianza explicada: {diag_fin['varianza_explicada']:.4f}")
        rep.log("  FIN-1-4 (única variable financiera en el modelo actual) no tiene")
        rep.log("  KMO propio fuera de su fusión en institucional_productivo_pca.")
        rep.log("  Evaluar si financiero_pca debe reemplazar a FIN-1-4 dentro de ese bloque,")
        rep.log("  o entrar como regresor independiente del Pilar 6.")

    # Pilar 6: mercado ampliado (TAM completo)
    diag_mkt = diagnosticos_pca.get("mercado_pca")
    if diag_mkt is not None and not np.isnan(diag_mkt["kmo_total"]):
        rep.log(f"\n{'-'*60}")
        rep.log("NOTA — Pilar 6: tamaño de mercado ampliado (mercado_pca)")
        rep.log(f"{'-'*60}")
        rep.log(f"  KMO global: {diag_mkt['kmo_total']:.4f}  "
                f"Varianza explicada: {diag_mkt['varianza_explicada']:.4f}")
        rep.log("  TAM-1-1 (única variable de mercado en el modelo actual) no tiene")
        rep.log("  KMO propio fuera de su fusión en institucional_productivo_pca.")
        rep.log("  v6 descartó TAM-2-1/TAM-2-2 por riesgo de colinealidad within con")
        rep.log("  TAM-1-1 — revisar si ese riesgo se confirma con el KMO/within de aquí.")


# ─── EVALUACIÓN DE CANDIDATOS NUEVOS (heredado v7 anterior + extendido) ──────

def evaluar_candidatos_v7(df, diagnosticos_pca, rep):
    """
    Compara TODOS los candidatos nuevos de v7 (heredados: salud, trámites;
    nuevos: los 13 bloques de los 7 pilares teóricos) contra los cinco
    regresores actuales del modelo en dos dimensiones clave:

      1. Validez factorial (KMO/Bartlett).
      2. Variación WITHIN (std promedio dentro de cada departamento).

    innovacion_pca e institucional_puro_pca ya están en VARS_MODELO (v8/v9)
    y se incluyen en la columna de referencia, no en la lista de candidatos.
    """
    candidatos = [
        # heredados
        "salud_cobertura_pca", "salud_oferta_pca", "tramites_pca",
        # nuevos v7 — 7 pilares teóricos (innovacion_pca e institucional_puro_pca excluidos: ya son regresores)
        "infraestructura_amplia_pca", "tic_puro_pca",
        "educacion_basica_pca", "educacion_superior_pca", "ds_ingles_pca",
        "salud_resultados_pca", "salud_calidad_pca",
        "laboral_ampliado_pca", "financiero_pca", "mercado_pca",
        "negocios_estructura_pca",
        "sofisticacion_pca",
    ]
    candidatos = [c for c in candidatos if c in diagnosticos_pca]
    if not candidatos:
        return

    rep.log(f"\n{'='*70}")
    rep.log("EVALUACIÓN DE CANDIDATOS v7 — nuevos determinantes potenciales")
    rep.log(f"{'='*70}")

    # ── Variación within: candidatos vs regresores actuales del modelo ──
    df_idx = df.set_index(["Departamento", "Año IDC"]) if "Departamento" in df.columns else df
    todas_vars = list(dict.fromkeys(VARS_MODELO + candidatos))
    todas_vars = [v for v in todas_vars if v in df_idx.columns]
    within_var = df_idx.groupby(level="Departamento")[todas_vars].std().mean()

    rep.log(f"\n{'─'*60}")
    rep.log("VARIACIÓN WITHIN — candidatos v7 vs regresores actuales")
    rep.log(f"{'─'*60}")
    rep.log("(Referencia: institucional_productivo_pca≈0.23 e ")
    rep.log(" infraestructura_territorial_pca≈0.07 ya muestran within bajo")
    rep.log(" y coeficientes inestables/no significativos en FE.)\n")
    for var in todas_vars:
        val = within_var.get(var, np.nan)
        es_candidato = " (en el modelo)" if var in VARS_MODELO else " ← candidato v7"
        bandera = "  ⚠ within muy bajo" if val < 0.10 else ("  ⚠ within bajo" if val < 0.25 else "")
        rep.log(f"  {var:<35} {val:.6f}{es_candidato}{bandera}")

    # ── Resumen de KMO/Bartlett de los candidatos con ≥2 variables ──
    rep.log(f"\n{'─'*60}")
    rep.log("VALIDEZ FACTORIAL — candidatos v7 (solo bloques con ≥2 variables)")
    rep.log(f"{'─'*60}")
    for nombre_var in candidatos:
        diag = diagnosticos_pca[nombre_var]
        if np.isnan(diag["kmo_total"]):
            rep.log(f"  {nombre_var:<25}: KMO/Bartlett no aplica (variable única o falló el cálculo)")
            continue
        bandera_kmo = ("  🔴 KMO<0.50 (inaceptable)" if diag["kmo_total"] < 0.50
                       else "  🟡 KMO marginal (0.50-0.60)" if diag["kmo_total"] < 0.60
                       else "  🟢 KMO aceptable")
        rep.log(f"  {nombre_var:<25}: KMO={diag['kmo_total']:.4f}  "
                f"var.explicada={diag['varianza_explicada']:.4f}  "
                f"Bartlett p={diag['bartlett_p']:.4g}{bandera_kmo}")

    # ── Recomendación textual (no automática) ──
    rep.log(f"\n{'─'*60}")
    rep.log("LECTURA SUGERIDA (decisión manual, no aplicada al modelo)")
    rep.log(f"{'─'*60}")
    for var in candidatos:
        val = within_var.get(var, np.nan)
        if np.isnan(val):
            continue
        if val < 0.10:
            rep.log(f"  {var}: within muy bajo ({val:.4f}) — probablemente no aportará")
            rep.log(f"    identificación adicional en FE, por la misma razón que")
            rep.log(f"    infraestructura_territorial_pca ya muestra coeficiente inestable.")
        elif val < 0.25:
            rep.log(f"  {var}: within bajo ({val:.4f}), similar a institucional_productivo_pca.")
            rep.log(f"    Podría entrar, pero anticipa el mismo problema de no-significancia en FE.")
        else:
            rep.log(f"  {var}: within razonable ({val:.4f}) — candidato viable para incorporar")
            rep.log(f"    si la teoría y el KMO (cuando aplique) lo respaldan.")


# ─── PRUEBA DE CANDIDATOS SELECCIONADOS COMO REGRESORES ADICIONALES ─────────
#
# innovacion_pca fue promovido en v8; institucional_puro_pca reemplazó a
# institucional_productivo_pca en v9. Ninguno aparece aquí.
# Se mantienen institucional_productivo_pca (para evaluar si debería volver)
# y tramites_pca para evaluación futura si el modelo se amplía a 6 regresores.

CANDIDATOS_A_PROBAR = ["institucional_productivo_pca", "tramites_pca", "tejido_empresarial_pca"]


def evaluar_candidato_como_regresor(df_indexado, y, candidato, rep):
    """
    Corre Pooled OLS, FE y RE sobre VARS_MODELO + [candidato] (modelo base
    + un regresor adicional) y reporta coeficiente, error estándar, p-valor
    y signo del candidato en los tres modelos, más un chequeo de VIF para
    detectar si el candidato introduce multicolinealidad relevante con los
    cinco regresores ya presentes.

    df_indexado : DataFrame ya indexado por (Departamento, Año IDC), con
                  todas las columnas PCA de TODOS_LOS_BLOQUES ya construidas
                  (el mismo df que devuelve estimar_modelos()).
    y           : Serie "informalidad", panel-indexada igual que df_indexado.
    candidato   : nombre de la columna candidata a probar (str).

    No modifica VARS_MODELO. Solo deja el reporte listo para decisión manual.
    """
    vars_extendido = VARS_MODELO + [candidato]
    X = sm.add_constant(df_indexado[vars_extendido]).dropna()
    y_alineado = y.loc[X.index]

    rep.log(f"\n{'='*70}")
    rep.log(f"PRUEBA DE REGRESOR ADICIONAL: {candidato}")
    rep.log(f"{'='*70}")
    rep.log(f"Modelo base (5 regresores actuales) + {candidato}")
    rep.log(f"N obs. válidas tras dropna conjunto: {len(X):,}")

    # --- VIF del modelo extendido ---
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    vif_data["Tolerancia"] = 1 / vif_data["VIF"]
    rep.log("\nVIF (modelo base + candidato):")
    rep.log(vif_data.to_string(index=False))
    vif_candidato = vif_data.loc[vif_data["Variable"] == candidato, "VIF"].iloc[0]
    if vif_candidato > 10:
        rep.log(f"  🔴 VIF del candidato ({vif_candidato:.2f}) > 10 — multicolinealidad alta, "
                f"interpretar coeficiente con cautela.")
    elif vif_candidato > 5:
        rep.log(f"  🟡 VIF del candidato ({vif_candidato:.2f}) moderado (5-10).")
    else:
        rep.log(f"  🟢 VIF del candidato ({vif_candidato:.2f}) OK (≤5).")

    # --- Pooled / FE / RE ---
    pooled = PooledOLS(y_alineado, X).fit()
    fe = PanelOLS(y_alineado, X, entity_effects=True).fit(cov_type="robust")
    re = RandomEffects(y_alineado, X).fit(cov_type="robust")

    modelos = {"Pooled": pooled, "FE": fe, "RE": re}
    filas = []
    for nombre_modelo, m in modelos.items():
        coef = m.params[candidato]
        se = m.std_errors[candidato]
        pval = m.pvalues[candidato]
        filas.append({
            "Modelo": nombre_modelo,
            "Coef.": coef,
            "Std.Err.": se,
            "p-value": pval,
            "Signo": "+" if coef > 0 else "−",
            "Sig. (5%)": "***" if pval < 0.05 else ("." if pval < 0.10 else ""),
        })
    tabla = pd.DataFrame(filas)

    rep.log(f"\nCoeficiente de {candidato} en cada modelo:")
    rep.log(tabla.to_string(index=False))

    # --- Estabilidad de signo ---
    signos = set(tabla["Signo"])
    signo_estable = len(signos) == 1
    rep.log(f"\nEstabilidad de signo entre Pooled/FE/RE: "
            f"{'ESTABLE (mismo signo en los 3)' if signo_estable else 'INESTABLE (cambia de signo)'}")

    # --- Significancia en FE/RE ---
    sig_fe = tabla.loc[tabla["Modelo"] == "FE", "p-value"].iloc[0] < 0.05
    sig_re = tabla.loc[tabla["Modelo"] == "RE", "p-value"].iloc[0] < 0.05
    rep.log(f"Significativo en FE (p<0.05): {'Sí' if sig_fe else 'No'}")
    rep.log(f"Significativo en RE (p<0.05): {'Sí' if sig_re else 'No'}")

    # --- Lectura sugerida ---
    rep.log(f"\n{'─'*60}")
    rep.log("LECTURA SUGERIDA (decisión manual, no aplicada al modelo)")
    rep.log(f"{'─'*60}")
    if not signo_estable:
        rep.log(f"  → Señal de alerta tipo criminalidad_pca/justicia_pca (v4): el efecto")
        rep.log(f"    between (Pooled) no coincide con el efecto within (FE). Descartar")
        rep.log(f"    como candidato a regresor, salvo evidencia adicional muy fuerte.")
    elif sig_fe or sig_re:
        rep.log(f"  → Signo estable y significativo en {'FE' if sig_fe else 'RE'}. Candidato")
        rep.log(f"    viable para promover a VARS_MODELO — confirmar también con VIF y con")
        rep.log(f"    el resto de diagnósticos (Hausman/Mundlak) antes de decidir.")
    else:
        rep.log(f"  → Signo estable pero no significativo ni en FE ni en RE. Mismo patrón")
        rep.log(f"    que institucional_productivo_pca/infraestructura_territorial_pca ya")
        rep.log(f"    muestran: dirección teóricamente razonable, pero el panel actual no")
        rep.log(f"    tiene suficiente variación within para identificarlo con confianza.")

    return tabla, vif_data, signo_estable, sig_fe, sig_re


def probar_candidatos_seleccionados(df_indexado, y, rep, candidatos=None):
    """
    Corre evaluar_candidato_como_regresor() para cada candidato en la lista
    (uno a la vez, nunca combinados) y cierra con una tabla resumen.
    """
    if candidatos is None:
        candidatos = CANDIDATOS_A_PROBAR
    candidatos = [c for c in candidatos if c in df_indexado.columns]

    rep.log(f"\n{'#'*70}")
    rep.log("# PRUEBA DE CANDIDATOS SELECCIONADOS COMO REGRESORES ADICIONALES")
    rep.log("# (uno a la vez sobre el modelo base de 5 regresores — v8)")
    rep.log(f"{'#'*70}")

    resumen = []
    for candidato in candidatos:
        tabla, vif_data, signo_estable, sig_fe, sig_re = \
            evaluar_candidato_como_regresor(df_indexado, y, candidato, rep)
        resumen.append({
            "Candidato": candidato,
            "Signo estable": "Sí" if signo_estable else "No",
            "Sig. FE": "Sí" if sig_fe else "No",
            "Sig. RE": "Sí" if sig_re else "No",
            "VIF": vif_data.loc[vif_data["Variable"] == candidato, "VIF"].iloc[0],
        })

    rep.log(f"\n{'='*70}")
    rep.log("RESUMEN — candidatos probados como regresor adicional")
    rep.log(f"{'='*70}")
    rep.log(pd.DataFrame(resumen).to_string(index=False))

    return resumen


# ─── ESTIMACIÓN DE MODELOS DE PANEL ──────────────────────────────────────────

def estimar_modelos(df, rep):
    df = df.set_index(["Departamento", "Año IDC"])

    y = df["informalidad"]
    X = sm.add_constant(df[VARS_MODELO])

    rep.log("\n### Estadística descriptiva ###")
    rep.log(df[["informalidad"] + VARS_MODELO].describe().to_string())

    # --- VIF y tolerancia ---
    X_vif = X.dropna()
    vif_data = pd.DataFrame({
        "Variable":   X_vif.columns,
        "VIF":        [variance_inflation_factor(X_vif.values, i)
                       for i in range(X_vif.shape[1])],
    })
    vif_data["Tolerancia"] = 1 / vif_data["VIF"]
    rep.log("\n### VIF y Tolerancia ###")
    rep.log(vif_data.to_string(index=False))

    # --- Modelos de panel ---
    pooled = PooledOLS(y, X).fit()
    rep.log("\n### Pooled OLS ###")
    rep.log(str(pooled.summary))

    fe = PanelOLS(y, X, entity_effects=True).fit(cov_type="robust")
    rep.log("\n### Fixed Effects ###")
    rep.log(str(fe.summary))

    re = RandomEffects(y, X).fit(cov_type="robust")
    rep.log("\n### Random Effects ###")
    rep.log(str(re.summary))

    rep.log("\n### Comparación FE vs RE ###")
    rep.log(str(compare({"FE": fe, "RE": re})))

    # --- Hausman formal ---
    diff = fe.params - re.params
    cov_diff = fe.cov - re.cov
    hausman_stat = diff.T @ np.linalg.inv(cov_diff) @ diff
    hausman_p = 1 - chi2.cdf(hausman_stat, df=len(diff))
    rep.log(f"\nHausman chi2: {hausman_stat:.4f}, p-value: {hausman_p:.4f}")

    # --- Mundlak test ---
    means = (df.groupby(level="Departamento")[VARS_MODELO]
               .transform("mean")
               .add_suffix("_mean"))
    X_mundlak = pd.concat([X, means], axis=1)
    re_mundlak = RandomEffects(y, X_mundlak).fit()
    rep.log("\n### Mundlak Test (RE con promedios grupales) ###")
    rep.log(str(re_mundlak.summary))

    cols_mean = [c for c in X_mundlak.columns if c.endswith("_mean")]
    try:
        params_mean = re_mundlak.params[cols_mean]
        cov_mean = re_mundlak.cov.loc[cols_mean, cols_mean]
        wald_stat = float(params_mean.values @ np.linalg.inv(cov_mean.values) @ params_mean.values)
        wald_df = len(cols_mean)
        wald_p = 1 - chi2.cdf(wald_stat, df=wald_df)
        rep.log(f"\nMundlak — Test conjunto (Wald) de medias grupales:")
        rep.log(f"   chi2({wald_df}) = {wald_stat:.4f}   p-value = {wald_p:.4f}")
        if wald_p > 0.05:
            rep.log("   → No se rechaza H0: medias grupales conjuntamente no significativas.")
            rep.log("     Evidencia a favor de Random Effects.")
            modelo_sugerido = "RE"
        else:
            rep.log("   → Se rechaza H0: medias grupales conjuntamente significativas.")
            rep.log("     Evidencia a favor de Fixed Effects.")
            modelo_sugerido = "FE"
    except np.linalg.LinAlgError:
        rep.log("\n   ⚠ No se pudo invertir la matriz de covarianza para el test conjunto de Mundlak.")
        wald_stat, wald_p = np.nan, np.nan
        modelo_sugerido = "RE" if hausman_p > 0.05 else "FE"

    rep.log(f"\n### Modelo sugerido para diagnóstico: {modelo_sugerido} ###")
    rep.log("(Basado en Hausman + test conjunto de Mundlak; si ambos coinciden en "
            "no rechazar H0, se usa RE por ser más eficiente; si alguno rechaza, FE.)")

    modelo_diag = re if modelo_sugerido == "RE" else fe

    return (df, y, X, pooled, fe, re, vif_data, hausman_stat, hausman_p,
            wald_stat, wald_p, modelo_sugerido, modelo_diag)


# ─── EXPORTACIÓN: base final usada en el modelamiento (CSV → futura migración a Postgres) ────

def exportar_base_modelo(df, rep, carpeta_salida=CARPETA_SALIDA,
                          nombre_archivo="panel_macro_final.csv"):
    """
    Exporta a CSV el dataframe final que entra al modelamiento econométrico
    (informalidad + los 4 regresores de VARS_MODELO), tal cual antes de
    correr Pooled/FE/RE — sin residuales ni valores predichos.

    Pensado como insumo directo para migrar después a PostgreSQL (tabla
    independiente, ej. `panel_macro_final`, dentro del esquema
    `geih_informalidad` o uno nuevo para el componente macro).

    Parámetros
    ----------
    df : DataFrame
        El dataframe retornado por estimar_modelos(), ya indexado por
        ["Departamento", "Año IDC"].
    rep : Reporte
        Objeto de logging ya usado en el resto del script.
    carpeta_salida : str
        Carpeta donde se guarda el CSV (por defecto, CARPETA_SALIDA).
    nombre_archivo : str
        Nombre del archivo CSV de salida.
    """
    cols_finales = ["informalidad"] + VARS_MODELO
    base_final = df[cols_finales].reset_index()

    carpeta_csv = os.path.join(carpeta_salida, "base_final")
    os.makedirs(carpeta_csv, exist_ok=True)
    ruta_csv = os.path.join(carpeta_csv, nombre_archivo)
    base_final.to_csv(ruta_csv, index=False, encoding="utf-8")

    # Diccionario mínimo de variables, para no perder el contexto de cada
    # columna al migrar a Postgres (sirve también como insumo del anexo
    # de variables en la sección de Metodología).
    diccionario = pd.DataFrame({
        "variable": ["Departamento", "Año IDC"] + cols_finales,
        "tipo": ["texto", "entero", "numérico (%)"]
                + ["numérico (PCA estandarizado)"] * len(VARS_MODELO),
        "descripcion": [
            "Departamento colombiano (33 unidades, incl. Bogotá)",
            "Año IDC, panel 2019-2025 (7 períodos)",
            "Tasa de informalidad laboral = 100 - LAB-1-3 (IDC)",
        ] + [f"Índice PCA estandarizado: {v}" for v in VARS_MODELO],
    })
    ruta_dicc = os.path.join(carpeta_csv, "diccionario_" + nombre_archivo)
    diccionario.to_csv(ruta_dicc, index=False, encoding="utf-8")

    rep.log(f"\n### Exportación de base final ###")
    rep.log(f"Filas: {base_final.shape[0]}  |  Columnas: {base_final.shape[1]}")
    rep.log(f"Generado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    rep.log(f"CSV base final: {ruta_csv}")
    rep.log(f"CSV diccionario de variables: {ruta_dicc}")

    return base_final


# ─── UTILIDAD INTERNA: guardar figura individual ─────────────────────────────

def _guardar_fig(fig, nombre, rep):
    """Guarda `fig` como PNG en CARPETA_SALIDA y registra la ruta en rep."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rep.log(f"  → Guardada: {ruta}")
    return ruta


# ─── DIAGNÓSTICOS ─────────────────────────────────────────────────────────────

def diagnostico(df, y, X, modelo_diag, modelo_sugerido, vif_data,
                 hausman_stat, hausman_p, wald_stat, wald_p, rep):
    """
    Calcula todos los diagnósticos y guarda CADA gráfica como archivo PNG
    independiente. El reporte .txt acumula todos los resultados numéricos.

    Figuras generadas
    -----------------
    01_correlaciones.png         — heatmap de correlaciones pool
    02_vif.png                   — VIF y tolerancia por regresor
    03_resid_vs_ajustados.png    — residuos vs valores ajustados
    04_histograma_residuos.png   — distribución de residuos + AD
    05_resid_vs_<var>.png        — un archivo por cada regresor (×5 en v8)
    NN_evolucion_temporal.png    — informalidad + índices PCA en el tiempo
    """
    resids        = modelo_diag.resids.values.flatten()
    fitted        = modelo_diag.fitted_values.values.flatten()
    X_aligned_idx = modelo_diag.resids.index
    X_aligned     = X.loc[X_aligned_idx].values
    X_plot        = X.loc[X_aligned_idx]

    # ── Tests numéricos ──────────────────────────────────────────────────────

    bp_test   = het_breuschpagan(resids, X_aligned)
    labels_bp = ["LM stat", "p-value", "F stat", "F p-value"]
    bp_dict   = dict(zip(labels_bp, bp_test))

    rep.log(f"\n{'='*70}")
    rep.log(f"DIAGNÓSTICOS DEL MODELO — {modelo_sugerido}")
    rep.log(f"{'='*70}")

    rep.log(f"\n{'─'*60}")
    rep.log(f"HOMOCEDASTICIDAD — Test de Breusch-Pagan (modelo {modelo_sugerido})")
    rep.log(f"{'─'*60}")
    rep.log(f"  LM stat       : {bp_dict['LM stat']:.6f}")
    rep.log(f"  p-value (LM)  : {bp_dict['p-value']:.6f}"
            + ("  *** heterocedástico" if bp_dict['p-value'] < 0.05 else "  (homocedasticidad no rechazada)"))
    rep.log(f"  F stat        : {bp_dict['F stat']:.6f}")
    rep.log(f"  p-value (F)   : {bp_dict['F p-value']:.6f}")
    rep.log(f"  Interpretación: {'Se rechaza homocedasticidad (p<0.05). Usar errores robustos.' if bp_dict['p-value'] < 0.05 else 'No se rechaza homocedasticidad (p≥0.05).'}")

    norm_test = normal_ad(resids)
    rep.log(f"\n{'─'*60}")
    rep.log(f"NORMALIDAD DE RESIDUOS — Test de Anderson-Darling (modelo {modelo_sugerido})")
    rep.log(f"{'─'*60}")
    rep.log(f"  Estadístico   : {norm_test[0]:.6f}")
    rep.log(f"  p-value       : {norm_test[1]:.6f}"
            + ("  *** no normal" if norm_test[1] < 0.05 else "  (normalidad no rechazada)"))
    rep.log(f"  Interpretación: {'Se rechaza normalidad (p<0.05). Inferencia basada en asintótica.' if norm_test[1] < 0.05 else 'No se rechaza normalidad (p≥0.05).'}")

    rep.log(f"\n{'─'*60}")
    rep.log("MULTICOLINEALIDAD — VIF y Tolerancia")
    rep.log(f"{'─'*60}")
    rep.log(f"  {'Variable':<35} {'VIF':>8}  {'Tolerancia':>10}  Semáforo")
    rep.log(f"  {'-'*35} {'-'*8}  {'-'*10}  {'-'*15}")
    for _, row in vif_data.iterrows():
        if row["Variable"] == "const":
            continue
        semaforo = ("🔴 ALTO (>10)"   if row["VIF"] > 10
                    else "🟡 MODERADO (>5)" if row["VIF"] > 5
                    else "🟢 OK (≤5)")
        rep.log(f"  {row['Variable']:<35} {row['VIF']:>8.4f}  {row['Tolerancia']:>10.4f}  {semaforo}")

    rep.log(f"\n{'─'*60}")
    rep.log("SELECCIÓN DE MODELO — Hausman + Mundlak")
    rep.log(f"{'─'*60}")
    rep.log(f"  Hausman chi²  : {hausman_stat:.6f}")
    rep.log(f"  Hausman p-val : {hausman_p:.6f}"
            + ("  → FE preferido" if hausman_p < 0.05 else "  → RE preferido"))
    if not np.isnan(wald_p):
        rep.log(f"  Mundlak Wald  : {wald_stat:.6f}")
        rep.log(f"  Mundlak p-val : {wald_p:.6f}"
                + ("  → FE preferido" if wald_p < 0.05 else "  → RE preferido"))
    rep.log(f"  Modelo elegido: {modelo_sugerido}")

    within_var = df.groupby(level="Departamento")[VARS_MODELO].std().mean()
    rep.log(f"\n{'─'*60}")
    rep.log("VARIACIÓN WITHIN — std promedio dentro de cada departamento")
    rep.log(f"{'─'*60}")
    for var, val in within_var.items():
        rep.log(f"  {var:<35} {val:.6f}"
                + ("  ⚠ variación muy baja" if val < 0.05 else ""))

    rep.log(f"\n{'='*70}")
    rep.log("ARCHIVOS DE GRÁFICAS GENERADOS")
    rep.log(f"{'='*70}")

    # ── Figura 1: Heatmap de correlaciones ───────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[["informalidad"] + VARS_MODELO].corr(),
        annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True,
        annot_kws={"size": 8}
    )
    ax.set_title("Matriz de correlaciones — informalidad e índices PCA\n"
                 "(pool completo, nivel departamento-año)", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    fig1.tight_layout()
    _guardar_fig(fig1, "01_correlaciones.png", rep)

    # ── Figura 2: VIF ─────────────────────────────────────────────────────────
    vif_plot = vif_data[vif_data["Variable"] != "const"].copy()
    colores  = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71"
                for v in vif_plot["VIF"]]
    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.barh(vif_plot["Variable"], vif_plot["VIF"], color=colores)
    ax.axvline(5,  color="#f39c12", linestyle="--", linewidth=1.2, label="VIF = 5")
    ax.axvline(10, color="#e74c3c", linestyle="--", linewidth=1.2, label="VIF = 10")
    ax.set_title(f"Multicolinealidad — VIF por regresor (modelo {modelo_sugerido})", fontsize=10)
    ax.set_xlabel("VIF")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for i, (v, tol) in enumerate(zip(vif_plot["VIF"], vif_plot["Tolerancia"])):
        ax.text(v + 0.05, i, f"{v:.2f}  (tol={tol:.2f})", va="center", fontsize=8)
    fig2.tight_layout()
    _guardar_fig(fig2, "02_vif.png", rep)

    # ── Figura 3: Residuos vs ajustados ──────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, resids, alpha=0.55, s=20, color="#7f8c8d", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(resids, fitted, frac=0.4)
        ax.plot(smooth[:, 0], smooth[:, 1], color="orange", linewidth=1.5,
                label="Lowess")
        ax.legend(fontsize=8)
    except Exception:
        pass
    bp_p = bp_dict['p-value']
    ax.set_title(
        f"Residuos vs valores ajustados ({modelo_sugerido})\n"
        f"Breusch-Pagan: LM={bp_dict['LM stat']:.3f}  p={bp_p:.4f}"
        + ("  *** heterocedástico" if bp_p < 0.05 else "  (homo. no rechazada)"),
        fontsize=9
    )
    ax.set_xlabel("Informalidad ajustada")
    ax.set_ylabel("Residuo")
    fig3.tight_layout()
    _guardar_fig(fig3, "03_resid_vs_ajustados.png", rep)

    # ── Figura 4: Histograma de residuos ─────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resids, bins=25, color="#3498db", alpha=0.75, edgecolor="white",
            density=True, label="Residuos")
    xr = np.linspace(resids.min(), resids.max(), 200)
    ax.plot(xr, (1 / (resids.std() * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((xr - resids.mean()) / resids.std())**2),
            color="red", linewidth=1.5, label="Normal teórica")
    ax.set_title(
        f"Distribución de residuos ({modelo_sugerido})\n"
        f"Anderson-Darling: stat={norm_test[0]:.4f}  p={norm_test[1]:.4f}"
        + ("  *** no normal" if norm_test[1] < 0.05 else "  (normalidad no rechazada)"),
        fontsize=9
    )
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    fig4.tight_layout()
    _guardar_fig(fig4, "04_histograma_residuos.png", rep)

    # ── Figuras 5+: Residuos vs cada regresor (una por variable) ────────────
    for idx, var in enumerate(VARS_MODELO, start=5):
        fig_r, ax = plt.subplots(figsize=(7, 5))
        x_col  = X_plot[var]
        mask   = x_col.notna().values
        ax.scatter(x_col, resids, alpha=0.6, s=20, color="steelblue",
                   edgecolors="none")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        if mask.sum() > 1:
            m, b = np.polyfit(x_col.values[mask], resids[mask], 1)
            xplot = np.linspace(x_col.min(), x_col.max(), 200)
            ax.plot(xplot, m * xplot + b, color="orange", linestyle="--",
                    linewidth=1.5, label=f"Tendencia lineal (β={m:.3f})")
            ax.legend(fontsize=8)
        ax.set_title(f"Residuos vs {var} ({modelo_sugerido})", fontsize=10)
        ax.set_xlabel(var)
        ax.set_ylabel("Residuo")
        fig_r.tight_layout()
        nombre_png = f"{idx:02d}_resid_vs_{var}.png"
        _guardar_fig(fig_r, nombre_png, rep)

    # ── Figura final: Evolución temporal con eje secundario ──────────────────
    evol = df.groupby(level="Año IDC")[["informalidad"] + VARS_MODELO].mean()

    fig_ev, ax_ev = plt.subplots(figsize=(13, 6))
    ax_ev.plot(evol.index, evol["informalidad"], marker="o", color="black",
               linewidth=2.4, label="Informalidad % (eje izq.)")
    ax_ev.set_xlabel("Año IDC")
    ax_ev.set_ylabel("Informalidad (%)", color="black")
    ax_ev.tick_params(axis="y", labelcolor="black")

    ax_ev2 = ax_ev.twinx()
    colores_pca = plt.cm.tab10(np.linspace(0, 1, len(VARS_MODELO)))
    for var, color in zip(VARS_MODELO, colores_pca):
        ax_ev2.plot(evol.index, evol[var], marker="s", linestyle="--",
                    color=color, linewidth=1.4, label=f"{var} (eje der.)")
    ax_ev2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_ev2.set_ylabel("Índices PCA (estandarizados, μ=0)")

    l1, lab1 = ax_ev.get_legend_handles_labels()
    l2, lab2 = ax_ev2.get_legend_handles_labels()
    ax_ev.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper center",
                 ncol=3, bbox_to_anchor=(0.5, -0.18))
    ax_ev.set_title(
        "Evolución temporal promedio (33 deptos.)\n"
        "Informalidad % (eje izq.) vs índices PCA estandarizados (eje der.)",
        fontsize=10
    )
    fig_ev.tight_layout()
    _guardar_fig(fig_ev, f"{len(VARS_MODELO) + 5:02d}_evolucion_temporal.png", rep)

    rep.log(f"\n{'='*70}")
    rep.log("FIN DEL REPORTE DE DIAGNÓSTICO")
    rep.log(f"{'='*70}\n")


# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rep = Reporte()

    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    df = cargar_datos(rep)

    # Construcción de todos los índices PCA. infraestructura_territorial_pca
    # (Variante A) e innovacion_pca conservan PCA puro; el método "ponderado"
    # se mantiene solo para las variantes de infraestructura territorial
    # por consistencia histórica con cobertura_pca (v3-v5).
    diagnosticos_pca = {}
    for nombre_var, bloque in TODOS_LOS_BLOQUES:
        metodo = ("ponderado"
                  if nombre_var in ("infraestructura_territorial_pca",
                                    "infraestructura_territorial_serv_pca",
                                    "infraestructura_amplia_pca")
                  else "pca")
        df, diag = construir_pca(df, nombre_var, bloque, rep, metodo=metodo)
        diagnosticos_pca[nombre_var] = diag

    comparar_variantes_infraestructura(diagnosticos_pca, rep)
    comparar_pilares_v7(diagnosticos_pca, rep)
    evaluar_candidatos_v7(df, diagnosticos_pca, rep)

    # ── DIAGNÓSTICO TEMPORAL #1: estado de VARS_MODELO justo tras el PCA ──
    print("\n=== DIAGNÓSTICO TEMPORAL #1 (tras construir_pca, antes de estimar_modelos) ===")
    print(df[VARS_MODELO].describe().to_string())
    print("=== FIN DIAGNÓSTICO #1 ===\n")

    (df, y, X, pooled, fe, re, vif_data, hausman_stat, hausman_p,
     wald_stat, wald_p, modelo_sugerido, modelo_diag) = estimar_modelos(df, rep)

    # ── DIAGNÓSTICO TEMPORAL #2: estado de VARS_MODELO tras estimar_modelos ──
    print("\n=== DIAGNÓSTICO TEMPORAL #2 (df retornado por estimar_modelos) ===")
    print(df[VARS_MODELO].describe().to_string())
    print("=== FIN DIAGNÓSTICO #2 ===\n")

    # Exporta la base final (informalidad + 4 regresores PCA) a CSV, lista
    # para una futura migración a PostgreSQL.
    exportar_base_modelo(df, rep)

    # Prueba de los 2 candidatos restantes (institucional_productivo_pca,
    # tramites_pca) sobre el modelo base de 6 regresores (v11).
    # innovacion_pca, institucional_puro_pca y tejido_empresarial_pca ya están en el modelo.
    resumen_candidatos = probar_candidatos_seleccionados(df, y, rep)

    diagnostico(df, y, X, modelo_diag, modelo_sugerido, vif_data,
                hausman_stat, hausman_p, wald_stat, wald_p, rep)

    ruta_txt = os.path.join(CARPETA_SALIDA, "reporte_panel.txt")
    rep.guardar(ruta_txt)
    print(f"\nReporte de texto guardado como: {ruta_txt}")