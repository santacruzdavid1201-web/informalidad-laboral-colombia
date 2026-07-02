-- ============================================================
-- CONSULTAS DE ANÁLISIS EXPLORATORIO
-- Tablas: pipeline_macroeconomico, pipeline_microeconomico, ingresos_por_ano
-- Nota: varias columnas vienen como TEXT por la importación (probablemente
-- coma decimal). Se usa REPLACE(col, ',', '.')::numeric para castearlas
-- sin necesidad de alterar la tabla. Si en algún caso ya están con punto,
-- el REPLACE no hace daño (no encuentra comas y deja el valor igual).
-- ============================================================


-- ------------------------------------------------------------
-- 1. VISTA MAESTRA: macro + ingresos por departamento-año
-- ------------------------------------------------------------
SELECT
    m.departamento,
    m.ano_idc AS ano,
    REPLACE(m.informalidad_pct, ',', '.')::numeric                       AS informalidad_pct,
    REPLACE(m.capital_humano_digital_pca_bi100, ',', '.')::numeric        AS capital_humano_digital,
    REPLACE(m.infraestructura_territorial_pca_bi100, ',', '.')::numeric   AS infraestructura_territorial,
    REPLACE(m.laboral_pca_bi100, ',', '.')::numeric                       AS laboral,
    REPLACE(m.innovacion_pca_bi100, ',', '.')::numeric                    AS innovacion,
    i.ingreso_mediana,
    REPLACE(i.ingreso_promedio, ',', '.')::numeric                        AS ingreso_promedio,
    i.n_obs
FROM pipeline_macroeconomico m
LEFT JOIN ingresos_por_ano i
    ON m.id_departamento = i.id_departamento AND m.ano_idc = i.ano
ORDER BY m.departamento, m.ano_idc;


-- ------------------------------------------------------------
-- 2. RANKING: informalidad promedio del periodo por departamento
-- ------------------------------------------------------------
SELECT
    departamento,
    ROUND(AVG(REPLACE(informalidad_pct, ',', '.')::numeric), 2) AS informalidad_promedio,
    COUNT(*) AS n_anios
FROM pipeline_macroeconomico
GROUP BY departamento
ORDER BY informalidad_promedio DESC;


-- ------------------------------------------------------------
-- 3. EVOLUCIÓN TEMPORAL de informalidad (para graficar tendencia)
-- ------------------------------------------------------------
SELECT
    departamento,
    ano_idc AS ano,
    REPLACE(informalidad_pct, ',', '.')::numeric AS informalidad_pct
FROM pipeline_macroeconomico
ORDER BY departamento, ano_idc;


-- ------------------------------------------------------------
-- 4. COMPOSICIÓN SECTORIAL (% de ocupados por rama) por depto-año
--    Usa pipeline_microeconomico (conteos por categoría)
-- ------------------------------------------------------------
WITH total_rama AS (
    SELECT
        departamento, ano, id_departamento,
        rama_seccion_a_agro + rama_seccion_b_minas + rama_seccion_c_manuf +
        rama_seccion_d_energ + rama_seccion_e_agua + rama_seccion_f_constr +
        rama_seccion_g_comerc + rama_seccion_h_transp + rama_seccion_i_aloj +
        rama_seccion_j_infocom + rama_seccion_k_finan + rama_seccion_l_inmob +
        rama_seccion_m_prof + rama_seccion_n_admin + rama_seccion_otro +
        rama_seccion_p_educ + rama_seccion_q_salud + rama_seccion_r_entret +
        rama_seccion_s_otrosserv + rama_seccion_t_hogar AS total_obs
    FROM pipeline_microeconomico
)
SELECT
    p.departamento,
    p.ano,
    ROUND(100.0 * p.rama_seccion_g_comerc / NULLIF(t.total_obs,0), 2) AS pct_comercio,
    ROUND(100.0 * p.rama_seccion_a_agro   / NULLIF(t.total_obs,0), 2) AS pct_agro,
    ROUND(100.0 * p.rama_seccion_c_manuf  / NULLIF(t.total_obs,0), 2) AS pct_manufactura,
    ROUND(100.0 * p.rama_seccion_f_constr / NULLIF(t.total_obs,0), 2) AS pct_construccion
FROM pipeline_microeconomico p
JOIN total_rama t USING (departamento, ano, id_departamento)
ORDER BY p.departamento, p.ano;


-- ------------------------------------------------------------
-- 5. BRECHA DE GÉNERO: % mujeres y % mujer con carga de cuidado
--    por departamento-año (insumo para discusión del Heckman)
-- ------------------------------------------------------------
SELECT
    departamento,
    ano,
    ROUND(100.0 * mujer_1 / NULLIF(mujer_0 + mujer_1, 0), 2)                       AS pct_mujeres,
    ROUND(100.0 * mujer_cuidado_1 / NULLIF(mujer_cuidado_0 + mujer_cuidado_1, 0),2) AS pct_mujer_cuidado
FROM pipeline_microeconomico
ORDER BY departamento, ano;


-- ------------------------------------------------------------
-- 6. RELACIÓN INGRESO vs INFORMALIDAD (para scatter / correlación)
-- ------------------------------------------------------------
SELECT
    m.departamento,
    m.ano_idc AS ano,
    REPLACE(m.informalidad_pct, ',', '.')::numeric AS informalidad_pct,
    i.ingreso_mediana
FROM pipeline_macroeconomico m
JOIN ingresos_por_ano i
    ON m.id_departamento = i.id_departamento AND m.ano_idc = i.ano
WHERE i.ingreso_mediana IS NOT NULL
ORDER BY departamento, ano DESC;


-- ------------------------------------------------------------
-- 7. COMPOSICIÓN EDUCATIVA: % con educación superior por depto-año
-- ------------------------------------------------------------
SELECT
    departamento,
    ano,
    ROUND(
        100.0 * educacion_cat_3_superior /
        NULLIF(educacion_cat_1_primaria_menos + educacion_cat_2_secundaria + educacion_cat_3_superior, 0)
    , 2) AS pct_educacion_superior
FROM pipeline_microeconomico
ORDER BY departamento, ano;


-- ============================================================
-- 8. VISTA MAESTRA COMPLETA: macro + micro (todo en %) + ingresos
--    Una fila por departamento-año, pensada como fact table única
--    para Power BI. Asume que el total de observaciones es el
--    mismo en todas las categorías de pipeline_microeconomico
--    (mismo conjunto de encuestados, solo partido distinto).
--    El query 9 valida ese supuesto.
-- ============================================================
WITH base_micro AS (
    SELECT
        *,
        rama_seccion_a_agro + rama_seccion_b_minas + rama_seccion_c_manuf +
        rama_seccion_d_energ + rama_seccion_e_agua + rama_seccion_f_constr +
        rama_seccion_g_comerc + rama_seccion_h_transp + rama_seccion_i_aloj +
        rama_seccion_j_infocom + rama_seccion_k_finan + rama_seccion_l_inmob +
        rama_seccion_m_prof + rama_seccion_n_admin + rama_seccion_otro +
        rama_seccion_p_educ + rama_seccion_q_salud + rama_seccion_r_entret +
        rama_seccion_s_otrosserv + rama_seccion_t_hogar AS total_obs
    FROM pipeline_microeconomico
),
micro_cnt AS (
    SELECT
        departamento,
        ano,
        id_departamento,
        total_obs,
        mujer_1                AS cnt_mujeres,
        mujer_cuidado_1         AS cnt_mujer_cuidado,
        etnia_1                 AS cnt_etnia,
        no_hetero_1             AS cnt_no_hetero,
        en_pareja_1              AS cnt_en_pareja,
        nacido_extranjero_1     AS cnt_nacido_extranjero,
        migrante_reciente_1_0   AS cnt_migrante_reciente,
        campesino_1_0            AS cnt_campesino,
        zona_rural_1_0           AS cnt_zona_rural,
        trans_1                  AS cnt_persona_trans,          
        ocupacion_calif_1_0      AS cnt_ocupacion_calificada,
        desea_cambio_1_0         AS cnt_desea_cambio_empleo,
        subsidio_gob_1           AS cnt_subsidio_gob,
        tiene_ing_no_lab_1       AS cnt_ingreso_no_laboral,
        educacion_cat_3_superior   AS cnt_educacion_superior,
        educacion_cat_2_secundaria AS cnt_educacion_secundaria,
        horas_cat_1_parcial_menos35    AS cnt_tiempo_parcial,
        horas_cat_3_sobretrabajo_mas48 AS cnt_sobretrabajo,
        -- Ramas de actividad económica (las 20 secciones completas)
        rama_seccion_a_agro       AS cnt_agro,
        rama_seccion_b_minas      AS cnt_minas,
        rama_seccion_c_manuf      AS cnt_manufactura,
        rama_seccion_d_energ      AS cnt_energia,
        rama_seccion_e_agua       AS cnt_agua,
        rama_seccion_f_constr     AS cnt_construccion,
        rama_seccion_g_comerc     AS cnt_comercio,
        rama_seccion_h_transp     AS cnt_transporte,
        rama_seccion_i_aloj       AS cnt_alojamiento_comida,
        rama_seccion_j_infocom    AS cnt_infocomunicaciones,
        rama_seccion_k_finan      AS cnt_financiero,
        rama_seccion_l_inmob      AS cnt_inmobiliario,
        rama_seccion_m_prof       AS cnt_profesional_cientif,
        rama_seccion_n_admin      AS cnt_administrativo,
        rama_seccion_otro         AS cnt_otro,
        rama_seccion_p_educ       AS cnt_educacion_sector,
        rama_seccion_q_salud      AS cnt_salud,
        rama_seccion_r_entret     AS cnt_entretenimiento,
        rama_seccion_s_otrosserv  AS cnt_otros_servicios,
        rama_seccion_t_hogar      AS cnt_servicio_domestico
    FROM base_micro
)
SELECT
    mc2.departamento,
    LPAD(mc2.id_departamento::text, 2, '0')                             AS id_departamento_geo,
    mc2.ano,
    mc2.total_obs                                                       AS n_obs_micro,
    REPLACE(mc.informalidad_pct, ',', '.')::numeric                     AS informalidad_pct,
    REPLACE(mc.capital_humano_digital_pca_bi100, ',', '.')::numeric     AS capital_humano_digital,
    REPLACE(mc.infraestructura_territorial_pca_bi100, ',', '.')::numeric AS infraestructura_territorial,
    REPLACE(mc.laboral_pca_bi100, ',', '.')::numeric                    AS laboral_pca,
    REPLACE(mc.innovacion_pca_bi100, ',', '.')::numeric                 AS innovacion_pca,
    ing.ingreso_mediana,
    REPLACE(ing.ingreso_promedio, ',', '.')::numeric                    AS ingreso_promedio,
    ing.n_obs                                                           AS n_obs_ingresos,
    mc2.cnt_mujeres,
    mc2.cnt_mujer_cuidado,
    mc2.cnt_etnia,
    mc2.cnt_no_hetero,
    mc2.cnt_en_pareja,
    mc2.cnt_nacido_extranjero,
    mc2.cnt_migrante_reciente,
    mc2.cnt_campesino,
    mc2.cnt_zona_rural,
    mc2.cnt_persona_trans,
    mc2.cnt_ocupacion_calificada,
    mc2.cnt_desea_cambio_empleo,
    mc2.cnt_subsidio_gob,
    mc2.cnt_ingreso_no_laboral,
    mc2.cnt_educacion_superior,
    mc2.cnt_educacion_secundaria,
    mc2.cnt_tiempo_parcial,
    mc2.cnt_sobretrabajo,
    mc2.cnt_agro,
    mc2.cnt_minas,
    mc2.cnt_manufactura,
    mc2.cnt_energia,
    mc2.cnt_agua,
    mc2.cnt_construccion,
    mc2.cnt_comercio,
    mc2.cnt_transporte,
    mc2.cnt_alojamiento_comida,
    mc2.cnt_infocomunicaciones,
    mc2.cnt_financiero,
    mc2.cnt_inmobiliario,
    mc2.cnt_profesional_cientif,
    mc2.cnt_administrativo,
    mc2.cnt_otro,
    mc2.cnt_educacion_sector,
    mc2.cnt_salud,
    mc2.cnt_entretenimiento,
    mc2.cnt_otros_servicios,
    mc2.cnt_servicio_domestico
FROM micro_cnt mc2
LEFT JOIN pipeline_macroeconomico mc
    ON mc2.id_departamento = mc.id_departamento AND mc2.ano = mc.ano_idc
LEFT JOIN ingresos_por_ano ing
    ON mc2.id_departamento = ing.id_departamento AND mc2.ano = ing.ano
ORDER BY mc2.departamento, mc2.ano;


-- ------------------------------------------------------------
-- 9. VALIDACIÓN: ¿el total de obs por categoría coincide entre
--    pipeline_microeconomico y n_obs de ingresos_por_ano?
--    Si hay diferencias grandes, revisa antes de confiar en el
--    query 8 (los pct ahí asumen un único total_obs por fila).
-- ------------------------------------------------------------
SELECT
    p.departamento,
    p.ano,
    (p.rama_seccion_a_agro + p.rama_seccion_b_minas + p.rama_seccion_c_manuf +
     p.rama_seccion_d_energ + p.rama_seccion_e_agua + p.rama_seccion_f_constr +
     p.rama_seccion_g_comerc + p.rama_seccion_h_transp + p.rama_seccion_i_aloj +
     p.rama_seccion_j_infocom + p.rama_seccion_k_finan + p.rama_seccion_l_inmob +
     p.rama_seccion_m_prof + p.rama_seccion_n_admin + p.rama_seccion_otro +
     p.rama_seccion_p_educ + p.rama_seccion_q_salud + p.rama_seccion_r_entret +
     p.rama_seccion_s_otrosserv + p.rama_seccion_t_hogar)             AS total_obs_micro,
    (p.mujer_0 + p.mujer_1)                                           AS total_obs_mujer,
    (p.educacion_cat_1_primaria_menos + p.educacion_cat_2_secundaria + p.educacion_cat_3_superior) AS total_obs_educacion,
    i.n_obs                                                           AS n_obs_ingresos
FROM pipeline_microeconomico p
LEFT JOIN ingresos_por_ano i
    ON p.id_departamento = i.id_departamento AND p.ano = i.ano
ORDER BY p.departamento, p.ano;


