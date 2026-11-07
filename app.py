import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import altair as alt

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(
    page_title="Prediccion mensual de ventas",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Consulta acumulable de ventas y proyecciones (LightGBM)")
st.caption(
    "Consulta acumulable por categoria, combo y descripcion, potenciada con "
    "LightGBM para capturar la estacionalidad extrema de noviembre y diciembre."
)

MESES = {
    "ENERO": 1,
    "FEBRERO": 2,
    "MARZO": 3,
    "ABRIL": 4,
    "MAYO": 5,
    "JUNIO": 6,
    "JULIO": 7,
    "AGOSTO": 8,
    "SEPTIEMBRE": 9,
    "OCTUBRE": 10,
    "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}

FEATURES = [
    "lag1", "lag2", "lag3", "lag4", "lag6", "lag12",
    "mean3", "std3", "mean6", "std6", "mean12", "std12",
    "sin_mes", "cos_mes",
]

# =========================================================
# FUNCIONES DEL MODELO (LIGHTGBM)
# =========================================================
def completar_calendario(g):
    salida = []
    meta = [
        c for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
        if c in g.columns
    ]

    for codigo, x in g.groupby("CODIGO"):
        x = x.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            x["AÑO"].astype(int).astype(str)
            + "-"
            + x["MES_NUM"].astype(int).astype(str)
            + "-01"
        )
        x.index = fechas
        idx = pd.date_range(fechas.min(), fechas.max(), freq="MS")
        x = x.reindex(idx)
        x["CODIGO"] = codigo

        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["MES_NUM"] = x.index.month
        x["AÑO"] = x.index.year
        x["Periodo"] = x.index.year * 100 + x.index.month

        for c in meta:
            x[c] = x[c].ffill().bfill()

        salida.append(x.reset_index(drop=True))

    return pd.concat(salida, ignore_index=True)


def crear_features(df):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grp = z.groupby("CODIGO")["VENTA"]

    for lag in [1, 2, 3, 4, 6, 12]:
        z[f"lag{lag}"] = grp.shift(lag)

    for w in [3, 6, 12]:
        z[f"mean{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).mean()
        )
        z[f"std{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).std()
        ).fillna(0)

    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    return z


def nuevo_modelo():
    # Reemplazo de ExtraTrees por LightGBM para potenciar Gradient Boosting
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


@st.cache_resource(show_spinner=False)
def entrenar(df):
    f = crear_features(df).dropna(
        subset=["lag1", "lag2", "VENTA"]
    ).copy()

    medianas = f[FEATURES].median().fillna(0)
    X = f[FEATURES].fillna(medianas)
    modelo = nuevo_modelo()

    modelo.fit(X, np.log1p(f["VENTA"].clip(lower=0)))
    return modelo, medianas


def vector_futuro(hist, mes_num, medianas):
    vals = (
        hist.sort_values("Periodo")["VENTA"]
        .astype(float)
        .tolist()
    )

    q = {
        f"lag{lag}": vals[-lag] if len(vals) >= lag else np.nan
        for lag in [1, 2, 3, 4, 6, 12]
    }

    for w in [3, 6, 12]:
        a = np.asarray(vals[-w:], dtype=float)
        q[f"mean{w}"] = a.mean() if len(a) >= 2 else np.nan
        q[f"std{w}"] = a.std(ddof=1) if len(a) >= 2 else 0.0

    q["sin_mes"] = np.sin(2 * np.pi * mes_num / 12)
    q["cos_mes"] = np.cos(2 * np.pi * mes_num / 12)

    return pd.DataFrame([q], columns=FEATURES).fillna(medianas)


def predecir_recursivo(modelo, medianas, historico, periodos):
    hist = historico.copy()
    preds = []

    for periodo in periodos:
        mes = periodo % 100
        p = float(
            np.expm1(
                modelo.predict(
                    vector_futuro(hist, mes, medianas)
                )[0]
            )
        )
        p = max(0.0, p)
        preds.append(p)

        base = hist.iloc[-1].copy()
        base["Periodo"] = periodo
        base["AÑO"] = periodo // 100
        base["MES_NUM"] = mes
        base["VENTA"] = p
        hist = pd.concat(
            [hist, pd.DataFrame([base])],
            ignore_index=True,
        )

    return preds


def calcular_indice_estacional_nov_dic(historial, anio_limite=None):
    h = historial.sort_values("Periodo").copy()
    if anio_limite is not None:
        h = h[h["AÑO"] < anio_limite].copy()

    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"].astype(float)
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"].astype(float)

    if len(temporada) < 2 or len(resto) < 6:
        return 1.0
    promedio_resto = float(resto.mean())
    if promedio_resto <= 0:
        return 1.0
    return float(temporada.mean() / promedio_resto)


def ajustar_prediccion_estacional(
    prediccion_modelo,
    historial,
    periodo_objetivo,
    peso_modelo=0.60,
    umbral_estacional=1.40,
):
    h = historial.sort_values("Periodo").copy()
    mes_objetivo = int(periodo_objetivo % 100)
    anio_objetivo = int(periodo_objetivo // 100)

    if mes_objetivo not in [11, 12]:
        return float(prediccion_modelo), 1.0, np.nan, False

    indice = calcular_indice_estacional_nov_dic(
        h, anio_limite=anio_objetivo
    )
    es_estacional = indice >= umbral_estacional
    if not es_estacional:
        return float(prediccion_modelo), indice, np.nan, False

    mismo_mes = h[
        (h["MES_NUM"] == mes_objetivo)
        & (h["AÑO"] < anio_objetivo)
    ].sort_values("AÑO")

    if mismo_mes.empty:
        return float(prediccion_modelo), indice, np.nan, False

    venta_anio_anterior = float(mismo_mes.iloc[-1]["VENTA"])
    pred_ajustada = (
        peso_modelo * float(prediccion_modelo)
        + (1.0 - peso_modelo) * venta_anio_anterior
    )
    return max(0.0, pred_ajustada), indice, venta_anio_anterior, True


def ajustar_lista_estacional(
    predicciones_base,
    historial,
    periodos,
    peso_modelo=0.60,
    umbral_estacional=1.40,
):
    ajustadas, diagnostico = [], []
    for pred_base, periodo in zip(predicciones_base, periodos):
        pred_aj, indice, referencia, aplicado = ajustar_prediccion_estacional(
            pred_base, historial, periodo, peso_modelo, umbral_estacional
        )
        ajustadas.append(pred_aj)
        diagnostico.append({
            "Periodo": periodo,
            "Pred_Base": float(pred_base),
            "Pred_Ajustada": float(pred_aj),
            "Indice_Nov_Dic": float(indice),
            "Venta_Mismo_Mes_Anterior": referencia,
            "Ajuste_Aplicado": aplicado,
        })
    return ajustadas, diagnostico


@st.cache_data(show_spinner="Leyendo y normalizando archivos...", max_entries=3)
def cargar(archivos, version_cache="v3-lightgbm-1"):
    partes = []
    columnas_utiles = {
        "CODIGO", "VENTA", "MES", "AÑO",
        "DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS",
    }

    for archivo in archivos:
        hojas = pd.read_excel(
            archivo,
            sheet_name=None,
            engine="openpyxl",
        )
        for h in hojas.values():
            if h is None or h.empty:
                continue
            h = h.copy()
            h.columns = [str(c).strip().upper() for c in h.columns]
            conservar = [c for c in h.columns if c in columnas_utiles]
            if conservar:
                partes.append(h[conservar])

    if not partes:
        raise ValueError("No se encontraron hojas con columnas compatibles.")

    df = pd.concat(partes, ignore_index=True, copy=False)

    oblig = ["CODIGO", "VENTA", "MES", "AÑO"]
    falt = [c for c in oblig if c not in df.columns]
    if falt:
        raise ValueError("Faltan columnas: " + ", ".join(falt))

    df["MES_NUM"] = (
        df["MES"].astype(str).str.strip().str.upper().map(MESES)
    )
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce")
    if "COMBOS" not in df.columns:
        df["COMBOS"] = 0
    df["COMBOS"] = (
        pd.to_numeric(df["COMBOS"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )

    df["CODIGO"] = (
        df["CODIGO"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df = df.dropna(subset=["MES_NUM", "AÑO", "VENTA"])
    df["Periodo"] = (df["AÑO"] * 100 + df["MES_NUM"]).astype(int)

    meta = {
        c: "first"
        for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
        if c in df.columns
    }

    g = (
        df.groupby(
            ["CODIGO", "Periodo", "MES_NUM", "AÑO"],
            as_index=False,
        )
        .agg({"VENTA": "sum", **meta})
    )

    return (
        completar_calendario(g)
        .sort_values(["CODIGO", "Periodo"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def crear_indices_rapidos(df):
    historicos = {
        codigo: grupo.sort_values("Periodo").copy()
        for codigo, grupo in df.groupby("CODIGO", sort=False)
    }
    ultimos = pd.concat(
        [grupo.tail(1) for grupo in historicos.values()],
        ignore_index=True,
    )
    return historicos, ultimos


# =========================================================
# INTERFAZ
# =========================================================
archivos = st.file_uploader(
    "Sube ventas 2024, 2025 y 2026",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True,
)

with st.sidebar:
    st.header("Ajuste estacional")
    usar_ajuste_estacional = st.checkbox(
        "Activar ajuste Nov-Dic", value=True
    )
    peso_modelo_estacional = st.slider(
        "Peso de la prediccion LightGBM",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="El resto del peso se asigna a la venta del mismo mes del año anterior.",
    )
    umbral_estacional = st.slider(
        "Umbral indice Nov-Dic",
        min_value=1.10,
        max_value=2.50,
        value=1.40,
        step=0.05,
    )

if archivos:
    try:
        df = cargar(archivos)
        st.success(
            f"{len(df):,} observaciones mensuales normalizadas | "
            f"{df.CODIGO.nunique():,} productos"
        )

        historicos_por_codigo, ult = crear_indices_rapidos(df)
        modelo, medianas = entrenar(df)

        st.subheader("Consulta acumulable por categoria, combo y descripcion")
        
        ult_consulta = ult.copy()
        for col in ["CATEGORIA", "DESCRIPCION"]:
            if col not in ult_consulta.columns:
                ult_consulta[col] = "Sin dato"
            ult_consulta[col] = (
                ult_consulta[col]
                .fillna("Sin dato")
                .astype(str)
                .str.strip()
            )
            ult_consulta.loc[ult_consulta[col].eq(""), col] = "Sin dato"

        if "COMBOS" not in ult_consulta.columns:
            ult_consulta["COMBOS"] = 0
        ult_consulta["COMBOS"] = (
            pd.to_numeric(ult_consulta["COMBOS"], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            categoria_consulta = st.selectbox(
                "CATEGORIA",
                ["Todas"] + sorted(ult_consulta["CATEGORIA"].unique().tolist()),
                key="consulta_categoria",
            )
        if categoria_consulta != "Todas":
            ult_consulta = ult_consulta[
                ult_consulta["CATEGORIA"] == categoria_consulta
            ].copy()

        with fc2:
            combo_consulta = st.selectbox(
                "COMBO",
                ["Todos", "No combo (0)", "Combo (1)"],
                key="consulta_combo",
            )
        if combo_consulta == "No combo (0)":
            ult_consulta = ult_consulta[ult_consulta["COMBOS"] == 0].copy()
        elif combo_consulta == "Combo (1)":
            ult_consulta = ult_consulta[ult_consulta["COMBOS"] == 1].copy()

        descripciones_disponibles = sorted(
            ult_consulta["DESCRIPCION"].dropna().unique().tolist()
        )
        modo_descripcion = st.radio(
            "Seleccion de DESCRIPCION",
            ["Todas las descripciones disponibles", "Seleccion manual"],
            horizontal=True,
            key="consulta_modo_descripcion",
        )

        if modo_descripcion == "Todas las descripciones disponibles":
            descripciones_sel = descripciones_disponibles
        else:
            descripciones_sel = st.multiselect(
                "DESCRIPCION",
                options=descripciones_disponibles,
                default=[],
                placeholder="Selecciona una o varias descripciones",
                key="consulta_descripciones",
            )

        if not descripciones_sel:
            st.warning("Selecciona al menos una descripción.")
            st.stop()

        seleccion_sku = ult_consulta[
            ult_consulta["DESCRIPCION"].isin(descripciones_sel)
        ].copy()
        codigos_seleccionados = seleccion_sku["CODIGO"].drop_duplicates().tolist()

        if not codigos_seleccionados:
            st.warning("No hay productos para la combinación seleccionada.")
            st.stop()

        ultimo = int(df["Periodo"].max())
        fecha = pd.Timestamp(ultimo // 100, ultimo % 100, 1)
        periodos = []
        for i in [1, 2]:
            f = fecha + pd.DateOffset(months=i)
            periodos.append(f.year * 100 + f.month)

        historicos_seleccionados = [
            historicos_por_codigo[codigo]
            for codigo in codigos_seleccionados
            if codigo in historicos_por_codigo
        ]
        hist_acumulado = (
            pd.concat(historicos_seleccionados, ignore_index=True)
            .groupby("Periodo", as_index=False)
            .agg(VENTA=("VENTA", "sum"))
            .sort_values("Periodo")
        )

        predicciones_sku = []
        for codigo in codigos_seleccionados:
            hist_producto = historicos_por_codigo[codigo]
            pred_producto = predecir_recursivo(
                modelo,
                medianas,
                hist_producto,
                periodos,
            )
            if usar_ajuste_estacional:
                pred_ajustada, diag_est = ajustar_lista_estacional(
                    pred_producto,
                    hist_producto,
                    periodos,
                    peso_modelo=peso_modelo_estacional,
                    umbral_estacional=umbral_estacional,
                )
            else:
                pred_ajustada = list(pred_producto)
                diag_est = [
                    {
                        "Indice_Nov_Dic": calcular_indice_estacional_nov_dic(hist_producto),
                        "Venta_Mismo_Mes_Anterior": np.nan,
                        "Ajuste_Aplicado": False,
                    }
                    for _ in periodos
                ]
            meta_producto = seleccion_sku[
                seleccion_sku["CODIGO"] == codigo
            ].iloc[0]
            predicciones_sku.append({
                "CODIGO": codigo,
                "DESCRIPCION": meta_producto.get("DESCRIPCION", ""),
                "CATEGORIA": meta_producto.get("CATEGORIA", ""),
                "COMBOS": int(meta_producto.get("COMBOS", 0)),
                f"Pred Base {periodos[0]}": pred_producto[0],
                f"Pred Ajustada {periodos[0]}": pred_ajustada[0],
                f"Pred Base {periodos[1]}": pred_producto[1],
                f"Pred Ajustada {periodos[1]}": pred_ajustada[1],
                "Indice_Nov_Dic": diag_est[0]["Indice_Nov_Dic"],
                "Ajuste_Estacional": any(d["Ajuste_Aplicado"] for d in diag_est),
            })

        detalle_pred = pd.DataFrame(predicciones_sku)
        pred_base_acum = [
            float(detalle_pred[f"Pred Base {periodos[0]}"].sum()),
            float(detalle_pred[f"Pred Base {periodos[1]}"].sum()),
        ]
        pred_acum = [
            float(detalle_pred[f"Pred Ajustada {periodos[0]}"].sum()),
            float(detalle_pred[f"Pred Ajustada {periodos[1]}"].sum()),
        ]

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("SKU acumulados", f"{len(codigos_seleccionados):,}")
        pc2.metric(
            f"Prediccion ajustada {periodos[0]}",
            f"${pred_acum[0]:,.2f}",
            delta=f"Base LightGBM: ${pred_base_acum[0]:,.2f}",
            delta_color="off",
        )
        pc3.metric(
            f"Prediccion ajustada {periodos[1]}",
            f"${pred_acum[1]:,.2f}",
            delta=f"Base LightGBM: ${pred_base_acum[1]:,.2f}",
            delta_color="off",
        )

        real_graf = hist_acumulado.rename(columns={"VENTA": "Valor"}).copy()
        real_graf["Serie"] = "Venta historica acumulada"

        ultimo_real = real_graf.sort_values("Periodo").iloc[-1]
        futuro_graf = pd.DataFrame({
            "Periodo": [int(ultimo_real["Periodo"])] + periodos,
            "Valor": [float(ultimo_real["Valor"])] + pred_acum,
            "Serie": "Proyeccion acumulada",
        })

        graf_consulta = pd.concat(
            [real_graf, futuro_graf],
            ignore_index=True,
        ).sort_values(["Serie", "Periodo"])

        chart_consulta = (
            alt.Chart(graf_consulta)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X(
                    "Periodo:O",
                    title="Periodo",
                    sort="ascending",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(
                    "Valor:Q",
                    title="Venta acumulada",
                    axis=alt.Axis(format="$,.0f"),
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Serie:N",
                    title="Serie",
                    scale=alt.Scale(
                        domain=[
                            "Venta historica acumulada",
                            "Proyeccion acumulada",
                        ],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                detail="Serie:N",
                order=alt.Order("Periodo:O", sort="ascending"),
                tooltip=[
                    "Periodo:O",
                    "Serie:N",
                    alt.Tooltip("Valor:Q", format="$,.2f"),
                ],
            )
            .properties(
                height=420,
                title="Venta historica y proyeccion acumuladas (LightGBM)",
            )
            .interactive()
        )
        st.altair_chart(chart_consulta, use_container_width=True)

    except Exception as e:
        st.exception(e)
