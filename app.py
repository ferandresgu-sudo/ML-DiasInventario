import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(
    page_title="Prediccion mensual de ventas",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Simulador predictivo mensual optimizado, consulta acumulable por categoria, combo y descripcion")
st.caption(
    "Modelo global Extra Trees sobre log(venta), estacionalidad, rezagos "
    "y promedios moviles. Las graficas, segmentadores y segmentaciones son diagnosticas "
    "y no modifican las predicciones."
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
# FUNCIONES DEL MODELO V2, SIN MODIFICACIONES DE COMPORTAMIENTO
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

        # Conserva el comportamiento original: ausencia mensual del SKU = venta 0.
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
    # Mismos hiperparametros de V2.
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )


@st.cache_resource(show_spinner=False)
def entrenar(df):
    f = crear_features(df).dropna(
        subset=["lag1", "lag2", "VENTA"]
    ).copy()

    medianas = f[FEATURES].median().fillna(0)
    X = f[FEATURES].fillna(medianas)
    modelo = nuevo_modelo()

    # Misma transformacion logaritmica de V2.
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
    base = float(resto.mean())
    return float(temporada.mean() / base) if base > 0 else 1.0


def ajustar_prediccion_estacional(pred_base, historial, periodo, peso_v2, umbral):
    mes = int(periodo % 100)
    anio = int(periodo // 100)
    indice = calcular_indice_estacional_nov_dic(historial, anio)
    if mes not in [11, 12] or indice < umbral:
        return float(pred_base), indice, np.nan, False
    referencia = historial[
        (historial["MES_NUM"] == mes) & (historial["AÑO"] < anio)
    ].sort_values("AÑO")
    if referencia.empty:
        return float(pred_base), indice, np.nan, False
    venta_anterior = float(referencia.iloc[-1]["VENTA"])
    ajustada = peso_v2 * float(pred_base) + (1 - peso_v2) * venta_anterior
    return max(0.0, ajustada), indice, venta_anterior, True


def ajustar_lista_estacional(pred_base, historial, periodos, peso_v2, umbral):
    finales, diagnostico = [], []
    for p, periodo in zip(pred_base, periodos):
        final, indice, referencia, aplicado = ajustar_prediccion_estacional(
            p, historial, periodo, peso_v2, umbral
        )
        finales.append(final)
        diagnostico.append({
            "Indice_Nov_Dic": indice,
            "Venta_Mismo_Mes_Anterior": referencia,
            "Ajuste_Aplicado": aplicado,
        })
    return finales, diagnostico


@st.cache_data(show_spinner="Leyendo y normalizando archivos...", max_entries=3)
def cargar(archivos, version_cache="v2-rapida-1"):
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

    # Mantener CODIGO como texto evita perdida de precision/notacion cientifica.
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
def backtest(df, n_holdout=2, usar_ajuste=True, peso_v2=0.60, umbral=1.40):
    tamanos = df.groupby("CODIGO").size()
    codigos = tamanos[tamanos >= n_holdout + 3].index
    base = df[df.CODIGO.isin(codigos)].copy()
    held = base.groupby("CODIGO", group_keys=False).tail(n_holdout)
    train = base.drop(index=held.index)
    modelo, medianas = entrenar(train)
    filas = []
    for codigo, real in held.groupby("CODIGO"):
        real = real.sort_values("Periodo")
        hist = train[train.CODIGO == codigo].sort_values("Periodo")
        periodos = real.Periodo.astype(int).tolist()
        pred_base = predecir_recursivo(modelo, medianas, hist, periodos)
        if usar_ajuste:
            pred_final, diag = ajustar_lista_estacional(
                pred_base, hist, periodos, peso_v2, umbral
            )
        else:
            pred_final = list(pred_base)
            diag = [{
                "Indice_Nov_Dic": calcular_indice_estacional_nov_dic(hist, int(per//100)),
                "Venta_Mismo_Mes_Anterior": np.nan,
                "Ajuste_Aplicado": False,
            } for per in periodos]
        for paso, ((_, r), pb, pf, d) in enumerate(
            zip(real.iterrows(), pred_base, pred_final, diag), 1
        ):
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "TIPO": r.get("TIPO", ""),
                "COMBOS": int(r.get("COMBOS", 0)),
                "Periodo_Real": int(r.Periodo),
                "Mes_Predecido": paso,
                "Prediccion_Base": float(pb),
                "Prediccion": float(pf),
                "Actual": float(r.VENTA),
                "Indice_Nov_Dic": float(d["Indice_Nov_Dic"]),
                "Venta_Mismo_Mes_Anterior": d["Venta_Mismo_Mes_Anterior"],
                "Ajuste_Estacional_Aplicado": bool(d["Ajuste_Aplicado"]),
            })
    return pd.DataFrame(filas)

# =========================================================
# FUNCIONES DIAGNOSTICAS, NO INTERVIENEN EN EL MODELO
# =========================================================
def clasificar_producto_diagnostico(historial):
    """
    Clasificacion solo informativa.
    Intermitente: 3 o mas ceros en los ultimos 6 meses.
    Volatil: CV de los ultimos 6 meses >= 1.
    Estable: cualquier otro caso.
    """
    ventas = (
        historial.sort_values("Periodo")["VENTA"]
        .tail(6)
        .astype(float)
        .to_numpy()
    )

    if len(ventas) == 0:
        return "Intermitente"

    ceros = int(np.sum(ventas == 0))
    if ceros >= 3:
        return "Intermitente"

    media = float(np.mean(ventas))
    desviacion = (
        float(np.std(ventas, ddof=1))
        if len(ventas) >= 2
        else 0.0
    )
    cv = desviacion / media if media > 0 else np.inf

    if cv >= 1.0:
        return "Volatil"

    return "Estable"


@st.cache_data(show_spinner=False)
def crear_tabla_segmentacion(df):
    filas = []
    for codigo, hist in df.groupby("CODIGO"):
        filas.append({
            "CODIGO": codigo,
            "SEGMENTO": clasificar_producto_diagnostico(hist),
        })

    detalle = pd.DataFrame(filas)
    resumen = (
        detalle.groupby("SEGMENTO", as_index=False)
        .agg(Productos=("CODIGO", "nunique"))
    )

    orden = pd.CategoricalDtype(
        ["Estable", "Intermitente", "Volatil"],
        ordered=True,
    )
    resumen["SEGMENTO"] = resumen["SEGMENTO"].astype(orden)
    resumen = resumen.sort_values("SEGMENTO").reset_index(drop=True)
    return detalle, resumen


def preparar_diagnostico_backtest(bt, segmentos):
    diag = bt.merge(segmentos, on="CODIGO", how="left")
    diag["Error_Abs"] = (
        diag["Prediccion"] - diag["Actual"]
    ).abs()
    diag["Error_Signo"] = diag["Prediccion"] - diag["Actual"]
    diag["Error_Cuadrado"] = diag["Error_Signo"] ** 2
    diag["Variacion_%"] = np.where(
        diag["Actual"] > 0,
        100 * diag["Error_Signo"] / diag["Actual"],
        np.nan,
    )
    diag["Situacion"] = np.where(
        diag["Error_Signo"] >= 0,
        "Sobreestimacion",
        "Subestimacion",
    )
    return diag


def aplicar_segmentadores_diagnostico(bt_diag):
    """Filtra tablas y graficas sin reentrenar ni modificar predicciones."""
    st.subheader("Segmentadores del diagnostico")
    st.caption(
        "Los filtros de CATEGORIA, COMBO y PRODUCTO solo cambian las metricas, "
        "tablas y graficas mostradas. El modelo V2 y sus predicciones permanecen iguales."
    )

    datos = bt_diag.copy()

    for col in ["CATEGORIA", "DESCRIPCION"]:
        if col not in datos.columns:
            datos[col] = "Sin dato"
        datos[col] = datos[col].fillna("Sin dato").astype(str).str.strip()
        datos.loc[datos[col].eq(""), col] = "Sin dato"

    if "COMBOS" not in datos.columns:
        datos["COMBOS"] = 0
    datos["COMBOS"] = (
        pd.to_numeric(datos["COMBOS"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        categorias = sorted(datos["CATEGORIA"].unique().tolist())
        categorias_sel = st.multiselect(
            "CATEGORIA",
            options=categorias,
            default=[],
            placeholder="Todas las categorias",
            key="seg_categoria_bt",
        )
    if categorias_sel:
        datos = datos[datos["CATEGORIA"].isin(categorias_sel)].copy()

    with f2:
        combos_sel = st.selectbox(
            "COMBO",
            options=["Todos", "No combo (0)", "Combo (1)"],
            index=0,
            key="seg_combo_bt",
        )
    if combos_sel == "No combo (0)":
        datos = datos[datos["COMBOS"] == 0].copy()
    elif combos_sel == "Combo (1)":
        datos = datos[datos["COMBOS"] == 1].copy()

    opciones_producto = (
        datos[["CODIGO", "DESCRIPCION"]]
        .drop_duplicates()
        .assign(
            Producto=lambda x: (
                x["CODIGO"].astype(str)
                + " - "
                + x["DESCRIPCION"].astype(str)
            )
        )
        .sort_values("Producto")
    )
    mapa_producto = dict(
        zip(opciones_producto["Producto"], opciones_producto["CODIGO"])
    )

    with f3:
        productos_sel = st.multiselect(
            "PRODUCTO",
            options=list(mapa_producto.keys()),
            default=[],
            placeholder="Todos los productos",
            key="seg_producto_bt",
        )
    if productos_sel:
        codigos_sel = [mapa_producto[x] for x in productos_sel]
        datos = datos[datos["CODIGO"].isin(codigos_sel)].copy()

    if datos.empty:
        st.warning("No hay registros para la combinacion de filtros seleccionada.")
        return datos

    y_f = datos["Actual"].to_numpy(float)
    p_f = datos["Prediccion"].to_numpy(float)
    wape_f = 100 * np.abs(y_f - p_f).sum() / max(np.abs(y_f).sum(), 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Registros filtrados", f"{len(datos):,}")
    m2.metric("WAPE filtrado", f"{wape_f:.2f}%")
    m3.metric("MAE filtrado", f"${mean_absolute_error(y_f, p_f):,.2f}")
    m4.metric("RMSE filtrado", f"${mean_squared_error(y_f, p_f) ** 0.5:,.2f}")
    r2_f = r2_score(y_f, p_f) if len(datos) >= 2 else np.nan
    m5.metric("R² filtrado", f"{r2_f:.3f}" if pd.notna(r2_f) else "N/D")

    return datos


def mostrar_graficas_backtest(bt_diag):
    st.subheader("Diagnostico visual del backtest")

    # -----------------------------------------------------
    # 1. Venta real acumulada vs prediccion acumulada
    # -----------------------------------------------------
    st.markdown("### Venta real acumulada vs prediccion acumulada")

    totales = pd.DataFrame({
        "Serie": ["Venta real", "Prediccion"],
        "Monto": [
            float(bt_diag["Actual"].sum()),
            float(bt_diag["Prediccion"].sum()),
        ],
    })

    total_real = float(bt_diag["Actual"].sum())
    total_pred = float(bt_diag["Prediccion"].sum())
    diferencia = total_pred - total_real
    diferencia_pct = (
        100 * diferencia / total_real
        if total_real != 0
        else np.nan
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Venta real acumulada", f"${total_real:,.2f}")
    c2.metric("Prediccion acumulada", f"${total_pred:,.2f}")
    c3.metric(
        "Diferencia acumulada",
        f"${diferencia:,.2f}",
        delta=(
            f"{diferencia_pct:+.2f}%"
            if pd.notna(diferencia_pct)
            else None
        ),
        delta_color="inverse",
    )

    barras_totales = (
        alt.Chart(totales)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Serie:N", title=None, sort=["Venta real", "Prediccion"]),
            y=alt.Y(
                "Monto:Q",
                title="Monto acumulado",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "Serie:N",
                scale=alt.Scale(
                    domain=["Venta real", "Prediccion"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
                legend=None,
            ),
            tooltip=[
                "Serie:N",
                alt.Tooltip("Monto:Q", format="$,.2f"),
            ],
        )
        .properties(height=380)
    )

    etiquetas = barras_totales.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        color="white",
    ).encode(text=alt.Text("Monto:Q", format="$,.0f"))

    st.altair_chart(
        barras_totales + etiquetas,
        use_container_width=True,
    )

    # Comparacion acumulada por horizonte del backtest.
    por_horizonte = (
        bt_diag.groupby("Mes_Predecido", as_index=False)
        .agg(
            Venta_Real=("Actual", "sum"),
            Prediccion=("Prediccion", "sum"),
        )
        .melt(
            id_vars="Mes_Predecido",
            value_vars=["Venta_Real", "Prediccion"],
            var_name="Serie",
            value_name="Monto",
        )
    )
    por_horizonte["Serie"] = por_horizonte["Serie"].replace(
        {"Venta_Real": "Venta real", "Prediccion": "Prediccion"}
    )

    barras_horizonte = (
        alt.Chart(por_horizonte)
        .mark_bar()
        .encode(
            x=alt.X(
                "Mes_Predecido:O",
                title="Horizonte pronosticado",
            ),
            xOffset="Serie:N",
            y=alt.Y(
                "Monto:Q",
                title="Monto acumulado",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "Serie:N",
                scale=alt.Scale(
                    domain=["Venta real", "Prediccion"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
                title=None,
            ),
            tooltip=[
                "Mes_Predecido:O",
                "Serie:N",
                alt.Tooltip("Monto:Q", format="$,.2f"),
            ],
        )
        .properties(
            height=340,
            title="Venta real y prediccion acumuladas por horizonte",
        )
    )
    st.altair_chart(barras_horizonte, use_container_width=True)

    # -----------------------------------------------------
    # 2. Real vs prediccion
    # -----------------------------------------------------
    st.markdown("### Venta real vs prediccion por producto-mes")
    maximo = float(max(
        bt_diag["Actual"].max(),
        bt_diag["Prediccion"].max(),
        1,
    ))

    puntos = (
        alt.Chart(bt_diag)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X(
                "Actual:Q",
                title="Venta real",
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "Prediccion:Q",
                title="Prediccion",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color("SEGMENTO:N", title="Segmento"),
            size=alt.Size(
                "Error_Abs:Q",
                title="Error absoluto",
                scale=alt.Scale(range=[20, 500]),
            ),
            tooltip=[
                "CODIGO:N",
                "DESCRIPCION:N",
                "CATEGORIA:N",
                "SEGMENTO:N",
                "Periodo_Real:O",
                alt.Tooltip("Actual:Q", format="$,.2f"),
                alt.Tooltip("Prediccion:Q", format="$,.2f"),
                alt.Tooltip("Error_Abs:Q", format="$,.2f"),
            ],
        )
    )

    diagonal = (
        alt.Chart(pd.DataFrame({
            "Actual": [0, maximo],
            "Prediccion": [0, maximo],
        }))
        .mark_line(color="#d62728", strokeDash=[6, 4])
        .encode(x="Actual:Q", y="Prediccion:Q")
    )

    st.altair_chart(
        (puntos + diagonal).properties(height=430),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # 3 y 4. Top RMSE y sesgo
    # -----------------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        top_rmse = (
            bt_diag.groupby(
                ["CODIGO", "DESCRIPCION"],
                dropna=False,
                as_index=False,
            )
            .agg(
                Error_Absoluto=("Error_Abs", "sum"),
                Error_Cuadrado=("Error_Cuadrado", "sum"),
                Venta_Real=("Actual", "sum"),
                Prediccion=("Prediccion", "sum"),
            )
            .nlargest(15, "Error_Cuadrado")
        )
        top_rmse["Producto"] = (
            top_rmse["CODIGO"].astype(str)
            + " - "
            + top_rmse["DESCRIPCION"].fillna("").astype(str)
        )

        graf_rmse = (
            alt.Chart(top_rmse)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Error_Cuadrado:Q",
                    title="Error cuadratico acumulado",
                ),
                y=alt.Y(
                    "Producto:N",
                    sort="-x",
                    title=None,
                ),
                color=alt.Color(
                    "Error_Absoluto:Q",
                    scale=alt.Scale(scheme="yelloworangered"),
                    title="Error absoluto",
                ),
                tooltip=[
                    "Producto:N",
                    alt.Tooltip("Venta_Real:Q", format="$,.2f"),
                    alt.Tooltip("Prediccion:Q", format="$,.2f"),
                    alt.Tooltip("Error_Absoluto:Q", format="$,.2f"),
                ],
            )
            .properties(
                height=500,
                title="Productos que mas elevan el RMSE",
            )
        )
        st.altair_chart(graf_rmse, use_container_width=True)

    with col_b:
        sesgo = (
            bt_diag.groupby(
                ["CODIGO", "DESCRIPCION"],
                dropna=False,
                as_index=False,
            )
            .agg(
                Error_Signo=("Error_Signo", "sum"),
                Error_Abs=("Error_Abs", "sum"),
            )
            .nlargest(15, "Error_Abs")
        )
        sesgo["Producto"] = (
            sesgo["CODIGO"].astype(str)
            + " - "
            + sesgo["DESCRIPCION"].fillna("").astype(str)
        )
        sesgo["Situacion"] = np.where(
            sesgo["Error_Signo"] >= 0,
            "Sobreestimacion",
            "Subestimacion",
        )

        graf_sesgo = (
            alt.Chart(sesgo)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Error_Signo:Q",
                    title="Prediccion menos venta real",
                    axis=alt.Axis(format="$,.0f"),
                ),
                y=alt.Y(
                    "Producto:N",
                    sort="-x",
                    title=None,
                ),
                color=alt.Color(
                    "Situacion:N",
                    scale=alt.Scale(
                        domain=["Sobreestimacion", "Subestimacion"],
                        range=["#d62728", "#1f77b4"],
                    ),
                    title=None,
                ),
                tooltip=[
                    "Producto:N",
                    "Situacion:N",
                    alt.Tooltip("Error_Signo:Q", format="$,.2f"),
                ],
            )
            .properties(
                height=500,
                title="Sobreestimacion y subestimacion",
            )
        )
        st.altair_chart(graf_sesgo, use_container_width=True)

    # -----------------------------------------------------
    # 5 y 6. Histograma y Pareto
    # -----------------------------------------------------
    col_c, col_d = st.columns(2)

    with col_c:
        hist_error = bt_diag[
            bt_diag["Variacion_%"].between(-200, 200)
        ].copy()

        graf_hist = (
            alt.Chart(hist_error)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X(
                    "Variacion_%:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Error porcentual",
                ),
                y=alt.Y("count():Q", title="Numero de predicciones"),
                tooltip=[
                    alt.Tooltip("count():Q", title="Predicciones")
                ],
            )
            .properties(
                height=350,
                title="Distribucion del error porcentual (-200% a 200%)",
            )
        )
        st.altair_chart(graf_hist, use_container_width=True)

    with col_d:
        pareto = (
            bt_diag.groupby(
                ["CODIGO", "DESCRIPCION"],
                dropna=False,
                as_index=False,
            )
            .agg(Error_Abs=("Error_Abs", "sum"))
            .sort_values("Error_Abs", ascending=False)
        )
        pareto["Producto"] = (
            pareto["CODIGO"].astype(str)
            + " - "
            + pareto["DESCRIPCION"].fillna("").astype(str)
        )
        pareto["Porcentaje_Error"] = (
            100 * pareto["Error_Abs"]
            / max(pareto["Error_Abs"].sum(), 1)
        )
        pareto["Porcentaje_Acumulado"] = (
            pareto["Porcentaje_Error"].cumsum()
        )
        pareto["Orden"] = np.arange(1, len(pareto) + 1)
        p30 = pareto.head(30)

        barras = (
            alt.Chart(p30)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X(
                    "Orden:O",
                    title="Productos ordenados por error",
                ),
                y=alt.Y(
                    "Error_Abs:Q",
                    title="Error absoluto",
                    axis=alt.Axis(format="$,.0f"),
                ),
                tooltip=[
                    "Producto:N",
                    alt.Tooltip("Error_Abs:Q", format="$,.2f"),
                    alt.Tooltip(
                        "Porcentaje_Acumulado:Q",
                        format=".1f",
                    ),
                ],
            )
        )

        linea = (
            alt.Chart(p30)
            .mark_line(color="#e45756", point=True)
            .encode(
                x="Orden:O",
                y=alt.Y(
                    "Porcentaje_Acumulado:Q",
                    title="% acumulado",
                    scale=alt.Scale(domain=[0, 100]),
                ),
            )
        )

        graf_pareto = (
            alt.layer(barras, linea)
            .resolve_scale(y="independent")
            .properties(
                height=350,
                title="Pareto de contribucion al error",
            )
        )
        st.altair_chart(graf_pareto, use_container_width=True)

@st.cache_data(show_spinner=False)
def crear_indices_rapidos(df):
    """Precalcula historiales y ultimos registros para evitar filtros repetidos."""
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
    st.header("Ajuste estacional Nov-Dic")
    usar_ajuste_estacional = st.checkbox("Activar ajuste", value=True)
    peso_modelo_estacional = st.slider(
        "Peso de la prediccion V2", 0.50, 0.90, 0.60, 0.05
    )
    umbral_estacional = st.slider(
        "Umbral indice Nov-Dic", 1.10, 2.50, 1.40, 0.05
    )

if archivos:
    try:
        df = cargar(archivos)
        st.success(
            f"{len(df):,} observaciones mensuales normalizadas | "
            f"{df.CODIGO.nunique():,} productos"
        )
        st.caption(
            "La primera lectura de los Excel puede tardar. Mientras los archivos no cambien, "
            "la lectura, el backtest, el entrenamiento y la segmentacion se reutilizan desde cache."
        )

        # Indices reutilizados por selector individual y proyeccion masiva.
        historicos_por_codigo, ult = crear_indices_rapidos(df)

        # Tabla diagnostica de productos por segmento.
        segmentos_detalle, tabla_segmentos = crear_tabla_segmentacion(df)
        st.subheader("Cantidad de productos por comportamiento de venta")
        st.caption(
            "Esta segmentacion es solo diagnostica y no modifica el modelo: "
            "Intermitente = 3 o mas ceros en los ultimos 6 meses; "
            "Volatil = coeficiente de variacion de 6 meses mayor o igual a 1; "
            "Estable = resto de productos."
        )
        st.dataframe(
            tabla_segmentos,
            use_container_width=True,
            hide_index=True,
        )

        graf_segmentos = (
            alt.Chart(tabla_segmentos)
            .mark_bar()
            .encode(
                x=alt.X("SEGMENTO:N", title=None),
                y=alt.Y("Productos:Q", title="Cantidad de productos"),
                color=alt.Color(
                    "SEGMENTO:N",
                    scale=alt.Scale(
                        domain=["Estable", "Intermitente", "Volatil"],
                        range=["#2ca02c", "#ffbf00", "#d62728"],
                    ),
                    legend=None,
                ),
                tooltip=["SEGMENTO:N", "Productos:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(graf_segmentos, use_container_width=True)

        holdout = st.slider("Meses de backtest", 1, 4, 2)

        with st.spinner("Entrenando y validando..."):
            bt = backtest(
                df, holdout,
                usar_ajuste=usar_ajuste_estacional,
                peso_v2=peso_modelo_estacional,
                umbral=umbral_estacional,
            )

            if bt.empty:
                raise ValueError(
                    "El backtest no genero resultados. Revisa el historial disponible."
                )

            y = bt.Actual.to_numpy()
            p = bt.Prediccion.to_numpy()
            wape = (
                100 * np.abs(y - p).sum()
                / max(np.abs(y).sum(), 1)
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("WAPE", f"{wape:.2f}%")
            c2.metric("MAE", f"${mean_absolute_error(y, p):,.2f}")
            c3.metric(
                "RMSE",
                f"${mean_squared_error(y, p) ** 0.5:,.2f}",
            )
            c4.metric("R²", f"{r2_score(y, p):.3f}")

            p_base_bt = bt["Prediccion_Base"].to_numpy(float)
            wape_base = 100 * np.abs(y-p_base_bt).sum() / max(np.abs(y).sum(),1)
            mae_base = mean_absolute_error(y,p_base_bt)
            rmse_base = mean_squared_error(y,p_base_bt)**0.5
            r2_base = r2_score(y,p_base_bt)
            ajustes_n = int(bt["Ajuste_Estacional_Aplicado"].sum())
            st.info(
                f"Las metricas principales usan la prediccion ajustada. "
                f"Ajuste aplicado en {ajustes_n:,} registros del backtest."
            )
            with st.expander("Comparar metricas ajustadas vs V2 base"):
                comp = pd.DataFrame({
                    "Metrica":["WAPE","MAE","RMSE","R2"],
                    "V2 Base":[wape_base,mae_base,rmse_base,r2_base],
                    "Ajustada":[wape,mean_absolute_error(y,p),mean_squared_error(y,p)**0.5,r2_score(y,p)],
                })
                comp["Cambio"] = comp["Ajustada"]-comp["V2 Base"]
                st.dataframe(comp,use_container_width=True,hide_index=True)

            bt_diag = preparar_diagnostico_backtest(
                bt,
                segmentos_detalle,
            )

            bt_visual = aplicar_segmentadores_diagnostico(bt_diag)
            if not bt_visual.empty:
                mostrar_graficas_backtest(bt_visual)

            st.subheader("Predicciones con mayor error absoluto")
            columnas_error = [
                "CODIGO", "DESCRIPCION", "CATEGORIA", "TIPO", "SEGMENTO",
                "Periodo_Real", "Mes_Predecido", "Actual",
                "Prediccion_Base", "Prediccion", "Ajuste_Estacional_Aplicado",
                "Indice_Nov_Dic", "Venta_Mismo_Mes_Anterior",
                "Error_Abs", "Variacion_%",
            ]
            st.dataframe(
                bt_visual.nlargest(15, "Error_Abs")[columnas_error]
                if not bt_visual.empty else bt_visual,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Descargar detalle del backtest",
                bt_diag.to_csv(index=False).encode("utf-8-sig"),
                "backtest_v2_con_diagnostico.csv",
                "text/csv",
            )

        # Entrenamiento final original, sin cambios.
        modelo, medianas = entrenar(df)

        # =====================================================
        # CONSULTA ACUMULABLE: CATEGORIA, COMBO Y DESCRIPCION
        # =====================================================
        st.subheader("Consulta acumulable por categoria, combo y descripcion")
        st.caption(
            "Los filtros acumulan la venta historica y las predicciones de los SKU "
            "seleccionados. Las metricas se calculan con las observaciones del backtest "
            "correspondientes a esos mismos SKU. El modelo V2 no se reentrena ni cambia."
        )

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

        # Filtros encadenados.
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
            st.warning(
                "No hay descripciones seleccionadas. Elige al menos una o selecciona "
                "'Todas las descripciones disponibles'."
            )
            st.stop()

        seleccion_sku = ult_consulta[
            ult_consulta["DESCRIPCION"].isin(descripciones_sel)
        ].copy()
        codigos_seleccionados = seleccion_sku["CODIGO"].drop_duplicates().tolist()

        if not codigos_seleccionados:
            st.warning("No hay productos para la combinacion de filtros seleccionada.")
            st.stop()

        st.info(
            f"Consulta formada por {len(codigos_seleccionados):,} SKU y "
            f"{len(descripciones_sel):,} descripciones."
        )

        # -----------------------------------------------------
        # METRICAS DEL BACKTEST PARA LA SELECCION
        # -----------------------------------------------------
        bt_consulta = bt_diag[
            bt_diag["CODIGO"].isin(codigos_seleccionados)
        ].copy()

        st.markdown("### Metricas de la seleccion")
        if bt_consulta.empty:
            st.warning(
                "Los SKU seleccionados no tienen observaciones suficientes en el "
                "backtest actual. Las predicciones futuras sí se muestran."
            )
        else:
            y_sel = bt_consulta["Actual"].to_numpy(float)
            p_sel = bt_consulta["Prediccion"].to_numpy(float)
            wape_sel = (
                100 * np.abs(y_sel - p_sel).sum()
                / max(np.abs(y_sel).sum(), 1)
            )
            mae_sel = mean_absolute_error(y_sel, p_sel)
            rmse_sel = mean_squared_error(y_sel, p_sel) ** 0.5
            r2_sel = r2_score(y_sel, p_sel) if len(bt_consulta) >= 2 else np.nan

            ms1, ms2, ms3, ms4, ms5 = st.columns(5)
            ms1.metric("Registros de backtest", f"{len(bt_consulta):,}")
            ms2.metric("WAPE", f"{wape_sel:.2f}%")
            ms3.metric("MAE", f"${mae_sel:,.2f}")
            ms4.metric("RMSE", f"${rmse_sel:,.2f}")
            ms5.metric("R²", f"{r2_sel:.3f}" if pd.notna(r2_sel) else "N/D")

            with st.expander("Ver detalle del backtest de la seleccion"):
                columnas_bt = [
                    "CODIGO", "DESCRIPCION", "CATEGORIA", "TIPO", "COMBOS",
                    "Periodo_Real", "Mes_Predecido", "Actual", "Prediccion",
                    "Error_Abs", "Variacion_%",
                ]
                columnas_bt = [c for c in columnas_bt if c in bt_consulta.columns]
                st.dataframe(
                    bt_consulta[columnas_bt].sort_values(
                        "Error_Abs", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # -----------------------------------------------------
        # VENTA HISTORICA Y PREDICCION ACUMULADAS
        # -----------------------------------------------------
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
            pred_producto_base = predecir_recursivo(
                modelo, medianas, hist_producto, periodos
            )
            if usar_ajuste_estacional:
                pred_producto, diag_est = ajustar_lista_estacional(
                    pred_producto_base, hist_producto, periodos,
                    peso_modelo_estacional, umbral_estacional
                )
            else:
                pred_producto = list(pred_producto_base)
                diag_est = [{"Indice_Nov_Dic": calcular_indice_estacional_nov_dic(hist_producto), "Ajuste_Aplicado": False} for _ in periodos]
            meta_producto = seleccion_sku[
                seleccion_sku["CODIGO"] == codigo
            ].iloc[0]
            predicciones_sku.append({
                "CODIGO": codigo,
                "DESCRIPCION": meta_producto.get("DESCRIPCION", ""),
                "CATEGORIA": meta_producto.get("CATEGORIA", ""),
                "COMBOS": int(meta_producto.get("COMBOS", 0)),
                f"Pred Base {periodos[0]}": pred_producto_base[0],
                f"Pred {periodos[0]}": pred_producto[0],
                f"Pred Base {periodos[1]}": pred_producto_base[1],
                f"Pred {periodos[1]}": pred_producto[1],
                "Indice_Nov_Dic": diag_est[0]["Indice_Nov_Dic"],
                "Ajuste_Estacional": any(d["Ajuste_Aplicado"] for d in diag_est),
            })

        detalle_pred = pd.DataFrame(predicciones_sku)
        pred_acum = [
            float(detalle_pred[f"Pred {periodos[0]}"].sum()),
            float(detalle_pred[f"Pred {periodos[1]}"].sum()),
        ]

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("SKU acumulados", f"{len(codigos_seleccionados):,}")
        pc2.metric(
            f"Prediccion acumulada {periodos[0]}",
            f"${pred_acum[0]:,.2f}",
        )
        pc3.metric(
            f"Prediccion acumulada {periodos[1]}",
            f"${pred_acum[1]:,.2f}",
        )

        real_graf = hist_acumulado.rename(columns={"VENTA": "Valor"})
        real_graf["Serie"] = "Venta historica acumulada"
        futuro_graf = pd.DataFrame({
            "Periodo": periodos,
            "Valor": pred_acum,
            "Serie": "Proyeccion acumulada",
        })
        graf_consulta = pd.concat(
            [real_graf, futuro_graf],
            ignore_index=True,
        )

        chart_consulta = (
            alt.Chart(graf_consulta)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X(
                    "Periodo:O",
                    title="Periodo",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(
                    "Valor:Q",
                    title="Venta acumulada",
                    axis=alt.Axis(format="$,.0f"),
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
                tooltip=[
                    "Periodo:O",
                    "Serie:N",
                    alt.Tooltip("Valor:Q", format="$,.2f"),
                ],
            )
            .properties(
                height=420,
                title="Venta historica y proyeccion acumuladas",
            )
            .interactive()
        )
        st.altair_chart(chart_consulta, use_container_width=True)

        # Participacion de cada SKU en la proyeccion acumulada.
        detalle_pred[f"Participacion {periodos[0]} %"] = (
            100 * detalle_pred[f"Pred {periodos[0]}"]
            / max(pred_acum[0], 1)
        )
        detalle_pred[f"Participacion {periodos[1]} %"] = (
            100 * detalle_pred[f"Pred {periodos[1]}"]
            / max(pred_acum[1], 1)
        )
        detalle_pred = detalle_pred.sort_values(
            f"Pred {periodos[0]}", ascending=False
        )

        with st.expander("Ver aportacion de cada SKU al acumulado"):
            st.dataframe(
                detalle_pred.style.format({
                    f"Pred {periodos[0]}": "${:,.2f}",
                    f"Pred {periodos[1]}": "${:,.2f}",
                    f"Participacion {periodos[0]} %": "{:.2f}%",
                    f"Participacion {periodos[1]} %": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # Proyeccion masiva original, sin cambios en la prediccion.
        filas = []
        for _, r in ult.iterrows():
            h = historicos_por_codigo[r.CODIGO]
            pr_base = predecir_recursivo(modelo, medianas, h, periodos)
            if usar_ajuste_estacional:
                pr, diag_pr = ajustar_lista_estacional(
                    pr_base, h, periodos,
                    peso_modelo_estacional, umbral_estacional
                )
            else:
                pr = list(pr_base)
                diag_pr = [{"Indice_Nov_Dic": calcular_indice_estacional_nov_dic(h), "Ajuste_Aplicado": False} for _ in periodos]
            segmento = segmentos_detalle.loc[
                segmentos_detalle["CODIGO"] == r.CODIGO,
                "SEGMENTO",
            ]
            segmento = (
                segmento.iloc[0]
                if not segmento.empty
                else "Sin clasificar"
            )

            filas.append({
                "CODIGO": r.CODIGO,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "TIPO": r.get("TIPO", ""),
                "COMBOS": int(r.get("COMBOS", 0)),
                "SEGMENTO_DIAGNOSTICO": segmento,
                f"Pred Base {periodos[0]}": pr_base[0],
                f"Pred {periodos[0]}": pr[0],
                f"Pred Base {periodos[1]}": pr_base[1],
                f"Pred {periodos[1]}": pr[1],
                "INDICE_NOV_DIC": diag_pr[0]["Indice_Nov_Dic"],
                "AJUSTE_ESTACIONAL": any(d["Ajuste_Aplicado"] for d in diag_pr),
            })

        resultado = pd.DataFrame(filas)
        st.subheader("Proyeccion masiva")
        st.dataframe(
            resultado,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Descargar proyeccion CSV",
            resultado.to_csv(index=False).encode("utf-8-sig"),
            "proyeccion_ventas.csv",
            "text/csv",
        )

    except Exception as e:
        st.exception(e)
