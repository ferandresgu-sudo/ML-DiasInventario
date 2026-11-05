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
st.title("📦 Simulador predictivo mensual optimizado")
st.caption(
    "Modelo global Extra Trees sobre log(venta), estacionalidad, rezagos "
    "y promedios moviles. Las graficas y segmentaciones son diagnosticas "
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
        c for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]
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


@st.cache_data(show_spinner="Leyendo y normalizando archivos...")
def cargar(archivos):
    partes = []

    for archivo in archivos:
        hojas = pd.read_excel(
            archivo,
            sheet_name=None,
            engine="openpyxl",
        )
        for _, h in hojas.items():
            partes.append(h)

    df = pd.concat(partes, ignore_index=True)
    df.columns = [str(c).strip().upper() for c in df.columns]

    oblig = ["CODIGO", "VENTA", "MES", "AÑO"]
    falt = [c for c in oblig if c not in df.columns]
    if falt:
        raise ValueError("Faltan columnas: " + ", ".join(falt))

    df["MES_NUM"] = (
        df["MES"].astype(str).str.strip().str.upper().map(MESES)
    )
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce")

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
        for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]
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


def backtest(df, n_holdout=2):
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
        pred = predecir_recursivo(
            modelo,
            medianas,
            hist,
            real.Periodo.astype(int).tolist(),
        )

        for paso, ((_, r), p) in enumerate(
            zip(real.iterrows(), pred),
            1,
        ):
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "Periodo_Real": int(r.Periodo),
                "Mes_Predecido": paso,
                "Prediccion": p,
                "Actual": float(r.VENTA),
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

# =========================================================
# INTERFAZ
# =========================================================
archivos = st.file_uploader(
    "Sube ventas 2024, 2025 y 2026",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True,
)

if archivos:
    try:
        df = cargar(archivos)
        st.success(
            f"{len(df):,} observaciones mensuales normalizadas | "
            f"{df.CODIGO.nunique():,} productos"
        )

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
            bt = backtest(df, holdout)

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

            bt_diag = preparar_diagnostico_backtest(
                bt,
                segmentos_detalle,
            )

            mostrar_graficas_backtest(bt_diag)

            st.subheader("Predicciones con mayor error absoluto")
            columnas_error = [
                "CODIGO", "DESCRIPCION", "CATEGORIA", "SEGMENTO",
                "Periodo_Real", "Mes_Predecido", "Actual", "Prediccion",
                "Error_Abs", "Variacion_%",
            ]
            st.dataframe(
                bt_diag.nlargest(15, "Error_Abs")[columnas_error],
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
        ult = (
            df.sort_values("Periodo")
            .groupby("CODIGO")
            .tail(1)
        )
        opciones = {
            r.CODIGO: f'{r.CODIGO} - {r.get("DESCRIPCION", "")}'
            for _, r in ult.iterrows()
        }

        codigo = st.selectbox(
            "Producto",
            list(opciones),
            format_func=lambda x: opciones[x],
        )

        hist = (
            df[df.CODIGO == codigo]
            .sort_values("Periodo")
        )
        ultimo = int(df.Periodo.max())
        fecha = pd.Timestamp(ultimo // 100, ultimo % 100, 1)
        periodos = []

        for i in [1, 2]:
            f = fecha + pd.DateOffset(months=i)
            periodos.append(f.year * 100 + f.month)

        pred = predecir_recursivo(
            modelo,
            medianas,
            hist,
            periodos,
        )

        st.write(
            f"**{periodos[0]}:** ${pred[0]:,.2f} | "
            f"**{periodos[1]}:** ${pred[1]:,.2f}"
        )

        # Historial individual con la misma prediccion original.
        hist_graf = hist[["Periodo", "VENTA"]].copy()
        hist_graf["Media 6 meses"] = (
            hist_graf["VENTA"].rolling(6, min_periods=2).mean()
        )
        hist_graf["Mediana 6 meses"] = (
            hist_graf["VENTA"].rolling(6, min_periods=2).median()
        )

        real_largo = hist_graf.melt(
            id_vars="Periodo",
            value_vars=["VENTA", "Media 6 meses", "Mediana 6 meses"],
            var_name="Tipo",
            value_name="Valor",
        )
        futuro = pd.DataFrame({
            "Periodo": periodos,
            "Valor": pred,
            "Tipo": "Proyectado",
        })
        graf = pd.concat(
            [real_largo, futuro],
            ignore_index=True,
        )

        ch = (
            alt.Chart(graf)
            .mark_line(point=True)
            .encode(
                x=alt.X("Periodo:O", title="Periodo"),
                y=alt.Y(
                    "Valor:Q",
                    title="Venta",
                    axis=alt.Axis(format="$,.0f"),
                ),
                color=alt.Color("Tipo:N", title="Serie"),
                tooltip=[
                    "Periodo:O",
                    "Tipo:N",
                    alt.Tooltip("Valor:Q", format="$,.2f"),
                ],
            )
            .properties(
                height=380,
                title="Historial, tendencia movil y proyeccion",
            )
        )
        st.altair_chart(ch, use_container_width=True)

        # Proyeccion masiva original, sin cambios en la prediccion.
        filas = []
        for _, r in ult.iterrows():
            h = df[df.CODIGO == r.CODIGO]
            pr = predecir_recursivo(
                modelo,
                medianas,
                h,
                periodos,
            )
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
                "SEGMENTO_DIAGNOSTICO": segmento,
                f"Pred {periodos[0]}": pr[0],
                f"Pred {periodos[1]}": pr[1],
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
