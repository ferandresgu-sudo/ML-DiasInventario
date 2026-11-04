import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Prediccion mensual V3", page_icon="📦", layout="wide")
st.title("📦 Pronostico mensual V3")
st.caption("Segmentacion de productos, estacionalidad, faltantes, atipicos, pesos y ensamble temporal.")

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12,
}
META = ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]
LAGS = [1, 2, 3, 4, 6, 9, 12]
FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [f"{m}{w}" for w in (3, 6, 12) for m in ("mean", "median", "std", "min", "max")]
    + ["trend3", "trend6", "cero3", "cero6", "registro3", "registro6",
       "venta_mismo_mes_anterior", "sin_mes", "cos_mes"]
)


def periodo_siguiente(periodo, pasos=1):
    fecha = pd.Timestamp(periodo // 100, periodo % 100, 1) + pd.DateOffset(months=pasos)
    return fecha.year * 100 + fecha.month


@st.cache_data(show_spinner="Leyendo y normalizando archivos...")
def cargar_datos(archivos):
    partes = []
    for archivo in archivos:
        hojas = pd.read_excel(archivo, sheet_name=None, engine="openpyxl")
        for nombre_hoja, hoja in hojas.items():
            hoja = hoja.copy()
            hoja["ARCHIVO_ORIGEN"] = archivo.name
            hoja["HOJA_ORIGEN"] = nombre_hoja
            partes.append(hoja)

    df = pd.concat(partes, ignore_index=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    requeridas = ["CODIGO", "VENTA", "MES", "AÑO"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes))

    df["MES_NUM"] = df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce")
    df["CODIGO"] = df["CODIGO"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df = df.dropna(subset=["MES_NUM", "AÑO", "VENTA", "CODIGO"])
    df = df[df["CODIGO"].ne("")]
    df["VENTA"] = df["VENTA"].clip(lower=0)
    df["Periodo"] = (df["AÑO"] * 100 + df["MES_NUM"]).astype(int)

    agregaciones = {"VENTA": "sum", **{c: "first" for c in META if c in df.columns}}
    base = df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False).agg(agregaciones)

    # Calendario completo. REGISTRO_EXISTENTE diferencia cero real de ausencia en el archivo.
    salida = []
    for codigo, grupo in base.groupby("CODIGO"):
        grupo = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            grupo["AÑO"].astype(int).astype(str) + "-" +
            grupo["MES_NUM"].astype(int).astype(str) + "-01"
        )
        grupo.index = fechas
        grupo["REGISTRO_EXISTENTE"] = 1
        indice = pd.date_range(fechas.min(), fechas.max(), freq="MS")
        grupo = grupo.reindex(indice)
        grupo["CODIGO"] = codigo
        grupo["REGISTRO_EXISTENTE"] = grupo["REGISTRO_EXISTENTE"].fillna(0).astype(int)
        grupo["VENTA"] = grupo["VENTA"].fillna(0.0)
        grupo["MES_NUM"] = grupo.index.month
        grupo["AÑO"] = grupo.index.year
        grupo["Periodo"] = grupo.index.year * 100 + grupo.index.month
        for c in META:
            if c in grupo.columns:
                grupo[c] = grupo[c].ffill().bfill()
        salida.append(grupo.reset_index(drop=True))

    completo = pd.concat(salida, ignore_index=True)
    return completo.sort_values(["CODIGO", "Periodo"]).reset_index(drop=True)


def clasificar_productos(df):
    resumen = df.groupby("CODIGO").agg(
        PROMEDIO=("VENTA", "mean"),
        DESVIACION=("VENTA", "std"),
        PROP_CEROS=("VENTA", lambda s: float((s == 0).mean())),
        MESES=("VENTA", "size"),
    ).reset_index()
    resumen["CV"] = np.where(
        resumen["PROMEDIO"] > 0,
        resumen["DESVIACION"].fillna(0) / resumen["PROMEDIO"],
        999.0,
    )
    resumen["SEGMENTO"] = np.select(
        [resumen["PROP_CEROS"] >= 0.25, resumen["CV"] >= 0.50],
        ["Intermitente", "Volatil"],
        default="Estable",
    )
    return df.merge(resumen[["CODIGO", "PROMEDIO", "CV", "PROP_CEROS", "SEGMENTO"]], on="CODIGO", how="left")


def marcar_atipicos(df):
    def procesar(grupo):
        grupo = grupo.sort_values("Periodo").copy()
        historico = grupo["VENTA"].shift(1)
        mediana = historico.rolling(6, min_periods=3).median()
        mad = (historico - mediana).abs().rolling(6, min_periods=3).median()
        limite = mediana + 4 * mad.replace(0, np.nan)
        grupo["VENTA_EXTRAORDINARIA"] = (grupo["VENTA"] > limite).fillna(False).astype(int)
        return grupo
    return df.groupby("CODIGO", group_keys=False).apply(procesar, include_groups=False).reset_index(drop=True)


def crear_features(df):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    venta = z.groupby("CODIGO")["VENTA"]
    registro = z.groupby("CODIGO")["REGISTRO_EXISTENTE"]

    for lag in LAGS:
        z[f"lag{lag}"] = venta.shift(lag)

    for ventana in (3, 6, 12):
        z[f"mean{ventana}"] = venta.transform(lambda s: s.shift(1).rolling(ventana, min_periods=2).mean())
        z[f"median{ventana}"] = venta.transform(lambda s: s.shift(1).rolling(ventana, min_periods=2).median())
        z[f"std{ventana}"] = venta.transform(lambda s: s.shift(1).rolling(ventana, min_periods=2).std()).fillna(0)
        z[f"min{ventana}"] = venta.transform(lambda s: s.shift(1).rolling(ventana, min_periods=2).min())
        z[f"max{ventana}"] = venta.transform(lambda s: s.shift(1).rolling(ventana, min_periods=2).max())

    z["trend3"] = (z["lag1"] - z["lag3"]) / 3
    z["trend6"] = (z["lag1"] - z["lag6"]) / 6
    z["cero3"] = venta.transform(lambda s: s.shift(1).rolling(3, min_periods=2).apply(lambda a: (a == 0).mean()))
    z["cero6"] = venta.transform(lambda s: s.shift(1).rolling(6, min_periods=2).apply(lambda a: (a == 0).mean()))
    z["registro3"] = registro.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    z["registro6"] = registro.transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    z["venta_mismo_mes_anterior"] = z["lag12"]
    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    return z


def modelos_nuevos():
    modelo_log = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=1.0,
        n_jobs=-1, random_state=42,
    )
    modelo_raw = RandomForestRegressor(
        n_estimators=400, max_depth=16, min_samples_leaf=2,
        max_features=0.8, n_jobs=-1, random_state=42,
    )
    return modelo_log, modelo_raw


def entrenar(df):
    f = crear_features(df).dropna(subset=["lag1", "lag2", "VENTA"]).copy()
    medianas = f[FEATURES].median().fillna(0)
    X = f[FEATURES].fillna(medianas)
    y = f["VENTA"].clip(lower=0)

    # Menor peso a ausencias convertidas a cero y a ventas extraordinarias.
    pesos = np.ones(len(f), dtype=float)
    pesos[f["REGISTRO_EXISTENTE"].eq(0).to_numpy()] *= 0.25
    pesos[f["VENTA_EXTRAORDINARIA"].eq(1).to_numpy()] *= 0.40
    pesos[f["SEGMENTO"].eq("Intermitente").to_numpy()] *= 0.80

    modelo_log, modelo_raw = modelos_nuevos()
    modelo_log.fit(X, np.log1p(y), sample_weight=pesos)
    modelo_raw.fit(X, y, sample_weight=pesos)
    return modelo_log, modelo_raw, medianas


def vector_futuro(hist, mes, medianas):
    hist = hist.sort_values("Periodo")
    valores = hist["VENTA"].astype(float).tolist()
    registros = hist["REGISTRO_EXISTENTE"].astype(float).tolist()
    q = {}
    for lag in LAGS:
        q[f"lag{lag}"] = valores[-lag] if len(valores) >= lag else np.nan
    for ventana in (3, 6, 12):
        a = np.asarray(valores[-ventana:], dtype=float)
        q[f"mean{ventana}"] = a.mean() if len(a) >= 2 else np.nan
        q[f"median{ventana}"] = np.median(a) if len(a) >= 2 else np.nan
        q[f"std{ventana}"] = a.std(ddof=1) if len(a) >= 2 else 0
        q[f"min{ventana}"] = a.min() if len(a) >= 2 else np.nan
        q[f"max{ventana}"] = a.max() if len(a) >= 2 else np.nan
    q["trend3"] = (q["lag1"] - q["lag3"]) / 3 if pd.notna(q["lag3"]) else 0
    q["trend6"] = (q["lag1"] - q["lag6"]) / 6 if pd.notna(q["lag6"]) else 0
    q["cero3"] = np.mean(np.asarray(valores[-3:]) == 0) if len(valores) >= 2 else 0
    q["cero6"] = np.mean(np.asarray(valores[-6:]) == 0) if len(valores) >= 2 else 0
    q["registro3"] = np.mean(registros[-3:]) if registros else 0
    q["registro6"] = np.mean(registros[-6:]) if registros else 0
    q["venta_mismo_mes_anterior"] = q["lag12"]
    q["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    q["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    return pd.DataFrame([q], columns=FEATURES).fillna(medianas)


def croston_tsb(ventas, alpha=0.25, beta=0.15):
    ventas = np.asarray(ventas, dtype=float)
    if len(ventas) == 0 or np.all(ventas <= 0):
        return 0.0
    demanda = ventas[ventas > 0]
    z = demanda[0]
    p = float(ventas[0] > 0)
    for valor in ventas[1:]:
        ocurrio = float(valor > 0)
        p = beta * ocurrio + (1 - beta) * p
        if valor > 0:
            z = alpha * valor + (1 - alpha) * z
    return max(0.0, p * z)


def predecir_recursivo(modelo_log, modelo_raw, medianas, historico, periodos, alpha_log=0.8):
    hist = historico.copy()
    predicciones = []
    segmento = hist["SEGMENTO"].iloc[-1]

    for periodo in periodos:
        X = vector_futuro(hist, periodo % 100, medianas)
        pred_log = max(0.0, float(np.expm1(modelo_log.predict(X)[0])))
        pred_raw = max(0.0, float(modelo_raw.predict(X)[0]))
        pred_ml = alpha_log * pred_log + (1 - alpha_log) * pred_raw

        estacional = X["venta_mismo_mes_anterior"].iloc[0]
        reciente = X["median3"].iloc[0]

        if segmento == "Intermitente":
            pred = 0.60 * croston_tsb(hist["VENTA"].values) + 0.40 * pred_log
        elif segmento == "Volatil":
            pred = 0.70 * pred_ml + 0.20 * estacional + 0.10 * reciente
        else:
            pred = 0.75 * pred_ml + 0.20 * estacional + 0.05 * reciente

        pred = max(0.0, float(pred))
        predicciones.append(pred)
        fila = hist.iloc[-1].copy()
        fila["Periodo"] = periodo
        fila["AÑO"] = periodo // 100
        fila["MES_NUM"] = periodo % 100
        fila["VENTA"] = pred
        fila["REGISTRO_EXISTENTE"] = 1
        fila["VENTA_EXTRAORDINARIA"] = 0
        hist = pd.concat([hist, pd.DataFrame([fila])], ignore_index=True)
    return predicciones


def separar_holdout(df, meses):
    valido = df.groupby("CODIGO").filter(lambda g: len(g) >= meses + 4)
    held = valido.groupby("CODIGO", group_keys=False).tail(meses)
    train = valido.drop(index=held.index)
    return train, held


def ejecutar_backtest(train, held, alpha_log):
    modelo_log, modelo_raw, medianas = entrenar(train)
    filas = []
    for codigo, reales in held.groupby("CODIGO"):
        reales = reales.sort_values("Periodo")
        historico = train[train["CODIGO"].eq(codigo)].sort_values("Periodo")
        periodos = reales["Periodo"].astype(int).tolist()
        preds = predecir_recursivo(modelo_log, modelo_raw, medianas, historico, periodos, alpha_log)
        for paso, ((_, real), pred) in enumerate(zip(reales.iterrows(), preds), start=1):
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": real.get("DESCRIPCION", ""),
                "CATEGORIA": real.get("CATEGORIA", ""),
                "SEGMENTO": real["SEGMENTO"],
                "Periodo_Real": int(real["Periodo"]),
                "Mes_Predecido": paso,
                "Actual": float(real["VENTA"]),
                "Prediccion": float(pred),
            })
    return pd.DataFrame(filas)


def calcular_metricas(df):
    y = df["Actual"].to_numpy()
    p = df["Prediccion"].to_numpy()
    denominador = np.abs(y).sum()
    return {
        "WAPE": 100 * np.abs(y - p).sum() / denominador if denominador > 0 else np.nan,
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p),
    }


archivos = st.file_uploader(
    "Sube los archivos mensuales de ventas",
    type=["xlsx", "xlsm"], accept_multiple_files=True,
)

if archivos:
    try:
        df = cargar_datos(archivos)
        df = clasificar_productos(df)
        df = marcar_atipicos(df)

        st.success(f"{df['CODIGO'].nunique():,} productos y {len(df):,} observaciones normalizadas")
        st.dataframe(
            df.groupby("SEGMENTO")["CODIGO"].nunique().rename("Productos").reset_index(),
            hide_index=True, use_container_width=True,
        )

        meses_holdout = st.slider("Meses reservados para backtest", 1, 4, 2)
        alpha_log = st.slider("Peso del modelo logaritmico", 0.0, 1.0, 0.8, 0.05)
        train, held = separar_holdout(df, meses_holdout)

        with st.spinner("Entrenando y ejecutando backtest temporal..."):
            bt = ejecutar_backtest(train, held, alpha_log)

        metricas = calcular_metricas(bt)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WAPE", f"{metricas['WAPE']:.2f}%")
        c2.metric("MAE", f"${metricas['MAE']:,.2f}")
        c3.metric("RMSE", f"${metricas['RMSE']:,.2f}")
        c4.metric("R²", f"{metricas['R2']:.3f}")

        st.subheader("Resultados por segmento")
        por_segmento = []
        for segmento, grupo in bt.groupby("SEGMENTO"):
            m = calcular_metricas(grupo)
            por_segmento.append({"SEGMENTO": segmento, "Registros": len(grupo), **m})
        st.dataframe(pd.DataFrame(por_segmento), hide_index=True, use_container_width=True)

        bt["Error_Abs"] = (bt["Actual"] - bt["Prediccion"]).abs()
        bt["Variacion_%"] = np.where(
            bt["Actual"] > 0,
            100 * (bt["Prediccion"] - bt["Actual"]) / bt["Actual"],
            np.nan,
        )
        st.subheader("20 predicciones con mayor error")
        st.dataframe(bt.nlargest(20, "Error_Abs"), hide_index=True, use_container_width=True)

        # ========================================================
        # ANALISIS DE SKU QUE GENERAN ERRORES EXTREMOS
        # ========================================================
        st.subheader("Analisis de SKU con errores extremos")
        st.caption(
            "Un error se considera extremo cuando supera el mayor valor entre "
            "el percentil seleccionado y el multiplicador del MAE global."
        )

        col_ext1, col_ext2 = st.columns(2)
        with col_ext1:
            percentil_extremo = st.slider(
                "Percentil para error extremo", 90, 99, 95, 1,
                help="95 identifica el 5% de observaciones con mayor error absoluto."
            )
        with col_ext2:
            multiplicador_mae = st.slider(
                "Multiplicador del MAE", 1.0, 5.0, 2.0, 0.25,
                help="2.0 exige que el error sea al menos dos veces el MAE global."
            )

        umbral_percentil = bt["Error_Abs"].quantile(percentil_extremo / 100)
        umbral_mae = metricas["MAE"] * multiplicador_mae
        umbral_extremo = max(umbral_percentil, umbral_mae)
        bt["ES_ERROR_EXTREMO"] = bt["Error_Abs"].ge(umbral_extremo)
        bt["Error_Signado"] = bt["Prediccion"] - bt["Actual"]
        bt["TIPO_ERROR"] = np.select(
            [bt["Error_Signado"] > 0, bt["Error_Signado"] < 0],
            ["Sobreestimacion", "Subestimacion"],
            default="Exacto"
        )

        errores_extremos = bt[bt["ES_ERROR_EXTREMO"]].copy()
        error_total = bt["Error_Abs"].sum()
        error_extremo_total = errores_extremos["Error_Abs"].sum()
        aporte_extremos = 100 * error_extremo_total / error_total if error_total > 0 else 0

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Umbral extremo", f"${umbral_extremo:,.2f}")
        e2.metric("Registros extremos", f"{len(errores_extremos):,}")
        e3.metric("SKU involucrados", f"{errores_extremos['CODIGO'].nunique():,}")
        e4.metric("Aporte al error total", f"{aporte_extremos:.1f}%")

        if errores_extremos.empty:
            st.info("No se detectaron errores extremos con los umbrales seleccionados.")
        else:
            sku_extremos = errores_extremos.groupby(
                ["CODIGO", "DESCRIPCION", "CATEGORIA", "SEGMENTO"],
                dropna=False, as_index=False
            ).agg(
                Veces_Extremo=("ES_ERROR_EXTREMO", "sum"),
                Error_Abs_Total=("Error_Abs", "sum"),
                Error_Abs_Promedio=("Error_Abs", "mean"),
                Error_Abs_Maximo=("Error_Abs", "max"),
                Venta_Real=("Actual", "sum"),
                Venta_Predicha=("Prediccion", "sum"),
                Sesgo=("Error_Signado", "sum"),
            )
            sku_extremos["WAPE_SKU"] = np.where(
                sku_extremos["Venta_Real"].abs() > 0,
                100 * sku_extremos["Error_Abs_Total"] / sku_extremos["Venta_Real"].abs(),
                np.nan
            )
            sku_extremos["Aporte_Error_%"] = np.where(
                error_total > 0,
                100 * sku_extremos["Error_Abs_Total"] / error_total,
                0.0
            )
            sku_extremos["Diagnostico"] = np.select(
                [
                    sku_extremos["Sesgo"] > 0.20 * sku_extremos["Error_Abs_Total"],
                    sku_extremos["Sesgo"] < -0.20 * sku_extremos["Error_Abs_Total"],
                    sku_extremos["SEGMENTO"].eq("Intermitente"),
                    sku_extremos["SEGMENTO"].eq("Volatil"),
                ],
                [
                    "Sobreestimacion recurrente: revisar ceros, bajas o descontinuacion",
                    "Subestimacion recurrente: revisar promocion, estacionalidad o quiebre previo",
                    "Demanda intermitente: revisar Croston/TSB y disponibilidad",
                    "Alta volatilidad: revisar promociones, precio e inventario",
                ],
                default="Error puntual: revisar dato, precio, promocion e inventario"
            )
            sku_extremos = sku_extremos.sort_values("Error_Abs_Total", ascending=False)
            sku_extremos["Aporte_Acumulado_%"] = sku_extremos["Aporte_Error_%"].cumsum()
            sku_extremos["PRIORIDAD"] = np.select(
                [
                    sku_extremos["Aporte_Acumulado_%"] <= 50,
                    sku_extremos["Aporte_Acumulado_%"] <= 80,
                ],
                ["A - Critica", "B - Alta"],
                default="C - Seguimiento"
            )

            st.markdown("#### Ranking de SKU por contribucion al error")
            st.dataframe(
                sku_extremos[[
                    "PRIORIDAD", "CODIGO", "DESCRIPCION", "CATEGORIA", "SEGMENTO",
                    "Veces_Extremo", "Error_Abs_Total", "Error_Abs_Maximo", "WAPE_SKU",
                    "Aporte_Error_%", "Aporte_Acumulado_%", "Sesgo", "Diagnostico"
                ]],
                hide_index=True, use_container_width=True,
                column_config={
                    "Error_Abs_Total": st.column_config.NumberColumn(format="$%,.2f"),
                    "Error_Abs_Maximo": st.column_config.NumberColumn(format="$%,.2f"),
                    "WAPE_SKU": st.column_config.NumberColumn(format="%.2f%%"),
                    "Aporte_Error_%": st.column_config.NumberColumn(format="%.2f%%"),
                    "Aporte_Acumulado_%": st.column_config.NumberColumn(format="%.2f%%"),
                    "Sesgo": st.column_config.NumberColumn(format="$%,.2f"),
                }
            )

            st.markdown("#### Pareto de contribucion al error")
            top_pareto = sku_extremos.head(20).set_index("CODIGO")[["Aporte_Error_%"]]
            st.bar_chart(top_pareto, horizontal=True)

            codigos_extremos = sku_extremos["CODIGO"].tolist()
            sku_seleccionado = st.selectbox(
                "Revisar detalle de un SKU extremo",
                codigos_extremos,
                format_func=lambda codigo: (
                    f"{codigo} - " + str(
                        sku_extremos.loc[sku_extremos['CODIGO'].eq(codigo), 'DESCRIPCION'].iloc[0]
                    )
                ),
                key="sku_extremo_detalle"
            )
            detalle_sku = bt[bt["CODIGO"].eq(sku_seleccionado)].sort_values("Periodo_Real").copy()
            st.dataframe(
                detalle_sku[[
                    "CODIGO", "DESCRIPCION", "Periodo_Real", "Mes_Predecido",
                    "Actual", "Prediccion", "Error_Abs", "Variacion_%",
                    "TIPO_ERROR", "ES_ERROR_EXTREMO"
                ]],
                hide_index=True, use_container_width=True
            )

            st.download_button(
                "Descargar analisis de SKU extremos",
                sku_extremos.to_csv(index=False).encode("utf-8-sig"),
                "analisis_sku_errores_extremos.csv",
                "text/csv",
            )
            st.download_button(
                "Descargar detalle de observaciones extremas",
                errores_extremos.to_csv(index=False).encode("utf-8-sig"),
                "detalle_errores_extremos.csv",
                "text/csv",
            )

        with st.spinner("Entrenando modelo final y generando proyeccion..."):
            modelo_log, modelo_raw, medianas = entrenar(df)
            ultimo_periodo = int(df["Periodo"].max())
            periodos_futuros = [periodo_siguiente(ultimo_periodo, 1), periodo_siguiente(ultimo_periodo, 2)]
            ultimos = df.groupby("CODIGO", group_keys=False).tail(1)
            filas = []
            for _, registro in ultimos.iterrows():
                codigo = registro["CODIGO"]
                historico = df[df["CODIGO"].eq(codigo)]
                preds = predecir_recursivo(
                    modelo_log, modelo_raw, medianas, historico, periodos_futuros, alpha_log
                )
                filas.append({
                    "CODIGO": codigo,
                    "DESCRIPCION": registro.get("DESCRIPCION", ""),
                    "CATEGORIA": registro.get("CATEGORIA", ""),
                    "SEGMENTO": registro["SEGMENTO"],
                    f"Pred {periodos_futuros[0]}": preds[0],
                    f"Pred {periodos_futuros[1]}": preds[1],
                })
            pronostico = pd.DataFrame(filas)

        st.subheader("Proyeccion masiva")
        st.dataframe(pronostico, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar proyeccion CSV",
            pronostico.to_csv(index=False).encode("utf-8-sig"),
            "proyeccion_ventas_v3.csv",
            "text/csv",
        )
        st.download_button(
            "Descargar backtest CSV",
            bt.to_csv(index=False).encode("utf-8-sig"),
            "backtest_ventas_v3.csv",
            "text/csv",
        )

    except Exception as exc:
        st.exception(exc)



