import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(page_title="Prediccion mensual V3", page_icon="📦", layout="wide")
st.title("📦 Prediccion mensual de ventas V3")
st.caption("Pronostico directo H+1/H+2, Extra Trees, control de promociones, picos e intermitencia.")

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12,
}

LAGS = [1, 2, 3, 4, 6, 12]
VENTANAS = [3, 6, 12]
BASE_FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [f"mean{x}" for x in VENTANAS]
    + [f"median{x}" for x in VENTANAS]
    + [f"std{x}" for x in VENTANAS]
    + ["max3", "max6", "q75_12", "q90_12", "ceros6", "ceros12",
       "cv6", "ratio_lag1_mediana6", "pico_reciente", "es_intermitente",
       "es_promocion", "sin_mes_obj", "cos_mes_obj"]
)

# =========================================================
# PREPARACION DE DATOS
# =========================================================
def normalizar_codigo(serie):
    return (
        serie.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[eE]\+?", "E", regex=True)
    )


def detectar_promocion(df):
    texto = (
        df.get("DESCRIPCION", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("TIPO", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.upper()
    patron = r"PACK|TRIPACK|COMBO|PROMO|PROMOCIONAL|GRATIS|BONUS|\+|IN ?& ?OUT|IN ?OUT|OUT"
    return texto.str.contains(patron, regex=True, na=False).astype(int)


def completar_calendario(df):
    salida = []
    meta = [c for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"] if c in df.columns]
    for codigo, grupo in df.groupby("CODIGO"):
        grupo = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            grupo["AÑO"].astype(int).astype(str) + "-"
            + grupo["MES_NUM"].astype(int).astype(str) + "-01"
        )
        grupo.index = fechas
        indice = pd.date_range(fechas.min(), fechas.max(), freq="MS")
        grupo = grupo.reindex(indice)
        grupo["CODIGO"] = codigo
        grupo["VENTA"] = grupo["VENTA"].fillna(0.0)
        grupo["MES_NUM"] = grupo.index.month
        grupo["AÑO"] = grupo.index.year
        grupo["Periodo"] = grupo.index.year * 100 + grupo.index.month
        for col in meta:
            grupo[col] = grupo[col].ffill().bfill()
        salida.append(grupo.reset_index(drop=True))
    return pd.concat(salida, ignore_index=True)


@st.cache_data(show_spinner="Leyendo y normalizando archivos...")
def cargar_archivos(archivos):
    partes = []
    for archivo in archivos:
        hojas = pd.read_excel(archivo, sheet_name=None, engine="openpyxl")
        for nombre, hoja in hojas.items():
            if hoja is not None and not hoja.empty:
                partes.append(hoja)
    if not partes:
        raise ValueError("No se encontraron datos en los archivos.")

    df = pd.concat(partes, ignore_index=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    requeridas = ["CODIGO", "VENTA", "MES", "AÑO"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(faltantes))

    df["MES_NUM"] = df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce")
    df["CODIGO"] = normalizar_codigo(df["CODIGO"])
    df = df.dropna(subset=["MES_NUM", "AÑO", "VENTA"])
    df = df[df["CODIGO"].ne("") & df["CODIGO"].ne("nan")]
    df["Periodo"] = (df["AÑO"] * 100 + df["MES_NUM"]).astype(int)

    meta = {c: "first" for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"] if c in df.columns}
    mensual = (
        df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
        .agg({"VENTA": "sum", **meta})
    )
    mensual = completar_calendario(mensual)
    mensual["ES_PROMOCION"] = detectar_promocion(mensual)
    return mensual.sort_values(["CODIGO", "Periodo"]).reset_index(drop=True)

# =========================================================
# FEATURES Y MODELOS DIRECTOS
# =========================================================
def crear_features(df, horizonte):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")["VENTA"]

    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)

    for ventana in VENTANAS:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s, w=ventana: s.shift(1).rolling(w, min_periods=2).mean()
        )
        z[f"median{ventana}"] = grupo.transform(
            lambda s, w=ventana: s.shift(1).rolling(w, min_periods=2).median()
        )
        z[f"std{ventana}"] = grupo.transform(
            lambda s, w=ventana: s.shift(1).rolling(w, min_periods=2).std()
        ).fillna(0.0)

    z["max3"] = grupo.transform(lambda s: s.shift(1).rolling(3, min_periods=2).max())
    z["max6"] = grupo.transform(lambda s: s.shift(1).rolling(6, min_periods=3).max())
    z["q75_12"] = grupo.transform(lambda s: s.shift(1).rolling(12, min_periods=4).quantile(0.75))
    z["q90_12"] = grupo.transform(lambda s: s.shift(1).rolling(12, min_periods=4).quantile(0.90))
    z["ceros6"] = grupo.transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).apply(lambda x: np.sum(x == 0), raw=True)
    )
    z["ceros12"] = grupo.transform(
        lambda s: s.shift(1).rolling(12, min_periods=4).apply(lambda x: np.sum(x == 0), raw=True)
    )
    z["cv6"] = (z["std6"] / z["mean6"].replace(0, np.nan)).clip(0, 10)
    z["ratio_lag1_mediana6"] = (z["lag1"] / z["median6"].replace(0, np.nan)).clip(0, 10)
    z["pico_reciente"] = (z["ratio_lag1_mediana6"] > 1.8).astype(int)
    z["es_intermitente"] = (z["ceros6"] >= 3).astype(int)
    z["es_promocion"] = z["ES_PROMOCION"].fillna(0).astype(int)

    # El mes objetivo cambia según H+1 o H+2.
    mes_obj = ((z["MES_NUM"].astype(int) - 1 + horizonte) % 12) + 1
    z["sin_mes_obj"] = np.sin(2 * np.pi * mes_obj / 12)
    z["cos_mes_obj"] = np.cos(2 * np.pi * mes_obj / 12)
    z["TARGET"] = grupo.shift(-horizonte)
    z["PERIODO_OBJ"] = z.groupby("CODIGO")["Periodo"].shift(-horizonte)
    return z


def crear_modelo(tipo, parametros):
    if tipo == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=parametros["n_estimators"],
            max_depth=parametros["max_depth"],
            min_samples_leaf=parametros["min_samples_leaf"],
            max_features=parametros["max_features"],
            n_jobs=-1,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def entrenar_horizonte(df_train, horizonte, tipo, parametros):
    f = crear_features(df_train, horizonte)
    f = f.dropna(subset=["lag1", "lag2", "TARGET"]).copy()
    if f.empty:
        raise ValueError(f"No hay suficientes datos para entrenar H+{horizonte}.")
    medianas = f[BASE_FEATURES].median().fillna(0.0)
    X = f[BASE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medianas)
    y = np.log1p(f["TARGET"].clip(lower=0))
    modelo = crear_modelo(tipo, parametros)
    modelo.fit(X, y)
    return modelo, medianas


def fila_prediccion(historial, horizonte, medianas):
    # Añadir una fila ficticia para que shift(1) use el último mes real como lag1.
    hist = historial.sort_values("Periodo").copy()
    ultimo = hist.iloc[-1].copy()
    fecha = pd.Timestamp(int(ultimo["AÑO"]), int(ultimo["MES_NUM"]), 1) + pd.DateOffset(months=1)
    ficticia = ultimo.copy()
    ficticia["AÑO"] = fecha.year
    ficticia["MES_NUM"] = fecha.month
    ficticia["Periodo"] = fecha.year * 100 + fecha.month
    ficticia["VENTA"] = np.nan
    temp = pd.concat([hist, pd.DataFrame([ficticia])], ignore_index=True)
    f = crear_features(temp, horizonte).iloc[-1]
    X = pd.DataFrame([{c: f.get(c, np.nan) for c in BASE_FEATURES}])
    return X.replace([np.inf, -np.inf], np.nan).fillna(medianas)


def pronostico_intermitente(historial):
    ultimos = historial.sort_values("Periodo")["VENTA"].tail(12).astype(float)
    if ultimos.empty:
        return 0.0
    positivos = ultimos[ultimos > 0]
    if positivos.empty:
        return 0.0
    probabilidad = float((ultimos > 0).mean())
    return float(probabilidad * positivos.median())


def ajustar_sobreestimacion(pred, historial, es_promocion, aplicar_ajuste=True):
    if not aplicar_ajuste:
        return max(0.0, float(pred))
    ventas = historial.sort_values("Periodo")["VENTA"].tail(12).astype(float)
    if len(ventas) < 6:
        return max(0.0, float(pred))

    mediana6 = float(ventas.tail(6).median())
    q90 = float(ventas.quantile(0.90))
    ultimo = float(ventas.iloc[-1])
    es_pico = mediana6 > 0 and ultimo > 1.8 * mediana6

    # El recorte solo aplica a promocionales/combo con pico reciente.
    if int(es_promocion) == 1 and es_pico:
        limite = max(1.40 * mediana6, q90)
        pred = min(float(pred), limite)
    return max(0.0, float(pred))


def predecir_producto(modelo, medianas, historial, horizonte, usar_intermitencia, usar_ajuste):
    ultimos = historial.sort_values("Periodo")["VENTA"].tail(6)
    es_intermitente = len(ultimos) >= 3 and int((ultimos == 0).sum()) >= 3
    if usar_intermitencia and es_intermitente:
        return pronostico_intermitente(historial), "Intermitente"

    X = fila_prediccion(historial, horizonte, medianas)
    pred = float(np.expm1(modelo.predict(X)[0]))
    es_promo = int(historial.iloc[-1].get("ES_PROMOCION", 0))
    pred = ajustar_sobreestimacion(pred, historial, es_promo, usar_ajuste)
    return pred, "Modelo"

# =========================================================
# BACKTEST DIRECTO SIN FUGA DE INFORMACION
# =========================================================
def backtest_corte(df, corte, tipo, parametros, usar_intermitencia, usar_ajuste):
    train = df[df["Periodo"] <= corte].copy()
    modelos = {}
    for h in [1, 2]:
        modelos[h] = entrenar_horizonte(train, h, tipo, parametros)

    fecha_corte = pd.Timestamp(corte // 100, corte % 100, 1)
    objetivos = []
    for h in [1, 2]:
        f = fecha_corte + pd.DateOffset(months=h)
        objetivos.append(f.year * 100 + f.month)

    filas = []
    for codigo, hist in train.groupby("CODIGO"):
        hist = hist.sort_values("Periodo")
        if len(hist) < 5:
            continue
        for h, periodo_obj in zip([1, 2], objetivos):
            real = df[(df["CODIGO"] == codigo) & (df["Periodo"] == periodo_obj)]
            if real.empty:
                continue
            modelo, medianas = modelos[h]
            pred, metodo = predecir_producto(
                modelo, medianas, hist, h, usar_intermitencia, usar_ajuste
            )
            r = real.iloc[0]
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "TIPO": r.get("TIPO", ""),
                "Corte": corte,
                "Horizonte": h,
                "Periodo_Real": periodo_obj,
                "Actual": float(r["VENTA"]),
                "Prediccion": pred,
                "Metodo": metodo,
            })
    return pd.DataFrame(filas)


def calcular_metricas(resultados):
    if resultados.empty:
        return None
    y = resultados["Actual"].to_numpy(float)
    p = resultados["Prediccion"].to_numpy(float)
    den = np.abs(y).sum()
    return {
        "WAPE": 100 * np.abs(y - p).sum() / den if den > 0 else np.nan,
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p) if len(y) > 1 else np.nan,
    }

# =========================================================
# INTERFAZ
# =========================================================
st.subheader("1. Datos y configuracion")
archivos = st.file_uploader(
    "Sube los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True,
)

with st.sidebar:
    st.header("Configuracion V3")
    algoritmo = st.selectbox("Algoritmo", ["Extra Trees", "HistGradientBoosting"])
    n_estimators = st.select_slider("Numero de arboles", [300, 400, 600, 800], value=600)
    max_depth_op = st.selectbox("Profundidad maxima", [10, 14, 16, 20, 24, "Sin limite"], index=2)
    min_samples_leaf = st.select_slider("Minimo por hoja", [2, 3, 4, 5, 8], value=3)
    max_features = st.select_slider("Proporcion de variables", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
    usar_intermitencia = st.checkbox("Regla para productos intermitentes", value=True)
    usar_ajuste = st.checkbox("Control de sobreestimacion promocional", value=True)

parametros = {
    "n_estimators": n_estimators,
    "max_depth": None if max_depth_op == "Sin limite" else int(max_depth_op),
    "min_samples_leaf": min_samples_leaf,
    "max_features": max_features,
}

if archivos:
    try:
        df = cargar_archivos(archivos)
        ultimo_periodo = int(df["Periodo"].max())
        st.success(
            f"{len(df):,} registros mensuales normalizados | "
            f"{df['CODIGO'].nunique():,} productos | ultimo periodo: {ultimo_periodo}"
        )

        st.subheader("2. Backtest temporal en varias ventanas")
        periodos = sorted(df["Periodo"].unique())
        cortes_validos = periodos[12:-2]
        cortes_default = cortes_validos[-3:] if len(cortes_validos) >= 3 else cortes_validos
        cortes = st.multiselect(
            "Selecciona meses de corte; se evaluan los dos meses siguientes",
            cortes_validos,
            default=cortes_default,
        )

        if st.button("Ejecutar backtest", type="primary"):
            if not cortes:
                st.warning("Selecciona al menos un corte temporal.")
            else:
                resultados = []
                barra = st.progress(0)
                for i, corte in enumerate(cortes, start=1):
                    r = backtest_corte(
                        df, int(corte), algoritmo, parametros,
                        usar_intermitencia, usar_ajuste
                    )
                    if not r.empty:
                        resultados.append(r)
                    barra.progress(i / len(cortes))
                barra.empty()

                if resultados:
                    bt = pd.concat(resultados, ignore_index=True)
                    metricas = calcular_metricas(bt)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("WAPE promedio", f"{metricas['WAPE']:.2f}%")
                    c2.metric("MAE", f"${metricas['MAE']:,.2f}")
                    c3.metric("RMSE", f"${metricas['RMSE']:,.2f}")
                    c4.metric("R²", f"{metricas['R2']:.3f}")

                    por_corte = []
                    for corte, grupo in bt.groupby("Corte"):
                        m = calcular_metricas(grupo)
                        por_corte.append({"Corte": corte, **m, "Registros": len(grupo)})
                    st.dataframe(pd.DataFrame(por_corte), use_container_width=True, hide_index=True)

                    bt["Error_Abs"] = (bt["Actual"] - bt["Prediccion"]).abs()
                    bt["Aporte_WAPE_pp"] = 100 * bt["Error_Abs"] / max(bt["Actual"].abs().sum(), 1)
                    st.markdown("#### Productos con mayor impacto")
                    top = (
                        bt.groupby(["CODIGO", "DESCRIPCION", "CATEGORIA"], dropna=False)
                        .agg(Actual=("Actual", "sum"), Prediccion=("Prediccion", "sum"),
                             Error_Abs=("Error_Abs", "sum"), Aporte_WAPE_pp=("Aporte_WAPE_pp", "sum"))
                        .reset_index().nlargest(20, "Error_Abs")
                    )
                    st.dataframe(top, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Descargar detalle del backtest",
                        bt.to_csv(index=False).encode("utf-8-sig"),
                        "backtest_v3.csv", "text/csv"
                    )
                else:
                    st.warning("No hubo observaciones suficientes para esos cortes.")

        st.subheader("3. Prediccion directa de los siguientes dos meses")
        with st.spinner("Entrenando modelos finales H+1 y H+2..."):
            modelos_finales = {
                h: entrenar_horizonte(df, h, algoritmo, parametros) for h in [1, 2]
            }

        ultimos = df.sort_values("Periodo").groupby("CODIGO").tail(1)
        opciones = {
            r["CODIGO"]: f"{r['CODIGO']} - {r.get('DESCRIPCION', '')}"
            for _, r in ultimos.iterrows()
        }
        codigo = st.selectbox("Selecciona un producto", list(opciones), format_func=lambda x: opciones[x])
        historial = df[df["CODIGO"] == codigo].sort_values("Periodo")
        fecha_ultima = pd.Timestamp(ultimo_periodo // 100, ultimo_periodo % 100, 1)
        futuros = []
        for h in [1, 2]:
            f = fecha_ultima + pd.DateOffset(months=h)
            modelo, medianas = modelos_finales[h]
            pred, metodo = predecir_producto(
                modelo, medianas, historial, h, usar_intermitencia, usar_ajuste
            )
            futuros.append({"Periodo": f.year * 100 + f.month, "VENTA": pred, "Tipo": "Proyectado", "Metodo": metodo})

        st.write(
            f"**{futuros[0]['Periodo']}:** ${futuros[0]['VENTA']:,.2f} | "
            f"**{futuros[1]['Periodo']}:** ${futuros[1]['VENTA']:,.2f}"
        )
        graf = pd.concat([
            historial[["Periodo", "VENTA"]].assign(Tipo="Real"),
            pd.DataFrame(futuros)[["Periodo", "VENTA", "Tipo"]],
        ], ignore_index=True)
        chart = alt.Chart(graf).mark_line(point=True).encode(
            x=alt.X("Periodo:O", title="Periodo"),
            y=alt.Y("VENTA:Q", title="Venta"),
            color="Tipo:N",
            tooltip=["Periodo:O", alt.Tooltip("VENTA:Q", format="$,.2f"), "Tipo:N"],
        ).properties(height=350).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.subheader("4. Proyeccion masiva")
        if st.button("Generar proyeccion masiva"):
            filas = []
            progreso = st.progress(0)
            total = len(ultimos)
            for i, (_, r) in enumerate(ultimos.iterrows(), start=1):
                hist = df[df["CODIGO"] == r["CODIGO"]].sort_values("Periodo")
                preds = []
                metodos = []
                for h in [1, 2]:
                    modelo, medianas = modelos_finales[h]
                    pred, metodo = predecir_producto(
                        modelo, medianas, hist, h, usar_intermitencia, usar_ajuste
                    )
                    preds.append(pred); metodos.append(metodo)
                filas.append({
                    "CODIGO": r["CODIGO"],
                    "DESCRIPCION": r.get("DESCRIPCION", ""),
                    "CATEGORIA": r.get("CATEGORIA", ""),
                    "TIPO": r.get("TIPO", ""),
                    f"Pred_{futuros[0]['Periodo']}": preds[0],
                    f"Pred_{futuros[1]['Periodo']}": preds[1],
                    "Metodo_H1": metodos[0],
                    "Metodo_H2": metodos[1],
                    "ES_PROMOCION": int(r.get("ES_PROMOCION", 0)),
                })
                progreso.progress(i / total)
            progreso.empty()
            salida = pd.DataFrame(filas)
            st.dataframe(salida, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar proyeccion V3",
                salida.to_csv(index=False).encode("utf-8-sig"),
                "proyeccion_ventas_v3.csv", "text/csv"
            )

    except Exception as e:
        st.exception(e)
