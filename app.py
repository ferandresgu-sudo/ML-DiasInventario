import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import altair as alt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(
    page_title="Prediccion mensual de ventas (Prophet)",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Consulta acumulable de ventas y proyecciones (Prophet)")
st.caption(
    "Consulta acumulable por categoria, combo y descripcion, con interpretacion "
    "del indice estacional y del peso aplicado a la prediccion."
)

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12,
}

# Ya no necesitamos la lista FEATURES de lags, Prophet maneja la estacionalidad internamente.

# =========================================================
# FUNCIONES DEL MODELO (ADAPTADAS A PROPHET)
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


def nuevo_modelo_prophet():
    """
    Crea y configura un modelo Prophet para series mensuales con picos en Nov-Dic.
    - Se desactiva la estacionalidad diaria y semanal (no aplica a datos mensuales).
    - Se activa la estacionalidad anual con un Fourier order alto para capturar picos.
    - Se añaden los festivos de fin de año (Navidad, Año Nuevo) como regresores.
    """
    m = Prophet(
        yearly_seasonality=False,  # La manejaremos manualmente o con festivos
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Mejor para picos porcentuales
    )
    # Añadimos una estacionalidad mensual personalizada para capturar los picos
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    # Añadimos los festivos de fin de año (asumiendo México, ajústalo a tu país)
    # Esto captura el efecto de Navidad y Año Nuevo.
    m.add_country_holidays(country_name='MX')
    # Para reforzar Nov-Dic, podemos añadir un regresor binario manual
    # (se hará en la función de entrenamiento)
    return m


@st.cache_resource(show_spinner=False)
def entrenar_prophet_por_sku(df_completo):
    """
    Entrena un modelo Prophet por cada SKU.
    Devuelve un diccionario {codigo: modelo_entrenado}.
    """
    modelos = {}
    for codigo, grupo in df_completo.groupby("CODIGO"):
        # Prophet requiere columnas 'ds' (fecha) y 'y' (valor)
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(grupo['AÑO'].astype(str) + '-' + grupo['MES_NUM'].astype(str) + '-01'),
            'y': grupo['VENTA'].astype(float)
        }).sort_values('ds')
        
        # Agregamos un regresor binario para Nov-Dic (refuerzo extra)
        df_prophet['es_nov_dic'] = df_prophet['ds'].dt.month.isin([11, 12]).astype(int)
        
        # Si hay muy pocos datos, Prophet puede fallar. Saltamos SKUs con < 6 meses.
        if len(df_prophet) < 6:
            continue
            
        m = nuevo_modelo_prophet()
        m.add_regressor('es_nov_dic')
        
        try:
            m.fit(df_prophet)
            modelos[codigo] = m
        except Exception:
            # Si falla el ajuste, se omite este SKU (se predecirá 0 o se usará otro método)
            continue
    return modelos


def predecir_con_prophet(modelo, periodos_futuros):
    """
    Genera predicciones para una lista de periodos (formato YYYYMM) usando un modelo Prophet.
    """
    # Creamos un dataframe futuro con las fechas
    fechas_futuras = pd.to_datetime([f"{p//100}-{p%100:02d}-01" for p in periodos_futuros])
    future = pd.DataFrame({'ds': fechas_futuras})
    future['es_nov_dic'] = future['ds'].dt.month.isin([11, 12]).astype(int)
    
    forecast = modelo.predict(future)
    # Prophet puede devolver valores negativos, los corregimos a 0
    preds = forecast['yhat'].clip(lower=0).tolist()
    return preds


# =========================================================
# FUNCIONES DE DIAGNOSTICO (SIN CAMBIOS)
# =========================================================
def calcular_indice_estacional_nov_dic(historial, anio_limite=None):
    """Calcula estacionalidad Nov-Dic usando solo años anteriores al objetivo."""
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
    prediccion_modelo, historial, periodo_objetivo,
    peso_modelo=0.60, umbral_estacional=1.40,
):
    """
    Combina la predicción del modelo con la venta del mismo mes del año anterior
    solo para SKU estacionales y objetivos de noviembre/diciembre.
    """
    h = historial.sort_values("Periodo").copy()
    mes_objetivo = int(periodo_objetivo % 100)
    anio_objetivo = int(periodo_objetivo // 100)

    if mes_objetivo not in [11, 12]:
        return float(prediccion_modelo), 1.0, np.nan, False

    indice = calcular_indice_estacional_nov_dic(h, anio_limite=anio_objetivo)
    es_estacional = indice >= umbral_estacional
    if not es_estacional:
        return float(prediccion_modelo), indice, np.nan, False

    mismo_mes = h[(h["MES_NUM"] == mes_objetivo) & (h["AÑO"] < anio_objetivo)].sort_values("AÑO")
    if mismo_mes.empty:
        return float(prediccion_modelo), indice, np.nan, False

    venta_anio_anterior = float(mismo_mes.iloc[-1]["VENTA"])
    pred_ajustada = (peso_modelo * float(prediccion_modelo) + (1.0 - peso_modelo) * venta_anio_anterior)
    return max(0.0, pred_ajustada), indice, venta_anio_anterior, True


def ajustar_lista_estacional(
    predicciones_base, historial, periodos,
    peso_modelo=0.60, umbral_estacional=1.40,
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
def cargar(archivos, version_cache="v2-rapida-1"):
    partes = []
    columnas_utiles = {
        "CODIGO", "VENTA", "MES", "AÑO",
        "DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS",
    }

    for archivo in archivos:
        hojas = pd.read_excel(archivo, sheet_name=None, engine="openpyxl")
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

    df["MES_NUM"] = df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce")
    if "COMBOS" not in df.columns:
        df["COMBOS"] = 0
    df["COMBOS"] = pd.to_numeric(df["COMBOS"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    df["CODIGO"] = df["CODIGO"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    df = df.dropna(subset=["MES_NUM", "AÑO", "VENTA"])
    df["Periodo"] = (df["AÑO"] * 100 + df["MES_NUM"]).astype(int)

    meta = {c: "first" for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"] if c in df.columns}
    g = df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False).agg({"VENTA": "sum", **meta})

    return completar_calendario(g).sort_values(["CODIGO", "Periodo"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def backtest(df, n_holdout=2):
    tamanos = df.groupby("CODIGO").size()
    codigos = tamanos[tamanos >= n_holdout + 3].index
    base = df[df.CODIGO.isin(codigos)].copy()

    held = base.groupby("CODIGO", group_keys=False).tail(n_holdout)
    train = base.drop(index=held.index)
    
    # Entrenamos modelos Prophet solo con los datos de train
    modelos_bt = entrenar_prophet_por_sku(train)

    filas = []
    for codigo, real in held.groupby("CODIGO"):
        real = real.sort_values("Periodo")
        if codigo not in modelos_bt:
            # Si no se pudo entrenar, predecimos 0 (o podrías usar la media)
            pred = [0.0] * len(real)
        else:
            periodos_reales = real.Periodo.astype(int).tolist()
            pred = predecir_con_prophet(modelos_bt[codigo], periodos_reales)

        for paso, ((_, r), p) in enumerate(zip(real.iterrows(), pred), 1):
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "TIPO": r.get("TIPO", ""),
                "COMBOS": int(r.get("COMBOS", 0)),
                "Periodo_Real": int(r.Periodo),
                "Mes_Predecido": paso,
                "Prediccion": p,
                "Actual": float(r.VENTA),
            })

    return pd.DataFrame(filas)


# =========================================================
# RESTO DE FUNCIONES DIAGNOSTICAS (SIN CAMBIOS)
# =========================================================
def clasificar_producto_diagnostico(historial):
    ventas = historial.sort_values("Periodo")["VENTA"].tail(6).astype(float).to_numpy()
    if len(ventas) == 0:
        return "Intermitente"
    ceros = int(np.sum(ventas == 0))
    if ceros >= 3:
        return "Intermitente"
    media = float(np.mean(ventas))
    desviacion = float(np.std(ventas, ddof=1)) if len(ventas) >= 2 else 0.0
    cv = desviacion / media if media > 0 else np.inf
    if cv >= 1.0:
        return "Volatil"
    return "Estable"


@st.cache_data(show_spinner=False)
def crear_tabla_segmentacion(df):
    filas = []
    for codigo, hist in df.groupby("CODIGO"):
        filas.append({"CODIGO": codigo, "SEGMENTO": clasificar_producto_diagnostico(hist)})
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("SEGMENTO", as_index=False).agg(Productos=("CODIGO", "nunique"))
    orden = pd.CategoricalDtype(["Estable", "Intermitente", "Volatil"], ordered=True)
    resumen["SEGMENTO"] = resumen["SEGMENTO"].astype(orden)
    resumen = resumen.sort_values("SEGMENTO").reset_index(drop=True)
    return detalle, resumen


def preparar_diagnostico_backtest(bt, segmentos):
    diag = bt.merge(segmentos, on="CODIGO", how="left")
    diag["Error_Abs"] = (diag["Prediccion"] - diag["Actual"]).abs()
    diag["Error_Signo"] = diag["Prediccion"] - diag["Actual"]
    diag["Error_Cuadrado"] = diag["Error_Signo"] ** 2
    diag["Variacion_%"] = np.where(diag["Actual"] > 0, 100 * diag["Error_Signo"] / diag["Actual"], np.nan)
    diag["Situacion"] = np.where(diag["Error_Signo"] >= 0, "Sobreestimacion", "Subestimacion")
    return diag


def aplicar_segmentadores_diagnostico(bt_diag):
    st.subheader("Segmentadores del diagnostico")
    st.caption("Los filtros de CATEGORIA, COMBO y PRODUCTO solo cambian las metricas, tablas y graficas mostradas. El modelo y sus predicciones permanecen iguales.")

    datos = bt_diag.copy()
    for col in ["CATEGORIA", "DESCRIPCION"]:
        if col not in datos.columns:
            datos[col] = "Sin dato"
        datos[col] = datos[col].fillna("Sin dato").astype(str).str.strip()
        datos.loc[datos[col].eq(""), col] = "Sin dato"

    if "COMBOS" not in datos.columns:
        datos["COMBOS"] = 0
    datos["COMBOS"] = pd.to_numeric(datos["COMBOS"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    f1, f2, f3 = st.columns(3)
    with f1:
        categorias = sorted(datos["CATEGORIA"].unique().tolist())
        categorias_sel = st.multiselect("CATEGORIA", options=categorias, default=[], placeholder="Todas las categorias", key="seg_categoria_bt")
    if categorias_sel:
        datos = datos[datos["CATEGORIA"].isin(categorias_sel)].copy()

    with f2:
        combos_sel = st.selectbox("COMBO", options=["Todos", "No combo (0)", "Combo (1)"], index=0, key="seg_combo_bt")
    if combos_sel == "No combo (0)":
        datos = datos[datos["COMBOS"] == 0].copy()
    elif combos_sel == "Combo (1)":
        datos = datos[datos["COMBOS"] == 1].copy()

    opciones_producto = datos[["CODIGO", "DESCRIPCION"]].drop_duplicates().assign(
        Producto=lambda x: x["CODIGO"].astype(str) + " - " + x["DESCRIPCION"].astype(str)
    ).sort_values("Producto")
    mapa_producto = dict(zip(opciones_producto["Producto"], opciones_producto["CODIGO"]))

    with f3:
        productos_sel = st.multiselect("PRODUCTO", options=list(mapa_producto.keys()), default=[], placeholder="Todos los productos", key="seg_producto_bt")
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
    st.markdown("### Venta real acumulada vs prediccion acumulada")

    totales = pd.DataFrame({
        "Serie": ["Venta real", "Prediccion"],
        "Monto": [float(bt_diag["Actual"].sum()), float(bt_diag["Prediccion"].sum())],
    })
    total_real = float(bt_diag["Actual"].sum())
    total_pred = float(bt_diag["Prediccion"].sum())
    diferencia = total_pred - total_real
    diferencia_pct = 100 * diferencia / total_real if total_real != 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Venta real acumulada", f"${total_real:,.2f}")
    c2.metric("Prediccion acumulada", f"${total_pred:,.2f}")
    c3.metric("Diferencia acumulada", f"${diferencia:,.2f}", delta=f"{diferencia_pct:+.2f}%" if pd.notna(diferencia_pct) else None, delta_color="inverse")

    barras_totales = alt.Chart(totales).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Serie:N", title=None, sort=["Venta real", "Prediccion"]),
        y=alt.Y("Monto:Q", title="Monto acumulado", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=["Venta real", "Prediccion"], range=["#1f77b4", "#ff7f0e"]), legend=None),
        tooltip=["Serie:N", alt.Tooltip("Monto:Q", format="$,.2f")],
    ).properties(height=380)
    etiquetas = barras_totales.mark_text(align="center", baseline="bottom", dy=-5, color="white").encode(text=alt.Text("Monto:Q", format="$,.0f"))
    st.altair_chart(barras_totales + etiquetas, use_container_width=True)

    por_horizonte = bt_diag.groupby("Mes_Predecido", as_index=False).agg(
        Venta_Real=("Actual", "sum"), Prediccion=("Prediccion", "sum")
    ).melt(id_vars="Mes_Predecido", value_vars=["Venta_Real", "Prediccion"], var_name="Serie", value_name="Monto")
    por_horizonte["Serie"] = por_horizonte["Serie"].replace({"Venta_Real": "Venta real", "Prediccion": "Prediccion"})

    barras_horizonte = alt.Chart(por_horizonte).mark_bar().encode(
        x=alt.X("Mes_Predecido:O", title="Horizonte pronosticado"),
        xOffset="Serie:N",
        y=alt.Y("Monto:Q", title="Monto acumulado", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=["Venta real", "Prediccion"], range=["#1f77b4", "#ff7f0e"]), title=None),
        tooltip=["Mes_Predecido:O", "Serie:N", alt.Tooltip("Monto:Q", format="$,.2f")],
    ).properties(height=340, title="Venta real y prediccion acumuladas por horizonte")
    st.altair_chart(barras_horizonte, use_container_width=True)

    st.markdown("### Venta real vs prediccion por producto-mes")
    maximo = float(max(bt_diag["Actual"].max(), bt_diag["Prediccion"].max(), 1))
    puntos = alt.Chart(bt_diag).mark_circle(opacity=0.65).encode(
        x=alt.X("Actual:Q", title="Venta real", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("Prediccion:Q", title="Prediccion", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("SEGMENTO:N", title="Segmento"),
        size=alt.Size("Error_Abs:Q", title="Error absoluto", scale=alt.Scale(range=[20, 500])),
        tooltip=["CODIGO:N", "DESCRIPCION:N", "CATEGORIA:N", "SEGMENTO:N", "Periodo_Real:O", alt.Tooltip("Actual:Q", format="$,.2f"), alt.Tooltip("Prediccion:Q", format="$,.2f"), alt.Tooltip("Error_Abs:Q", format="$,.2f")],
    )
    diagonal = alt.Chart(pd.DataFrame({"Actual": [0, maximo], "Prediccion": [0, maximo]})).mark_line(color="#d62728", strokeDash=[6, 4]).encode(x="Actual:Q", y="Prediccion:Q")
    st.altair_chart((puntos + diagonal).properties(height=430), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        top_rmse = bt_diag.groupby(["CODIGO", "DESCRIPCION"], dropna=False, as_index=False).agg(
            Error_Absoluto=("Error_Abs", "sum"), Error_Cuadrado=("Error_Cuadrado", "sum"), Venta_Real=("Actual", "sum"), Prediccion=("Prediccion", "sum")
        ).nlargest(15, "Error_Cuadrado")
        top_rmse["Producto"] = top_rmse["CODIGO"].astype(str) + " - " + top_rmse["DESCRIPCION"].fillna("").astype(str)
        graf_rmse = alt.Chart(top_rmse).mark_bar().encode(
            x=alt.X("Error_Cuadrado:Q", title="Error cuadratico acumulado"),
            y=alt.Y("Producto:N", sort="-x", title=None),
            color=alt.Color("Error_Absoluto:Q", scale=alt.Scale(scheme="yelloworangered"), title="Error absoluto"),
            tooltip=["Producto:N", alt.Tooltip("Venta_Real:Q", format="$,.2f"), alt.Tooltip("Prediccion:Q", format="$,.2f"), alt.Tooltip("Error_Absoluto:Q", format="$,.2f")],
        ).properties(height=500, title="Productos que mas elevan el RMSE")
        st.altair_chart(graf_rmse, use_container_width=True)

    with col_b:
        sesgo = bt_diag.groupby(["CODIGO", "DESCRIPCION"], dropna=False, as_index=False).agg(
            Error_Signo=("Error_Signo", "sum"), Error_Abs=("Error_Abs", "sum")
        ).nlargest(15, "Error_Abs")
        sesgo["Producto"] = sesgo["CODIGO"].astype(str) + " - " + sesgo["DESCRIPCION"].fillna("").astype(str)
        sesgo["Situacion"] = np.where(sesgo["Error_Signo"] >= 0, "Sobreestimacion", "Subestimacion")
        graf_sesgo = alt.Chart(sesgo).mark_bar().encode(
            x=alt.X("Error_Signo:Q", title="Prediccion menos venta real", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("Producto:N", sort="-x", title=None),
            color=alt.Color("Situacion:N", scale=alt.Scale(domain=["Sobreestimacion", "Subestimacion"], range=["#d62728", "#1f77b4"]), title=None),
            tooltip=["Producto:N", "Situacion:N", alt.Tooltip("Error_Signo:Q", format="$,.2f")],
        ).properties(height=500, title="Sobreestimacion y subestimacion")
        st.altair_chart(graf_sesgo, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        hist_error = bt_diag[bt_diag["Variacion_%"].between(-200, 200)].copy()
        graf_hist = alt.Chart(hist_error).mark_bar(color="#4c78a8").encode(
            x=alt.X("Variacion_%:Q", bin=alt.Bin(maxbins=40), title="Error porcentual"),
            y=alt.Y("count():Q", title="Numero de predicciones"),
            tooltip=[alt.Tooltip("count():Q", title="Predicciones")],
        ).properties(height=350, title="Distribucion del error porcentual (-200% a 200%)")
        st.altair_chart(graf_hist, use_container_width=True)

    with col_d:
        pareto = bt_diag.groupby(["CODIGO", "DESCRIPCION"], dropna=False, as_index=False).agg(Error_Abs=("Error_Abs", "sum")).sort_values("Error_Abs", ascending=False)
        pareto["Producto"] = pareto["CODIGO"].astype(str) + " - " + pareto["DESCRIPCION"].fillna("").astype(str)
        pareto["Porcentaje_Error"] = 100 * pareto["Error_Abs"] / max(pareto["Error_Abs"].sum(), 1)
        pareto["Porcentaje_Acumulado"] = pareto["Porcentaje_Error"].cumsum()
        pareto["Orden"] = np.arange(1, len(pareto) + 1)
        p30 = pareto.head(30)
        barras = alt.Chart(p30).mark_bar(color="#4c78a8").encode(
            x=alt.X("Orden:O", title="Productos ordenados por error"),
            y=alt.Y("Error_Abs:Q", title="Error absoluto", axis=alt.Axis(format="$,.0f")),
            tooltip=["Producto:N", alt.Tooltip("Error_Abs:Q", format="$,.2f"), alt.Tooltip("Porcentaje_Acumulado:Q", format=".1f")],
        )
        linea = alt.Chart(p30).mark_line(color="#e45756", point=True).encode(
            x="Orden:O", y=alt.Y("Porcentaje_Acumulado:Q", title="% acumulado", scale=alt.Scale(domain=[0, 100]))
        )
        graf_pareto = alt.layer(barras, linea).resolve_scale(y="independent").properties(height=350, title="Pareto de contribucion al error")
        st.altair_chart(graf_pareto, use_container_width=True)


@st.cache_data(show_spinner=False)
def crear_indices_rapidos(df):
    historicos = {codigo: grupo.sort_values("Periodo").copy() for codigo, grupo in df.groupby("CODIGO", sort=False)}
    ultimos = pd.concat([grupo.tail(1) for grupo in historicos.values()], ignore_index=True)
    return historicos, ultimos


# =========================================================
# INTERFAZ
# =========================================================
archivos = st.file_uploader("Sube ventas 2024, 2025 y 2026", type=["xlsx", "xlsm"], accept_multiple_files=True)

with st.sidebar:
    st.header("Ajuste estacional")
    usar_ajuste_estacional = st.checkbox("Activar ajuste Nov-Dic", value=True)
    peso_modelo_estacional = st.slider("Peso de la prediccion del modelo", min_value=0.50, max_value=0.90, value=0.60, step=0.05, help="El resto del peso se asigna a la venta del mismo mes del año anterior.")
    umbral_estacional = st.slider("Umbral indice Nov-Dic", min_value=1.10, max_value=2.50, value=1.40, step=0.05)

if archivos:
    try:
        df = cargar(archivos)
        st.success(f"{len(df):,} observaciones mensuales normalizadas | {df.CODIGO.nunique():,} productos")
        st.caption("La primera lectura y el entrenamiento pueden tardar. Mientras los archivos no cambien, Streamlit reutiliza la informacion almacenada en cache.")

        historicos_por_codigo, ult = crear_indices_rapidos(df)
        
        # Entrenamos los modelos Prophet para todos los SKUs
        with st.spinner("Entrenando modelos Prophet por SKU..."):
            modelos_prophet = entrenar_prophet_por_sku(df)

        st.subheader("Consulta acumulable por categoria, combo y descripcion")
        st.caption("Los filtros acumulan la venta historica y las predicciones de los SKU seleccionados. El modelo Prophet no se reentrena ni cambia.")

        ult_consulta = ult.copy()
        for col in ["CATEGORIA", "DESCRIPCION"]:
            if col not in ult_consulta.columns:
                ult_consulta[col] = "Sin dato"
            ult_consulta[col] = ult_consulta[col].fillna("Sin dato").astype(str).str.strip()
            ult_consulta.loc[ult_consulta[col].eq(""), col] = "Sin dato"

        if "COMBOS" not in ult_consulta.columns:
            ult_consulta["COMBOS"] = 0
        ult_consulta["COMBOS"] = pd.to_numeric(ult_consulta["COMBOS"], errors="coerce").fillna(0).clip(0, 1).astype(int)

        fc1, fc2 = st.columns(2)
        with fc1:
            categoria_consulta = st.selectbox("CATEGORIA", ["Todas"] + sorted(ult_consulta["CATEGORIA"].unique().tolist()), key="consulta_categoria")
        if categoria_consulta != "Todas":
            ult_consulta = ult_consulta[ult_consulta["CATEGORIA"] == categoria_consulta].copy()

        with fc2:
            combo_consulta = st.selectbox("COMBO", ["Todos", "No combo (0)", "Combo (1)"], key="consulta_combo")
        if combo_consulta == "No combo (0)":
            ult_consulta = ult_consulta[ult_consulta["COMBOS"] == 0].copy()
        elif combo_consulta == "Combo (1)":
            ult_consulta = ult_consulta[ult_consulta["COMBOS"] == 1].copy()

        descripciones_disponibles = sorted(ult_consulta["DESCRIPCION"].dropna().unique().tolist())
        modo_descripcion = st.radio("Seleccion de DESCRIPCION", ["Todas las descripciones disponibles", "Seleccion manual"], horizontal=True, key="consulta_modo_descripcion")

        if modo_descripcion == "Todas las descripciones disponibles":
            descripciones_sel = descripciones_disponibles
        else:
            descripciones_sel = st.multiselect("DESCRIPCION", options=descripciones_disponibles, default=[], placeholder="Selecciona una o varias descripciones", key="consulta_descripciones")

        if not descripciones_sel:
            st.warning("No hay descripciones seleccionadas. Elige al menos una o selecciona 'Todas las descripciones disponibles'.")
            st.stop()

        seleccion_sku = ult_consulta[ult_consulta["DESCRIPCION"].isin(descripciones_sel)].copy()
        codigos_seleccionados = seleccion_sku["CODIGO"].drop_duplicates().tolist()

        if not codigos_seleccionados:
            st.warning("No hay productos para la combinacion de filtros seleccionada.")
            st.stop()

        st.info(f"Consulta formada por {len(codigos_seleccionados):,} SKU y {len(descripciones_sel):,} descripciones.")

        ultimo = int(df["Periodo"].max())
        fecha = pd.Timestamp(ultimo // 100, ultimo % 100, 1)
        periodos = []
        for i in [1, 2]:
            f = fecha + pd.DateOffset(months=i)
            periodos.append(f.year * 100 + f.month)

        historicos_seleccionados = [historicos_por_codigo[codigo] for codigo in codigos_seleccionados if codigo in historicos_por_codigo]
        hist_acumulado = pd.concat(historicos_seleccionados, ignore_index=True).groupby("Periodo", as_index=False).agg(VENTA=("VENTA", "sum")).sort_values("Periodo")

        predicciones_sku = []
        for codigo in codigos_seleccionados:
            hist_producto = historicos_por_codigo[codigo]
            
            # Predicción con Prophet
            if codigo in modelos_prophet:
                pred_producto = predecir_con_prophet(modelos_prophet[codigo], periodos)
            else:
                # Fallback: si no hay modelo, usamos la media de los últimos 3 meses
                pred_producto = [hist_producto["VENTA"].tail(3).mean()] * len(periodos)
            
            if usar_ajuste_estacional:
                pred_ajustada, diag_est = ajustar_lista_estacional(pred_producto, hist_producto, periodos, peso_modelo=peso_modelo_estacional, umbral_estacional=umbral_estacional)
            else:
                pred_ajustada = list(pred_producto)
                diag_est = [{"Indice_Nov_Dic": calcular_indice_estacional_nov_dic(hist_producto), "Venta_Mismo_Mes_Anterior": np.nan, "Ajuste_Aplicado": False} for _ in periodos]
            
            meta_producto = seleccion_sku[seleccion_sku["CODIGO"] == codigo].iloc[0]
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
                f"Referencia {periodos[0]}": diag_est[0]["Venta_Mismo_Mes_Anterior"],
                f"Referencia {periodos[1]}": diag_est[1]["Venta_Mismo_Mes_Anterior"],
                "Ajuste_Estacional": any(d["Ajuste_Aplicado"] for d in diag_est),
            })

        detalle_pred = pd.DataFrame(predicciones_sku)
        pred_base_acum = [float(detalle_pred[f"Pred Base {periodos[0]}"].sum()), float(detalle_pred[f"Pred Base {periodos[1]}"].sum())]
        pred_acum = [float(detalle_pred[f"Pred Ajustada {periodos[0]}"].sum()), float(detalle_pred[f"Pred Ajustada {periodos[1]}"].sum())]

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("SKU acumulados", f"{len(codigos_seleccionados):,}")
        pc2.metric(f"Prediccion ajustada {periodos[0]}", f"${pred_acum[0]:,.2f}", delta=f"Base Prophet: ${pred_base_acum[0]:,.2f}", delta_color="off")
        pc3.metric(f"Prediccion ajustada {periodos[1]}", f"${pred_acum[1]:,.2f}", delta=f"Base Prophet: ${pred_base_acum[1]:,.2f}", delta_color="off")

        real_graf = hist_acumulado.rename(columns={"VENTA": "Valor"}).copy()
        real_graf["Serie"] = "Venta historica acumulada"

        ultimo_real = real_graf.sort_values("Periodo").iloc[-1]
        futuro_graf = pd.DataFrame({
            "Periodo": [int(ultimo_real["Periodo"])] + periodos,
            "Valor": [float(ultimo_real["Valor"])] + pred_acum,
            "Serie": "Proyeccion acumulada",
        })

        graf_consulta = pd.concat([real_graf, futuro_graf], ignore_index=True).sort_values(["Serie", "Periodo"])

        chart_consulta = alt.Chart(graf_consulta).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Periodo:O", title="Periodo", sort="ascending", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Valor:Q", title="Venta acumulada", axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False)),
            color=alt.Color("Serie:N", title="Serie", scale=alt.Scale(domain=["Venta historica acumulada", "Proyeccion acumulada"], range=["#1f77b4", "#ff7f0e"])),
            detail="Serie:N",
            order=alt.Order("Periodo:O", sort="ascending"),
            tooltip=["Periodo:O", "Serie:N", alt.Tooltip("Valor:Q", format="$,.2f")],
        ).properties(height=420, title="Venta historica y proyeccion acumuladas").interactive()
        st.altair_chart(chart_consulta, use_container_width=True)

        detalle_pred[f"Participacion {periodos[0]} %"] = 100 * detalle_pred[f"Pred Ajustada {periodos[0]}"] / max(pred_acum[0], 1)
        detalle_pred[f"Participacion {periodos[1]} %"] = 100 * detalle_pred[f"Pred Ajustada {periodos[1]}"] / max(pred_acum[1], 1)
        detalle_pred = detalle_pred.sort_values(f"Pred Ajustada {periodos[0]}", ascending=False)

        with st.expander("Ver aportacion de cada SKU al acumulado"):
            st.dataframe(detalle_pred.style.format({
                f"Pred Ajustada {periodos[0]}": "${:,.2f}",
                f"Pred Ajustada {periodos[1]}": "${:,.2f}",
                f"Participacion {periodos[0]} %": "{:.2f}%",
                f"Participacion {periodos[1]} %": "{:.2f}%",
            }), use_container_width=True, hide_index=True)

        st.markdown("### Interpretacion del ajuste estacional")
        porcentaje_v2 = int(round(peso_modelo_estacional * 100))
        porcentaje_anterior = 100 - porcentaje_v2
        incremento_umbral = int(round((umbral_estacional - 1.0) * 100))

        if peso_modelo_estacional >= 0.90:
            interpretacion_peso = "Ajuste muy ligero"
        elif peso_modelo_estacional >= 0.80:
            interpretacion_peso = "Ajuste conservador"
        elif peso_modelo_estacional >= 0.70:
            interpretacion_peso = "Ajuste moderado"
        elif peso_modelo_estacional >= 0.60:
            interpretacion_peso = "Ajuste estacional fuerte"
        else:
            interpretacion_peso = "Alta dependencia del año anterior"

        if umbral_estacional < 1.20:
            interpretacion_umbral = "Detecta estacionalidad ligera"
        elif umbral_estacional < 1.40:
            interpretacion_umbral = "Detecta estacionalidad moderada"
        elif umbral_estacional < 2.00:
            interpretacion_umbral = "Exige estacionalidad clara"
        else:
            interpretacion_umbral = "Solo detecta estacionalidad muy fuerte"

        i1, i2 = st.columns(2)
        with i1:
            st.metric("Peso actual de la prediccion del modelo", f"{peso_modelo_estacional:.2f}", help="Porcentaje de confianza asignado al modelo Prophet.")
            st.write(f"**{porcentaje_v2}%** proviene de la prediccion de Prophet y **{porcentaje_anterior}%** de la venta del mismo mes del año anterior.")
            st.caption(interpretacion_peso)

        with i2:
            st.metric("Umbral actual del indice Nov-Dic", f"{umbral_estacional:.2f}", help="Nivel minimo de estacionalidad requerido para aplicar el ajuste.")
            st.write(f"El ajuste se aplica cuando Nov-Dic vende al menos **{incremento_umbral}% mas** que el resto del año.")
            st.caption(interpretacion_umbral)

        with st.expander("Ver como interpretar el indice Nov-Dic", expanded=False):
            tabla_indice = pd.DataFrame({
                "Indice": [1.00, 1.20, 1.40, 1.80, 2.00, 3.00],
                "Significado": ["Nov-Dic vende igual que el resto del año", "Nov-Dic vende 20% mas", "Nov-Dic vende 40% mas", "Nov-Dic vende 80% mas", "Nov-Dic vende el doble", "Nov-Dic vende tres veces mas"],
            })
            st.dataframe(tabla_indice.style.format({"Indice": "{:.2f}"}), use_container_width=True, hide_index=True)
            st.info(f"Configuracion actual: el producto debe alcanzar un indice de {umbral_estacional:.2f} o superior para recibir el ajuste.")

        with st.expander("Ver como interpretar el peso del modelo", expanded=False):
            tabla_pesos = pd.DataFrame({
                "Peso Modelo": [0.90, 0.80, 0.70, 0.60, 0.50],
                "Modelo": ["90%", "80%", "70%", "60%", "50%"],
                "Año anterior": ["10%", "20%", "30%", "40%", "50%"],
                "Interpretacion": ["Ajuste muy ligero", "Ajuste conservador", "Ajuste moderado", "Ajuste estacional fuerte", "Alta dependencia del año anterior"],
            })
            st.dataframe(tabla_pesos.style.format({"Peso Modelo": "{:.2f}"}), use_container_width=True, hide_index=True)
            st.info(f"Configuracion actual: {porcentaje_v2}% Modelo + {porcentaje_anterior}% mismo mes del año anterior.")

        st.caption("La formula aplicada es: Prediccion ajustada = (Peso Modelo x Prediccion Prophet) + ((1 - Peso Modelo) x Venta del mismo mes del año anterior).")

    except Exception as e:
        st.exception(e)
