import streamlit as st
import pandas as pd
import numpy as np
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(
    page_title="Forecast estacional CatBoost",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Consulta acumulable de ventas y proyecciones")
st.caption(
    "Consulta acumulable por categoria, combo y descripcion, con interpretacion "
    "del indice estacional y del peso aplicado a la forecast CatBoost."
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

NUM_FEATURES = [
    "lag1", "lag2", "lag3", "lag4", "lag6", "lag12", "lag24",
    "mean3", "std3", "mean6", "std6", "mean12", "std12",
    "sin_mes", "cos_mes", "MES_NUM", "AÑO", "tendencia_3_12",
]
CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
FEATURES = NUM_FEATURES + CAT_FEATURES

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

    for lag in [1, 2, 3, 4, 6, 12, 24]:
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
    z["tendencia_3_12"] = np.where(
        z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0
    )
    for c in CAT_FEATURES:
        if c not in z.columns:
            z[c] = "SIN_DATO"
        z[c] = z[c].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError(
            "Falta CatBoost. Instala una vez con: pip install catboost"
        )
    return CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.035,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=8.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

@st.cache_resource(show_spinner=False)
def entrenar(df):
    f = crear_features(df).dropna(subset=["lag1", "lag2", "VENTA"]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    for c in CAT_FEATURES:
        f[c] = f[c].fillna("SIN_DATO").astype(str)
    X = f[FEATURES]
    # Nov-Dic y los años recientes pesan más, porque son el objetivo real.
    peso_temporada = np.where(f["MES_NUM"].isin([11, 12]), 2.75, 1.0)
    anio_min = int(f["AÑO"].min())
    peso_recencia = 1.0 + 0.20 * (f["AÑO"].astype(int) - anio_min)
    sample_weight = peso_temporada * peso_recencia
    modelo = nuevo_modelo()
    modelo.fit(
        X, np.log1p(f["VENTA"].clip(lower=0)),
        cat_features=CAT_FEATURES, sample_weight=sample_weight,
    )
    return modelo, medianas

def vector_futuro(hist, periodo, medianas):
    mes_num = int(periodo % 100)
    anio = int(periodo // 100)
    hist = hist.sort_values("Periodo")
    vals = hist["VENTA"].astype(float).tolist()
    q = {
        f"lag{lag}": vals[-lag] if len(vals) >= lag else np.nan
        for lag in [1, 2, 3, 4, 6, 12, 24]
    }
    for w in [3, 6, 12]:
        a = np.asarray(vals[-w:], dtype=float)
        q[f"mean{w}"] = a.mean() if len(a) >= 2 else np.nan
        q[f"std{w}"] = a.std(ddof=1) if len(a) >= 2 else 0.0
    q["sin_mes"] = np.sin(2 * np.pi * mes_num / 12)
    q["cos_mes"] = np.cos(2 * np.pi * mes_num / 12)
    q["MES_NUM"] = mes_num
    q["AÑO"] = anio
    q["tendencia_3_12"] = (
        q["mean3"] / q["mean12"]
        if pd.notna(q["mean12"]) and abs(q["mean12"]) > 1e-9 else 1.0
    )
    ultimo = hist.iloc[-1]
    for c in CAT_FEATURES:
        q[c] = str(ultimo.get(c, "SIN_DATO"))
    x = pd.DataFrame([q], columns=FEATURES)
    x[NUM_FEATURES] = x[NUM_FEATURES].fillna(medianas)
    for c in CAT_FEATURES:
        x[c] = x[c].fillna("SIN_DATO").astype(str)
    return x

def predecir_recursivo(modelo, medianas, historico, periodos):
    hist = historico.copy()
    preds = []
    for periodo in periodos:
        p = float(np.expm1(modelo.predict(vector_futuro(hist, periodo, medianas))[0]))
        p = max(0.0, p)
        preds.append(p)
        base = hist.iloc[-1].copy()
        base["Periodo"] = periodo
        base["AÑO"] = periodo // 100
        base["MES_NUM"] = periodo % 100
        base["VENTA"] = p
        hist = pd.concat([hist, pd.DataFrame([base])], ignore_index=True)
    return preds

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


def calcular_crecimiento_ytd(historial, anio_objetivo):
    h = historial.copy()
    actual = h[(h["AÑO"] == anio_objetivo) & (h["MES_NUM"] <= 10)]
    previo = h[(h["AÑO"] == anio_objetivo - 1) & (h["MES_NUM"] <= 10)]
    meses = sorted(set(actual["MES_NUM"]).intersection(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    if vp <= 0:
        return 1.0
    return float(np.clip(va / vp, 0.80, 1.40))

def ajustar_prediccion_estacional(
    prediccion_modelo, historial, periodo_objetivo,
    peso_modelo=0.60, umbral_estacional=1.10,
):
    """Ensamble CatBoost + mismo mes histórico escalado por tendencia YTD.

    A diferencia de la V2, no copia 2025 sin crecimiento. La referencia del
    mismo mes se multiplica por el crecimiento comparable Ene-Oct de 2026.
    """
    h = historial.sort_values("Periodo").copy()
    mes_objetivo = int(periodo_objetivo % 100)
    anio_objetivo = int(periodo_objetivo // 100)
    if mes_objetivo not in [11, 12]:
        return float(prediccion_modelo), 1.0, np.nan, False
    indice = calcular_indice_estacional_nov_dic(h, anio_limite=anio_objetivo)
    mismo_mes = h[(h["MES_NUM"] == mes_objetivo) & (h["AÑO"] < anio_objetivo)].sort_values("AÑO")
    if mismo_mes.empty:
        return float(prediccion_modelo), indice, np.nan, False
    # 65% año reciente y 35% año previo, cuando ambos existen.
    refs = mismo_mes.tail(2)["VENTA"].astype(float).to_numpy()
    referencia_hist = refs[-1] if len(refs) == 1 else 0.65 * refs[-1] + 0.35 * refs[-2]
    crecimiento = calcular_crecimiento_ytd(h, anio_objetivo)
    referencia_tendencia = referencia_hist * crecimiento
    es_estacional = indice >= umbral_estacional
    if not es_estacional:
        return float(prediccion_modelo), indice, referencia_tendencia, False
    pred_ensamble = peso_modelo * float(prediccion_modelo) + (1.0 - peso_modelo) * referencia_tendencia
    # Piso prudente: si 2026 crece vs 2025, Nov-Dic no queda debajo de 2025.
    referencia_2025 = float(refs[-1])
    piso = referencia_2025 * (1.0 + 0.35 * max(crecimiento - 1.0, 0.0))
    pred_ajustada = max(pred_ensamble, piso)
    return max(0.0, pred_ajustada), indice, referencia_tendencia, True

def ajustar_lista_estacional(
    predicciones_base,
    historial,
    periodos,
    peso_modelo=0.60,
    umbral_estacional=1.10,
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
def cargar(archivos, version_cache="catboost-estacional-v1"):
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
                "TIPO": r.get("TIPO", ""),
                "COMBOS": int(r.get("COMBOS", 0)),
                "Periodo_Real": int(r.Periodo),
                "Mes_Predecido": paso,
                "Prediccion": p,
                "Actual": float(r.VENTA),
            })

    return pd.DataFrame(filas)


def calcular_metricas_forecast(resultados):
    """Metricas globales sobre pares producto-mes reales y pronosticados."""
    if resultados is None or resultados.empty:
        return {"WAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "SESGO_PCT": np.nan, "REAL": 0.0, "PRED": 0.0}
    y = resultados["Actual"].to_numpy(dtype=float)
    p = resultados["Prediccion"].to_numpy(dtype=float)
    denom = max(np.abs(y).sum(), 1.0)
    return {
        "WAPE": 100.0 * np.abs(y - p).sum() / denom,
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "R2": r2_score(y, p) if len(y) >= 2 and np.var(y) > 0 else np.nan,
        "SESGO_PCT": 100.0 * (p.sum() - y.sum()) / max(abs(y.sum()), 1.0),
        "REAL": float(y.sum()),
        "PRED": float(p.sum()),
    }

@st.cache_data(show_spinner=False)
def backtest_periodos_objetivo(df, periodos_objetivo, peso_modelo=0.60,
                                umbral_estacional=1.10,
                                aplicar_ajuste=True):
    """Simula un forecast real: entrena solo con datos anteriores al primer objetivo.

    Evita fuga de informacion y evalua exactamente los meses indicados.
    """
    periodos_objetivo = sorted(int(x) for x in periodos_objetivo)
    corte = periodos_objetivo[0]
    train = df[df["Periodo"] < corte].copy()
    real = df[df["Periodo"].isin(periodos_objetivo)].copy()
    if train.empty or real.empty:
        return pd.DataFrame()
    modelo_bt, medianas_bt = entrenar(train)
    filas = []
    codigos = sorted(set(train["CODIGO"]).intersection(real["CODIGO"]))
    for codigo in codigos:
        hist = train[train["CODIGO"] == codigo].sort_values("Periodo")
        real_sku = real[real["CODIGO"] == codigo].sort_values("Periodo")
        disponibles = real_sku["Periodo"].astype(int).tolist()
        if hist.empty or not disponibles:
            continue
        pred_base = predecir_recursivo(modelo_bt, medianas_bt, hist, disponibles)
        if aplicar_ajuste:
            pred_final, diagnostico = ajustar_lista_estacional(
                pred_base, hist, disponibles,
                peso_modelo=peso_modelo,
                umbral_estacional=umbral_estacional,
            )
        else:
            pred_final = pred_base
            diagnostico = [{"Ajuste_Aplicado": False} for _ in disponibles]
        for paso, ((_, r), pb, pf, dg) in enumerate(
            zip(real_sku.iterrows(), pred_base, pred_final, diagnostico), 1
        ):
            filas.append({
                "Prueba": f"{periodos_objetivo[0]}-{periodos_objetivo[-1]}",
                "CODIGO": codigo,
                "DESCRIPCION": r.get("DESCRIPCION", ""),
                "CATEGORIA": r.get("CATEGORIA", ""),
                "COMBOS": int(r.get("COMBOS", 0)),
                "Periodo_Real": int(r["Periodo"]),
                "Mes_Predecido": paso,
                "Pred_Base": float(pb),
                "Prediccion": float(pf),
                "Actual": float(r["VENTA"]),
                "Ajuste_Estacional": bool(dg.get("Ajuste_Aplicado", False)),
            })
    return pd.DataFrame(filas)

def mostrar_metricas_backtest(nombre, resultados):
    st.markdown(f"#### {nombre}")
    if resultados.empty:
        st.warning("No hay suficientes registros para ejecutar esta prueba.")
        return
    m = calcular_metricas_forecast(resultados)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WAPE", f"{m['WAPE']:.2f}%")
    c2.metric("MAE", f"${m['MAE']:,.2f}")
    c3.metric("RMSE", f"${m['RMSE']:,.2f}")
    c4.metric("R²", f"{m['R2']:.3f}" if pd.notna(m['R2']) else "N/D")
    c5.metric("Sesgo acumulado", f"{m['SESGO_PCT']:+.2f}%")
    st.caption(
        f"Venta real: ${m['REAL']:,.2f} | Pronóstico: ${m['PRED']:,.2f} | "
        f"Registros producto-mes: {len(resultados):,}"
    )
    por_mes = resultados.groupby("Periodo_Real", as_index=False).agg(
        Venta_Real=("Actual", "sum"), Pronostico=("Prediccion", "sum")
    )
    por_mes["Error_pct"] = np.where(
        por_mes["Venta_Real"].abs() > 0,
        100 * (por_mes["Pronostico"] - por_mes["Venta_Real"]) / por_mes["Venta_Real"],
        np.nan,
    )
    st.dataframe(
        por_mes.style.format({
            "Venta_Real": "${:,.2f}", "Pronostico": "${:,.2f}",
            "Error_pct": "{:+.2f}%"
        }), use_container_width=True, hide_index=True
    )

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
    st.header("Ajuste estacional")
    usar_ajuste_estacional = st.checkbox(
        "Activar ajuste Nov-Dic", value=True
    )
    peso_modelo_estacional = st.slider(
        "Peso de la forecast CatBoost",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="El resto del peso se asigna a la venta del mismo mes del año anterior.",
    )
    umbral_estacional = st.slider(
        "Umbral indice Nov-Dic",
        min_value=1.00,
        max_value=2.00,
        value=1.10,
        step=0.05,
    )

if archivos:
    try:
        df = cargar(archivos)
        st.success(
            f"{len(df):,} observaciones mensuales normalizadas | "
            f"{df.CODIGO.nunique():,} productos"
        )
        st.caption(
            "La primera lectura y el entrenamiento pueden tardar. Mientras los archivos "
            "no cambien, Streamlit reutiliza la informacion almacenada en cache."
        )

        # Preparacion requerida para la consulta acumulable.
        historicos_por_codigo, ult = crear_indices_rapidos(df)

        # Entrenamiento final original, sin cambios.
        modelo, medianas = entrenar(df)

        # =====================================================
        # VALIDACION TEMPORAL SIN FUGA DE INFORMACION
        st.subheader("Validacion temporal del forecast")
        st.caption("La prueba principal es Nov-Dic 2025: entrena solo hasta octubre de 2025 y compara contra las ventas reales. Sep-Oct 2026 es una prueba secundaria de estabilidad reciente.")
        pruebas_disponibles = {
            "Nov-Dic 2025 (principal)": [202511, 202512],
            "Sep-Oct 2026 (secundaria)": [202609, 202610],
            "Nov-Dic 2024 (historial corto)": [202411, 202412],
        }
        pruebas_sel = st.multiselect(
            "Pruebas de backtesting", list(pruebas_disponibles),
            default=["Nov-Dic 2025 (principal)", "Sep-Oct 2026 (secundaria)"],
        )
        if st.button("Ejecutar validacion temporal", type="primary"):
            if not pruebas_sel:
                st.warning("Selecciona al menos una prueba.")
            for nombre_prueba in pruebas_sel:
                with st.spinner(f"Calculando {nombre_prueba}..."):
                    r_bt = backtest_periodos_objetivo(
                        df, pruebas_disponibles[nombre_prueba],
                        peso_modelo=peso_modelo_estacional,
                        umbral_estacional=umbral_estacional,
                        aplicar_ajuste=usar_ajuste_estacional,
                    )
                mostrar_metricas_backtest(nombre_prueba, r_bt)
        with st.expander("Como interpretar estas pruebas"):
            st.markdown(
                "- **Nov-Dic 2025 es la medicion decisiva**, porque reproduce la temporada objetivo.\n"
                "- **Sep-Oct 2026 es secundaria**: comprueba nivel reciente, no los picos de fin de ano.\n"
                "- **Nov-Dic 2024 es una prueba de arranque**: el modelo aun no conoce una temporada Nov-Dic anterior.\n"
                "- Prioriza WAPE y sesgo para el total; RMSE detecta errores extremos por SKU. R2 es complementario."
            )

        # CONSULTA ACUMULABLE: CATEGORIA, COMBO Y DESCRIPCION
        # =====================================================
        st.subheader("Consulta acumulable por categoria, combo y descripcion")
        st.caption(
            "Los filtros acumulan la venta historica y las predicciones de los SKU "
            "seleccionados. El modelo V2 no se reentrena ni cambia."
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
                f"Referencia {periodos[0]}": diag_est[0]["Venta_Mismo_Mes_Anterior"],
                f"Referencia {periodos[1]}": diag_est[1]["Venta_Mismo_Mes_Anterior"],
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
            delta=f"Base CatBoost: ${pred_base_acum[0]:,.2f}",
            delta_color="off",
        )
        pc3.metric(
            f"Prediccion ajustada {periodos[1]}",
            f"${pred_acum[1]:,.2f}",
            delta=f"Base CatBoost: ${pred_base_acum[1]:,.2f}",
            delta_color="off",
        )

        real_graf = hist_acumulado.rename(columns={"VENTA": "Valor"}).copy()
        real_graf["Serie"] = "Venta historica acumulada"

        # Conexion visual: la proyeccion comienza en el ultimo punto historico.
        # Este punto compartido solo afecta la apariencia de la grafica.
        # No modifica las predicciones, metricas ni acumulados.
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
                title="Venta historica y proyeccion acumuladas",
            )
            .interactive()
        )
        st.altair_chart(chart_consulta, use_container_width=True)

        # Participacion de cada SKU en la proyeccion acumulada.
        detalle_pred[f"Participacion {periodos[0]} %"] = (
            100 * detalle_pred[f"Pred Ajustada {periodos[0]}"]
            / max(pred_acum[0], 1)
        )
        detalle_pred[f"Participacion {periodos[1]} %"] = (
            100 * detalle_pred[f"Pred Ajustada {periodos[1]}"]
            / max(pred_acum[1], 1)
        )
        detalle_pred = detalle_pred.sort_values(
            f"Pred Ajustada {periodos[0]}", ascending=False
        )

        with st.expander("Ver aportacion de cada SKU al acumulado"):
            st.dataframe(
                detalle_pred.style.format({
                    f"Pred Ajustada {periodos[0]}": "${:,.2f}",
                    f"Pred Ajustada {periodos[1]}": "${:,.2f}",
                    f"Participacion {periodos[0]} %": "{:.2f}%",
                    f"Participacion {periodos[1]} %": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )


        # -----------------------------------------------------
        # INTERPRETACION DEL AJUSTE ESTACIONAL
        # -----------------------------------------------------
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
            st.metric(
                "Peso actual de la forecast CatBoost",
                f"{peso_modelo_estacional:.2f}",
                help="Porcentaje de confianza asignado al modelo CatBoost estacional.",
            )
            st.write(
                f"**{porcentaje_v2}%** proviene de la forecast CatBoost y "
                f"**{porcentaje_anterior}%** de la venta del mismo mes del año anterior."
            )
            st.caption(interpretacion_peso)

        with i2:
            st.metric(
                "Umbral actual del indice Nov-Dic",
                f"{umbral_estacional:.2f}",
                help="Nivel minimo de estacionalidad requerido para aplicar el ajuste.",
            )
            st.write(
                f"El ajuste se aplica cuando Nov-Dic vende al menos "
                f"**{incremento_umbral}% mas** que el resto del año."
            )
            st.caption(interpretacion_umbral)

        with st.expander("Ver como interpretar el indice Nov-Dic", expanded=False):
            tabla_indice = pd.DataFrame({
                "Indice": [1.00, 1.20, 1.40, 1.80, 2.00, 3.00],
                "Significado": [
                    "Nov-Dic vende igual que el resto del año",
                    "Nov-Dic vende 20% mas",
                    "Nov-Dic vende 40% mas",
                    "Nov-Dic vende 80% mas",
                    "Nov-Dic vende el doble",
                    "Nov-Dic vende tres veces mas",
                ],
            })
            st.dataframe(
                tabla_indice.style.format({"Indice": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                f"Configuracion actual: el producto debe alcanzar un indice de "
                f"{umbral_estacional:.2f} o superior para recibir el ajuste."
            )

        with st.expander("Ver como interpretar el peso V2", expanded=False):
            tabla_pesos = pd.DataFrame({
                "Peso CatBoost": [0.90, 0.80, 0.70, 0.60, 0.50],
                "V2": ["90%", "80%", "70%", "60%", "50%"],
                "Año anterior": ["10%", "20%", "30%", "40%", "50%"],
                "Interpretacion": [
                    "Ajuste muy ligero",
                    "Ajuste conservador",
                    "Ajuste moderado",
                    "Ajuste estacional fuerte",
                    "Alta dependencia del año anterior",
                ],
            })
            st.dataframe(
                tabla_pesos.style.format({"Peso CatBoost": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                f"Configuracion actual: {porcentaje_v2}% CatBoost + "
                f"{porcentaje_anterior}% mismo mes del año anterior."
            )

        st.caption(
            "La formula aplicada es: Prediccion ajustada = "
            "(Peso CatBoost x Forecast CatBoost) + "
            "((1 - Peso CatBoost) x Venta del mismo mes del año anterior)."
        )



    except Exception as e:
        st.exception(e)
