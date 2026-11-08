import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast venta, unidades y precio", layout="wide")
st.title("Forecast CatBoost: venta + unidades x precio")

MESES = {m: i + 1 for i, m in enumerate([
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
])}
CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
BASE_NUM = ([f"lag{x}" for x in LAGS] +
            ["mean3", "std3", "mean6", "std6", "mean12", "std12",
             "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12"])
FEATURES = BASE_NUM + CAT_FEATURES


def _texto_codigo(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


@st.cache_data(show_spinner=False)
def cargar_archivos(contenidos, nombres):
    partes = []
    requeridas = {"CODIGO", "VENTA", "UNIDADES", "MES", "AÑO"}
    utiles = requeridas | {"PRECIO PROMEDIO", "DESCRIPCION", "CATEGORIA",
                           "FAMILIA", "TIPO", "COMBOS"}
    for contenido, nombre in zip(contenidos, nombres):
        hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None, engine="openpyxl")
        for hoja, h in hojas.items():
            if h is None or h.empty:
                continue
            h = h.copy()
            h.columns = [str(c).strip().upper() for c in h.columns]
            if not requeridas.issubset(h.columns):
                continue
            partes.append(h[[c for c in h.columns if c in utiles]])
    if not partes:
        raise ValueError("No se encontraron hojas con CODIGO, VENTA, UNIDADES, MES y AÑO.")

    df = pd.concat(partes, ignore_index=True)
    df["CODIGO"] = _texto_codigo(df["CODIGO"])
    df["MES_NUM"] = df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    for c in ["VENTA", "UNIDADES", "PRECIO PROMEDIO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["VENTA"] = df["VENTA"].fillna(0).clip(lower=0)
    df["UNIDADES"] = df["UNIDADES"].fillna(0).clip(lower=0)
    df["COMBOS"] = pd.to_numeric(df.get("COMBOS", 0), errors="coerce").fillna(0).clip(0, 1).astype(int)
    df = df.dropna(subset=["CODIGO", "MES_NUM", "AÑO"])
    df["AÑO"] = df["AÑO"].astype(int)
    df["Periodo"] = df["AÑO"] * 100 + df["MES_NUM"].astype(int)

    meta_cols = [c for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"] if c in df]
    agg = {"VENTA": "sum", "UNIDADES": "sum", **{c: "first" for c in meta_cols}}
    g = df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False).agg(agg)

    # El precio se recalcula de forma ponderada. Nunca se suma ni se promedia sin ponderar.
    g["PRECIO_PROMEDIO"] = np.where(g["UNIDADES"] > 0, g["VENTA"] / g["UNIDADES"], np.nan)
    for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
        if c not in g:
            g[c] = "SIN_DATO"
        g[c] = g[c].fillna("SIN_DATO").astype(str)
    return completar_calendario(g)


def completar_calendario(df):
    salida = []
    for codigo, x in df.groupby("CODIGO"):
        x = x.sort_values("Periodo").copy()
        fechas = pd.to_datetime(x["AÑO"].astype(str) + "-" + x["MES_NUM"].astype(str) + "-01")
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[c] = x[c].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan)
        # Precio faltante se imputa solo desde el pasado del SKU, luego mediana de categoria.
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))
    z = pd.concat(salida, ignore_index=True)
    med_cat = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = z["PRECIO_PROMEDIO"].fillna(med_cat).fillna(z["PRECIO_PROMEDIO"].median()).clip(lower=0)
    return z


def crear_features_objetivo(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grp = z.groupby("CODIGO")[objetivo]
    for lag in LAGS:
        z[f"lag{lag}"] = grp.shift(lag)
    for w in [3, 6, 12]:
        z[f"mean{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
        z[f"std{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).std()).fillna(0)
    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0)
    for c in CAT_FEATURES:
        z[c] = z[c].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo(iterations=650):
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost: python -m pip install catboost")
    return CatBoostRegressor(iterations=iterations, depth=8, learning_rate=0.04,
        loss_function="MAE", eval_metric="MAE", l2_leaf_reg=10,
        random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1)


def entrenar_objetivo(df, objetivo):
    f = crear_features_objetivo(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[BASE_NUM].median().fillna(0)
    f[BASE_NUM] = f[BASE_NUM].fillna(medianas)
    y = np.log1p(f[objetivo].clip(lower=0))
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())
    m = nuevo_modelo()
    m.fit(f[FEATURES], y, cat_features=CAT_FEATURES, sample_weight=pesos)
    return m, medianas


def vector_futuro(hist, periodo, objetivo, medianas):
    hist = hist.sort_values("Periodo")
    vals = hist[objetivo].astype(float).tolist()
    q = {f"lag{lag}": vals[-lag] if len(vals) >= lag else np.nan for lag in LAGS}
    for w in [3, 6, 12]:
        a = np.asarray(vals[-w:], float)
        q[f"mean{w}"] = a.mean() if len(a) >= 2 else np.nan
        q[f"std{w}"] = a.std(ddof=1) if len(a) >= 2 else 0.0
    mes, ano = periodo % 100, periodo // 100
    q.update({"sin_mes": np.sin(2*np.pi*mes/12), "cos_mes": np.cos(2*np.pi*mes/12),
              "MES_NUM": mes, "ANO": ano})
    q["tendencia_3_12"] = q["mean3"] / q["mean12"] if pd.notna(q["mean12"]) and q["mean12"] > 0 else 1.0
    ultimo = hist.iloc[-1]
    for c in CAT_FEATURES:
        q[c] = str(ultimo.get(c, "SIN_DATO"))
    X = pd.DataFrame([q], columns=FEATURES)
    X[BASE_NUM] = X[BASE_NUM].fillna(medianas)
    return X


def predecir_objetivo(modelo, medianas, hist, periodos, objetivo):
    h = hist.copy()
    preds = []
    for periodo in periodos:
        pred = max(0.0, float(np.expm1(modelo.predict(vector_futuro(h, periodo, objetivo, medianas))[0])))
        preds.append(pred)
        fila = h.iloc[-1].copy()
        fila["Periodo"], fila["AÑO"], fila["MES_NUM"], fila[objetivo] = periodo, periodo//100, periodo%100, pred
        h = pd.concat([h, pd.DataFrame([fila])], ignore_index=True)
    return preds


def entrenar_tres_modelos(df):
    return {obj: entrenar_objetivo(df, obj) for obj in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]}


def pronosticar_sku(modelos, hist, periodos, peso_directo=0.65):
    pv = predecir_objetivo(*modelos["VENTA"], hist, periodos, "VENTA")
    pu = predecir_objetivo(*modelos["UNIDADES"], hist, periodos, "UNIDADES")
    pp = predecir_objetivo(*modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO")
    estructural = [u * p for u, p in zip(pu, pp)]
    final = [peso_directo*v + (1-peso_directo)*e for v, e in zip(pv, estructural)]
    return pv, pu, pp, estructural, final


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = max(np.abs(y).sum(), 1.0)
    return {"WAPE": 100*np.abs(y-p).sum()/den,
            "MAE": mean_absolute_error(y,p),
            "RMSE": mean_squared_error(y,p)**0.5,
            "R2": r2_score(y,p) if len(y)>1 and np.var(y)>0 else np.nan,
            "SESGO": 100*(p.sum()-y.sum())/max(abs(y.sum()),1.0)}


def backtest(df, periodos, peso_directo):
    corte = min(periodos)
    train, real = df[df["Periodo"] < corte].copy(), df[df["Periodo"].isin(periodos)].copy()
    modelos = entrenar_tres_modelos(train)
    filas = []
    for codigo in sorted(set(train.CODIGO) & set(real.CODIGO)):
        h = train[train.CODIGO == codigo]
        r = real[real.CODIGO == codigo].sort_values("Periodo")
        ps = r.Periodo.astype(int).tolist()
        if not ps: continue
        pv, pu, pp, pe, pf = pronosticar_sku(modelos, h, ps, peso_directo)
        for (_, rr), a,b,c,d,e in zip(r.iterrows(), pv,pu,pp,pe,pf):
            filas.append({"CODIGO": codigo, "Periodo": int(rr.Periodo), "Actual": rr.VENTA,
                          "Venta_directa": a, "Unidades_pred": b, "Precio_pred": c,
                          "Venta_estructural": d, "Prediccion": e, "CATEGORIA": rr.CATEGORIA})
    return pd.DataFrame(filas)


archivos = st.file_uploader("Carga VENTAS 2024, 2025 y 2026", type=["xlsx"], accept_multiple_files=True)
peso_directo = st.sidebar.slider("Peso forecast directo de venta", 0.50, 0.90, 0.65, 0.05)
st.sidebar.caption(f"Venta final = {peso_directo:.0%} venta directa + {1-peso_directo:.0%} unidades x precio")

if archivos:
    try:
        df = cargar_archivos([a.getvalue() for a in archivos], [a.name for a in archivos])
        st.success(f"{len(df):,} registros mensuales | {df.CODIGO.nunique():,} productos")
        inconsist = df[(df.UNIDADES == 0) & (df.VENTA > 0)]
        if len(inconsist):
            st.warning(f"Hay {len(inconsist):,} producto-mes con venta positiva y cero unidades. Revisa esos casos.")

        st.subheader("Backtesting comparable")
        prueba = st.selectbox("Periodo", ["Nov-Dic 2025", "Sep-Oct 2026"])
        periodos_bt = [202511, 202512] if prueba == "Nov-Dic 2025" else [202609, 202610]
        if st.button("Ejecutar backtest", type="primary"):
            with st.spinner("Entrenando tres modelos y evaluando..."):
                bt = backtest(df, periodos_bt, peso_directo)
            if bt.empty:
                st.error("No hubo datos suficientes.")
            else:
                m_final = metricas(bt.Actual, bt.Prediccion)
                m_directo = metricas(bt.Actual, bt.Venta_directa)
                comp = pd.DataFrame([{"Modelo":"Venta directa", **m_directo},
                                     {"Modelo":"Venta + unidades x precio", **m_final}])
                st.dataframe(comp.style.format({"WAPE":"{:.2f}%", "MAE":"${:,.2f}",
                    "RMSE":"${:,.2f}", "R2":"{:.3f}", "SESGO":"{:+.2f}%"}),
                    use_container_width=True, hide_index=True)
                st.caption("Conserva el componente estructural solo si baja WAPE y RMSE y acerca el sesgo a cero.")
                st.dataframe(bt, use_container_width=True, hide_index=True)

        st.subheader("Forecast noviembre-diciembre 2026")
        if st.button("Generar forecast 2026"):
            with st.spinner("Entrenando venta, unidades y precio..."):
                modelos = entrenar_tres_modelos(df)
                filas=[]
                for codigo, h in df.groupby("CODIGO"):
                    pv,pu,pp,pe,pf = pronosticar_sku(modelos,h,[202611,202612],peso_directo)
                    meta=h.sort_values("Periodo").iloc[-1]
                    for periodo,a,b,c,d,e in zip([202611,202612],pv,pu,pp,pe,pf):
                        filas.append({"CODIGO":codigo,"DESCRIPCION":meta.DESCRIPCION,
                            "CATEGORIA":meta.CATEGORIA,"Periodo":periodo,"Venta_directa":a,
                            "Unidades_pred":b,"Precio_pred":c,"Venta_estructural":d,"Venta_final":e})
                pron=pd.DataFrame(filas)
            st.dataframe(pron, use_container_width=True, hide_index=True)
            st.download_button("Descargar forecast CSV", pron.to_csv(index=False).encode("utf-8-sig"),
                               "forecast_2026_unidades_precio.csv", "text/csv")
    except Exception as e:
        st.exception(e)
else:
    st.info("Carga los tres archivos para comenzar.")


