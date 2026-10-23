import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# ========================================================
# CONFIGURACION DE LA PAGINA
# ========================================================
st.set_page_config(
    page_title="Prediccion de Inventario", page_icon="📦", layout="wide"
)

st.title("📦 Simulador Predictivo de Dias de Inventario")
st.markdown("Sube todos los archivos de tu carpeta historica para entrenar el modelo y analizar el desempeño por producto.")

# ========================================================
# CONSTANTES: 3 FEATURES MINIMALISTAS
# ========================================================
FEATURES = ['Venta_lag1', 'Venta_lag2', 'Venta_MA3']


# ========================================================
# FUNCIONES AUXILIARES
# ========================================================
def obtener_rango_fechas(semana_num):
    base_date = datetime.date(2026, 6, 15)
    semana_base = 25
    diferencia_semanas = int(semana_num) - semana_base
    start_date = base_date + datetime.timedelta(weeks=diferencia_semanas)
    end_date = start_date + datetime.timedelta(days=6)
    meses = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
             7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
    return f"{start_date.day:02d} {meses[start_date.month]} - {end_date.day:02d} {meses[end_date.month]}"


def construir_features(df):
    """Version minimalista: solo 3 features para no perder filas por NaN."""
    df = df.sort_values(['Codigo', 'Semana']).copy()
    df['Venta_lag1'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_lag2'] = df.groupby('Codigo')['Monto de ventas'].shift(2)
    df['Venta_MA3'] = df.groupby('Codigo')['Monto de ventas'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    return df


def es_bajo_volumen(hist_ventas, umbral_ceros=0.6):
    """True si >60% de semanas son cero/casi cero -> usar mediana simple."""
    if len(hist_ventas) == 0:
        return False
    return float((hist_ventas <= 0.01).mean()) > umbral_ceros


def construir_modelo_conservador():
    """RandomForest pensado para datos escasos (18 semanas)."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=8,
        min_samples_split=12,
        max_features=1.0,
        criterion='absolute_error',
        random_state=42,
        n_jobs=-1,
    )


def entrenar_modelos_directos(df, max_horizonte, min_filas=15):
    """
    Entrena UN modelo por horizonte h = 1..max_horizonte.
    Devuelve (modelos_dict, filas_por_horizonte_dict).
    """
    df_fe = construir_features(df)
    modelos = {}
    filas_por_horizonte = {}

    for h in range(1, max_horizonte + 1):
        target = f'target_h{h}'
        df_fe[target] = df_fe.groupby('Codigo')['Monto de ventas'].shift(-h)
        sub = df_fe.dropna(subset=FEATURES + [target])
        filas_por_horizonte[h] = len(sub)
        if len(sub) < min_filas:
            continue
        m = construir_modelo_conservador()
        m.fit(sub[FEATURES], sub[target])
        modelos[h] = m

    return modelos, filas_por_horizonte


def predecir_directo(modelos, df_historial, codigo, n_semanas):
    """
    Predice sin realimentar. Aplica bajo volumen y clipping amplio [0.5x, 2x].
    """
    df_fe = construir_features(df_historial)
    estado = df_fe[df_fe['Codigo'] == codigo].sort_values('Semana').tail(1)
    hist = df_historial[df_historial['Codigo'] == codigo]['Monto de ventas'].dropna()

    if estado.empty or len(hist) == 0:
        return [0.01] * n_semanas

    if es_bajo_volumen(hist, umbral_ceros=0.6):
        mediana = float(hist[hist > 0].median()) if (hist > 0).any() else 0.01
        return [max(0.01, mediana)] * n_semanas

    X_input = estado[FEATURES]
    piso = max(0.01, float(hist.min()) * 0.5)
    techo = float(hist.max()) * 2.0
    media = float(hist.mean())

    preds = []
    for h in range(1, n_semanas + 1):
        if h in modelos:
            try:
                p = float(modelos[h].predict(X_input)[0])
            except Exception:
                p = media
        else:
            p = media
        preds.append(max(0.01, float(np.clip(p, piso, techo))))
    return preds


# ========================================================
# CARGA Y ENTRENAMIENTO GLOBAL
# ========================================================
@st.cache_resource(show_spinner="Procesando archivos y entrenando modelos, esto puede tomar unos segundos.")
def cargar_y_entrenar(archivos_subidos):
    lista_dfs = []
    for archivo in archivos_subidos:
        try:
            df_temp = pd.read_excel(archivo, engine='openpyxl')
            lista_dfs.append(df_temp)
        except Exception:
            pass
    if not lista_dfs:
        raise ValueError("No se pudo leer ningun archivo correctamente.")
    df = pd.concat(lista_dfs, ignore_index=True)

    nuevas_columnas = [str(col).strip().split('[')[-1].replace(']', '') for col in df.columns]
    df.columns = nuevas_columnas

    for col in df.columns:
        col_lower = col.lower()
        if 'descrip' in col_lower: df.rename(columns={col: 'Descripcion'}, inplace=True)
        elif 'categoria' in col_lower: df.rename(columns={col: 'Categoria'}, inplace=True)
        elif 'inventario' in col_lower and 'monto' not in col_lower and 'total' not in col_lower and 'dias' not in col_lower: df.rename(columns={col: 'Inventario'}, inplace=True)
        elif 'venta' in col_lower: df.rename(columns={col: 'Monto de ventas'}, inplace=True)
        elif 'codigo' in col_lower or 'código' in col_lower: df.rename(columns={col: 'Codigo'}, inplace=True)
        elif 'semana' in col_lower: df.rename(columns={col: 'Semana'}, inplace=True)
        elif 'dias' in col_lower and 'inv' in col_lower: df.rename(columns={col: 'Dias Inv'}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    df['Codigo'] = pd.to_numeric(df['Codigo'], errors='coerce')
    df['Semana'] = pd.to_numeric(df['Semana'], errors='coerce')
    df['Monto de ventas'] = pd.to_numeric(df['Monto de ventas'], errors='coerce')
    df['Inventario'] = pd.to_numeric(df['Inventario'], errors='coerce')

    if 'Dias Inv' in df.columns:
        df['Dias Inv'] = pd.to_numeric(df['Dias Inv'], errors='coerce')
    else:
        df['Dias Inv'] = 0.0

    df = df.sort_values(['Codigo', 'Semana']).reset_index(drop=True)

    # Compatibilidad para secciones visuales
    df['Venta_1_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_2_Semanas_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(2)

    if df.dropna(subset=['Monto de ventas']).empty:
        raise ValueError("No hay datos de ventas validos.")

    # Entrenar modelos directos (1..12 horizontes)
    modelos_directos, filas_por_horizonte = entrenar_modelos_directos(
        df, max_horizonte=12, min_filas=10
    )

    if not modelos_directos:
        raise ValueError("No hay suficientes datos para entrenar (se necesitan al menos 5-6 semanas por producto).")

    # Metricas in-sample (referencia rapida, sesgadas)
    df_fe = construir_features(df)
    df_fe['target_h1'] = df_fe.groupby('Codigo')['Monto de ventas'].shift(-1)
    sub = df_fe.dropna(subset=FEATURES + ['target_h1'])

    if len(sub) > 0 and 1 in modelos_directos:
        y_pred = modelos_directos[1].predict(sub[FEATURES])
        y_true = sub['target_h1'].values
        suma_abs_err = float(np.sum(np.abs(y_true - y_pred)))
        suma_real = float(np.sum(np.abs(y_true)))
        wape = (suma_abs_err / suma_real) * 100 if suma_real > 0 else 0.0
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
    else:
        wape = mae = rmse = r2 = 0.0

    return df, modelos_directos, filas_por_horizonte, wape, mae, rmse, r2


# ========================================================
# BACKTEST CON MODELOS DIRECTOS
# ========================================================
@st.cache_resource(show_spinner=False)
def entrenar_backtest(df, n_semanas_holdout, codigos_tuple):
    codigos_filtro = list(codigos_tuple)
    df_work = df.dropna(subset=['Codigo', 'Semana', 'Monto de ventas']).copy()
    df_work = df_work[df_work['Codigo'].isin(codigos_filtro)]
    df_work = df_work.sort_values(['Codigo', 'Semana']).reset_index(drop=True)

    train_rows, held_rows = [], []
    for codigo, group in df_work.groupby('Codigo'):
        group = group.sort_values('Semana').reset_index(drop=True)
        if len(group) <= n_semanas_holdout + 2:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:-n_semanas_holdout])
            held_rows.append(group.iloc[-n_semanas_holdout:])

    if not train_rows:
        raise ValueError("No hay datos de entrenamiento suficientes.")

    df_train = pd.concat(train_rows, ignore_index=True)
    df_held = pd.concat(held_rows, ignore_index=True) if held_rows else pd.DataFrame()

    modelos_bt, filas_bt = entrenar_modelos_directos(
        df_train, max_horizonte=n_semanas_holdout, min_filas=6
    )

    return modelos_bt, df_held, df_train, filas_bt


# ========================================================
# INTERFAZ DE USUARIO
# ========================================================
st.subheader("1. Origen de los Datos")
archivos_subidos = st.file_uploader(
    "Selecciona o arrastra todos los archivos Excel de la carpeta (Ctrl + E para seleccionar todos):",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

if archivos_subidos:
    try:
        df_global, modelos_directos, filas_por_horizonte, wape_val, mae_val, rmse_val, r2_val = cargar_y_entrenar(archivos_subidos)

        st.success("Archivos cargados y modelos entrenados con exito!")

        st.markdown("##### Metricas de Ajuste (in-sample — referencia rapida)")
        st.caption(
            "⚠️ Estas metricas son IN-SAMPLE (optimistas). Usa la seccion 1.5 (Backtest) para la evaluacion honesta out-of-sample."
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="WAPE (in-sample)", value=f"{wape_val:.2f}%")
        with col_m2:
            st.metric(label="MAE (in-sample)", value=f"${mae_val:,.2f}")
        with col_m3:
            st.metric(label="RMSE (in-sample)", value=f"${rmse_val:,.2f}")
        with col_m4:
            st.metric(label="R² (in-sample)", value=f"{max(0, r2_val):.2f}")

        # Diagnostico de horizontes entrenados
        with st.expander("🔧 Diagnostico del modelo (horizontes entrenados)"):
            horizontes_ok = sorted(modelos_directos.keys())
            st.write(f"**Horizontes entrenados**: {horizontes_ok}")
            st.write(f"**Total productos**: {df_global['Codigo'].nunique()}")
            st.write(f"**Total semanas historicas**: {df_global['Semana'].nunique()}")
            df_filas = pd.DataFrame({
                'Horizonte': list(filas_por_horizonte.keys()),
                'Filas utiles': list(filas_por_horizonte.values()),
                'Entrenado': [h in modelos_directos for h in filas_por_horizonte.keys()],
            })
            st.dataframe(df_filas, use_container_width=True, hide_index=True)

        st.markdown("---")

        CODIGOS_PREDICCION = [
7506475104722, 7506475124614, 7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 
            7506475104708, 7501059211209, 7501058619563, 7501059278868, 7501058638076, 7501001600426, 7613030695295, 
            7501059234321, 7501058615541, 7501058611062, 7501058629517, 7506475121231, 7506475128179, 7506475100250, 
            7506475111690, 7506475131179, 7506475116367, 7501000913367, 7501000911967, 7501058628831, 7506475119443, 
            7501059240681, 7501058619228, 7506475113915, 7501059235038, 7501058619211, 7501058632227, 7501000913299, 
            7501058619235, 7501058645296, 7501058645302, 7501058655172, 7506475121156, 7506475117364, 7501058654205, 
            7501058642127, 7501058642134, 7501058642141, 7501058642165, 7506475130295, 7501058642172, 7501058642158, 
            7506475108829, 7506475112970, 7506475122764, 7506475101981, 7506475101264, 7506475120500, 7506475127158, 
            7501058655219, 7501058644848, 7501058644824, 7501058644862, 7501000912803, 7501058620101, 7506475123051, 
            7501058620095, 7501000912612, 7501059224827, 7506475131438, 7506475125253, 7506475113946, 7506475102353, 
            7501058610942, 7501058624635, 7506475110112, 7506475114998, 7506475125543, 7506475113700, 7501059284111, 
            7501058620002, 7501058620019, 7501000912605, 7506475123389, 7501058616548, 7506475128162, 7501058622099, 
            7506475123792, 7501059231962, 7501059298941, 7501058628466, 7501058628473, 7501058616470, 7501058648020, 
            7506475110105, 7506475115254, 7501058646217, 7501058618924, 7501058618931, 7501058618917, 7506475114936, 
            7506475111119, 7506475103855, 7501058629135, 7501058629173, 7501058629159, 7506475126731, 7501058624130, 
            7501058624147, 7501058624154, 7501058624161, 7506475120289, 7506475120265, 7506475120272, 7501058654793, 
            8445292308465, 8445292308540, 7501058654212, 7506475130707, 7506475130134, 7501058642608, 7501058642592, 
            7501058618597, 7506475128292, 7501059214590, 7506475118217, 7501059214385, 7501059229211, 7501059229228, 
            7501059289239, 7501059239630, 7506475113168, 7506475111713, 7501058643902, 75000011, 7506475125673, 
            7506475125680, 7506475117876, 7506475114172, 7506475113564, 7506475105606, 7501058610959, 7501058611857, 
            7501058613554, 7506475102834, 7501073411173, 7506475123617, 7506475124874, 7501058628299, 7501058624666, 
            7501059233980, 7501059289659, 7506475102148, 7613287207197, 7613287216458, 7613287218162, 8445290925299, 
            7613037012637, 8585002432315, 88169052054, 7501001604318, 7501001604325, 7501058628664, 7501058628503, 
            7501059240742, 7501001604103, 7506475120814, 7501001604110, 7501059297586, 7506475120821, 7506475118446, 
            7501058650665, 7506475108935, 7506475127080, 7506475130325, 7506475130332, 7506475130318, 7506475114356, 
            7506475102476, 7506475102421, 7506475102452, 7506475102490, 7506475102469, 7506475102520, 7506475102506, 
            7506475102483, 7506475102537, 75004712, 75004705, 75004767, 75004729, 75004743, 7501000906246, 7501000906253, 
            7501000906284, 7501000906680, 7501058651136, 7501058651129, 7506475114073, 7506475115438, 7506475119665, 
            7506475119672, 7506475115421, 7506475122450, 7506475122436, 7506475122443, 7506475117081, 7506475122122, 
            7506475122139, 7506475122115, 7501058637659, 7506475122092, 7506475121996, 7891000395745, 7501000909568, 
            7501058626226, 7501058639493, 7506475103220, 7506475103213, 7501058616678, 7501058616715, 7501058614193, 
            7501000909612, 7501059278721, 7501059278691, 7501058626530, 7613030884361, 7506475106917, 7501058625212, 
            7501058623256, 7506475106924, 7501058625229, 7501058623249, 7501058625205, 7501058625236, 7506475127295, 
            7501058623232, 7506475106801, 7506475106771, 7506475106153, 7506475106818, 7506475106788, 7506475106146, 
            7506475103053, 7506475103275, 7501059275133, 7501059282117, 7501058615138, 7506475126090, 7501059234390, 
            7506475112888, 7506475112956, 7506475112895, 7506475112963, 7501059225411, 7501059225350, 7506475122955, 
            7506475103244, 7506475129664, 7506475118675, 7501059233072, 7501058617439, 7501058652683, 7501058651266, 
            7501058651273, 7501058637727, 7501059243880, 7501059243873, 7501058637734, 7501059241060, 7501058637758, 
            7501058637741, 7501058652652, 16000135437, 16000221284, 7501001625337, 7501058652690, 7506475111546, 
            7501001625214, 7506475111812, 7506475115544, 7506475117197, 7506475126984, 7506475126977, 7506475126960, 
            7506475125413, 7506475128117, 7506475128858, 7506475128285, 7501001604004, 7501059240216, 7501001604387, 
            7501059239845, 7613034161086, 7506475131629, 7501059287938, 7501058615596, 7501058627292, 7501059219106, 
            7501059242883, 7501059284623, 7501059284630, 7506475106078, 7501073419117, 7501001600198, 7501000912889, 
            10722776200640, 7501059281165, 7501058618566, 7501058617736, 3800020423547, 7891000277119, 7501059288904, 
            7501059223905, 28000170707, 7891000248362, 7501058623348, 7501058654861, 7501058656247, 7501058638090, 
            7506475128384, 7506475126694, 7502252484285, 7502252482694, 7502252481796, 7502252480928, 7502252480676, 
            7502252484247, 7502252482038, 7502252482045, 7502252480263, 7502252480287, 7502252483929, 7502252483936, 
            7502252483943, 7502252484469, 7502252482236, 7502252482359, 7502252482250, 7502252487637, 7502252487675, 
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077
        ]
        CODIGOS_TABLA = [
     7506475104722, 7506475124614, 7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 
            7506475104708, 7501059211209, 7501058619563, 7501059278868, 7501058638076, 7501001600426, 7613030695295, 
            7501059234321, 7501058615541, 7501058611062, 7501058629517, 7506475121231, 7506475128179, 7506475100250, 
            7506475111690, 7506475131179, 7506475116367, 7501000913367, 7501000911967, 7501058628831, 7506475119443, 
            7501059240681, 7501058619228, 7506475113915, 7501059235038, 7501058619211, 7501058632227, 7501000913299, 
            7501058619235, 7501058645296, 7501058645302, 7501058655172, 7506475121156, 7506475117364, 7501058654205, 
            7501058642127, 7501058642134, 7501058642141, 7501058642165, 7506475130295, 7501058642172, 7501058642158, 
            7506475108829, 7506475112970, 7506475122764, 7506475101981, 7506475101264, 7506475120500, 7506475127158, 
            7501058655219, 7501058644848, 7501058644824, 7501058644862, 7501000912803, 7501058620101, 7506475123051, 
            7501058620095, 7501000912612, 7501059224827, 7506475131438, 7506475125253, 7506475113946, 7506475102353, 
            7501058610942, 7501058624635, 7506475110112, 7506475114998, 7506475125543, 7506475113700, 7501059284111, 
            7501058620002, 7501058620019, 7501000912605, 7506475123389, 7501058616548, 7506475128162, 7501058622099, 
            7506475123792, 7501059231962, 7501059298941, 7501058628466, 7501058628473, 7501058616470, 7501058648020, 
            7506475110105, 7506475115254, 7501058646217, 7501058618924, 7501058618931, 7501058618917, 7506475114936, 
            7506475111119, 7506475103855, 7501058629135, 7501058629173, 7501058629159, 7506475126731, 7501058624130, 
            7501058624147, 7501058624154, 7501058624161, 7506475120289, 7506475120265, 7506475120272, 7501058654793, 
            8445292308465, 8445292308540, 7501058654212, 7506475130707, 7506475130134, 7501058642608, 7501058642592, 
            7501058618597, 7506475128292, 7501059214590, 7506475118217, 7501059214385, 7501059229211, 7501059229228, 
            7501059289239, 7501059239630, 7506475113168, 7506475111713, 7501058643902, 75000011, 7506475125673, 
            7506475125680, 7506475117876, 7506475114172, 7506475113564, 7506475105606, 7501058610959, 7501058611857, 
            7501058613554, 7506475102834, 7501073411173, 7506475123617, 7506475124874, 7501058628299, 7501058624666, 
            7501059233980, 7501059289659, 7506475102148, 7613287207197, 7613287216458, 7613287218162, 8445290925299, 
            7613037012637, 8585002432315, 88169052054, 7501001604318, 7501001604325, 7501058628664, 7501058628503, 
            7501059240742, 7501001604103, 7506475120814, 7501001604110, 7501059297586, 7506475120821, 7506475118446, 
            7501058650665, 7506475108935, 7506475127080, 7506475130325, 7506475130332, 7506475130318, 7506475114356, 
            7506475102476, 7506475102421, 7506475102452, 7506475102490, 7506475102469, 7506475102520, 7506475102506, 
            7506475102483, 7506475102537, 75004712, 75004705, 75004767, 75004729, 75004743, 7501000906246, 7501000906253, 
            7501000906284, 7501000906680, 7501058651136, 7501058651129, 7506475114073, 7506475115438, 7506475119665, 
            7506475119672, 7506475115421, 7506475122450, 7506475122436, 7506475122443, 7506475117081, 7506475122122, 
            7506475122139, 7506475122115, 7501058637659, 7506475122092, 7506475121996, 7891000395745, 7501000909568, 
            7501058626226, 7501058639493, 7506475103220, 7506475103213, 7501058616678, 7501058616715, 7501058614193, 
            7501000909612, 7501059278721, 7501059278691, 7501058626530, 7613030884361, 7506475106917, 7501058625212, 
            7501058623256, 7506475106924, 7501058625229, 7501058623249, 7501058625205, 7501058625236, 7506475127295, 
            7501058623232, 7506475106801, 7506475106771, 7506475106153, 7506475106818, 7506475106788, 7506475106146, 
            7506475103053, 7506475103275, 7501059275133, 7501059282117, 7501058615138, 7506475126090, 7501059234390, 
            7506475112888, 7506475112956, 7506475112895, 7506475112963, 7501059225411, 7501059225350, 7506475122955, 
            7506475103244, 7506475129664, 7506475118675, 7501059233072, 7501058617439, 7501058652683, 7501058651266, 
            7501058651273, 7501058637727, 7501059243880, 7501059243873, 7501058637734, 7501059241060, 7501058637758, 
            7501058637741, 7501058652652, 16000135437, 16000221284, 7501001625337, 7501058652690, 7506475111546, 
            7501001625214, 7506475111812, 7506475115544, 7506475117197, 7506475126984, 7506475126977, 7506475126960, 
            7506475125413, 7506475128117, 7506475128858, 7506475128285, 7501001604004, 7501059240216, 7501001604387, 
            7501059239845, 7613034161086, 7506475131629, 7501059287938, 7501058615596, 7501058627292, 7501059219106, 
            7501059242883, 7501059284623, 7501059284630, 7506475106078, 7501073419117, 7501001600198, 7501000912889, 
            10722776200640, 7501059281165, 7501058618566, 7501058617736, 3800020423547, 7891000277119, 7501059288904, 
            7501059223905, 28000170707, 7891000248362, 7501058623348, 7501058654861, 7501058656247, 7501058638090, 
            7506475128384, 7506475126694, 7502252484285, 7502252482694, 7502252481796, 7502252480928, 7502252480676, 
            7502252484247, 7502252482038, 7502252482045, 7502252480263, 7502252480287, 7502252483929, 7502252483936, 
            7502252483943, 7502252484469, 7502252482236, 7502252482359, 7502252482250, 7502252487637, 7502252487675, 
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077
        ]

        # ========================================================
        # 1.5. BACKTEST
        # ========================================================
        st.subheader("1.5. Validación Histórica por Semana (Backtest)")
        st.markdown(
            "Entrena modelos **excluyendo las últimas N semanas** de historia real, predice esas semanas "
            "**sin realimentar** (prediccion directa multi-horizonte) y compara contra los valores reales."
        )

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            semanas_holdout = st.slider(
                "Semanas historicas a reservar para validacion:",
                min_value=1, max_value=12, value=4, key="slider_holdout",
            )
        with col_bt2:
            semana_inicio_metrica = st.slider(
                "Calcular metricas a partir de la semana predecida #:",
                min_value=1, max_value=semanas_holdout, value=1, key="slider_semana_inicio",
            )

        try:
            modelos_bt, df_held_bt, df_train_bt, filas_bt = entrenar_backtest(
                df_global, semanas_holdout, tuple(CODIGOS_PREDICCION)
            )

            if df_held_bt.empty:
                st.warning("No hay suficientes semanas historicas por producto para hacer el backtest.")
            elif not modelos_bt:
                st.warning(
                    "No hay datos suficientes para entrenar ningun modelo en el backtest. "
                    "Prueba con menos semanas reservadas (ej. 2-3)."
                )
            else:
                resultados_bt = []
                for codigo, group in df_held_bt.groupby('Codigo'):
                    group = group.sort_values('Semana').reset_index(drop=True)
                    train_group = df_train_bt[df_train_bt['Codigo'] == codigo]
                    if train_group.empty or len(train_group) < 4:
                        continue

                    n_semanas = len(group)
                    preds = predecir_directo(modelos_bt, df_train_bt, codigo, n_semanas)

                    for i, (pred, actual) in enumerate(zip(preds, group['Monto de ventas'].tolist()), start=1):
                        if pd.isna(actual):
                            continue
                        resultados_bt.append({
                            'Codigo': codigo,
                            'Descripcion': group.iloc[i-1].get('Descripcion', 'Desconocido'),
                            'Semana_Real': int(group.iloc[i-1]['Semana']),
                            'Semana_Predecida': i,
                            'Prediccion': float(pred),
                            'Actual': float(actual),
                        })

                if not resultados_bt:
                    st.warning("No se pudieron generar predicciones de backtest.")
                else:
                    df_bt = pd.DataFrame(resultados_bt)
                    df_bt_filt = df_bt[df_bt['Semana_Predecida'] >= semana_inicio_metrica].copy()

                    if df_bt_filt.empty:
                        st.warning("No hay datos para la semana seleccionada.")
                    else:
                        y_true = df_bt_filt['Actual'].values
                        y_pred = df_bt_filt['Prediccion'].values

                        suma_abs_err = np.sum(np.abs(y_true - y_pred))
                        suma_real = np.sum(np.abs(y_true))
                        wape_bt = (suma_abs_err / suma_real) * 100 if suma_real > 0 else 0.0
                        mae_bt = mean_absolute_error(y_true, y_pred)
                        rmse_bt = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2_bt = r2_score(y_true, y_pred) if len(y_true) >= 2 else float('nan')

                        st.markdown(
                            f"#### 📊 Metricas de Backtest — Semanas +{semana_inicio_metrica} a +{semanas_holdout}"
                        )
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("WAPE", f"{wape_bt:.2f}%", help="Error ponderado por volumen")
                        c2.metric("MAE", f"${mae_bt:,.2f}", help="Error absoluto promedio")
                        c3.metric("RMSE", f"${rmse_bt:,.2f}", help="Penaliza errores grandes")
                        r2_display_bt = f"{max(0, r2_bt):.2f}" if not np.isnan(r2_bt) else "N/A"
                        c4.metric("R²", r2_display_bt, help="Capacidad explicativa (0 a 1)")

                        st.caption(
                            f"Total de predicciones evaluadas: **{len(y_true)}** "
                            f"({df_bt_filt['Codigo'].nunique()} productos × semanas seleccionadas)"
                        )

                        # Diagnostico backtest: que horizontes se pudieron entrenar
                        with st.expander("🔧 Diagnostico del backtest (horizontes entrenados)"):
                            st.write(f"**Horizontes entrenados en backtest**: {sorted(modelos_bt.keys())}")
                            df_filas_bt = pd.DataFrame({
                                'Horizonte': list(filas_bt.keys()),
                                'Filas utiles': list(filas_bt.values()),
                                'Entrenado': [h in modelos_bt for h in filas_bt.keys()],
                            })
                            st.dataframe(df_filas_bt, use_container_width=True, hide_index=True)

                        st.markdown("##### 🔍 Comparativa Real vs Predicho (suma agregada por semana)")
                        df_chart = df_bt_filt.groupby('Semana_Predecida').agg({
                            'Prediccion': 'sum', 'Actual': 'sum'
                        }).reset_index()
                        df_chart['Semana_Etiqueta'] = df_chart['Semana_Predecida'].apply(lambda x: f"Sem +{int(x)}")
                        df_chart_melt = df_chart.melt(
                            id_vars=['Semana_Etiqueta'],
                            value_vars=['Prediccion', 'Actual'],
                            var_name='Tipo', value_name='Monto'
                        )

                        chart_bt = alt.Chart(df_chart_melt).mark_bar().encode(
                            x=alt.X('Semana_Etiqueta:O', title='Semana Predecida'),
                            y=alt.Y('Monto:Q', title='Monto Total ($)', axis=alt.Axis(format='$,.2f')),
                            color=alt.Color('Tipo:N', scale=alt.Scale(
                                domain=['Actual', 'Prediccion'],
                                range=['#2ca02c', '#ff7f0e']
                            )),
                            xOffset='Tipo:N',
                            tooltip=['Semana_Etiqueta:O', 'Tipo:N', alt.Tooltip('Monto:Q', format='$,.2f')]
                        ).properties(height=300)
                        st.altair_chart(chart_bt, use_container_width=True)

                        with st.expander("🔎 Ver detalle por producto (prediccion vs real)"):
                            df_diag = df_bt_filt.copy()
                            df_diag['Error_Abs'] = (df_diag['Prediccion'] - df_diag['Actual']).abs()
                            st.dataframe(
                                df_diag.sort_values(['Codigo', 'Semana_Predecida']).style.format({
                                    'Prediccion': '${:,.2f}',
                                    'Actual': '${:,.2f}',
                                    'Error_Abs': '${:,.2f}',
                                }),
                                use_container_width=True
                            )

                            st.markdown("##### 🚨 Top 10 peores predicciones (por error absoluto)")
                            st.dataframe(
                                df_diag.nlargest(10, 'Error_Abs')[
                                    ['Codigo', 'Descripcion', 'Semana_Real', 'Semana_Predecida',
                                     'Prediccion', 'Actual', 'Error_Abs']
                                ].style.format({
                                    'Prediccion': '${:,.2f}',
                                    'Actual': '${:,.2f}',
                                    'Error_Abs': '${:,.2f}',
                                }),
                                use_container_width=True
                            )

        except Exception as e:
            st.error(f"Error en backtest: {e}")

        st.markdown("---")

        # ========================================================
        # 2. PREDICCION Y GRAFICA
        # ========================================================
        st.subheader("2. Simulacion y Tendencia de Producto")

        df_pred = df_global.dropna(subset=['Codigo']).copy()
        df_pred = df_pred[df_pred['Codigo'].isin(CODIGOS_PREDICCION)]

        if df_pred.empty:
            st.warning("No se encontraron los codigos de CODIGOS_PREDICCION en los archivos subidos.")
        else:
            df_ultimos = df_pred.sort_values('Semana').groupby('Codigo').tail(1)
            opciones_productos = {
                int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}"
                for _, row in df_ultimos.iterrows()
            }

            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                codigo_seleccionado = st.selectbox(
                    "Selecciona el Producto a analizar:",
                    options=list(opciones_productos.keys()),
                    format_func=lambda x: opciones_productos[x]
                )
            with col_sim2:
                monto_enviar = st.number_input(
                    "Monto de mercancia a enviar ($):",
                    min_value=0.0, value=0.0, step=1000.0
                )
            with col_sim3:
                semanas_proyectar = st.slider(
                    "Semanas a proyectar al futuro (Gráfica):",
                    min_value=1, max_value=8, value=4, key="slider_grafica"
                )

            df_hist_prod = df_pred[df_pred['Codigo'] == codigo_seleccionado].sort_values('Semana')

            if not df_hist_prod.empty:
                semana_max = df_hist_prod['Semana'].max()
                df_ultimo_registro = df_hist_prod[df_hist_prod['Semana'] == semana_max].iloc[0]

                preds_futuras = predecir_directo(
                    modelos_directos, df_global, codigo_seleccionado, semanas_proyectar
                )

                df_real = df_hist_prod[['Semana', 'Monto de ventas']].copy()
                df_real['Tipo'] = 'Historico Real'

                filas_futuras = [{
                    'Semana': semana_max,
                    'Monto de ventas': df_ultimo_registro['Monto de ventas'],
                    'Tipo': 'Proyectado',
                }]
                for idx, pred_val in enumerate(preds_futuras, start=1):
                    filas_futuras.append({
                        'Semana': semana_max + idx,
                        'Monto de ventas': pred_val,
                        'Tipo': 'Proyectado',
                    })
                df_futuro = pd.DataFrame(filas_futuras)

                df_grafica_comb = pd.concat([df_real, df_futuro], ignore_index=True)

                st.markdown(f"#### 📈 Tendencia Historica y Proyeccion Directa ({semanas_proyectar} Semanas)")

                grafica = alt.Chart(df_grafica_comb).mark_line(point=True).encode(
                    x=alt.X('Semana:O', title='Semana', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto de ventas:Q', title='Monto de Ventas ($)', axis=alt.Axis(format='$,.2f')),
                    color=alt.Color('Tipo:N', title='Origen', scale=alt.Scale(
                        domain=['Historico Real', 'Proyectado'], range=['#1f77b4', '#ff7f0e']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.Tipo == 'Proyectado',
                        alt.value([4, 4]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('Semana:O', title='Semana'),
                        alt.Tooltip('Monto de ventas:Q', title='Monto ($)', format='$,.2f'),
                        alt.Tooltip('Tipo:N', title='Estado')
                    ]
                ).properties(height=350).interactive()

                st.altair_chart(grafica, use_container_width=True)

                inv_actual = df_ultimo_registro['Inventario']
                inv_actual = 0 if pd.isna(inv_actual) else inv_actual
                inv_futuro = inv_actual + monto_enviar

                venta_prom_est = np.mean(preds_futuras)
                dias_inv = (inv_futuro / venta_prom_est) * 30 if venta_prom_est > 0 else 0

                st.markdown("### 📊 Resultados de la Proyeccion Multi-Semana")
                r1, r2, r3, r4 = st.columns(4)
                r1.info(f"Inv. en Sistema:\n\n${inv_actual:,.2f}")
                r2.warning(f"Inv. Total (Simulado):\n\n${inv_futuro:,.2f}")
                r3.info(f"Venta Est. Prom. ({semanas_proyectar} sem):\n\n${venta_prom_est:,.2f}")
                r4.success(f"Dias Cobertura Prom.:\n\n{dias_inv:.1f} dias")

        st.markdown("---")

        # ========================================================
        # 3. REPORTE SEMANAL
        # ========================================================
        st.subheader("3. Reporte Semanal (Dias de Inventario)")

        df_tabla = df_global.dropna(subset=['Codigo']).copy()
        df_tabla = df_tabla[df_tabla['Codigo'].isin(CODIGOS_TABLA)]

        if df_tabla.empty:
            st.warning("No se encontraron los codigos de CODIGOS_TABLA en los archivos subidos.")
        else:
            cols_indice = ['Codigo', 'Descripcion']
            if 'Categoria' in df_tabla.columns:
                cols_indice.append('Categoria')

            df_pivot = df_tabla.pivot_table(
                index=cols_indice, columns='Semana', values='Dias Inv', aggfunc='sum'
            ).reset_index().fillna(0)

            semanas_cols = sorted([c for c in df_pivot.columns if isinstance(c, (int, float))])

            if len(semanas_cols) > 0:
                col_antigua = semanas_cols[0]
                col_reciente = semanas_cols[-1]
                df_pivot['Variacion'] = df_pivot[col_reciente] - df_pivot[col_antigua]
                df_pivot['% Meta'] = (df_pivot[col_reciente] / 60 * 100).round(1).astype(str) + '%'
            else:
                df_pivot['Variacion'] = 0.0
                df_pivot['% Meta'] = '0%'

            mapeo_fechas = {sem: obtener_rango_fechas(sem) for sem in semanas_cols}
            df_pivot = df_pivot.rename(columns=mapeo_fechas)

            st.dataframe(df_pivot, use_container_width=True)

        st.markdown("---")

        # ========================================================
        # 4. TABLA MASIVA EDITABLE
        # ========================================================
        st.subheader("4. Proyeccion Masiva de Inventario (Codigos Seleccionados)")
        st.markdown(
            "Ajusta manualmente los montos de inventario por semana en la **tabla editable**. "
            "Los **Dias de Inventario** se recalculan automaticamente con base en tus valores editados."
        )

        CODIGOS_PERMITIDOS = list(set(CODIGOS_PREDICCION + CODIGOS_TABLA))
        df_todos = df_global.dropna(subset=['Codigo']).copy()
        df_todos = df_todos[df_todos['Codigo'].isin(CODIGOS_PERMITIDOS)]

        df_ultimos_masivo = df_todos.sort_values('Semana').groupby('Codigo').tail(1)

        opciones_todos = {
            int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}"
            for _, row in df_ultimos_masivo.iterrows()
        }
        opciones_filtro_tabla = ["Todos"] + list(opciones_todos.keys())

        col_masiva1, col_masiva2 = st.columns(2)
        with col_masiva1:
            semanas_proyectar_masivo = st.slider(
                "Semanas a predecir en la tabla masiva:",
                min_value=1, max_value=12, value=4, key="slider_masiva"
            )
        with col_masiva2:
            filtro_seleccionado = st.selectbox(
                "Filtra la tabla por un producto (o selecciona 'Todos'):",
                options=opciones_filtro_tabla,
                format_func=lambda x: "Todos los productos" if x == "Todos" else opciones_todos[x],
                key="filtro_seccion_4"
            )

        with st.spinner("Calculando predicciones de venta... esto puede tomar unos segundos."):
            df_a_procesar = (
                df_ultimos_masivo if filtro_seleccionado == "Todos"
                else df_ultimos_masivo[df_ultimos_masivo['Codigo'] == filtro_seleccionado]
            )

            if df_a_procesar.empty:
                st.info("No hay informacion de productos para predecir.")
            else:
                predicciones_por_codigo = {}
                base_por_codigo = {}

                for _, row in df_a_procesar.iterrows():
                    codigo = int(row['Codigo'])
                    inv_actual = row['Inventario'] if pd.notna(row['Inventario']) else 0.0

                    preds = predecir_directo(
                        modelos_directos, df_global, codigo, semanas_proyectar_masivo
                    )
                    predicciones_por_codigo[codigo] = preds
                    base_por_codigo[codigo] = {
                        'Descripcion': row.get('Descripcion', 'Desconocido'),
                        'Inv_Actual': inv_actual,
                    }

                editor_rows = []
                for codigo, preds in predicciones_por_codigo.items():
                    info = base_por_codigo[codigo]
                    row_data = {
                        'Codigo': codigo,
                        'Descripcion': info['Descripcion'],
                        'Inv. Actual (Inicial)': float(info['Inv_Actual']),
                    }
                    inv_sugerido = float(info['Inv_Actual'])
                    for sem_idx, p_val in enumerate(preds, start=1):
                        row_data[f'Inv. Sem +{sem_idx} ($)'] = inv_sugerido
                        inv_sugerido = max(0.0, inv_sugerido - float(p_val))
                    editor_rows.append(row_data)

                editor_df = pd.DataFrame(editor_rows)

                column_config = {
                    'Codigo': st.column_config.NumberColumn('Codigo', disabled=True, format='%d'),
                    'Descripcion': st.column_config.TextColumn('Descripcion', disabled=True),
                    'Inv. Actual (Inicial)': st.column_config.NumberColumn(
                        'Inv. Actual (Inicial)', disabled=True, format='$%.2f'
                    ),
                }
                for sem_idx in range(1, semanas_proyectar_masivo + 1):
                    column_config[f'Inv. Sem +{sem_idx} ($)'] = st.column_config.NumberColumn(
                        f'Inv. Sem +{sem_idx} ($)',
                        min_value=0.0, step=100.0, format='$%.2f',
                        help=f"Edita el inventario proyectado para la semana +{sem_idx}"
                    )

                st.markdown("##### ✏️ Tabla editable — ajusta los montos de inventario por semana:")
                edited_df = st.data_editor(
                    editor_df,
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key=f"editor_inv_{semanas_proyectar_masivo}_{filtro_seleccionado}"
                )

                resultados_finales = []
                for _, row in edited_df.iterrows():
                    codigo = int(row['Codigo'])
                    preds = predicciones_por_codigo[codigo]
                    res_dict = {
                        'Codigo': codigo,
                        'Descripcion': row['Descripcion'],
                        'Inv. Actual (Inicial)': float(row['Inv. Actual (Inicial)']) if pd.notna(row['Inv. Actual (Inicial)']) else 0.0,
                    }
                    for sem_idx, p_val in enumerate(preds, start=1):
                        col_name = f'Inv. Sem +{sem_idx} ($)'
                        inv_editado = float(row[col_name]) if pd.notna(row[col_name]) else 0.0
                        dias_inv = (inv_editado / p_val) * 30 if p_val > 0 else 0.0
                        res_dict[col_name] = inv_editado
                        res_dict[f'Pred. Sem +{sem_idx} ($)'] = float(p_val)
                        res_dict[f'Dias Inv Sem +{sem_idx}'] = dias_inv
                    resultados_finales.append(res_dict)

                df_resultados = pd.DataFrame(resultados_finales)

                format_dict = {'Inv. Actual (Inicial)': '${:,.2f}'}
                for sem_idx in range(1, semanas_proyectar_masivo + 1):
                    format_dict[f'Inv. Sem +{sem_idx} ($)'] = '${:,.2f}'
                    format_dict[f'Pred. Sem +{sem_idx} ($)'] = '${:,.2f}'
                    format_dict[f'Dias Inv Sem +{sem_idx}'] = '{:.1f}'

                st.markdown("##### 📊 Resultado final — Dias de inventario recalculados con tus ajustes:")
                st.dataframe(
                    df_resultados.style.format(format_dict),
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error al procesar: {e}")
