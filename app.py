import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import altair as alt

# ========================================================
# CONFIGURACION DE LA PAGINA
# ========================================================
st.set_page_config(page_title="Prediccion de Inventario", page_icon="📦", layout="wide")
st.title("📦 Simulador Predictivo de Dias de Inventario")
st.markdown("Sube todos los archivos de tu carpeta historica para entrenar el modelo y analizar el desempeño por producto.")

# ========================================================
# FUNCIONES AUXILIARES
# ========================================================
def obtener_rango_fechas(semana_num):
    base_date = datetime.date(2026, 6, 15)
    start_date = base_date + datetime.timedelta(weeks=int(semana_num) - 25)
    end_date = start_date + datetime.timedelta(days=6)
    meses = {1: 'ene', 2: 'feb', 3: 'mar', 4: 'abr', 5: 'may', 6: 'jun',
             7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'}
    return f"{start_date.day:02d} {meses[start_date.month]} - {end_date.day:02d} {meses[end_date.month]}"

def es_producto_muerto(df_prod, ventana_semanas=4):
    """NUEVO: Detecta si el producto no ha vendido en las ultimas semanas."""
    if df_prod.empty: return True
    ultimas = df_prod.sort_values('Semana').tail(ventana_semanas)['Monto de ventas']
    return bool(ultimas.fillna(0).sum() <= 0.01)

def predecir_futuro_recursivo(modelo, hist_ventas, n_semanas, hist_max, hist_median):
    """
    NUEVO: Prediccion Logaritmica + Clipping + Calculo dinamico de features.
    Recibe el historial completo de ventas del producto.
    """
    predicciones = []
    historial = list(hist_ventas) # Lista viva para actualizar los lags

    for _ in range(n_semanas):
        # Extraer lags del historial (protegido contra indices fuera de rango)
        v1 = historial[-1] if len(historial) >= 1 else 0
        v2 = historial[-2] if len(historial) >= 2 else v1
        v3 = historial[-3] if len(historial) >= 3 else v2
        med4 = np.median(historial[-4:]) if len(historial) >= 4 else np.median(historial)
        
        # Transformacion Logaritmica
        X_input = pd.DataFrame({
            'Venta_1_Semana_Atras': [np.log1p(v1)], 
            'Venta_2_Semanas_Atras': [np.log1p(v2)],
            'Venta_3_Semana_Atras': [np.log1p(v3)],
            'Venta_Mediana_4': [np.log1p(med4)]
        })
        
        # Prediccion y Destransformacion
        pred_log = modelo.predict(X_input)[0]
        pred = np.expm1(pred_log)
        pred = 0.01 if pd.isna(pred) or pred <= 0 else float(pred)
        
        # CLIPPING DINAMICO
        if hist_max > 0: pred = min(pred, hist_max * 1.3)
        if hist_median > 0: pred = max(pred, hist_median * 0.5)
        
        predicciones.append(pred)
        historial.append(pred)
        
    return predicciones

# ========================================================
# CARGA Y ENTRENAMIENTO GLOBAL
# ========================================================
@st.cache_resource(show_spinner="Entrenando modelo con Logaritmo y nuevas features...")
def cargar_y_entrenar(archivos_subidos):
    lista_dfs = []
    for archivo in archivos_subidos:
        try: lista_dfs.append(pd.read_excel(archivo, engine='openpyxl'))
        except Exception: pass 
    if not lista_dfs: raise ValueError("No se pudo leer ningun archivo.")
    
    df = pd.concat(lista_dfs, ignore_index=True)
    df.columns = [str(col).strip().split('[')[-1].replace(']', '') for col in df.columns]
    for col in df.columns:
        cl = col.lower()
        if 'descrip' in cl: df.rename(columns={col: 'Descripcion'}, inplace=True)
        elif 'categoria' in cl: df.rename(columns={col: 'Categoria'}, inplace=True)
        elif 'inventario' in cl and 'monto' not in cl and 'total' not in cl and 'dias' not in cl: df.rename(columns={col: 'Inventario'}, inplace=True) 
        elif 'venta' in cl: df.rename(columns={col: 'Monto de ventas'}, inplace=True)
        elif 'codigo' in cl or 'código' in cl: df.rename(columns={col: 'Codigo'}, inplace=True)
        elif 'semana' in cl: df.rename(columns={col: 'Semana'}, inplace=True)
        elif 'dias' in cl and 'inv' in cl: df.rename(columns={col: 'Dias Inv'}, inplace=True)
        
    df = df.loc[:, ~df.columns.duplicated()]
    df['Codigo'] = pd.to_numeric(df['Codigo'], errors='coerce')
    df['Semana'] = pd.to_numeric(df['Semana'], errors='coerce')
    df['Monto de ventas'] = pd.to_numeric(df['Monto de ventas'], errors='coerce')
    df['Inventario'] = pd.to_numeric(df['Inventario'], errors='coerce')
    df['Dias Inv'] = pd.to_numeric(df['Dias Inv'], errors='coerce') if 'Dias Inv' in df.columns else 0.0
    
    df = df.sort_values(by=['Codigo', 'Semana'])
    
    # NUEVAS FEATURES
    df['Venta_1_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_2_Semanas_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(2)
    df['Venta_3_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(3)
    df['Venta_Mediana_4'] = df.groupby('Codigo')['Monto de ventas'].transform(lambda s: s.shift(1).rolling(4, min_periods=1).median())
    
    df_modelo = df.dropna(subset=['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Venta_3_Semana_Atras', 'Venta_Mediana_4', 'Monto de ventas']).copy()
    
    if df_modelo.empty:
        raise ValueError("No hay historial suficiente (se requieren al menos 4 semanas).")
        
    features = ['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Venta_3_Semana_Atras', 'Venta_Mediana_4']
    
    # APLICAMOS LOGARITMO A LA VARIABLE OBJETIVO (Y)
    X = df_modelo[features]
    y = np.log1p(df_modelo['Monto de ventas'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_eval = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
    modelo_eval.fit(X_train, y_train)
    
    # Destransformamos las predicciones del test para calcular metricas reales
    preds_test_real = np.expm1(modelo_eval.predict(X_test))
    y_test_real = np.expm1(y_test)
    
    suma_errores = np.sum(np.abs(y_test_real - preds_test_real))
    suma_ventas = np.sum(y_test_real)
    
    wape = (suma_errores / suma_ventas) * 100 if suma_ventas > 0 else 0.0
    mae = mean_absolute_error(y_test_real, preds_test_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_test_real))
    r2 = r2_score(y_test_real, preds_test_real)

    modelo_final = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
    modelo_final.fit(X, y)
    
    return df, modelo_final, wape, mae, rmse, r2

@st.cache_resource(show_spinner=False)
def entrenar_backtest(df, n_semanas_holdout, codigos_tuple):
    codigos = list(codigos_tuple)
    df_work = df.dropna(subset=['Codigo', 'Semana', 'Monto de ventas']).copy()
    df_work = df_work[df_work['Codigo'].isin(codigos)].sort_values(['Codigo', 'Semana']).reset_index(drop=True)

    train_rows = []
    held_rows = []

    for codigo, group in df_work.groupby('Codigo'):
        group = group.sort_values('Semana').reset_index(drop=True)
        if len(group) <= n_semanas_holdout + 3:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:-n_semanas_holdout])
        held_rows.append(group.iloc[-n_semanas_holdout:])

    if not train_rows: raise ValueError("No hay datos de entrenamiento suficientes.")

    df_train = pd.concat(train_rows, ignore_index=True)
    df_held = pd.concat(held_rows, ignore_index=True) if held_rows else pd.DataFrame()

    df_train = df_train.sort_values(['Codigo', 'Semana'])
    df_train['Venta_1_Semana_Atras'] = df_train.groupby('Codigo')['Monto de ventas'].shift(1)
    df_train['Venta_2_Semanas_Atras'] = df_train.groupby('Codigo')['Monto de ventas'].shift(2)
    df_train['Venta_3_Semana_Atras'] = df_train.groupby('Codigo')['Monto de ventas'].shift(3)
    df_train['Venta_Mediana_4'] = df_train.groupby('Codigo')['Monto de ventas'].transform(lambda s: s.shift(1).rolling(4, min_periods=1).median())

    features = ['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Venta_3_Semana_Atras', 'Venta_Mediana_4']
    df_tr_model = df_train.dropna(subset=features + ['Monto de ventas'])

    if df_tr_model.empty: raise ValueError("Sin datos suficientes para entrenar el backtest.")

    X = df_tr_model[features]
    y = np.log1p(df_tr_model['Monto de ventas']) # Log transform en Backtest

    modelo = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
    modelo.fit(X, y)

    return modelo, df_held, df_train

# ========================================================
# INTERFAZ
# ========================================================
st.subheader("1. Origen de los Datos")
archivos_subidos = st.file_uploader("Selecciona todos los archivos Excel:", type=["xlsx", "xls", "xlsm"], accept_multiple_files=True)

if archivos_subidos:
    try:
        df_global, modelo_rf, wape_val, mae_val, rmse_val, r2_val = cargar_y_entrenar(archivos_subidos)
        
        st.success("Archivos cargados y modelo optimizado entrenado con exito!")
        
        st.markdown("##### Metricas de Ajuste (Split Aleatorio con correccion Logaritmica)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Error WAPE", f"{wape_val:.2f}%")
        col_m2.metric("Error Promedio (MAE)", f"${mae_val:,.2f}")
        col_m3.metric("Penalización (RMSE)", f"${rmse_val:,.2f}")
        col_m4.metric("Explicabilidad (R²)", f"{max(0, r2_val):.2f}")

        st.markdown("---")

        CODIGOS_PREDICCION = [7506475104722, 7506475124614, 7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 
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
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077]
        CODIGOS_TABLA = [7506475104722, 7506475124614, 7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 
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
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077]

        # ========================================================
        # 1.5. BACKTEST
        # ========================================================
        st.subheader("1.5. Validación Histórica por Semana (Backtest)")
        
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            semanas_holdout = st.slider("Semanas historicas a reservar:", 1, 12, 4, key="slider_holdout")
        with col_bt2:
            semana_inicio_metrica = st.slider("Calcular a partir de semana #:", 1, semanas_holdout, 1, key="slider_semana_inicio")

        try:
            modelo_bt, df_held_bt, df_train_bt = entrenar_backtest(df_global, semanas_holdout, tuple(CODIGOS_PREDICCION))

            if df_held_bt.empty:
                st.warning("No hay suficientes semanas para hacer el backtest.")
            else:
                resultados_bt = []
                for codigo, group in df_held_bt.groupby('Codigo'):
                    group = group.sort_values('Semana').reset_index(drop=True)
                    train_group = df_train_bt[df_train_bt['Codigo'] == codigo].sort_values('Semana')

                    if train_group.empty or len(train_group) < 4: continue

                    n_semanas = len(group)
                    hist_real = train_group['Monto de ventas'].dropna().values
                    
                    # FILTRO DE PRODUCTO MUERTO
                    if es_producto_muerto(train_group):
                        preds = [0.0] * n_semanas
                    else:
                        h_max = hist_real.max() if len(hist_real) > 0 else 0
                        h_med = np.median(hist_real) if len(hist_real) > 0 else 0
                        preds = predecir_futuro_recursivo(modelo_bt, hist_real, n_semanas, h_max, h_med)

                    for i, (pred, actual) in enumerate(zip(preds, group['Monto de ventas'].tolist()), start=1):
                        if pd.isna(actual): continue
                        resultados_bt.append({
                            'Codigo': codigo, 'Descripcion': group.iloc[i-1].get('Descripcion', 'Desc.'),
                            'Semana_Predecida': i, 'Prediccion': float(pred), 'Actual': float(actual)
                        })

                if not resultados_bt:
                    st.warning("No se pudieron generar predicciones de backtest.")
                else:
                    df_bt = pd.DataFrame(resultados_bt)
                    df_bt['Error_Abs'] = (df_bt['Prediccion'] - df_bt['Actual']).abs()
                    df_bt['Variacion_%'] = np.where(df_bt['Actual'] > 0, ((df_bt['Prediccion'] - df_bt['Actual']) / df_bt['Actual']) * 100, 0.0)
                    df_bt_filt = df_bt[df_bt['Semana_Predecida'] >= semana_inicio_metrica].copy()

                    if not df_bt_filt.empty:
                        y_true = df_bt_filt['Actual'].values
                        y_pred = df_bt_filt['Prediccion'].values

                        wape_bt = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100 if np.sum(y_true) > 0 else 0.0
                        st.markdown(f"#### 📊 Metricas de Backtest — Semanas +{semana_inicio_metrica} a +{semanas_holdout}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("WAPE", f"{wape_bt:.2f}%")
                        c2.metric("MAE", f"${mean_absolute_error(y_true, y_pred):,.2f}")
                        c3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
                        c4.metric("R²", f"{max(0, r2_score(y_true, y_pred)):.2f}")

                        with st.expander("🔎 Ver detalle y Top 10 Peores Predicciones"):
                            st.markdown("##### 🚨 Top 10 Peores Predicciones")
                            top_10 = df_bt_filt.nlargest(10, 'Error_Abs')
                            st.dataframe(top_10.style.format({'Prediccion': '${:,.2f}', 'Actual': '${:,.2f}', 'Error_Abs': '${:,.2f}', 'Variacion_%': '{:+.1f}%'}), use_container_width=True, hide_index=True)
                            
                            st.markdown("##### 📋 Detalle Completo")
                            st.dataframe(df_bt_filt.sort_values(['Codigo', 'Semana_Predecida']).style.format({'Prediccion': '${:,.2f}', 'Actual': '${:,.2f}', 'Error_Abs': '${:,.2f}', 'Variacion_%': '{:+.1f}%'}), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error en backtest: {e}")

        st.markdown("---")

        # ========================================================
        # 2. PREDICCIÓN Y GRÁFICA
        # ========================================================
        st.subheader("2. Simulacion y Tendencia de Producto")
        df_pred = df_global.dropna(subset=['Codigo']).copy()
        df_pred = df_pred[df_pred['Codigo'].isin(CODIGOS_PREDICCION)]
        
        if not df_pred.empty:
            df_ultimos = df_pred.sort_values('Semana').groupby('Codigo').tail(1)
            opciones_productos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos.iterrows()}
                
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                codigo_seleccionado = st.selectbox("Selecciona el Producto:", options=list(opciones_productos.keys()), format_func=lambda x: opciones_productos[x])
            with col_sim2:
                monto_enviar = st.number_input("Monto de mercancia a enviar ($):", min_value=0.0, value=0.0, step=1000.0)
            with col_sim3:
                semanas_proyectar = st.slider("Semanas a proyectar (Gráfica):", 1, 8, 4, key="slider_grafica")

            df_hist_prod = df_pred[df_pred['Codigo'] == codigo_seleccionado].sort_values('Semana')
            
            if not df_hist_prod.empty:
                semana_max = df_hist_prod['Semana'].max()
                df_ultimo_registro = df_hist_prod[df_hist_prod['Semana'] == semana_max].iloc[0]
                
                hist_ventas = df_hist_prod['Monto de ventas'].dropna().values
                
                if es_producto_muerto(df_hist_prod):
                    preds_futuras = [0.0] * semanas_proyectar
                    st.warning("⚠️ Este producto no ha registrado ventas recientes. Se proyecta en $0.")
                else:
                    h_max = hist_ventas.max() if len(hist_ventas) > 0 else 0
                    h_med = np.median(hist_ventas) if len(hist_ventas) > 0 else 0
                    preds_futuras = predecir_futuro_recursivo(modelo_rf, hist_ventas, semanas_proyectar, h_max, h_med)
                
                df_real = df_hist_prod[['Semana', 'Monto de ventas']].copy()
                df_real['Tipo'] = 'Historico Real'
                
                filas_futuras = [{'Semana': semana_max, 'Monto de ventas': df_ultimo_registro['Monto de ventas'], 'Tipo': 'Proyectado'}]
                for idx, pred_val in enumerate(preds_futuras, start=1):
                    filas_futuras.append({'Semana': semana_max + idx, 'Monto de ventas': pred_val, 'Tipo': 'Proyectado'})
                
                df_grafica_comb = pd.concat([df_real, pd.DataFrame(filas_futuras)], ignore_index=True)

                st.markdown(f"#### 📈 Tendencia Historica y Proyeccion ({semanas_proyectar} Semanas)")
                grafica = alt.Chart(df_grafica_comb).mark_line(point=True).encode(
                    x=alt.X('Semana:O', title='Semana', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto de ventas:Q', title='Monto de Ventas ($)', axis=alt.Axis(format='$,.2f')),
                    color=alt.Color('Tipo:N', title='Origen', scale=alt.Scale(domain=['Historico Real', 'Proyectado'], range=['#1f77b4', '#ff7f0e'])),
                    strokeDash=alt.condition(alt.datum.Tipo == 'Proyectado', alt.value([4, 4]), alt.value([0])),
                    tooltip=['Semana:O', alt.Tooltip('Monto de ventas:Q', format='$,.2f'), 'Tipo:N']
                ).properties(height=350).interactive()
                
                st.altair_chart(grafica, use_container_width=True)

                inv_actual = df_ultimo_registro['Inventario'] if pd.notna(df_ultimo_registro['Inventario']) else 0
                inv_futuro = inv_actual + monto_enviar
                venta_prom_est = np.mean(preds_futuras) if len(preds_futuras) > 0 else 0
                dias_inv = (inv_futuro / venta_prom_est) * 30 if venta_prom_est > 0 else 0
                
                st.markdown("### 📊 Resultados de la Proyeccion")
                r1, r2, r3, r4 = st.columns(4)
                r1.info(f"Inv. en Sistema:\n\n${inv_actual:,.2f}")
                r2.warning(f"Inv. Total (Simulado):\n\n${inv_futuro:,.2f}")
                r3.info(f"Venta Est. Prom.:\n\n${venta_prom_est:,.2f}")
                r4.success(f"Dias Cobertura Prom.:\n\n{dias_inv:.1f} dias")

        st.markdown("---")

        # ========================================================
        # 3. REPORTE SEMANAL
        # ========================================================
        st.subheader("3. Reporte Semanal (Dias de Inventario)")
        df_tabla = df_global.dropna(subset=['Codigo']).copy()
        df_tabla = df_tabla[df_tabla['Codigo'].isin(CODIGOS_TABLA)]
        
        if not df_tabla.empty:
            cols_indice = ['Codigo', 'Descripcion']
            if 'Categoria' in df_tabla.columns: cols_indice.append('Categoria')
            df_pivot = df_tabla.pivot_table(index=cols_indice, columns='Semana', values='Dias Inv', aggfunc='sum').reset_index().fillna(0)
            semanas_cols = sorted([c for c in df_pivot.columns if isinstance(c, (int, float))])
            
            if len(semanas_cols) > 0:
                df_pivot['Variacion'] = df_pivot[semanas_cols[-1]] - df_pivot[semanas_cols[0]]
                df_pivot['% Meta'] = (df_pivot[semanas_cols[-1]] / 60 * 100).round(1).astype(str) + '%'
            
            df_pivot = df_pivot.rename(columns={sem: obtener_rango_fechas(sem) for sem in semanas_cols})
            st.dataframe(df_pivot, use_container_width=True)

        st.markdown("---")

        # ========================================================
        # 4. TABLA MASIVA
        # ========================================================
        st.subheader("4. Proyeccion Masiva de Inventario")
        CODIGOS_PERMITIDOS = list(set(CODIGOS_PREDICCION + CODIGOS_TABLA))
        df_todos = df_global.dropna(subset=['Codigo']).copy()
        df_ultimos_masivo = df_todos[df_todos['Codigo'].isin(CODIGOS_PERMITIDOS)].sort_values('Semana').groupby('Codigo').tail(1)
        
        opciones_todos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos_masivo.iterrows()}
        
        col_masiva1, col_masiva2 = st.columns(2)
        with col_masiva1:
            semanas_proyectar_masivo = st.slider("Semanas a predecir:", 1, 12, 4, key="slider_masiva")
        with col_masiva2:
            filtro_seleccionado = st.selectbox("Filtra por producto:", ["Todos"] + list(opciones_todos.keys()), format_func=lambda x: "Todos" if x == "Todos" else opciones_todos[x])
        
        with st.spinner("Calculando predicciones con Log y Clipping..."):
            df_a_procesar = df_ultimos_masivo if filtro_seleccionado == "Todos" else df_ultimos_masivo[df_ultimos_masivo['Codigo'] == filtro_seleccionado]
            
            if not df_a_procesar.empty:
                predicciones_por_codigo = {}
                base_por_codigo = {}
                
                for _, row in df_a_procesar.iterrows():
                    codigo = int(row['Codigo'])
                    df_prod_hist = df_global[df_global['Codigo'] == codigo]
                    hist_ventas = df_prod_hist['Monto de ventas'].dropna().values
                    
                    if es_producto_muerto(df_prod_hist):
                        preds = [0.0] * semanas_proyectar_masivo
                    else:
                        h_max = hist_ventas.max() if len(hist_ventas) > 0 else 0
                        h_med = np.median(hist_ventas) if len(hist_ventas) > 0 else 0
                        preds = predecir_futuro_recursivo(modelo_rf, hist_ventas, semanas_proyectar_masivo, h_max, h_med)
                        
                    predicciones_por_codigo[codigo] = preds
                    base_por_codigo[codigo] = {'Descripcion': row.get('Descripcion', 'Desc.'), 'Inv_Actual': row['Inventario'] if pd.notna(row['Inventario']) else 0.0}
                
                editor_rows = []
                for codigo, preds in predicciones_por_codigo.items():
                    info = base_por_codigo[codigo]
                    row_data = {'Codigo': codigo, 'Descripcion': info['Descripcion'], 'Inv. Actual (Inicial)': float(info['Inv_Actual'])}
                    inv_sugerido = float(info['Inv_Actual'])
                    for sem_idx, p_val in enumerate(preds, start=1):
                        row_data[f'Inv. Sem +{sem_idx} ($)'] = inv_sugerido
                        inv_sugerido = max(0.0, inv_sugerido - float(p_val))
                    editor_rows.append(row_data)
                
                editor_df = pd.DataFrame(editor_rows)
                column_config = {
                    'Codigo': st.column_config.NumberColumn('Codigo', disabled=True, format='%d'),
                    'Descripcion': st.column_config.TextColumn('Descripcion', disabled=True),
                    'Inv. Actual (Inicial)': st.column_config.NumberColumn('Inv. Actual (Inicial)', disabled=True, format='$%.2f'),
                }
                for sem_idx in range(1, semanas_proyectar_masivo + 1):
                    column_config[f'Inv. Sem +{sem_idx} ($)'] = st.column_config.NumberColumn(f'Inv. Sem +{sem_idx} ($)', min_value=0.0, step=100.0, format='$%.2f')
                
                st.markdown("##### ✏️ Tabla editable — ajusta los montos de inventario:")
                edited_df = st.data_editor(editor_df, column_config=column_config, use_container_width=True, hide_index=True, num_rows="fixed", key=f"editor_inv_{semanas_proyectar_masivo}")
                
                resultados_finales = []
                for _, row in edited_df.iterrows():
                    codigo = int(row['Codigo'])
                    preds = predicciones_por_codigo[codigo]
                    res_dict = {'Codigo': codigo, 'Descripcion': row['Descripcion'], 'Inv. Actual (Inicial)': float(row['Inv. Actual (Inicial)']) if pd.notna(row['Inv. Actual (Inicial)']) else 0.0}
                    
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
                
                st.markdown("##### 📊 Resultado final — Dias de inventario recalculados:")
                st.dataframe(df_resultados.style.format(format_dict), use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar: {e}")
