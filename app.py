import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import altair as alt

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

# ========================================================
# CONFIGURACION
# ========================================================
st.set_page_config(page_title="Prediccion de Inventario", page_icon="📦", layout="wide")
st.title("📦 Simulador Predictivo de Dias de Inventario (Optimizado para Picos)")

# ========================================================
# MODELO CUANTÍLICO (El secreto para los picos altos)
# ========================================================
def construir_modelo():
    # Usamos alpha=0.8 (Percentil 80) para forzar al modelo a mirar los picos altos
    if LGB_AVAILABLE:
        return lgb.LGBMRegressor(
            objective='quantile', alpha=0.8, 
            n_estimators=200, max_depth=4, num_leaves=8, learning_rate=0.05,
            min_child_samples=4, subsample=0.9, random_state=42, n_jobs=-1, verbose=-1
        )
    # Fallback si no hay LightGBM
    return GradientBoostingRegressor(
        loss='quantile', alpha=0.8,
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )

def obtener_rango_fechas(semana_num):
    base_date = datetime.date(2026, 6, 15)
    start = base_date + datetime.timedelta(weeks=int(semana_num) - 25)
    end = start + datetime.timedelta(days=6)
    meses = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
    return f"{start.day:02d} {meses[start.month]} - {end.day:02d} {meses[end.month]}"

def es_producto_muerto(df_prod, ventana_semanas=3):
    if df_prod.empty: return True
    ultimas = df_prod.sort_values('Semana').tail(ventana_semanas)['Monto de ventas']
    return bool(ultimas.fillna(0).sum() <= 0.01)

def predecir_recursivo_seguro(modelo, hist_ventas, n_semanas):
    """
    Prediccion recursiva con Tendencia Multiplicativa.
    Evita las lineas planas y respeta la escala real sin limites artificiales (Clipping).
    """
    predicciones = []
    
    v_prev1 = float(hist_ventas[-1]) if len(hist_ventas) >= 1 else 0.0
    v_prev2 = float(hist_ventas[-2]) if len(hist_ventas) >= 2 else v_prev1
    
    # Calcular factor de tendencia (Multiplicativo)
    tendencia = (v_prev1 / v_prev2) if v_prev2 > 0 else 1.0
    # Limitar la tendencia para que no explote (entre 0.8 y 1.25)
    tendencia = np.clip(tendencia, 0.8, 1.25)
    
    for i in range(n_semanas):
        X_input = pd.DataFrame({'Venta_1_Semana_Atras': [v_prev1], 'Venta_2_Semanas_Atras': [v_prev2]})
        try:
            pred_modelo = float(modelo.predict(X_input)[0])
        except:
            pred_modelo = v_prev1
            
        # Aplicar la tendencia suavemente a partir de la semana 2
        if i > 0:
            pred_final = pred_modelo * (tendencia ** i)
        else:
            pred_final = pred_modelo
            
        pred_final = max(0.01, pred_final)
        predicciones.append(pred_final)
        
        v_prev2 = v_prev1
        v_prev1 = pred_final
        
    return predicciones

# ========================================================
# ENTRENAMIENTO (Retorna un Diccionario para evitar errores)
# ========================================================
@st.cache_resource(show_spinner="Entrenando modelo con regresión cuantílica y pesos...")
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
    for c in ['Codigo', 'Semana', 'Monto de ventas', 'Inventario', 'Dias Inv']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Dias Inv' not in df.columns: df['Dias Inv'] = 0.0
    
    df = df.sort_values(['Codigo', 'Semana'])
    df['Venta_1_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_2_Semanas_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(2)
    
    df_modelo = df.dropna(subset=['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Monto de ventas']).copy()
    if df_modelo.empty: raise ValueError("No hay historial suficiente.")
    
    X = df_modelo[['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras']]
    y = df_modelo['Monto de ventas']
    
    # PONDERACION DE MUESTRAS (Dar mas peso a ventas altas)
    pesos = 1.0 + (y / y.mean()).clip(0, 3.5)
    
    X_train, X_test, y_train, y_test, w_train = train_test_split(X, y, pesos, test_size=0.2, random_state=42)
    
    modelo_eval = construir_modelo()
    if LGB_AVAILABLE:
        modelo_eval.fit(X_train, y_train, sample_weight=w_train)
    else:
        modelo_eval.fit(X_train, y_train, sample_weight=w_train)
        
    preds_test = modelo_eval.predict(X_test)
    
    wape = (np.sum(np.abs(y_test - preds_test)) / y_test.sum()) * 100 if y_test.sum() > 0 else 0.0
    
    modelo_final = construir_modelo()
    modelo_final.fit(X, y, sample_weight=pesos)
    
    # RETORNO COMO DICCIONARIO (Evita el error "too many values to unpack")
    return {
        'df': df,
        'modelo': modelo_final,
        'wape': wape,
        'mae': mean_absolute_error(y_test, preds_test),
        'rmse': np.sqrt(mean_squared_error(y_test, preds_test)),
        'r2': r2_score(y_test, preds_test)
    }

@st.cache_resource(show_spinner=False)
def entrenar_backtest(df, n_semanas_holdout, codigos_tuple):
    codigos = list(codigos_tuple)
    df_work = df.dropna(subset=['Codigo', 'Semana', 'Monto de ventas']).copy()
    df_work = df_work[df_work['Codigo'].isin(codigos)].sort_values(['Codigo', 'Semana']).reset_index(drop=True)

    train_rows, held_rows = [], []
    for codigo, group in df_work.groupby('Codigo'):
        if len(group) <= n_semanas_holdout + 2:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:-n_semanas_holdout])
        held_rows.append(group.iloc[-n_semanas_holdout:])

    df_train = pd.concat(train_rows, ignore_index=True)
    df_held = pd.concat(held_rows, ignore_index=True) if held_rows else pd.DataFrame()

    df_train = df_train.sort_values(['Codigo', 'Semana'])
    df_train['Venta_1_Semana_Atras'] = df_train.groupby('Codigo')['Monto de ventas'].shift(1)
    df_train['Venta_2_Semanas_Atras'] = df_train.groupby('Codigo')['Monto de ventas'].shift(2)

    df_tr_model = df_train.dropna(subset=['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Monto de ventas'])
    
    X = df_tr_model[['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras']]
    y = df_tr_model['Monto de ventas']
    pesos = 1.0 + (y / y.mean()).clip(0, 3.5)
    
    modelo = construir_modelo()
    modelo.fit(X, y, sample_weight=pesos)
    
    return modelo, df_held, df_train

# ========================================================
# INTERFAZ
# ========================================================
st.subheader("1. Origen de los Datos")
archivos_subidos = st.file_uploader("Selecciona archivos Excel:", type=["xlsx", "xls", "xlsm"], accept_multiple_files=True)

if archivos_subidos:
    try:
        # Extraemos los valores del diccionario de forma segura
        resultado = cargar_y_entrenar(archivos_subidos)
        df_global = resultado['df']
        modelo_rf = resultado['modelo']
        wape_val = resultado['wape']
        mae_val = resultado['mae']
        rmse_val = resultado['rmse']
        r2_val = resultado['r2']
        
        st.success("Modelo Cuantílico entrenado con exito.")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("WAPE Base", f"{wape_val:.2f}%")
        col_m2.metric("MAE Base", f"${mae_val:,.2f}")
        col_m3.metric("RMSE Base", f"${rmse_val:,.2f}")
        col_m4.metric("R² Base", f"{max(0, r2_val):.2f}")

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

        st.subheader("1.5. Validación Histórica (Backtest)")
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1: semanas_holdout = st.slider("Semanas a reservar:", 1, 12, 4, key="s_hold")
        with col_bt2: semana_inicio_metrica = st.slider("Calcular a partir de semana #:", 1, semanas_holdout, 1, key="s_ini")

        try:
            modelo_bt, df_held_bt, df_train_bt = entrenar_backtest(df_global, semanas_holdout, tuple(CODIGOS_PREDICCION))

            if not df_held_bt.empty:
                resultados_bt = []
                for codigo, group in df_held_bt.groupby('Codigo'):
                    group = group.sort_values('Semana').reset_index(drop=True)
                    train_group = df_train_bt[df_train_bt['Codigo'] == codigo].sort_values('Semana')
                    if train_group.empty or len(train_group) < 3: continue

                    hist_real = train_group['Monto de ventas'].dropna().values
                    
                    if es_producto_muerto(train_group):
                        preds = [0.0] * len(group)
                    else:
                        preds = predecir_recursivo_seguro(modelo_bt, hist_real, len(group))

                    for i, (pred, actual) in enumerate(zip(preds, group['Monto de ventas'].tolist()), start=1):
                        if pd.isna(actual): continue
                        resultados_bt.append({
                            'Codigo': codigo, 'Descripcion': group.iloc[i-1].get('Descripcion', 'Desc.'),
                            'Semana_Predecida': i, 'Prediccion': float(pred), 'Actual': float(actual)
                        })

                if resultados_bt:
                    df_bt = pd.DataFrame(resultados_bt)
                    df_bt['Error_Abs'] = (df_bt['Prediccion'] - df_bt['Actual']).abs()
                    df_bt['Variacion_%'] = np.where(df_bt['Actual'] > 0, ((df_bt['Prediccion'] - df_bt['Actual']) / df_bt['Actual']) * 100, 0.0)
                    df_bt_filt = df_bt[df_bt['Semana_Predecida'] >= semana_inicio_metrica].copy()

                    if not df_bt_filt.empty:
                        y_t, y_p = df_bt_filt['Actual'].values, df_bt_filt['Prediccion'].values
                        wape_bt = (np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t))) * 100 if np.sum(y_t) > 0 else 0.0
                        
                        st.markdown(f"#### 📊 Métricas Backtest — Semanas +{semana_inicio_metrica} a +{semanas_holdout}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("WAPE", f"{wape_bt:.2f}%")
                        c2.metric("MAE", f"${mean_absolute_error(y_t, y_p):,.2f}")
                        c3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_t, y_p)):,.2f}")
                        c4.metric("R²", f"{max(0, r2_score(y_t, y_p)):.2f}")

                        with st.expander("🔎 Ver detalle y Top 10 Peores Predicciones"):
                            st.dataframe(df_bt_filt.nlargest(10, 'Error_Abs').style.format({'Prediccion': '${:,.2f}', 'Actual': '${:,.2f}', 'Error_Abs': '${:,.2f}', 'Variacion_%': '{:+.1f}%'}), use_container_width=True, hide_index=True)

        except Exception as e: st.error(f"Error en backtest: {e}")

        st.markdown("---")

        st.subheader("2. Simulacion y Tendencia de Producto")
        df_pred = df_global.dropna(subset=['Codigo']).copy()
        df_pred = df_pred[df_pred['Codigo'].isin(CODIGOS_PREDICCION)]
        
        if not df_pred.empty:
            df_ultimos = df_pred.sort_values('Semana').groupby('Codigo').tail(1)
            opciones_productos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos.iterrows()}
                
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1: codigo_sel = st.selectbox("Producto:", options=list(opciones_productos.keys()), format_func=lambda x: opciones_productos[x])
            with col_sim2: monto_enviar = st.number_input("Mercancia ($):", min_value=0.0, value=0.0, step=1000.0)
            with col_sim3: sem_proy = st.slider("Semanas a proyectar:", 1, 8, 4, key="s_graf")

            df_hist_prod = df_pred[df_pred['Codigo'] == codigo_sel].sort_values('Semana')
            if not df_hist_prod.empty:
                semana_max = df_hist_prod['Semana'].max()
                df_ult = df_hist_prod[df_hist_prod['Semana'] == semana_max].iloc[0]
                hist_ventas = df_hist_prod['Monto de ventas'].dropna().values
                
                if es_producto_muerto(df_hist_prod):
                    preds_futuras = [0.0] * sem_proy
                    st.warning("⚠️ Producto sin ventas recientes. Proyeccion en $0.")
                else:
                    preds_futuras = predecir_recursivo_seguro(modelo_rf, hist_ventas, sem_proy)
                
                df_real = df_hist_prod[['Semana', 'Monto de ventas']].copy()
                df_real['Tipo'] = 'Historico Real'
                
                filas_futuras = [{'Semana': semana_max, 'Monto de ventas': df_ult['Monto de ventas'], 'Tipo': 'Proyectado'}]
                for idx, p_val in enumerate(preds_futuras, start=1):
                    filas_futuras.append({'Semana': semana_max + idx, 'Monto de ventas': p_val, 'Tipo': 'Proyectado'})
                
                df_graf = pd.concat([df_real, pd.DataFrame(filas_futuras)], ignore_index=True)
                grafica = alt.Chart(df_graf).mark_line(point=True).encode(
                    x=alt.X('Semana:O', title='Semana', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto de ventas:Q', title='Ventas ($)', axis=alt.Axis(format='$,.2f')),
                    color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Historico Real', 'Proyectado'], range=['#1f77b4', '#ff7f0e'])),
                    strokeDash=alt.condition(alt.datum.Tipo == 'Proyectado', alt.value([4, 4]), alt.value([0])),
                    tooltip=['Semana:O', alt.Tooltip('Monto de ventas:Q', format='$,.2f'), 'Tipo:N']
                ).properties(height=350).interactive()
                st.altair_chart(grafica, use_container_width=True)

                inv_act = df_ult['Inventario'] if pd.notna(df_ult['Inventario']) else 0
                inv_fut = inv_act + monto_enviar
                venta_prom = np.mean(preds_futuras) if len(preds_futuras) > 0 else 0
                dias_inv = (inv_fut / venta_prom) * 30 if venta_prom > 0 else 0
                
                r1, r2, r3, r4 = st.columns(4)
                r1.info(f"Inv. Sistema:\n\n${inv_act:,.2f}")
                r2.warning(f"Inv. Total:\n\n${inv_fut:,.2f}")
                r3.info(f"Venta Prom. Est.:\n\n${venta_prom:,.2f}")
                r4.success(f"Dias Cobertura:\n\n{dias_inv:.1f} dias")

        st.markdown("---")

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

        st.subheader("4. Proyeccion Masiva de Inventario")
        CODIGOS_PERMITIDOS = list(set(CODIGOS_PREDICCION + CODIGOS_TABLA))
        df_ultimos_masivo = df_global[df_global['Codigo'].isin(CODIGOS_PERMITIDOS)].sort_values('Semana').groupby('Codigo').tail(1)
        opciones_todos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos_masivo.iterrows()}
        
        col_m1, col_m2 = st.columns(2)
        with col_m1: sem_proy_mas = st.slider("Semanas a predecir:", 1, 12, 4, key="slider_mas")
        with col_m2: filtro = st.selectbox("Filtrar:", ["Todos"] + list(opciones_todos.keys()), format_func=lambda x: "Todos" if x == "Todos" else opciones_todos[x])
        
        with st.spinner("Calculando..."):
            df_proc = df_ultimos_masivo if filtro == "Todos" else df_ultimos_masivo[df_ultimos_masivo['Codigo'] == filtro]
            if not df_proc.empty:
                preds_dict, base_dict = {}, {}
                for _, row in df_proc.iterrows():
                    cod = int(row['Codigo'])
                    df_hist = df_global[df_global['Codigo'] == cod]
                    hist = df_hist['Monto de ventas'].dropna().values
                    
                    if es_producto_muerto(df_hist):
                        preds = [0.0] * sem_proy_mas
                    else:
                        preds = predecir_recursivo_seguro(modelo_rf, hist, sem_proy_mas)
                        
                    preds_dict[cod] = preds
                    base_dict[cod] = {'Desc': row.get('Descripcion', 'Desc.'), 'Inv': row['Inventario'] if pd.notna(row['Inventario']) else 0.0}
                
                editor_rows = []
                for cod, preds in preds_dict.items():
                    info = base_dict[cod]
                    r_data = {'Codigo': cod, 'Descripcion': info['Desc'], 'Inv. Actual': float(info['Inv'])}
                    inv_sug = float(info['Inv'])
                    for i, p in enumerate(preds, start=1):
                        r_data[f'Inv. Sem +{i}'] = inv_sug
                        inv_sug = max(0.0, inv_sug - p)
                    editor_rows.append(r_data)
                
                editor_df = pd.DataFrame(editor_rows)
                cfg = {'Codigo': st.column_config.NumberColumn(disabled=True), 'Descripcion': st.column_config.TextColumn(disabled=True), 'Inv. Actual': st.column_config.NumberColumn('Inv. Actual', disabled=True, format='$%.2f')}
                for i in range(1, sem_proy_mas + 1): cfg[f'Inv. Sem +{i}'] = st.column_config.NumberColumn(f'Inv. Sem +{i}', min_value=0.0, step=100.0, format='$%.2f')
                
                edited = st.data_editor(editor_df, column_config=cfg, use_container_width=True, hide_index=True, key=f"ed_{sem_proy_mas}")
                
                res = []
                for _, row in edited.iterrows():
                    cod = int(row['Codigo'])
                    r_dict = {'Codigo': cod, 'Descripcion': row['Descripcion'], 'Inv. Actual': float(row['Inv. Actual'])}
                    for i, p in enumerate(preds_dict[cod], start=1):
                        col = f'Inv. Sem +{i}'
                        inv_ed = float(row[col]) if pd.notna(row[col]) else 0.0
                        r_dict[col] = inv_ed
                        r_dict[f'Pred. Sem +{i}'] = p
                        r_dict[f'Dias Inv +{i}'] = (inv_ed / p) * 30 if p > 0 else 0.0
                    res.append(r_dict)
                
                df_final = pd.DataFrame(res)
                fmt = {'Inv. Actual': '${:,.2f}'}
                for i in range(1, sem_proy_mas + 1):
                    fmt[f'Inv. Sem +{i}'] = '${:,.2f}'
                    fmt[f'Pred. Sem +{i}'] = '${:,.2f}'
                    fmt[f'Dias Inv +{i}'] = '{:.1f}'
                
                st.dataframe(df_final.style.format(fmt), use_container_width=True)

    except Exception as e:
        st.error(f"Error crítico en la app: {e}")
