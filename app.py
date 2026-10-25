import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import altair as alt

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

st.set_page_config(page_title="Prediccion de Inventario", page_icon="📦", layout="wide")
st.title("📦 Simulador Predictivo de Dias de Inventario")

if not LGB_AVAILABLE:
    st.warning("⚠️ LightGBM no esta instalado. Usando RandomForest como fallback.")

def construir_modelo_principal():
    if LGB_AVAILABLE:
        return lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=3,
            num_leaves=6,
            learning_rate=0.05,
            min_child_samples=6,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.5,
            objective='regression_l1',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    return RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

def obtener_rango_fechas(semana_num):
    base_date = datetime.date(2026, 6, 15)
    start = base_date + datetime.timedelta(weeks=int(semana_num) - 25)
    end = start + datetime.timedelta(days=6)
    meses = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
             7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
    return f"{start.day:02d} {meses[start.month]} - {end.day:02d} {meses[end.month]}"

def es_producto_muerto(df_prod, ventana_semanas=4):
    if df_prod.empty: return True
    ultimas = df_prod.sort_values('Semana').tail(ventana_semanas)['Monto de ventas']
    return bool(ultimas.fillna(0).sum() <= 0.01)

def clasificar_producto(df_prod):
    if df_prod.empty or es_producto_muerto(df_prod): return 'muerto'
    hist = df_prod['Monto de ventas'].dropna()
    if len(hist) < 4: return 'volatil'
    pct_ceros = float((hist <= 0.01).mean())
    if pct_ceros > 0.5: return 'intermitente'
    mean_val = float(hist.mean())
    if mean_val <= 0: return 'intermitente'
    cv = float(hist.std() / mean_val) if len(hist) > 1 else 0.0
    return 'estable' if cv < 0.4 else 'volatil'

# ========================================================
# PREDICCIÓN CON CLIPPING + INYECCIÓN DE TENDENCIA
# ========================================================
def predecir_futuro_recursivo(modelo, venta_actual, venta_1_atras, n_semanas, hist_stats, historial_reciente):
    predicciones = []
    v_prev1 = venta_actual
    v_prev2 = venta_1_atras
    
    # Calcular tendencia de las últimas 4 semanas (pendiente)
    if len(historial_reciente) >= 2:
        x_trend = np.arange(len(historial_reciente))
        # polyfit devuelve [pendiente, intercepto]
        pendiente = np.polyfit(x_trend, historial_reciente, 1)[0]
        # Limitamos la tendencia semanal para que no sea una explosión
        pendiente = np.clip(pendiente, -venta_actual*0.1, venta_actual*0.1)
    else:
        pendiente = 0.0
    
    for i in range(n_semanas):
        X_input = pd.DataFrame({
            'Venta_1_Semana_Atras': [v_prev1],
            'Venta_2_Semanas_Atras': [v_prev2],
        })
        try:
            pred_base = modelo.predict(X_input)[0]
        except Exception:
            pred_base = 0.01
        
        # --- INYECCIÓN DE TENDENCIA ---
        # Si el producto venía subiendo/bajando, aplicamos un pequeño % de esa tendencia
        # Multiplicamos por (i+1) para que el efecto crezca a futuro
        ajuste_tendencia = pendiente * (i + 1) * 0.5 
        pred = float(pred_base) + ajuste_tendencia
        
        pred = 0.01 if pd.isna(pred) or pred <= 0 else float(pred)
        
        # --- CLIPPING DINÁMICO ---
        if hist_stats['max'] > 0:
            pred = min(pred, hist_stats['max'] * 1.3)
        if hist_stats['median'] > 0:
            pred = max(pred, hist_stats['median'] * 0.5)
            
        predicciones.append(pred)
        v_prev2 = v_prev1
        v_prev1 = pred
        
    return predicciones

def predecir_por_segmento(modelo, df_historial, codigo, n_semanas):
    df_prod = df_historial[df_historial['Codigo'] == codigo].sort_values('Semana')
    if df_prod.empty: return [0.0] * n_semanas, 'muerto'

    segmento = clasificar_producto(df_prod)
    if segmento == 'muerto': return [0.0] * n_semanas, segmento

    hist = df_prod['Monto de ventas'].dropna()
    if len(hist) == 0: return [0.01] * n_semanas, 'intermitente'

    if segmento == 'intermitente':
        positivos = hist[hist > 0.01]
        base = float(positivos.median()) if len(positivos) > 0 else 0.01
        return [max(0.01, base)] * n_semanas, segmento

    if segmento == 'volatil':
        base = float(hist.tail(3).mean()) if len(hist) > 0 else 0.01
        return [max(0.01, base)] * n_semanas, segmento

    # Estable
    ultimo = df_prod.iloc[-1]
    venta_actual = ultimo['Monto de ventas'] if pd.notna(ultimo['Monto de ventas']) else 0.0
    venta_1_atras = ultimo['Venta_1_Semana_Atras'] if pd.notna(ultimo['Venta_1_Semana_Atras']) else 0.0
    
    hist_stats = {'max': float(hist.max()), 'median': float(hist.median())}
    # Pasamos las últimas 4 semanas para calcular la tendencia
    historial_reciente = hist.tail(4).values 
    
    preds = predecir_futuro_recursivo(modelo, venta_actual, venta_1_atras, n_semanas, hist_stats, historial_reciente)
    return preds, segmento

@st.cache_resource(show_spinner="Entrenando LightGBM con tendencia...")
def cargar_y_entrenar(archivos_subidos):
    lista_dfs = []
    for archivo in archivos_subidos:
        try: lista_dfs.append(pd.read_excel(archivo, engine='openpyxl'))
        except Exception: pass
    if not lista_dfs: raise ValueError("No se pudo leer ningun archivo.")
    
    df = pd.concat(lista_dfs, ignore_index=True)
    df.columns = [str(c).strip().split('[')[-1].replace(']', '') for c in df.columns]
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_eval = construir_modelo_principal()
    modelo_eval.fit(X_train, y_train)
    preds_test = modelo_eval.predict(X_test)
    
    suma_abs = np.sum(np.abs(y_test - preds_test))
    suma_real = y_test.sum()
    wape = (suma_abs / suma_real) * 100 if suma_real > 0 else 0.0
    mae = mean_absolute_error(y_test, preds_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    r2 = r2_score(y_test, preds_test)
    
    modelo_final = construir_modelo_principal()
    modelo_final.fit(X, y)
        
    return df, modelo_final, wape, mae, rmse, r2

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
    
    modelo = construir_modelo_principal()
    modelo.fit(X, y)
        
    return modelo, df_held, df_train

# ========================================================
# INTERFAZ
# ========================================================
st.subheader("1. Origen de los Datos")
archivos_subidos = st.file_uploader("Sube tus archivos Excel:", type=["xlsx", "xls", "xlsm"], accept_multiple_files=True)

if archivos_subidos:
    try:
        df_global, modelo_rf, wape_val, mae_val, rmse_val, r2_val = cargar_y_entrenar(archivos_subidos)
        etiqueta = "LightGBM (L1) + Clipping + Tendencia" if LGB_AVAILABLE else "RandomForest + Clipping + Tendencia"
        
        st.success(f"Modelo entrenado: **{etiqueta}**")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("WAPE Base", f"{wape_val:.2f}%")
        col_m2.metric("MAE Base", f"${mae_val:,.2f}")
        col_m3.metric("RMSE Base", f"${rmse_val:,.2f}")
        col_m4.metric("R² Base", f"{max(0, r2_val):.2f}")

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
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077]
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
            7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 7502252480584, 7502252485077]

        st.subheader("1.5. Validación Histórica (Backtest)")
        
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            semanas_holdout = st.slider("Semanas a reservar:", 1, 12, 8, key="slider_holdout")
        with col_bt2:
            semana_inicio = st.slider("Calcular a partir de semana #:", 1, semanas_holdout, 1, key="slider_inicio")

        try:
            modelo_bt, df_held_bt, df_train_bt = entrenar_backtest(df_global, semanas_holdout, tuple(CODIGOS_PREDICCION))
            
            if not df_held_bt.empty:
                resultados_bt = []
                for codigo, group in df_held_bt.groupby('Codigo'):
                    group = group.sort_values('Semana').reset_index(drop=True)
                    if df_train_bt[df_train_bt['Codigo'] == codigo].empty: continue
                    
                    n_semanas = len(group)
                    preds, segmento = predecir_por_segmento(modelo_bt, df_train_bt, codigo, n_semanas)
                    
                    for i, (pred, actual) in enumerate(zip(preds, group['Monto de ventas'].tolist()), start=1):
                        if pd.isna(actual): continue
                        resultados_bt.append({
                            'Codigo': codigo,
                            'Descripcion': group.iloc[i-1].get('Descripcion', 'Desconocido'),
                            'Segmento': segmento,
                            'Semana_Real': int(group.iloc[i-1]['Semana']),
                            'Semana_Predecida': i,
                            'Prediccion': float(pred),
                            'Actual': float(actual),
                        })

                if resultados_bt:
                    df_bt = pd.DataFrame(resultados_bt)
                    df_bt_filt = df_bt[df_bt['Semana_Predecida'] >= semana_inicio].copy()
                    
                    y_t = df_bt_filt['Actual'].values
                    y_p = df_bt_filt['Prediccion'].values
                    wape_bt = (np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t))) * 100 if np.sum(y_t) > 0 else 0.0
                    
                    st.markdown(f"#### 📊 Métricas Backtest — Semanas +{semana_inicio} a +{semanas_holdout}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("WAPE", f"{wape_bt:.2f}%")
                    c2.metric("MAE", f"${mean_absolute_error(y_t, y_p):,.2f}")
                    c3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_t, y_p)):,.2f}")
                    c4.metric("R²", f"{max(0, r2_score(y_t, y_p)):.2f}")

                    with st.expander("🔎 Ver detalle y Top 10 Peores Predicciones"):
                        df_diag = df_bt_filt.copy()
                        df_diag['Error_Abs'] = (df_diag['Prediccion'] - df_diag['Actual']).abs()
                        st.dataframe(df_diag.nlargest(10, 'Error_Abs'), use_container_width=True)

        except Exception as e:
            st.error(f"Error en backtest: {e}")

        st.markdown("---")

        st.subheader("4. Proyeccion Masiva (Con Dinamismo de Tendencia)")
        
        CODIGOS_PERMITIDOS = list(set(CODIGOS_PREDICCION + CODIGOS_TABLA))
        df_todos = df_global.dropna(subset=['Codigo']).copy()
        df_ultimos = df_todos[df_todos['Codigo'].isin(CODIGOS_PERMITIDOS)].sort_values('Semana').groupby('Codigo').tail(1)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            sem_proy = st.slider("Semanas a predecir:", 1, 12, 4, key="slider_masiva")
        with col_m2:
            filtro = st.selectbox("Filtrar producto:", ["Todos"] + list(df_ultimos['Codigo'].unique()))

        df_proc = df_ultimos if filtro == "Todos" else df_ultimos[df_ultimos['Codigo'] == filtro]
        
        if not df_proc.empty:
            rows = []
            for _, row in df_proc.iterrows():
                cod = int(row['Codigo'])
                preds, seg = predecir_por_segmento(modelo_rf, df_global, cod, sem_proy)
                
                r = {'Codigo': cod, 'Descripcion': row.get('Descripcion', ''), 'Segmento': seg, 'Inv. Inicial': row.get('Inventario', 0)}
                inv_s = float(row.get('Inventario', 0))
                
                for i, p in enumerate(preds, start=1):
                    r[f'Inv. Sem +{i}'] = inv_s
                    r[f'Pred. Sem +{i}'] = p
                    r[f'Días Inv +{i}'] = (inv_s / p) * 30 if p > 0 else 0.0
                    inv_s = max(0.0, inv_s - p)
                rows.append(r)
                
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    except Exception as e:
        st.error(f"Error crítico: {e}")
