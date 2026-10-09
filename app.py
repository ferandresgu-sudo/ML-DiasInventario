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
st.set_page_config(
    page_title="Prediccion de Inventario", page_icon="📦", layout="wide"
)

st.title("📦 Simulador Predictivo de Dias de Inventario")
st.markdown("Sube todos los archivos de tu carpeta historica para entrenar el modelo y analizar el desempeño por producto.")

# ========================================================
# FUNCION PARA CONVERTIR NUMERO DE SEMANA A RANGO DE FECHAS
# ========================================================
def obtener_rango_fechas(semana_num):
    base_date = datetime.date(2026, 6, 15)
    semana_base = 25
    diferencia_semanas = int(semana_num) - semana_base
    
    start_date = base_date + datetime.timedelta(weeks=diferencia_semanas)
    end_date = start_date + datetime.timedelta(days=6)
    
    meses = {1: 'ene', 2: 'feb', 3: 'mar', 4: 'abr', 5: 'may', 6: 'jun',
             7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'}
             
    return f"{start_date.day:02d} {meses[start_date.month]} - {end_date.day:02d} {meses[end_date.month]}"

# ========================================================
# FUNCION PARA PREDICCION RECURSIVA MULTI-SEMANA
# ========================================================
def predecir_futuro_recursivo(modelo, venta_actual, venta_1_atras, n_semanas):
    """
    Predice n_semanas hacia el futuro alimentando cada prediccion
    como entrada para la siguiente semana.
    """
    predicciones = []
    v_prev1 = venta_actual     # Venta t-1 para el paso futuro 1
    v_prev2 = venta_1_atras    # Venta t-2 para el paso futuro 1
    
    for _ in range(n_semanas):
        X_input = pd.DataFrame({
            'Venta_1_Semana_Atras': [v_prev1], 
            'Venta_2_Semanas_Atras': [v_prev2]
        })
        pred = modelo.predict(X_input)[0]
        pred = 0.01 if pd.isna(pred) or pred <= 0 else float(pred)
        predicciones.append(pred)
        
        # Actualizamos ventanas para el siguiente paso
        v_prev2 = v_prev1
        v_prev1 = pred
        
    return predicciones

# ========================================================
# FUNCION CACHEADA PARA PROCESAMIENTO Y ENTRENAMIENTO
# ========================================================
@st.cache_resource(show_spinner="Procesando archivos y entrenando modelo, esto puede tomar unos segundos.")
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

    df = df.sort_values(by=['Codigo', 'Semana'])
    df['Venta_1_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_2_Semanas_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(2)

    df_modelo = df.dropna(subset=['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Monto de ventas']).copy()

    if df_modelo.empty:
        raise ValueError("No hay historial suficiente (se requieren al menos 3 semanas consecutivas).")

    X = df_modelo[['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras']]
    y = df_modelo['Monto de ventas']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_evaluador = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    modelo_evaluador.fit(X_train, y_train)

    predicciones_test = modelo_evaluador.predict(X_test)
    
    suma_errores = np.sum(np.abs(y_test - predicciones_test))
    suma_ventas = np.sum(y_test)
    
    wape = (suma_errores / suma_ventas) * 100 if suma_ventas > 0 else 0.0
    mae = mean_absolute_error(y_test, predicciones_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones_test))
    r2 = r2_score(y_test, predicciones_test)

    modelo_final = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    modelo_final.fit(X, y)

    return df, modelo_final, wape, mae, rmse, r2

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
        df_global, modelo_rf, wape_val, mae_val, rmse_val, r2_val = cargar_y_entrenar(archivos_subidos)
        
        st.success("Archivos cargados y modelo entrenado con exito!")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="Error WAPE (Global)", value=f"{wape_val:.2f}%", help="Porcentaje de error ponderado.")
        with col_m2:
            st.metric(label="Error Promedio (MAE)", value=f"${mae_val:,.2f}", help="Promedio absoluto del error.")
        with col_m3:
            st.metric(label="Penalización (RMSE)", value=f"${rmse_val:,.2f}", help="Castiga errores grandes.")
        with col_m4:
            r2_display = f"{max(0, r2_val):.2f}"
            st.metric(label="Explicabilidad (R²)", value=r2_display, help="De 0 a 1. Capacidad real del modelo.")
            
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
        # 2. PREDICCIÓN Y GRÁFICA
        # ========================================================
        st.subheader("2. Simulacion y Tendencia de Producto")
        
        df_pred = df_global.dropna(subset=['Codigo']).copy()
        df_pred = df_pred[df_pred['Codigo'].isin(CODIGOS_PREDICCION)]
        
        if df_pred.empty:
            st.warning("No se encontraron los codigos de CODIGOS_PREDICCION en los archivos subidos.")
        else:
            df_ultimos = df_pred.sort_values('Semana').groupby('Codigo').tail(1)
            opciones_productos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos.iterrows()}
                
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
                    min_value=0.0, 
                    value=0.0, 
                    step=1000.0
                )
                
            with col_sim3:
                semanas_proyectar = st.slider(
                    "Semanas a proyectar al futuro (Gráfica):",
                    min_value=1,
                    max_value=8,
                    value=4,
                    key="slider_grafica"
                )

            df_hist_prod = df_pred[df_pred['Codigo'] == codigo_seleccionado].sort_values('Semana')
            
            if not df_hist_prod.empty:
                semana_max = df_hist_prod['Semana'].max()
                df_ultimo_registro = df_hist_prod[df_hist_prod['Semana'] == semana_max].iloc[0]
                
                venta_actual = df_ultimo_registro['Monto de ventas']
                venta_1_atras = df_ultimo_registro['Venta_1_Semana_Atras']
                
                preds_futuras = predecir_futuro_recursivo(modelo_rf, venta_actual, venta_1_atras, semanas_proyectar)
                
                df_real = df_hist_prod[['Semana', 'Monto de ventas']].copy()
                df_real['Tipo'] = 'Historico Real'
                
                filas_futuras = []
                for idx, pred_val in enumerate(preds_futuras, start=1):
                    sem_futura = semana_max + idx
                    filas_futuras.append({
                        'Semana': sem_futura,
                        'Monto de ventas': pred_val,
                        'Tipo': f'Proyeccion ({idx}a sem)'
                    })
                
                filas_futuras.insert(0, {
                    'Semana': semana_max,
                    'Monto de ventas': venta_actual,
                    'Tipo': 'Proyeccion'
                })
                
                df_futuro = pd.DataFrame(filas_futuras)
                df_futuro['Tipo'] = 'Proyectado'
                
                df_grafica_comb = pd.concat([df_real, df_futuro], ignore_index=True)

                st.markdown(f"#### 📈 Tendencia Historica y Proyeccion Autorregresiva ({semanas_proyectar} Semanas)")
                
                grafica = alt.Chart(df_grafica_comb).mark_line(point=True).encode(
                    x=alt.X('Semana:O', title='Semana', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto de ventas:Q', title='Monto de Ventas ($)', axis=alt.Axis(format='$,.2f')),
                    color=alt.Color('Tipo:N', title='Origen', scale=alt.Scale(domain=['Historico Real', 'Proyectado'], range=['#1f77b4', '#ff7f0e'])),
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
        # 3. REPORTE SEMANAL DE DÍAS DE INVENTARIO
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
                index=cols_indice,
                columns='Semana', 
                values='Dias Inv', 
                aggfunc='sum'
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
        # 4. TABLA GENERAL DE PREDICCIONES (TODOS LOS PRODUCTOS)
        # ========================================================
        st.subheader("4. Proyeccion Masiva de Inventario (Todos los Productos)")
        st.markdown("Tabla con el cálculo de ventas proyectadas y cobertura de inventario para todo el catálogo general.")
        
        # PREPARAR DATA COMPLETA (NO SOLO LOS CODIGOS FILTRADOS DE LA SECCION 2)
        df_todos = df_global.dropna(subset=['Codigo']).copy()
        df_ultimos_masivo = df_todos.sort_values('Semana').groupby('Codigo').tail(1)
        
        opciones_todos = {int(row['Codigo']): f"{int(row['Codigo'])} - {row.get('Descripcion', 'Desconocido')}" for _, row in df_ultimos_masivo.iterrows()}
        opciones_filtro_tabla = ["Todos"] + list(opciones_todos.keys())
        
        col_masiva1, col_masiva2 = st.columns(2)
        
        with col_masiva1:
            semanas_proyectar_masivo = st.slider(
                "Semanas a predecir en la tabla masiva:",
                min_value=1,
                max_value=12,
                value=4,
                key="slider_masiva"
            )
            
        with col_masiva2:
            filtro_seleccionado = st.selectbox(
                "Filtra la tabla por un producto (o selecciona 'Todos'):",
                options=opciones_filtro_tabla,
                format_func=lambda x: "Todos los productos" if x == "Todos" else opciones_todos[x],
                key="filtro_seccion_4"
            )
        
        # SPINNER MIENTRAS CALCULA TODOS LOS PRODUCTOS
        with st.spinner("Calculando proyecciones... esto puede tomar unos segundos."):
            resultados_masivos = []
            
            # Si se filtró, solo procesamos ese para mayor velocidad
            df_a_procesar = df_ultimos_masivo if filtro_seleccionado == "Todos" else df_ultimos_masivo[df_ultimos_masivo['Codigo'] == filtro_seleccionado]
            
            for _, row in df_a_procesar.iterrows():
                codigo = int(row['Codigo'])
                descripcion = row.get('Descripcion', 'Desconocido')
                
                # Manejar valores vacíos con 0
                venta_actual = row['Monto de ventas'] if pd.notna(row['Monto de ventas']) else 0.0
                venta_1_atras = row['Venta_1_Semana_Atras'] if pd.notna(row['Venta_1_Semana_Atras']) else 0.0
                inv_actual = row['Inventario'] if pd.notna(row['Inventario']) else 0.0
                
                # Predicciones encadenadas exclusivas para esta tabla
                preds_futuras = predecir_futuro_recursivo(modelo_rf, venta_actual, venta_1_atras, semanas_proyectar_masivo)
                venta_est_prox = preds_futuras[0]               
                venta_prom_horizonte = np.mean(preds_futuras)     
                
                dias_inv = (inv_actual / venta_prom_horizonte) * 30 if venta_prom_horizonte > 0 else 0
                
                res_dict = {
                    'Codigo': codigo,
                    'Descripcion': descripcion,
                    'Inv. Actual ($)': inv_actual,
                    'Venta Est. Sem +1 ($)': venta_est_prox,
                    f'Venta Prom. ({semanas_proyectar_masivo} sem) ($)': venta_prom_horizonte,
                    'Dias de Inventario (Proyectados)': dias_inv
                }
                
                # Agregar dinámicamente columnas según el slider
                for sem_idx, p_val in enumerate(preds_futuras, start=1):
                    res_dict[f'Pred. Sem +{sem_idx} ($)'] = p_val
                    
                resultados_masivos.append(res_dict)
                
            df_resultados = pd.DataFrame(resultados_masivos)
            
            if not df_resultados.empty:
                # Formato dinamico de moneda/decimales
                format_dict = {
                    'Inv. Actual ($)': '${:,.2f}',
                    'Venta Est. Sem +1 ($)': '${:,.2f}',
                    f'Venta Prom. ({semanas_proyectar_masivo} sem) ($)': '${:,.2f}',
                    'Dias de Inventario (Proyectados)': '{:.1f}'
                }
                for sem_idx in range(1, semanas_proyectar_masivo + 1):
                    format_dict[f'Pred. Sem +{sem_idx} ($)'] = '${:,.2f}'
                
                st.dataframe(df_resultados.style.format(format_dict), use_container_width=True)
            else:
                st.info("No hay información de productos para predecir.")
                
    except Exception as e:
        st.error(f"Error al procesar: {e}")
        
        st.dataframe(df_resultados.style.format(format_dict), use_container_width=True)
                
    except Exception as e:
        st.error(f"Error al procesar: {e}")
