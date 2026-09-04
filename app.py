import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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

    # Limpieza e identificacion de columnas
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

    # Conversion a numerico
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

    modelo = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    modelo.fit(X, y)

    predicciones = modelo.predict(X)
    suma_errores = np.sum(np.abs(y - predicciones))
    suma_ventas = np.sum(y)
    
    wape = (suma_errores / suma_ventas) * 100 if suma_ventas > 0 else 0.0
    mae = mean_absolute_error(y, predicciones)

    return df, modelo, wape, mae

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
        df_global, modelo_rf, wape_val, mae_val = cargar_y_entrenar(archivos_subidos)
        
        st.success("Archivos cargados y modelo entrenado con exito!")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Error WAPE (Global)", f"{wape_val:.2f}%")
        with col_m2:
            st.metric("Error Promedio (MAE)", f"${mae_val:,.2f} MXN")
            
        st.markdown("---")
        
        # ========================================================
        # CONFIGURACIÓN DE CÓDIGOS INDEPENDIENTES
        # ========================================================
        # Códigos EXCLUSIVOS para la Simulación y Gráfica
        CODIGOS_PREDICCION = [
            75000011, 7506475125673, 7506475125680, 7506475117876, 7506475114172, 7506475113564, 7506475105606, 7501058610959, 7501058611857, 7501058613554, 7506475102834, 7501059295193, 7501059282117, 7501058615138, 7506475126090, 7501058611420, 
            7506475112888, 7506475112956, 7506475112895, 7506475112963, 7501059225411, 7501059225350, 7506475122955, 7506475103244, 7506475118675, 7501059233072, 7506475103053, 7506475103275, 7506475106801, 7506475106771, 7506475106153, 7506475106818, 
            7506475106788, 7506475106146, 7501059234321, 7501001604004, 7501059240216, 7501001604387, 7501059239845, 7613034161086, 7501058615596, 7501058627292, 7501059219106, 7501059242883, 7501059284623, 7501059284630, 7506475106078, 7501073419117, 
            7501001600198, 7501000912889, 10722776200640, 7502252484285, 7502252482694, 7502252481796, 7502252480928, 7502252480676, 7502252481727, 7502252484247, 7502252482038, 7502252482045, 7502252480263, 7502252484902, 7502252480287, 7502252486937, 
            7502252486920, 7502252486913, 7502252483929, 7502252483936, 7502252483943, 7502252484469, 7502252482236, 7502252482359, 7502252482250, 7502252487637, 7502252487675, 7502252488184, 7502252488214, 7502252488887, 7502252488894, 7502252488900, 
            7502252480584, 7502252485077, 7501058618597, 7506475128292, 7501059296367, 7506475118217, 7501059214385, 7501059296374, 7501059296381, 7501059289239, 7501059296305, 7506475113168, 7506475111713, 7501058643902, 7506475104722, 7506475124614, 
            7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 7506475104708, 7501059211209, 7506475104470, 7501058619563, 7501059278868, 7501058638076, 7501001600426, 7501059297289, 7501058615541, 7506475115513, 7501058611062, 
            7501058629517, 7506475121231, 7506475128179, 7506475100250, 7506475111690, 7506475116367, 7501000913367, 7501000911967, 7501058628831, 7506475119443, 7501059240681, 7501059287938, 7506475106917, 7501058625212, 7501058623256, 7506475106924, 
            7501058625229, 7501058623249, 7501058625205, 7501058625236, 7506475127295, 7501058623232, 7613287207197, 7613287216458, 7613287218162, 8445290925299, 7613037012637, 8585002432315, 8585002432339, 7501001604318, 7501001604325, 7501058628664, 
            7501058628503, 7501059240742, 7501001604103, 7506475120814, 7501001604110, 7501059297586, 7506475120821, 7506475118446, 7501058628282, 7506475108935, 7506475127080, 7506475114356, 7501058619228, 7506475113915, 7501059235038, 7501058619211, 
            7501058632227, 7501000913299, 7501058619235, 7501058632197, 7501058645296, 7501058645302, 7501058655172, 7506475121156, 7506475117364, 7501058654205, 7501058642127, 7501058642134, 7501058642141, 7501058642165, 7501058642172, 7501058642158, 
            7506475108829, 7506475112970, 7506475122764, 7506475113748, 7506475101981, 7506475101264, 7506475120951, 7506475120494, 7506475120500, 7506475127158, 7501058655219, 7501073411173, 7506475123617, 7506475124874, 7501058628299, 7501058624666, 
            7501059233980, 7501059289659, 7506475102148, 7501059281172, 7501058618566, 7501058617736, 3800020423547, 7891000277119, 7501059288904, 7501059223905, 7501059223912, 7501058629913, 7501058623348, 7501058654861, 7501058656254, 7501058638090, 
            7506475126694, 7501058617439, 7501058652683, 7501058651266, 7501058651273, 7501058637727, 7501059243880, 7501059243873, 7501058637734, 7501059241060, 7501058637758, 7501058637741, 7501058652652, 16000135437, 16000221284, 7501001625337, 
            7501058652690, 7506475111546, 7501001625214, 7506475111812, 7506475115544, 7506475117197, 7506475126984, 7506475126977, 7506475126960, 7506475125413, 7506475128117, 7506475128858, 7506475128285, 7501058644848, 7501058644824, 7501058644862, 
            7501000912803, 7501058620101, 7506475123051, 7501058620095, 7501000912612, 7501059224827, 7506475125253, 7506475113946, 7506475102353, 7501058610942, 7501058624635, 7506475110112, 7506475114998, 7506475125543, 7506475113700, 7501059284111, 
            7501058620002, 7501058620019, 7501000912605, 7506475123389, 7501058616548, 7506475128162, 7501058622099, 7506475123792, 7501059231962, 7501059298941, 7501058628466, 7501058628473, 7501058616470, 7501058648020, 7506475110105, 7506475115254, 
            7501058646217, 7501058618924, 7501058618931, 7501058618917, 7506475114936, 7506475111119, 7506475103855, 7501058629135, 7501058629173, 7501058629159, 7506475126731, 7501058624130, 7501058624147, 7501058624154, 7501058624161, 7506475120289, 
            7506475120265, 7506475120272, 7501058654793, 7501058642608, 7501058642592, 75013394, 75013400, 75013332, 75015374, 7506475102476, 7506475102421, 7506475102452, 75003456, 7506475102490, 7506475102469, 7501000904235, 7506475102520, 7506475102506, 
            7506475102483, 75003258, 7506475102537, 75004712, 75004705, 75004767, 75004729, 75004743, 7501000906246, 7501000906253, 7501000906284, 7501000906680, 7501058651136, 7501058651129, 7506475114073, 7506475115438, 7506475119665, 7506475119672, 
            7506475115421, 7506475122450, 7506475122436, 7506475122443, 7506475117081, 7506475122122, 7506475122139, 7506475122115, 7501058637659, 7506475122092, 7506475121996, 7891000395745, 7501000909568, 7501058626226, 7501058639493, 7506475103220, 
            7506475103213, 7501058616678, 7501058616715, 7501058614193, 7501000909612, 7501059278721, 7501059278691, 7501058626530, 7501000910526
                             ] 
        
        # Códigos EXCLUSIVOS para la Tabla de Días de Inventario (al final)
        CODIGOS_TABLA = [
            7501058611062, 7506475104722, 7501058616548, 7506475117364, 7501059224827, 7501058618917, 7501058624635, 
            7501058652690, 7501058620101, 7501073411173, 7501058628831, 7501000912803, 7506475108829, 7501058654205, 
            7506475114172, 7501001600426, 7501058642134, 17501001619043, 7501058620019, 7501058624147, 7501059231962, 
            7506475112970, 7501058620002, 7501058628473, 7506475111546, 7501058616470, 7501058652683, 7501058624130, 
            7501059243873, 7501058624666, 7501058642141, 7891000248362, 7501058619211, 7506475103244, 7501059214590, 
            7501058618931, 7501058620095, 7501059243880, 7501058628466, 16000135437, 7501059209657, 7506475122764, 
            7506475128117, 7501058618924, 7501059281165, 7501058619228, 75010592096330, 7501059233980, 7501001614027, 7501058644824
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
                
            col_sim1, col_sim2 = st.columns(2)
            
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

            # GRÁFICA
            df_hist_prod = df_pred[df_pred['Codigo'] == codigo_seleccionado].sort_values('Semana')
            
            if not df_hist_prod.empty:
                st.markdown("#### 📈 Comportamiento Historico de Ventas ($)")
                chart_data = df_hist_prod[['Semana', 'Monto de ventas']].copy()
                
                grafica = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X('Semana:O', title='Semana', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto de ventas:Q', title='Monto de Ventas ($)', axis=alt.Axis(format='$,.2f')),
                    tooltip=[alt.Tooltip('Semana:O', title='Semana'), alt.Tooltip('Monto de ventas:Q', title='Monto ($)', format='$,.2f')]
                ).properties(height=350).interactive()
                
                st.altair_chart(grafica, use_container_width=True)
            
            # BOTÓN DE SIMULACIÓN
            if st.button("🔮 Calcular Proyeccion", type="primary"):
                df_producto = df_pred[df_pred['Codigo'] == codigo_seleccionado].copy()
                df_producto = df_producto[df_producto['Semana'] == df_producto['Semana'].max()]
                
                if not df_producto.empty:
                    venta_actual = df_producto['Monto de ventas'].values[0]
                    venta_1_atras = df_producto['Venta_1_Semana_Atras'].values[0]
                    
                    X_futuro = pd.DataFrame({'Venta_1_Semana_Atras': [venta_actual], 'Venta_2_Semanas_Atras': [venta_1_atras]})
                    prediccion = modelo_rf.predict(X_futuro)[0]
                    prediccion = 0.01 if pd.isna(prediccion) or prediccion <= 0 else prediccion
                    
                    inv_actual = df_producto['Inventario'].values[0]
                    inv_actual = 0 if pd.isna(inv_actual) else inv_actual
                    
                    inv_futuro = inv_actual + monto_enviar 
                    dias_inv = (inv_futuro / prediccion) * 30
                    
                    st.markdown("### 📊 Resultados de la Prediccion")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.info(f"*Inv. en Sistema:*\n\n${inv_actual:,.2f}")
                    r2.warning(f"*Inv. Total (Simulado):*\n\n${inv_futuro:,.2f}")
                    r3.info(f"*Venta Est. (Prox Sem):*\n\n${prediccion:,.2f}")
                    r4.success(f"*Dias de Inventario:*\n\n{dias_inv:.1f} dias")
                    
                    st.progress(min(int(dias_inv), 100) / 100, text="Nivel de Cobertura (hasta 100 dias)")
                else:
                    st.warning("No hay datos suficientes para realizar la prediccion de este producto.")

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
                
            # Tabla dinámica usando "Dias Inv"
            df_pivot = df_tabla.pivot_table(
                index=cols_indice,
                columns='Semana', 
                values='Dias Inv', 
                aggfunc='sum'
            ).reset_index().fillna(0)
            
            semanas_cols = sorted([c for c in df_pivot.columns if isinstance(c, (int, float))])
            
            if len(semanas_cols) >= 2:
                col_antigua = semanas_cols[0]
                col_reciente = semanas_cols[-1]
                df_pivot['Variacion'] = df_pivot[col_reciente] - df_pivot[col_antigua]
            else:
                df_pivot['Variacion'] = 0.0
                
            mapeo_fechas = {sem: obtener_rango_fechas(sem) for sem in semanas_cols}
            df_pivot = df_pivot.rename(columns=mapeo_fechas)
            
            st.dataframe(df_pivot, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error al procesar: {e}")
