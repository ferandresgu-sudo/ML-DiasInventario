import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import altair as alt

# ========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================================
st.set_page_config(
    page_title="Prediccion de Inventario", page_icon="📦", layout="wide"
)

st.title("📦 Prediccion de Dias de Inventario")
st.markdown("Sube todos los archivos de tu carpeta historica para entrenar el modelo y analizar el desempeño por producto.")

# ========================================================
# FUNCIÓN CACHEADA PARA PROCESAMIENTO Y ENTRENAMIENTO
# ========================================================
@st.cache_resource(show_spinner="Procesando archivos y entrenando modelo, esto puede tomar unos segundos.")
def cargar_y_entrenar(archivos_subidos):
    lista_dfs = []
    
    # 1. Consolidacion
    for archivo in archivos_subidos:
        try:
            df_temp = pd.read_excel(archivo, engine='openpyxl')
            lista_dfs.append(df_temp)
        except Exception:
            pass # Omite archivos dañados o no compatibles
            
    if not lista_dfs:
        raise ValueError("No se pudo leer ningun archivo correctamente.")
        
    df = pd.concat(lista_dfs, ignore_index=True)
    
    # 2. Limpieza Extrema
    nuevas_columnas = [str(col).strip().split('[')[-1].replace(']', '') for col in df.columns]
    df.columns = nuevas_columnas

    for col in df.columns:
        col_lower = col.lower()
        if 'descrip' in col_lower: df.rename(columns={col: 'Descripcion'}, inplace=True)
        elif 'inventario' in col_lower and 'monto' not in col_lower and 'total' not in col_lower: df.rename(columns={col: 'Inventario'}, inplace=True) 
        elif 'venta' in col_lower: df.rename(columns={col: 'Monto de ventas'}, inplace=True)
        elif 'codigo' in col_lower or 'código' in col_lower: df.rename(columns={col: 'Codigo'}, inplace=True)
        elif 'semana' in col_lower: df.rename(columns={col: 'Semana'}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Ingenieria de Caracteristicas
    df['Codigo'] = pd.to_numeric(df['Codigo'], errors='coerce')
    df['Semana'] = pd.to_numeric(df['Semana'], errors='coerce')
    df['Monto de ventas'] = pd.to_numeric(df['Monto de ventas'], errors='coerce')
    df['Inventario'] = pd.to_numeric(df['Inventario'], errors='coerce')

    df = df.sort_values(by=['Codigo', 'Semana'])
    df['Venta_1_Semana_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(1)
    df['Venta_2_Semanas_Atras'] = df.groupby('Codigo')['Monto de ventas'].shift(2)

    df_modelo = df.dropna(subset=['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras', 'Monto de ventas']).copy()

    if df_modelo.empty:
        raise ValueError("No hay historial suficiente (se requieren al menos 3 semanas consecutivas por producto).")

    # 4. Entrenamiento
    X = df_modelo[['Venta_1_Semana_Atras', 'Venta_2_Semanas_Atras']]
    y = df_modelo['Monto de ventas']

    modelo = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    modelo.fit(X, y)

    # Evaluacion
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
        
        st.subheader("2. Simulacion y Tendencia de Producto")
        
        df_validos = df_global.dropna(subset=['Codigo']).copy()
        df_ultimos = df_validos.sort_values('Semana').groupby('Codigo').tail(1)
        
        opciones_productos = {}
        for _, row in df_ultimos.iterrows():
            cod = int(row['Codigo'])
            desc = row['Descripcion'] if 'Descripcion' in df_ultimos.columns else 'Desconocido'
            opciones_productos[cod] = f"{cod} - {desc}"
            
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
                value=10000.0, 
                step=1000.0
            )

        # ----------------------------------------------------
        # GRÁFICA HISTÓRICA DE VENTAS SEMANALES CON MARCADORES
        # ----------------------------------------------------
        df_hist_prod = df_global[df_global['Codigo'] == codigo_seleccionado].sort_values('Semana')
        
        if not df_hist_prod.empty:
            st.markdown("#### 📈 Comportamiento Historico de Ventas ($)")
            
            chart_data = df_hist_prod[['Semana', 'Monto de ventas']].copy()
            
            # Grafica Altair con semanas horizontales y valores con comas
            grafica = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X(
                    'Semana:O', 
                    title='Semana', 
                    axis=alt.Axis(labelAngle=0)  # <-- Fuerza las semanas en horizontal
                ),
                y=alt.Y(
                    'Monto de ventas:Q', 
                    title='Monto de Ventas ($)', 
                    axis=alt.Axis(format='$,.2f')  # <-- Agrega comas y formato de moneda
                ),
                tooltip=[
                    alt.Tooltip('Semana:O', title='Semana'),
                    alt.Tooltip('Monto de ventas:Q', title='Monto ($)', format='$,.2f')
                ]
            ).properties(
                height=350
            ).interactive()
            
            st.altair_chart(grafica, use_container_width=True)
        
        # ----------------------------------------------------
        # BOTÓN DE SIMULACIÓN Y RESULTADOS
        # ----------------------------------------------------
        if st.button("🔮 Calcular Proyeccion", type="primary"):
            df_producto = df_global[df_global['Codigo'] == codigo_seleccionado].copy()
            df_producto = df_producto[df_producto['Semana'] == df_producto['Semana'].max()]
            
            if not df_producto.empty:
                venta_actual = df_producto['Monto de ventas'].values[0]
                venta_1_atras = df_producto['Venta_1_Semana_Atras'].values[0]
                
                X_futuro = pd.DataFrame({
                    'Venta_1_Semana_Atras': [venta_actual],
                    'Venta_2_Semanas_Atras': [venta_1_atras]
                })
                
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
                
    except Exception as e:
        st.error(f"Error al procesar: {e}")
