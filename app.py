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
    predicciones = []
    v_prev1 = venta_actual
    v_prev2 = venta_1_atras
    
    for _ in range(n_semanas):
        X_input = pd.DataFrame({
            'Venta_1_Semana_Atras': [v_prev1], 
            'Venta_2_Semanas_Atras': [v_prev2]
        })
        pred = modelo.predict(X_input)[0]
        pred = 0.01 if pd.isna(pred) or pred <= 0 else float(pred)
        predicciones.append(pred)
        
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
            7501059234321]
        CODIGOS_TABLA = [
            7506475104722, 7506475124614, 7506475128346, 7501058631961, 7506475112093, 7506475113861, 7506475120135, 
            7506475104708]
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
        # 4. TABLA GENERAL DE PREDICCIONES (CÓDIGOS PERMITIDOS)
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
        
        with st.spinner("Calculando predicciones de venta... esto puede tomar unos segundos."):
            df_a_procesar = (
                df_ultimos_masivo 
                if filtro_seleccionado == "Todos" 
                else df_ultimos_masivo[df_ultimos_masivo['Codigo'] == filtro_seleccionado]
            )
            
            if df_a_procesar.empty:
                st.info("No hay informacion de productos para predecir.")
            else:
                # --- 1) Calcular predicciones de venta por producto (una sola vez) ---
                predicciones_por_codigo = {}
                base_por_codigo = {}
                
                for _, row in df_a_procesar.iterrows():
                    codigo = int(row['Codigo'])
                    venta_actual = row['Monto de ventas'] if pd.notna(row['Monto de ventas']) else 0.0
                    venta_1_atras = row['Venta_1_Semana_Atras'] if pd.notna(row['Venta_1_Semana_Atras']) else 0.0
                    inv_actual = row['Inventario'] if pd.notna(row['Inventario']) else 0.0
                    
                    preds = predecir_futuro_recursivo(
                        modelo_rf, venta_actual, venta_1_atras, semanas_proyectar_masivo
                    )
                    predicciones_por_codigo[codigo] = preds
                    base_por_codigo[codigo] = {
                        'Descripcion': row.get('Descripcion', 'Desconocido'),
                        'Inv_Actual': inv_actual
                    }
                
                # --- 2) Construir tabla editable (valores sugeridos auto-calculados) ---
                editor_rows = []
                for codigo, preds in predicciones_por_codigo.items():
                    info = base_por_codigo[codigo]
                    row_data = {
                        'Codigo': codigo,
                        'Descripcion': info['Descripcion'],
                        'Inv. Actual (Inicial)': float(info['Inv_Actual']),
                    }
                    # Sugerencia: restar ventas semana a semana
                    inv_sugerido = float(info['Inv_Actual'])
                    for sem_idx, p_val in enumerate(preds, start=1):
                        row_data[f'Inv. Sem +{sem_idx} ($)'] = inv_sugerido
                        inv_sugerido = max(0.0, inv_sugerido - float(p_val))
                    editor_rows.append(row_data)
                
                editor_df = pd.DataFrame(editor_rows)
                
                # --- 3) Configuracion del editor: solo columnas "Inv. Sem +N" son editables ---
                column_config = {
                    'Codigo': st.column_config.NumberColumn(
                        'Codigo', disabled=True, format='%d'
                    ),
                    'Descripcion': st.column_config.TextColumn(
                        'Descripcion', disabled=True
                    ),
                    'Inv. Actual (Inicial)': st.column_config.NumberColumn(
                        'Inv. Actual (Inicial)', disabled=True, format='$%.2f'
                    ),
                }
                for sem_idx in range(1, semanas_proyectar_masivo + 1):
                    column_config[f'Inv. Sem +{sem_idx} ($)'] = st.column_config.NumberColumn(
                        f'Inv. Sem +{sem_idx} ($)',
                        min_value=0.0,
                        step=100.0,
                        format='$%.2f',
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
                
                # --- 4) Construir tabla final con dias recalculados usando valores editados ---
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
                
                # --- 5) Mostrar tabla final con formato ---
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
