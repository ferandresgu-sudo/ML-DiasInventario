import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# ========================================================
# CONFIGURACION DE LA PAGINA
# ========================================================
st.set_page_config(page_title="Prediccion Mensual de Inventario", page_icon="📦", layout="wide")
st.title("📦 Simulador Predictivo Mensual (Nov - Dic 2026)")
st.markdown("Sube los archivos de ventas 2024, 2025 y 2026. El sistema leerá todas las hojas mensuales automáticamente.")

# ========================================================
# FUNCIONES AUXILIARES
# ========================================================
def predecir_futuro_recursivo(modelo, venta_actual, venta_1_atras, n_meses):
    predicciones = []
    v_prev1 = venta_actual
    v_prev2 = venta_1_atras
    for _ in range(n_meses):
        X_input = pd.DataFrame({'Venta_Lag_1': [v_prev1], 'Venta_Lag_2': [v_prev2]})
        pred = modelo.predict(X_input)[0]
        pred = 0.01 if pd.isna(pred) or pred <= 0 else float(pred)
        predicciones.append(pred)
        v_prev2 = v_prev1
        v_prev1 = pred
    return predicciones

@st.cache_resource(show_spinner="Procesando archivos y leyendo hojas mensuales...")
def cargar_y_entrenar(archivos_subidos):
    lista_dfs = []
    
    # 1. Leer TODAS las hojas de TODOS los archivos subidos
    for archivo in archivos_subidos:
        try:
            # sheet_name=None lee todas las hojas en un diccionario
            hojas_dict = pd.read_excel(archivo, sheet_name=None, engine='openpyxl')
            for nombre_hoja, df_hoja in hojas_dict.items():
                lista_dfs.append(df_hoja)
        except Exception as e:
            st.warning(f"Error leyendo {archivo.name}: {e}")
            
    if not lista_dfs: 
        raise ValueError("No se pudo leer ningún dato de los archivos.")
    
    df = pd.concat(lista_dfs, ignore_index=True)
    
    # 2. Limpieza de columnas
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    # Mapeo de meses para crear un índice cronológico
    meses_map = {'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
                 'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12}
    
    # Limpiar MES y AÑO
    if 'MES' in df.columns:
        df['MES_NUM'] = df['MES'].astype(str).str.strip().str.upper().map(meses_map)
    if 'AÑO' in df.columns:
        df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce')
        
    # Si por alguna razón falta Año o Mes en alguna hoja, se descarta
    df = df.dropna(subset=['MES_NUM', 'AÑO', 'CODIGO', 'VENTA'])
    
    # Crear un índice de periodo (Ej: 202401, 202402, ... 202610)
    df['Periodo'] = (df['AÑO'] * 100) + df['MES_NUM']
    
    # Convertir Código (por la notación científica de Excel)
    df['CODIGO'] = pd.to_numeric(df['CODIGO'], errors='coerce').fillna(0).astype('int64')
    df['VENTA'] = pd.to_numeric(df['VENTA'], errors='coerce')
    
    # Agrupar por si hay ventas repetidas en el mismo mes
    df_group = df.groupby(['CODIGO', 'Periodo', 'MES', 'AÑO']).agg({
        'VENTA': 'sum',
        'DESCRIPCION': 'first',
        'CATEGORIA': 'first',
        'FAMILIA': 'first',
        'TIPO': 'first'
    }).reset_index().sort_values(['CODIGO', 'Periodo'])
    
    # 3. Crear Lags Mensuales
    df_group['Venta_Lag_1'] = df_group.groupby('CODIGO')['VENTA'].shift(1)
    df_group['Venta_Lag_2'] = df_group.groupby('CODIGO')['VENTA'].shift(2)
    
    df_modelo = df_group.dropna(subset=['Venta_Lag_1', 'Venta_Lag_2', 'VENTA']).copy()
    
    if df_modelo.empty: 
        raise ValueError("No hay suficiente historial (se requieren al menos 3 meses por producto).")
        
    X = df_modelo[['Venta_Lag_1', 'Venta_Lag_2']]
    y = df_modelo['VENTA']
    
    modelo_final = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    modelo_final.fit(X, y)
    
    return df_group, modelo_final

@st.cache_resource(show_spinner=False)
def entrenar_backtest(df, n_meses_holdout):
    train_rows, held_rows = [], []
    for codigo, group in df.groupby('CODIGO'):
        group = group.sort_values('Periodo').reset_index(drop=True)
        if len(group) <= n_meses_holdout + 2:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:-n_meses_holdout])
        held_rows.append(group.iloc[-n_meses_holdout:])

    df_train = pd.concat(train_rows, ignore_index=True)
    df_held = pd.concat(held_rows, ignore_index=True) if held_rows else pd.DataFrame()

    df_train = df_train.sort_values(['CODIGO', 'Periodo'])
    df_train['Venta_Lag_1'] = df_train.groupby('CODIGO')['VENTA'].shift(1)
    df_train['Venta_Lag_2'] = df_train.groupby('CODIGO')['VENTA'].shift(2)
    
    df_train_model = df_train.dropna(subset=['Venta_Lag_1', 'Venta_Lag_2', 'VENTA'])

    X = df_train_model[['Venta_Lag_1', 'Venta_Lag_2']]
    y = df_train_model['VENTA']
    modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    modelo.fit(X, y)
    return modelo, df_held, df_train

# ========================================================
# INTERFAZ DE USUARIO
# ========================================================
st.subheader("1. Origen de los Datos (Ventas Mensuales)")
archivos_subidos = st.file_uploader("Sube los archivos de ventas (2024, 2025, 2026):", type=["xlsx", "xls", "xlsm"], accept_multiple_files=True)

if archivos_subidos:
    try:
        df_global, modelo_rf = cargar_y_entrenar(archivos_subidos)
        
        # Obtener el último periodo (Octubre 2026 = 202610)
        ultimo_periodo = int(df_global['Periodo'].max())
        
        st.success(f"✅ Archivos cargados correctamente. Último periodo detectado: {ultimo_periodo} (Octubre 2026).")
        st.markdown("---")

        # ========================================================
        # 1.5. VALIDACION HISTORICA (BACKTEST)
        # ========================================================
        st.subheader("1.5. Validación Histórica (Backtest)")
        
        codigos_para_backtest = df_global['CODIGO'].unique().tolist()
        
        # Filtros dinámicos
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            if 'TIPO' in df_global.columns:
                tipos = ["Todos"] + list(df_global['TIPO'].dropna().unique())
                sel_tipo = st.selectbox("Filtrar por Tipo:", tipos, key="bt_tipo")
                if sel_tipo != "Todos":
                    codigos_para_backtest = df_global[df_global['TIPO'] == sel_tipo]['CODIGO'].unique().tolist()
        with col_f2:
            if 'CATEGORIA' in df_global.columns:
                df_para_cat = df_global[df_global['CODIGO'].isin(codigos_para_backtest)] if 'TIPO' in df_global.columns else df_global
                cats = ["Todas"] + list(df_para_cat['CATEGORIA'].dropna().unique())
                sel_cat = st.selectbox("Filtrar por Categoría:", cats, key="bt_cat")
                if sel_cat != "Todas":
                    codigos_para_backtest = df_para_cat[df_para_cat['CATEGORIA'] == sel_cat]['CODIGO'].unique().tolist()

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1: 
            meses_holdout = st.slider("Meses a reservar para validación:", 1, 6, 2, key="slider_holdout")
        with col_bt2: 
            mes_inicio_metrica = st.slider("Calcular a partir del mes predecido #:", 1, meses_holdout, 1, key="slider_mes_inicio")

        if not codigos_para_backtest:
            st.warning("No hay productos que coincidan con los filtros.")
        else:
            try:
                modelo_bt, df_held_bt, df_train_bt = entrenar_backtest(df_global[df_global['CODIGO'].isin(codigos_para_backtest)], meses_holdout)

                if not df_held_bt.empty:
                    resultados_bt = []
                    for codigo, group in df_held_bt.groupby('CODIGO'):
                        group = group.sort_values('Periodo').reset_index(drop=True)
                        train_group = df_train_bt[df_train_bt['CODIGO'] == codigo].sort_values('Periodo')
                        if train_group.empty or len(train_group) < 3: continue

                        last_train = train_group.iloc[-1]
                        venta_actual = last_train['VENTA'] if pd.notna(last_train['VENTA']) else 0.0
                        venta_1_atras = last_train['Venta_Lag_1'] if pd.notna(last_train['Venta_Lag_1']) else 0.0

                        preds = predecir_futuro_recursivo(modelo_bt, venta_actual, venta_1_atras, len(group))

                        for i, (pred, actual) in enumerate(zip(preds, group['VENTA'].tolist()), start=1):
                            if pd.isna(actual): continue
                            resultados_bt.append({
                                'CODIGO': codigo,
                                'DESCRIPCION': group.iloc[i-1].get('DESCRIPCION', 'Desconocido'),
                                'Periodo_Real': int(group.iloc[i-1]['Periodo']),
                                'Mes_Predecido': i,
                                'Prediccion': float(pred),
                                'Actual': float(actual),
                                'CATEGORIA': group.iloc[i-1].get('CATEGORIA', ''),
                                'TIPO': group.iloc[i-1].get('TIPO', '')
                            })

                    if resultados_bt:
                        df_bt = pd.DataFrame(resultados_bt)
                        df_bt['Error_Abs'] = (df_bt['Prediccion'] - df_bt['Actual']).abs()
                        df_bt['Variacion_%'] = np.where(df_bt['Actual'] > 0, ((df_bt['Prediccion'] - df_bt['Actual']) / df_bt['Actual']) * 100, 0.0)
                        df_bt_filt = df_bt[df_bt['Mes_Predecido'] >= mes_inicio_metrica].copy()

                        if not df_bt_filt.empty:
                            y_true = df_bt_filt['Actual'].values
                            y_pred = df_bt_filt['Prediccion'].values
                            wape_bt = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100 if np.sum(y_true) > 0 else 0.0
                            
                            st.markdown(f"#### 📊 Metricas de Backtest (Mensual)")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("WAPE", f"{wape_bt:.2f}%")
                            c2.metric("MAE", f"${mean_absolute_error(y_true, y_pred):,.2f}")
                            c3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
                            c4.metric("R²", f"{max(0, r2_score(y_true, y_pred)):.2f}")

                            with st.expander("🔎 Ver detalle y Top 10 Peores Predicciones"):
                                st.markdown("##### 🚨 Top 10 Peores Predicciones")
                                top_10 = df_bt_filt.nlargest(10, 'Error_Abs')
                                st.dataframe(top_10[['CODIGO', 'DESCRIPCION', 'CATEGORIA', 'Periodo_Real', 'Prediccion', 'Actual', 'Error_Abs', 'Variacion_%']].style.format({'Prediccion': '${:,.2f}', 'Actual': '${:,.2f}', 'Error_Abs': '${:,.2f}', 'Variacion_%': '{:+.1f}%'}), use_container_width=True)
            except Exception as e:
                st.error(f"Error en backtest: {e}")

        st.markdown("---")

        # ========================================================
        # 2. PREDICCIÓN INDIVIDUAL (NOV Y DIC 2026)
        # ========================================================
        st.subheader("2. Simulacion de Producto (Proyección Nov y Dic 2026)")
        
        df_ultimos = df_global.sort_values('Periodo').groupby('CODIGO').tail(1)
        opciones_productos = {int(row['CODIGO']): f"{int(row['CODIGO'])} - {row.get('DESCRIPCION', 'Desconocido')}" for _, row in df_ultimos.iterrows()}

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            codigo_seleccionado = st.selectbox("Selecciona el Producto:", options=list(opciones_productos.keys()), format_func=lambda x: opciones_productos[x])
        
        df_hist_prod = df_global[df_global['CODIGO'] == codigo_seleccionado].sort_values('Periodo')

        if not df_hist_prod.empty:
            ultimo_registro = df_hist_prod.iloc[-1]
            venta_actual = ultimo_registro['VENTA']
            venta_1_atras = ultimo_registro['Venta_Lag_1']

            # Forzamos la predicción de 2 meses exactos (Nov y Dic 2026)
            preds_futuras = predecir_futuro_recursivo(modelo_rf, venta_actual, venta_1_atras, n_meses=2)
            
            venta_nov = preds_futuras[0]
            venta_dic = preds_futuras[1]
            
            st.markdown(f"**Predicción Noviembre 2026:** ${venta_nov:,.2f} | **Predicción Diciembre 2026:** ${venta_dic:,.2f}")

            # Gráfica
            df_real = df_hist_prod[['Periodo', 'VENTA']].copy()
            df_real['Tipo'] = 'Historico Real'
            
            filas_futuras = [
                {'Periodo': 202611, 'VENTA': venta_nov, 'Tipo': 'Proyectado (Nov)'},
                {'Periodo': 202612, 'VENTA': venta_dic, 'Tipo': 'Proyectado (Dic)'}
            ]
            
            df_grafica_comb = pd.concat([df_real, pd.DataFrame(filas_futuras)], ignore_index=True)
            
            grafica = alt.Chart(df_grafica_comb).mark_line(point=True).encode(
                x=alt.X('Periodo:O', title='Periodo (AñoMes)', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('VENTA:Q', title='Venta ($)', axis=alt.Axis(format='$,.2f')),
                color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Historico Real', 'Proyectado (Nov)', 'Proyectado (Dic)'], range=['#1f77b4', '#ff7f0e', '#d62728'])),
                tooltip=['Periodo:O', alt.Tooltip('VENTA:Q', format='$,.2f'), 'Tipo:N']
            ).properties(height=350).interactive()
            st.altair_chart(grafica, use_container_width=True)

        st.markdown("---")

        # ========================================================
        # 4. TABLA MASIVA (NOV Y DIC 2026)
        # ========================================================
        st.subheader("4. Proyeccion Masiva (Noviembre y Diciembre 2026)")
        
        df_ultimos_masivo = df_global.sort_values('Periodo').groupby('CODIGO').tail(1)
        
        st.markdown("##### 🎯 Filtros de Segmentación")
        col_m1, col_m2, col_m3 = st.columns(3)
        df_filtrado = df_ultimos_masivo.copy()

        with col_m1:
            if 'TIPO' in df_filtrado.columns:
                opciones_tipo = ["Todos"] + list(df_filtrado['TIPO'].dropna().unique())
                filtro_tipo = st.selectbox("Filtrar por Tipo:", opciones_tipo, key="filtro_tipo_masivo")
                if filtro_tipo != "Todos":
                    df_filtrado = df_filtrado[df_filtrado['TIPO'] == filtro_tipo]

        with col_m2:
            if 'CATEGORIA' in df_filtrado.columns:
                opciones_cat = ["Todas"] + list(df_filtrado['CATEGORIA'].dropna().unique())
                filtro_cat = st.selectbox("Filtrar por Categoría:", opciones_cat, key="filtro_cat_masivo")
                if filtro_cat != "Todas":
                    df_filtrado = df_filtrado[df_filtrado['CATEGORIA'] == filtro_cat]

        with col_m3:
            opciones_todos = {int(row['CODIGO']): f"{int(row['CODIGO'])} - {row.get('DESCRIPCION', 'Desconocido')}" for _, row in df_filtrado.iterrows()}
            opciones_filtro_tabla = ["Todos"] + list(opciones_todos.keys())
            filtro_seleccionado = st.selectbox(
                "Filtrar por producto específico:",
                options=opciones_filtro_tabla,
                format_func=lambda x: "Todos los productos" if x == "Todos" else opciones_todos.get(x, str(x)),
                key="filtro_seccion_4"
            )

        if filtro_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['CODIGO'] == filtro_seleccionado]

        with st.spinner("Calculando proyecciones de Nov y Dic..."):
            if df_filtrado.empty:
                st.info("No hay productos que coincidan con los filtros.")
            else:
                editor_rows = []
                for _, row in df_filtrado.iterrows():
                    codigo = int(row['CODIGO'])
                    venta_act = row['VENTA'] if pd.notna(row['VENTA']) else 0.0
                    venta_lag = row['Venta_Lag_1'] if pd.notna(row['Venta_Lag_1']) else 0.0
                    
                    preds = predecir_futuro_recursivo(modelo_rf, venta_act, venta_lag, n_meses=2)
                    
                    # Solo mostramos Nov y Dic
                    editor_rows.append({
                        'CODIGO': codigo,
                        'DESCRIPCION': row.get('DESCRIPCION', 'Desconocido'),
                        'CATEGORIA': row.get('CATEGORIA', ''),
                        'Venta Oct 2026': venta_act,
                        'Pred Nov 2026': preds[0],
                        'Pred Dic 2026': preds[1],
                    })

                editor_df = pd.DataFrame(editor_rows)
                
                st.dataframe(
                    editor_df.style.format({
                        'Venta Oct 2026': '${:,.2f}',
                        'Pred Nov 2026': '${:,.2f}',
                        'Pred Dic 2026': '${:,.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:
        st.error(f"Error crítico al procesar los archivos: {e}")
