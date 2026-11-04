import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title='Prediccion por categoria', page_icon='📦', layout='wide')
st.title('📦 Prediccion Noviembre-Diciembre 2026 por CATEGORIA')

MESES={'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
LAGS=[1,2,3,4,6,12]
VENTANAS=[3,6,12]
FEATURES=([f'lag{x}' for x in LAGS]+[f'mean{x}' for x in VENTANAS]+[f'std{x}' for x in VENTANAS]+['sin_mes','cos_mes'])

# =========================================================
# CORRECCION PRINCIPAL DEL KEYERROR 'CODIGO'
# =========================================================
def limpiar_nombre_columna(nombre):
    """Normaliza BOM, acentos, saltos, espacios y caracteres invisibles."""
    texto=str(nombre).replace('\ufeff','').replace('\xa0',' ').strip().upper()
    texto=unicodedata.normalize('NFKD',texto).encode('ascii','ignore').decode('ascii')
    texto=re.sub(r'[\r\n\t]+',' ',texto)
    texto=re.sub(r'[^A-Z0-9]+','_',texto).strip('_')
    return texto


def normalizar_columnas(df):
    df=df.copy()
    df.columns=[limpiar_nombre_columna(c) for c in df.columns]

    # Alias comunes encontrados en archivos Excel.
    alias={
        'COD_PRODUCTO':'CODIGO','CODIGO_PRODUCTO':'CODIGO','CODIGOArtículo'.upper():'CODIGO',
        'COD_ARTICULO':'CODIGO','ID_PRODUCTO':'CODIGO','SKU':'CODIGO','MATERIAL':'CODIGO',
        'VENTAS':'VENTA','VENTA_NETA':'VENTA','MONTO_VENTA':'VENTA','IMPORTE':'VENTA',
        'ANO':'AÑO','YEAR':'AÑO','ANIO':'AÑO',
        'MONTH':'MES','NOMBRE_MES':'MES',
        'CATEGORY':'CATEGORIA','CAT':'CATEGORIA',
        'DESCRIPCION_PRODUCTO':'DESCRIPCION','PRODUCTO':'DESCRIPCION'
    }
    # Las claves también deben pasar por la misma limpieza.
    alias={limpiar_nombre_columna(k):v for k,v in alias.items()}
    df=df.rename(columns={c:alias.get(c,c) for c in df.columns})

    # Si una columna quedó repetida después de renombrar, conserva la primera no nula.
    if df.columns.duplicated().any():
        unicas={}
        for c in dict.fromkeys(df.columns):
            mismas=df.loc[:,df.columns==c]
            unicas[c]=mismas.bfill(axis=1).iloc[:,0]
        df=pd.DataFrame(unicas,index=df.index)
    return df


def buscar_fila_encabezado(raw, max_filas=15):
    """Detecta archivos donde el encabezado no está en la primera fila."""
    requeridas={'CODIGO','VENTA'}
    for i in range(min(max_filas,len(raw))):
        valores={limpiar_nombre_columna(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        valores_alias=set(normalizar_columnas(pd.DataFrame(columns=list(valores))).columns)
        if requeridas.issubset(valores_alias) and ('MES' in valores_alias or 'AÑO' in valores_alias or 'ANO' in valores_alias):
            return i
    return 0


def leer_hoja_segura(archivo,nombre_hoja):
    # Primero lee sin asumir encabezado para localizarlo correctamente.
    raw=pd.read_excel(archivo,sheet_name=nombre_hoja,header=None,engine='openpyxl')
    if raw.empty:
        return pd.DataFrame()
    fila=buscar_fila_encabezado(raw)
    encabezados=raw.iloc[fila].tolist()
    df=raw.iloc[fila+1:].copy()
    df.columns=encabezados
    df=df.dropna(how='all').reset_index(drop=True)
    return normalizar_columnas(df)


@st.cache_data(show_spinner='Leyendo y validando archivos...')
def cargar(archivos):
    partes=[]
    problemas=[]
    for archivo in archivos:
        try:
            excel=pd.ExcelFile(archivo,engine='openpyxl')
            for hoja in excel.sheet_names:
                try:
                    h=leer_hoja_segura(archivo,hoja)
                    if h.empty:
                        continue
                    faltantes=[c for c in ['CODIGO','VENTA','MES','AÑO','CATEGORIA'] if c not in h.columns]
                    if faltantes:
                        problemas.append(f'{archivo.name} / {hoja}: faltan {faltantes}; columnas={list(h.columns)}')
                        continue
                    h['ARCHIVO_ORIGEN']=archivo.name
                    h['HOJA_ORIGEN']=hoja
                    partes.append(h)
                except Exception as e:
                    problemas.append(f'{archivo.name} / {hoja}: {e}')
        except Exception as e:
            problemas.append(f'{archivo.name}: no se pudo abrir: {e}')

    if not partes:
        detalle='\n'.join(problemas[:15])
        raise ValueError('Ninguna hoja contiene CODIGO, VENTA, MES, AÑO y CATEGORIA después de normalizar.\n'+detalle)

    df=pd.concat(partes,ignore_index=True,sort=False)
    # Validación antes de cualquier df['CODIGO'].
    requeridas=['CODIGO','VENTA','MES','AÑO','CATEGORIA']
    faltantes=[c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f'Faltan columnas requeridas después de concatenar: {faltantes}. Disponibles: {list(df.columns)}')

    df['MES_NUM']=df['MES'].astype(str).str.strip().str.upper().map(MESES)
    # También admite meses numéricos.
    mes_numerico=pd.to_numeric(df['MES'],errors='coerce')
    df['MES_NUM']=df['MES_NUM'].fillna(mes_numerico)
    df['AÑO']=pd.to_numeric(df['AÑO'],errors='coerce')
    df['VENTA']=pd.to_numeric(df['VENTA'],errors='coerce')
    df['CODIGO']=df['CODIGO'].astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    df['CATEGORIA']=df['CATEGORIA'].astype(str).str.strip()
    df=df.dropna(subset=['MES_NUM','AÑO','VENTA'])
    df=df[~df['CODIGO'].isin(['','nan','None'])]
    df=df[df['MES_NUM'].between(1,12)]
    if df.empty:
        raise ValueError('Después de convertir MES, AÑO, CODIGO y VENTA no quedaron registros válidos.')
    df['Periodo']=(df['AÑO'].astype(int)*100+df['MES_NUM'].astype(int)).astype(int)

    meta={c:'first' for c in ['DESCRIPCION','CATEGORIA','FAMILIA','TIPO'] if c in df.columns}
    g=df.groupby(['CODIGO','Periodo','MES_NUM','AÑO'],as_index=False).agg({'VENTA':'sum',**meta})

    salida=[]
    for codigo,x in g.groupby('CODIGO',sort=False):
        x=x.sort_values('Periodo').copy()
        fechas=pd.to_datetime(x['AÑO'].astype(int).astype(str)+'-'+x['MES_NUM'].astype(int).astype(str)+'-01')
        x.index=fechas
        x=x.reindex(pd.date_range(fechas.min(),fechas.max(),freq='MS'))
        x['CODIGO']=codigo
        x['VENTA']=x['VENTA'].fillna(0.0)
        x['MES_NUM']=x.index.month
        x['AÑO']=x.index.year
        x['Periodo']=x.index.year*100+x.index.month
        for c in meta:
            x[c]=x[c].ffill().bfill()
        salida.append(x.reset_index(drop=True))

    resultado=pd.concat(salida,ignore_index=True).sort_values(['CODIGO','Periodo']).reset_index(drop=True)
    # Garantía final para impedir el KeyError reportado.
    if 'CODIGO' not in resultado.columns:
        raise ValueError(f'Error de estructura: CODIGO se perdió durante la normalización. Columnas: {list(resultado.columns)}')
    return resultado,problemas


def crear_features(df):
    z=df.sort_values(['CODIGO','Periodo']).copy()
    grp=z.groupby('CODIGO',sort=False)['VENTA']
    for lag in LAGS: z[f'lag{lag}']=grp.shift(lag)
    for w in VENTANAS:
        z[f'mean{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).mean())
        z[f'std{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z['sin_mes']=np.sin(2*np.pi*z['MES_NUM']/12)
    z['cos_mes']=np.cos(2*np.pi*z['MES_NUM']/12)
    return z


def entrenar(df,cfg):
    f=crear_features(df).dropna(subset=['lag1','lag2','VENTA']).copy()
    if f.empty: raise ValueError('No hay historial suficiente para entrenar esta categoría.')
    medianas=f[FEATURES].median().fillna(0)
    X=f[FEATURES].fillna(medianas).astype('float32')
    y=np.log1p(f['VENTA'].clip(lower=0))
    pesos=np.where(f['MES_NUM'].isin([11,12]) & f['AÑO'].isin([2024,2025]),cfg['peso'],1.0)
    modelo=ExtraTreesRegressor(n_estimators=cfg['arboles'],max_depth=cfg['profundidad'],
        min_samples_leaf=cfg['hoja'],max_features=cfg['variables'],n_jobs=-1,random_state=42)
    modelo.fit(X,y,sample_weight=pesos)
    return modelo,medianas


def vector(hist,mes,medianas):
    ventas=hist.sort_values('Periodo')['VENTA'].to_numpy(float); d={}
    for lag in LAGS: d[f'lag{lag}']=ventas[-lag] if len(ventas)>=lag else np.nan
    for w in VENTANAS:
        a=ventas[-w:]; d[f'mean{w}']=a.mean() if len(a)>=2 else np.nan; d[f'std{w}']=a.std(ddof=1) if len(a)>=2 else 0
    d['sin_mes']=np.sin(2*np.pi*mes/12); d['cos_mes']=np.cos(2*np.pi*mes/12)
    return pd.DataFrame([d],columns=FEATURES).fillna(medianas).astype('float32')


def predecir(modelo,medianas,hist,periodos):
    h=hist.copy(); pred=[]
    for periodo in periodos:
        p=max(0,float(np.expm1(modelo.predict(vector(h,periodo%100,medianas))[0])))
        pred.append(p); fila=h.iloc[-1].copy(); fila['Periodo']=periodo; fila['AÑO']=periodo//100; fila['MES_NUM']=periodo%100; fila['VENTA']=p
        h=pd.concat([h,pd.DataFrame([fila])],ignore_index=True)
    return pred


def metricas(r):
    y=r['Actual'].to_numpy(float); p=r['Prediccion'].to_numpy(float)
    return {'WAPE':100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),'MAE':mean_absolute_error(y,p),
            'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}


def backtest(df,cfg):
    tamanos=df.groupby('CODIGO').size(); codigos=tamanos[tamanos>=5].index
    base=df[df['CODIGO'].isin(codigos)].copy()
    held=base.groupby('CODIGO',group_keys=False).tail(2); train=base.drop(index=held.index)
    modelo,medianas=entrenar(train,cfg); historicos={c:g for c,g in train.groupby('CODIGO',sort=False)}; filas=[]
    for codigo,real in held.groupby('CODIGO',sort=False):
        if codigo not in historicos: continue
        real=real.sort_values('Periodo'); ps=predecir(modelo,medianas,historicos[codigo],real['Periodo'].astype(int).tolist())
        for (_,r),p in zip(real.iterrows(),ps):
            filas.append({'CODIGO':codigo,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),
                          'Periodo':int(r['Periodo']),'Actual':float(r['VENTA']),'Prediccion':p})
    return pd.DataFrame(filas)

archivos=st.file_uploader('Sube ventas 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
with st.sidebar.form('config'):
    arboles=st.select_slider('Árboles',[100,150,200,250,300,400],value=250)
    profundidad=st.selectbox('Profundidad',[10,14,16,20,None],index=3,format_func=lambda x:'Sin límite' if x is None else str(x))
    hoja=st.select_slider('Mínimo por hoja',[2,3,4,5],value=2)
    variables=st.select_slider('Variables por árbol',[0.7,0.8,0.9,1.0],value=1.0)
    peso=st.slider('Peso Nov-Dic 2024/2025',1.0,3.0,1.5,0.25)
    st.form_submit_button('Aplicar')

if archivos:
    try:
        total,problemas=cargar(archivos)
        # Esta línea ya es segura porque cargar() garantiza CODIGO.
        st.success(f"{total['CODIGO'].nunique():,} productos y {len(total):,} observaciones normalizadas")
        if problemas:
            with st.expander('Hojas omitidas o con estructura diferente'):
                for p in problemas: st.warning(p)

        categorias=sorted(total['CATEGORIA'].dropna().astype(str).unique())
        categoria=st.selectbox('CATEGORIA para métricas y entrenamiento',['Todas']+categorias)
        df=total.copy() if categoria=='Todas' else total[total['CATEGORIA'].astype(str)==categoria].copy()
        cfg={'arboles':arboles,'profundidad':profundidad,'hoja':hoja,'variables':variables,'peso':peso}
        firma=(tuple((a.name,a.size) for a in archivos),categoria,tuple(cfg.items()),len(df),int(df['Periodo'].max()))
        for k,v in [('modelo_cat',None),('firma_modelo_cat',None),('bt_cat',None),('firma_bt_cat',None)]:
            if k not in st.session_state: st.session_state[k]=v

        c1,c2=st.columns(2)
        with c1:
            if st.button('Calcular métricas de la categoría',type='primary'):
                with st.spinner('Calculando métricas...'):
                    st.session_state.bt_cat=backtest(df,cfg); st.session_state.firma_bt_cat=firma
        with c2:
            if st.button('Entrenar modelo de la categoría'):
                with st.spinner('Entrenando modelo...'):
                    st.session_state.modelo_cat=entrenar(df,cfg); st.session_state.firma_modelo_cat=firma

        if st.session_state.bt_cat is not None and st.session_state.firma_bt_cat==firma and not st.session_state.bt_cat.empty:
            bt=st.session_state.bt_cat.copy(); m=metricas(bt); a,b,c,d=st.columns(4)
            a.metric('WAPE',f"{m['WAPE']:.2f}%"); b.metric('MAE',f"${m['MAE']:,.2f}"); c.metric('RMSE',f"${m['RMSE']:,.2f}"); d.metric('R²',f"{m['R2']:.3f}")
            bt['Error_Abs']=(bt['Actual']-bt['Prediccion']).abs(); bt['Aporte_WAPE_pp']=100*bt['Error_Abs']/max(bt['Actual'].abs().sum(),1)
            st.dataframe(bt.nlargest(15,'Error_Abs'),use_container_width=True,hide_index=True)

        if st.session_state.modelo_cat is not None and st.session_state.firma_modelo_cat==firma:
            modelo,medianas=st.session_state.modelo_cat
            historicos={codigo:g.sort_values('Periodo') for codigo,g in df.groupby('CODIGO',sort=False)}
            ultimos=df.sort_values('Periodo').groupby('CODIGO').tail(1)
            opciones={r['CODIGO']:f"{r['CODIGO']} - {r.get('DESCRIPCION','')}" for _,r in ultimos.iterrows()}
            codigo=st.selectbox('Producto',list(opciones),format_func=lambda x:opciones[x])
            ultimo=int(df['Periodo'].max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
            periodos=[(fecha+pd.DateOffset(months=i)).year*100+(fecha+pd.DateOffset(months=i)).month for i in [1,2]]
            ps=predecir(modelo,medianas,historicos[codigo],periodos)
            st.write(f"**{periodos[0]}:** ${ps[0]:,.2f} | **{periodos[1]}:** ${ps[1]:,.2f}")
            graf=pd.concat([historicos[codigo][['Periodo','VENTA']].assign(Tipo='Real'),pd.DataFrame({'Periodo':periodos,'VENTA':ps,'Tipo':'Proyectado'})])
            st.altair_chart(alt.Chart(graf).mark_line(point=True).encode(x='Periodo:O',y='VENTA:Q',color='Tipo:N',tooltip=['Periodo','VENTA','Tipo']).properties(height=350),use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.exception(e)


