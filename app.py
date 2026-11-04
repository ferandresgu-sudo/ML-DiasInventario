import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title='Prediccion por categoria', page_icon='📦', layout='wide')
st.title('📦 Prediccion Noviembre-Diciembre 2026 V7, calibración automática de RMSE')
st.caption('Ensamble log/raw, suavizado por volatilidad y límites históricos para controlar errores extremos.')

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
    """Entrena dos Extra Trees: log reduce WAPE/MAE y raw protege productos de alto volumen."""
    f=crear_features(df).dropna(subset=['lag1','lag2','VENTA']).copy()
    if f.empty:
        raise ValueError('No hay historial suficiente para entrenar esta categoría.')
    medianas=f[FEATURES].median().fillna(0)
    X=f[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(medianas).astype('float32')
    y_raw=f['VENTA'].clip(lower=0).astype('float32')
    y_log=np.log1p(y_raw)
    pesos=np.where(f['MES_NUM'].isin([11,12]) & f['AÑO'].isin([2024,2025]),cfg['peso'],1.0)

    comunes=dict(
        n_estimators=cfg['arboles'],
        max_depth=cfg['profundidad'],
        min_samples_leaf=cfg['hoja'],
        max_features=cfg['variables'],
        n_jobs=-1,
        random_state=42
    )
    modelo_log=ExtraTreesRegressor(**comunes)
    modelo_raw=ExtraTreesRegressor(**comunes)
    modelo_log.fit(X,y_log,sample_weight=pesos)
    modelo_raw.fit(X,y_raw,sample_weight=pesos)
    return {'log':modelo_log,'raw':modelo_raw},medianas


def vector(hist,mes,medianas):
    ventas=hist.sort_values('Periodo')['VENTA'].to_numpy(float); d={}
    for lag in LAGS:
        d[f'lag{lag}']=ventas[-lag] if len(ventas)>=lag else np.nan
    for w in VENTANAS:
        a=ventas[-w:]
        d[f'mean{w}']=a.mean() if len(a)>=2 else np.nan
        d[f'std{w}']=a.std(ddof=1) if len(a)>=2 else 0
    d['sin_mes']=np.sin(2*np.pi*mes/12); d['cos_mes']=np.cos(2*np.pi*mes/12)
    return pd.DataFrame([d],columns=FEATURES).replace([np.inf,-np.inf],np.nan).fillna(medianas).astype('float32')


def ajustar_rmse(pred_modelo,hist,cfg):
    """Suaviza y recorta únicamente según el patrón histórico de cada SKU."""
    ventas=hist.sort_values('Periodo')['VENTA'].tail(12).astype(float)
    if len(ventas)<4:
        return max(0.0,float(pred_modelo)),'SIN_AJUSTE'

    ult6=ventas.tail(6)
    media6=float(ult6.mean())
    mediana6=float(ult6.median())
    std6=float(ult6.std(ddof=1)) if len(ult6)>1 else 0.0
    cv6=std6/media6 if media6>0 else 0.0
    ceros6=int((ult6==0).sum())

    if ceros6>=3:
        positivos=ventas[ventas>0]
        pred=(ventas.gt(0).mean()*positivos.median()) if not positivos.empty else 0.0
        return max(0.0,float(pred)),'INTERMITENTE'

    # Los SKU volátiles reciben más estabilización; los estables conservan el modelo.
    if cv6<=0.35:
        peso_modelo=0.90; segmento='ESTABLE'
    elif cv6<1.00:
        peso_modelo=0.80; segmento='VARIABLE'
    else:
        peso_modelo=0.65; segmento='VOLATIL'

    pred=peso_modelo*float(pred_modelo)+(1-peso_modelo)*mediana6

    if cfg['limites'] and len(ventas)>=6:
        q10=float(ventas.quantile(0.10)); q90=float(ventas.quantile(0.90))
        inferior=max(0.0,min(q10,0.50*mediana6))
        # Volátiles/estacionales mantienen más espacio que los SKU estables.
        multiplicador=2.50 if segmento=='VOLATIL' else (2.00 if segmento=='VARIABLE' else 1.75)
        superior=max(q90,multiplicador*mediana6,float(ventas.iloc[-1])*1.15)
        pred=float(np.clip(pred,inferior,superior))
    return max(0.0,float(pred)),segmento


def predecir(modelos,medianas,hist,periodos,cfg):
    h=hist.copy(); pred=[]; segmentos=[]
    for periodo in periodos:
        X=vector(h,periodo%100,medianas)
        p_log=max(0.0,float(np.expm1(modelos['log'].predict(X)[0])))
        p_raw=max(0.0,float(modelos['raw'].predict(X)[0]))
        # 70% log conserva WAPE; 30% raw reduce grandes errores de alto volumen.
        p_base=(1-cfg['peso_raw'])*p_log+cfg['peso_raw']*p_raw
        p,segmento=ajustar_rmse(p_base,h,cfg)
        pred.append(p); segmentos.append(segmento)
        fila=h.iloc[-1].copy(); fila['Periodo']=periodo; fila['AÑO']=periodo//100; fila['MES_NUM']=periodo%100; fila['VENTA']=p
        h=pd.concat([h,pd.DataFrame([fila])],ignore_index=True)
    return pred,segmentos


def metricas(r):
    y=r['Actual'].to_numpy(float); p=r['Prediccion'].to_numpy(float)
    return {'WAPE':100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),'MAE':mean_absolute_error(y,p),
            'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}


def backtest(df,cfg):
    """Entrena una vez y calibra automáticamente el peso raw para minimizar RMSE."""
    tamanos=df.groupby('CODIGO').size()
    codigos=tamanos[tamanos>=5].index
    base=df[df['CODIGO'].isin(codigos)].copy()
    held=base.groupby('CODIGO',group_keys=False).tail(2)
    train=base.drop(index=held.index)
    modelos,medianas=entrenar(train,cfg)
    historicos={c:g for c,g in train.groupby('CODIGO',sort=False)}

    candidatos=[]
    # Incluye modelo log puro y mezclas que prestan más atención a alto volumen.
    for peso_raw_cand in [0.00,0.10,0.20,0.30,0.40,0.50]:
        for usar_limites in [False,True]:
            cfg_cand=dict(cfg)
            cfg_cand['peso_raw']=peso_raw_cand
            cfg_cand['limites']=usar_limites
            filas=[]
            for codigo,real in held.groupby('CODIGO',sort=False):
                if codigo not in historicos:
                    continue
                real=real.sort_values('Periodo')
                ps,segmentos=predecir(
                    modelos,medianas,historicos[codigo],
                    real['Periodo'].astype(int).tolist(),cfg_cand
                )
                for ((_,r),pred,segmento) in zip(real.iterrows(),ps,segmentos):
                    filas.append({
                        'CODIGO':codigo,'DESCRIPCION':r.get('DESCRIPCION',''),
                        'CATEGORIA':r.get('CATEGORIA',''),'Periodo':int(r['Periodo']),
                        'Actual':float(r['VENTA']),'Prediccion':pred,'Segmento':segmento
                    })
            resultado=pd.DataFrame(filas)
            if resultado.empty:
                continue
            m=metricas(resultado)
            candidatos.append({
                'peso_raw':peso_raw_cand,'limites':usar_limites,
                'WAPE':m['WAPE'],'MAE':m['MAE'],'RMSE':m['RMSE'],'R2':m['R2'],
                'resultado':resultado
            })

    if not candidatos:
        return pd.DataFrame(),pd.DataFrame(),cfg

    comparacion=pd.DataFrame([{k:v for k,v in c.items() if k!='resultado'} for c in candidatos])
    base_log=min(candidatos,key=lambda x:abs(x['peso_raw'])+(1 if x['limites'] else 0))
    # Solo acepta menor RMSE si WAPE no aumenta más de 1 punto y MAE no sube más de 5%.
    elegibles=[c for c in candidatos
               if c['WAPE']<=base_log['WAPE']+1.0
               and c['MAE']<=base_log['MAE']*1.05]
    mejor=min(elegibles or candidatos,key=lambda x:x['RMSE'])
    cfg_opt=dict(cfg)
    cfg_opt['peso_raw']=mejor['peso_raw']
    cfg_opt['limites']=mejor['limites']
    return mejor['resultado'],comparacion.sort_values('RMSE'),cfg_opt

archivos=st.file_uploader('Sube ventas 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
with st.sidebar.form('config'):
    arboles=st.select_slider('Árboles por modelo',[75,100,125,150,200],value=150,help='Se entrenan dos modelos; 150 equivale a 300 árboles totales.')
    profundidad=st.selectbox('Profundidad',[10,12,14,16,20,None],index=2,format_func=lambda x:'Sin límite' if x is None else str(x))
    hoja=st.select_slider('Mínimo por hoja',[2,3,4,5,8],value=4)
    variables=st.select_slider('Variables por árbol',[0.6,0.7,0.8,0.9,1.0],value=0.8)
    peso=st.slider('Peso Nov-Dic 2024/2025',1.0,3.0,1.5,0.25)
    peso_raw=st.slider('Peso modelo raw para reducir RMSE',0.0,0.5,0.30,0.05)
    limites=st.checkbox('Suavizar y limitar errores extremos',value=True)
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
        cfg={'arboles':arboles,'profundidad':profundidad,'hoja':hoja,'variables':variables,'peso':peso,'peso_raw':peso_raw,'limites':limites}
        firma=(tuple((a.name,a.size) for a in archivos),categoria,tuple(cfg.items()),len(df),int(df['Periodo'].max()))
        for k,v in [('modelo_cat',None),('firma_modelo_cat',None),('bt_cat',None),('firma_bt_cat',None),('comparacion_rmse',None),('cfg_opt',None)]:
            if k not in st.session_state: st.session_state[k]=v

        c1,c2=st.columns(2)
        with c1:
            if st.button('Calcular métricas de la categoría',type='primary'):
                with st.spinner('Calculando métricas...'):
                    st.session_state.bt_cat,st.session_state.comparacion_rmse,st.session_state.cfg_opt=backtest(df,cfg)
                    st.session_state.firma_bt_cat=firma
        with c2:
            if st.button('Entrenar modelo de la categoría'):
                with st.spinner('Entrenando modelo...'):
                    cfg_entrenar=st.session_state.cfg_opt if st.session_state.cfg_opt is not None else cfg
                    st.session_state.modelo_cat=entrenar(df,cfg_entrenar); st.session_state.firma_modelo_cat=firma

        if st.session_state.bt_cat is not None and st.session_state.firma_bt_cat==firma and not st.session_state.bt_cat.empty:
            bt=st.session_state.bt_cat.copy(); m=metricas(bt); a,b,c,d=st.columns(4)
            a.metric('WAPE',f"{m['WAPE']:.2f}%"); b.metric('MAE',f"${m['MAE']:,.2f}"); c.metric('RMSE',f"${m['RMSE']:,.2f}"); d.metric('R²',f"{m['R2']:.3f}")
            bt['Error_Abs']=(bt['Actual']-bt['Prediccion']).abs(); bt['Aporte_WAPE_pp']=100*bt['Error_Abs']/max(bt['Actual'].abs().sum(),1)
            st.caption(f"Relación RMSE/MAE: {m['RMSE']/max(m['MAE'],1):.2f}x. El objetivo es reducirla sin elevar WAPE más de 1 punto.")
            if st.session_state.cfg_opt is not None:
                opt=st.session_state.cfg_opt
                st.success(f"Configuración automática seleccionada: peso raw {opt['peso_raw']:.2f}, límites {'activados' if opt['limites'] else 'desactivados'}.")
            if st.session_state.comparacion_rmse is not None:
                st.markdown('#### Comparación automática de configuraciones')
                st.dataframe(st.session_state.comparacion_rmse.style.format({'peso_raw':'{:.2f}','WAPE':'{:.2f}%','MAE':'${:,.2f}','RMSE':'${:,.2f}','R2':'{:.3f}'}),use_container_width=True,hide_index=True)
            st.dataframe(bt.nlargest(15,'Error_Abs'),use_container_width=True,hide_index=True)

        if st.session_state.modelo_cat is not None and st.session_state.firma_modelo_cat==firma:
            modelo,medianas=st.session_state.modelo_cat
            cfg_pred=st.session_state.cfg_opt if st.session_state.cfg_opt is not None else cfg
            historicos={codigo:g.sort_values('Periodo') for codigo,g in df.groupby('CODIGO',sort=False)}
            ultimos=df.sort_values('Periodo').groupby('CODIGO').tail(1)
            opciones={r['CODIGO']:f"{r['CODIGO']} - {r.get('DESCRIPCION','')}" for _,r in ultimos.iterrows()}
            codigo=st.selectbox('Producto',list(opciones),format_func=lambda x:opciones[x])
            ultimo=int(df['Periodo'].max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
            periodos=[(fecha+pd.DateOffset(months=i)).year*100+(fecha+pd.DateOffset(months=i)).month for i in [1,2]]
            ps,segmentos=predecir(modelo,medianas,historicos[codigo],periodos,cfg_pred)
            st.write(f"**{periodos[0]}:** ${ps[0]:,.2f} | **{periodos[1]}:** ${ps[1]:,.2f}")
            graf=pd.concat([historicos[codigo][['Periodo','VENTA']].assign(Tipo='Real'),pd.DataFrame({'Periodo':periodos,'VENTA':ps,'Tipo':'Proyectado'})])
            st.altair_chart(alt.Chart(graf).mark_line(point=True).encode(x='Periodo:O',y='VENTA:Q',color='Tipo:N',tooltip=['Periodo','VENTA','Tipo']).properties(height=350),use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.exception(e)


