import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title='Prediccion por categoria', page_icon='📦', layout='wide')
st.title('📦 Prediccion Noviembre-Diciembre 2026 por CATEGORIA')

MESES={'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
LAGS=[1,2,3,4,6,12]; VENTANAS=[3,6,12]
FEATURES=([f'lag{x}' for x in LAGS]+[f'mean{x}' for x in VENTANAS]+[f'std{x}' for x in VENTANAS]+['sin_mes','cos_mes'])

@st.cache_data(show_spinner='Leyendo archivos...')
def cargar(archivos):
    partes=[]
    for a in archivos:
        for h in pd.read_excel(a,sheet_name=None,engine='openpyxl').values():
            if h is not None and not h.empty: partes.append(h)
    if not partes: raise ValueError('No hay datos.')
    df=pd.concat(partes,ignore_index=True); df.columns=[str(c).strip().upper() for c in df.columns]
    req=['CODIGO','VENTA','MES','AÑO','CATEGORIA']; falt=[c for c in req if c not in df]
    if falt: raise ValueError('Faltan columnas: '+', '.join(falt))
    df['MES_NUM']=df.MES.astype(str).str.strip().str.upper().map(MESES)
    df['AÑO']=pd.to_numeric(df['AÑO'],errors='coerce'); df['VENTA']=pd.to_numeric(df.VENTA,errors='coerce')
    df['CODIGO']=df.CODIGO.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    df['CATEGORIA']=df.CATEGORIA.astype(str).str.strip()
    df=df.dropna(subset=['MES_NUM','AÑO','VENTA']); df=df[~df.CODIGO.isin(['','nan'])]
    df['Periodo']=(df.AÑO*100+df.MES_NUM).astype(int)
    meta={c:'first' for c in ['DESCRIPCION','CATEGORIA','FAMILIA','TIPO'] if c in df}
    g=df.groupby(['CODIGO','Periodo','MES_NUM','AÑO'],as_index=False).agg({'VENTA':'sum',**meta})
    out=[]
    for code,x in g.groupby('CODIGO',sort=False):
        x=x.sort_values('Periodo').copy(); dt=pd.to_datetime(x.AÑO.astype(int).astype(str)+'-'+x.MES_NUM.astype(int).astype(str)+'-01')
        x.index=dt; x=x.reindex(pd.date_range(dt.min(),dt.max(),freq='MS')); x['CODIGO']=code
        x['VENTA']=x.VENTA.fillna(0); x['MES_NUM']=x.index.month; x['AÑO']=x.index.year; x['Periodo']=x.index.year*100+x.index.month
        for c in meta: x[c]=x[c].ffill().bfill()
        out.append(x.reset_index(drop=True))
    return pd.concat(out,ignore_index=True).sort_values(['CODIGO','Periodo']).reset_index(drop=True)

def crear_features(df):
    z=df.sort_values(['CODIGO','Periodo']).copy(); grp=z.groupby('CODIGO',sort=False).VENTA
    for l in LAGS: z[f'lag{l}']=grp.shift(l)
    for w in VENTANAS:
        z[f'mean{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).mean())
        z[f'std{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z['sin_mes']=np.sin(2*np.pi*z.MES_NUM/12); z['cos_mes']=np.cos(2*np.pi*z.MES_NUM/12)
    return z

def entrenar(df,cfg):
    f=crear_features(df).dropna(subset=['lag1','lag2','VENTA']).copy()
    if f.empty: raise ValueError('La categoria no tiene historial suficiente para entrenar.')
    med=f[FEATURES].median().fillna(0); X=f[FEATURES].fillna(med).astype('float32'); y=np.log1p(f.VENTA.clip(lower=0))
    pesos=np.where(f.MES_NUM.isin([11,12]) & f.AÑO.isin([2024,2025]),cfg['peso'],1.0)
    modelo=ExtraTreesRegressor(n_estimators=cfg['arboles'],max_depth=cfg['profundidad'],min_samples_leaf=cfg['hoja'],max_features=cfg['variables'],n_jobs=-1,random_state=42)
    modelo.fit(X,y,sample_weight=pesos); return modelo,med

def vector(hist,mes,med):
    v=hist.sort_values('Periodo').VENTA.to_numpy(float); d={}
    for l in LAGS: d[f'lag{l}']=v[-l] if len(v)>=l else np.nan
    for w in VENTANAS:
        a=v[-w:]; d[f'mean{w}']=a.mean() if len(a)>=2 else np.nan; d[f'std{w}']=a.std(ddof=1) if len(a)>=2 else 0
    d['sin_mes']=np.sin(2*np.pi*mes/12); d['cos_mes']=np.cos(2*np.pi*mes/12)
    return pd.DataFrame([d],columns=FEATURES).fillna(med).astype('float32')

def predecir(modelo,med,hist,periodos):
    h=hist.copy(); out=[]
    for per in periodos:
        p=max(0,float(np.expm1(modelo.predict(vector(h,per%100,med))[0]))); out.append(p)
        r=h.iloc[-1].copy(); r['Periodo']=per; r['AÑO']=per//100; r['MES_NUM']=per%100; r['VENTA']=p
        h=pd.concat([h,pd.DataFrame([r])],ignore_index=True)
    return out

def metricas(r):
    y=r.Actual.to_numpy(float); p=r.Prediccion.to_numpy(float)
    return {'WAPE':100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**.5,'R2':r2_score(y,p)}

def backtest(df,cfg):
    sizes=df.groupby('CODIGO').size(); codes=sizes[sizes>=5].index; base=df[df.CODIGO.isin(codes)].copy()
    held=base.groupby('CODIGO',group_keys=False).tail(2); train=base.drop(index=held.index)
    modelo,med=entrenar(train,cfg); historicos={c:g for c,g in train.groupby('CODIGO',sort=False)}; rows=[]
    for code,real in held.groupby('CODIGO',sort=False):
        if code not in historicos: continue
        real=real.sort_values('Periodo'); ps=predecir(modelo,med,historicos[code],real.Periodo.astype(int).tolist())
        for (_,r),p in zip(real.iterrows(),ps): rows.append({'CODIGO':code,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),'Periodo':int(r.Periodo),'Actual':float(r.VENTA),'Prediccion':p})
    return pd.DataFrame(rows)

archivos=st.file_uploader('Sube ventas 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
with st.sidebar.form('config'):
    st.header('Configuracion')
    arboles=st.select_slider('Arboles',[100,150,200,250,300,400],value=250)
    profundidad=st.selectbox('Profundidad',[10,14,16,20,None],index=3,format_func=lambda x:'Sin limite' if x is None else str(x))
    hoja=st.select_slider('Minimo por hoja',[2,3,4,5],value=2)
    variables=st.select_slider('Variables por arbol',[0.7,0.8,0.9,1.0],value=1.0)
    peso=st.slider('Peso Nov-Dic 2024/2025',1.0,3.0,1.5,0.25)
    st.form_submit_button('Aplicar')

if archivos:
    try:
        total=cargar(archivos)
        categorias=sorted(total.CATEGORIA.dropna().astype(str).unique())
        categoria=st.selectbox('CATEGORIA para metricas y entrenamiento',['Todas']+categorias)
        df=total.copy() if categoria=='Todas' else total[total.CATEGORIA.astype(str)==categoria].copy()
        if df.empty: raise ValueError('La categoria seleccionada no tiene registros.')
        cfg={'arboles':arboles,'profundidad':profundidad,'hoja':hoja,'variables':variables,'peso':peso}
        firma=(tuple((a.name,a.size) for a in archivos),categoria,tuple(cfg.items()),len(df),int(df.Periodo.max()))
        st.success(f'Categoria: {categoria} | {df.CODIGO.nunique():,} productos | {len(df):,} registros')
        for k,v in [('modelo_cat',None),('firma_modelo_cat',None),('bt_cat',None),('firma_bt_cat',None)]:
            if k not in st.session_state: st.session_state[k]=v
        c1,c2=st.columns(2)
        with c1:
            if st.button('Calcular metricas de la categoria',type='primary'):
                with st.spinner(f'Validando {categoria}...'):
                    st.session_state.bt_cat=backtest(df,cfg); st.session_state.firma_bt_cat=firma
        with c2:
            if st.button('Entrenar modelo de la categoria'):
                with st.spinner(f'Entrenando {categoria}...'):
                    st.session_state.modelo_cat=entrenar(df,cfg); st.session_state.firma_modelo_cat=firma
        if st.session_state.bt_cat is not None and st.session_state.firma_bt_cat==firma and not st.session_state.bt_cat.empty:
            bt=st.session_state.bt_cat.copy(); m=metricas(bt); st.subheader(f'Metricas: {categoria}')
            a,b,c,d=st.columns(4); a.metric('WAPE',f"{m['WAPE']:.2f}%"); b.metric('MAE',f"${m['MAE']:,.2f}"); c.metric('RMSE',f"${m['RMSE']:,.2f}"); d.metric('R²',f"{m['R2']:.3f}")
            bt['Error_Abs']=(bt.Actual-bt.Prediccion).abs(); bt['Aporte_WAPE_pp']=100*bt.Error_Abs/max(bt.Actual.abs().sum(),1)
            st.dataframe(bt.nlargest(15,'Error_Abs'),use_container_width=True,hide_index=True)
            nombre='todas' if categoria=='Todas' else categoria.replace(' ','_')
            st.download_button('Descargar backtest',bt.to_csv(index=False).encode('utf-8-sig'),f'backtest_{nombre}.csv','text/csv')
        elif st.session_state.bt_cat is not None:
            st.info('La categoria o configuracion cambio. Calcula nuevamente las metricas.')
        if st.session_state.modelo_cat is not None and st.session_state.firma_modelo_cat==firma:
            modelo,med=st.session_state.modelo_cat; historicos={c:g.sort_values('Periodo') for c,g in df.groupby('CODIGO',sort=False)}
            ult=df.sort_values('Periodo').groupby('CODIGO').tail(1); opciones={r.CODIGO:f"{r.CODIGO} - {r.get('DESCRIPCION','')}" for _,r in ult.iterrows()}
            code=st.selectbox('Producto',list(opciones),format_func=lambda x:opciones[x]); ultimo=int(df.Periodo.max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
            periodos=[(fecha+pd.DateOffset(months=i)).year*100+(fecha+pd.DateOffset(months=i)).month for i in [1,2]]
            ps=predecir(modelo,med,historicos[code],periodos); st.write(f"**{periodos[0]}:** ${ps[0]:,.2f} | **{periodos[1]}:** ${ps[1]:,.2f}")
            graf=pd.concat([historicos[code][['Periodo','VENTA']].assign(Tipo='Real'),pd.DataFrame({'Periodo':periodos,'VENTA':ps,'Tipo':'Proyectado'})])
            st.altair_chart(alt.Chart(graf).mark_line(point=True).encode(x='Periodo:O',y='VENTA:Q',color='Tipo:N',tooltip=['Periodo','VENTA','Tipo']).properties(height=350),use_container_width=True)
            if st.button('Generar proyeccion masiva de la categoria'):
                rows=[]
                for codigo,h in historicos.items():
                    p=predecir(modelo,med,h,periodos); r=h.iloc[-1]
                    rows.append({'CODIGO':codigo,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),'Pred_Nov_2026':p[0],'Pred_Dic_2026':p[1]})
                salida=pd.DataFrame(rows); st.dataframe(salida,use_container_width=True,hide_index=True)
                nombre='todas' if categoria=='Todas' else categoria.replace(' ','_')
                st.download_button('Descargar proyeccion',salida.to_csv(index=False).encode('utf-8-sig'),f'prediccion_{nombre}.csv','text/csv')
        elif st.session_state.modelo_cat is not None:
            st.warning('La categoria o configuracion cambio. Entrena nuevamente el modelo.')
    except Exception as e: st.exception(e)
