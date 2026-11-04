import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title='Prediccion Nov-Dic V5', page_icon='📦', layout='wide')
st.title('📦 Prediccion Noviembre-Diciembre 2026 V5')
st.caption('Extra Trees logaritmico, 250 arboles, validacion comparable y peso estacional moderado.')

MESES={'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
LAGS=[1,2,3,4,6,12]; VENTANAS=[3,6,12]
FEATURES=([f'lag{x}' for x in LAGS]+[f'mean{x}' for x in VENTANAS]+[f'std{x}' for x in VENTANAS]
          +['sin_mes','cos_mes'])

@st.cache_data(show_spinner='Leyendo y normalizando archivos...')
def cargar(archivos):
    partes=[]
    for a in archivos:
        for h in pd.read_excel(a,sheet_name=None,engine='openpyxl').values():
            if h is not None and not h.empty: partes.append(h)
    df=pd.concat(partes,ignore_index=True); df.columns=[str(c).strip().upper() for c in df.columns]
    faltan=[c for c in ['CODIGO','VENTA','MES','AÑO'] if c not in df.columns]
    if faltan: raise ValueError('Faltan columnas: '+', '.join(faltan))
    df['MES_NUM']=df.MES.astype(str).str.strip().str.upper().map(MESES)
    df['AÑO']=pd.to_numeric(df['AÑO'],errors='coerce'); df['VENTA']=pd.to_numeric(df.VENTA,errors='coerce')
    df['CODIGO']=df.CODIGO.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    df=df.dropna(subset=['MES_NUM','AÑO','VENTA']); df=df[~df.CODIGO.isin(['','nan'])]
    df['Periodo']=(df['AÑO']*100+df.MES_NUM).astype(int)
    meta={c:'first' for c in ['DESCRIPCION','CATEGORIA','FAMILIA','TIPO'] if c in df.columns}
    g=df.groupby(['CODIGO','Periodo','MES_NUM','AÑO'],as_index=False).agg({'VENTA':'sum',**meta})
    out=[]
    for code,x in g.groupby('CODIGO',sort=False):
        x=x.sort_values('Periodo').copy(); dt=pd.to_datetime(x.AÑO.astype(int).astype(str)+'-'+x.MES_NUM.astype(int).astype(str)+'-01')
        x.index=dt; x=x.reindex(pd.date_range(dt.min(),dt.max(),freq='MS')); x['CODIGO']=code
        x['VENTA']=x.VENTA.fillna(0); x['MES_NUM']=x.index.month; x['AÑO']=x.index.year; x['Periodo']=x.index.year*100+x.index.month
        for c in meta: x[c]=x[c].ffill().bfill()
        out.append(x.reset_index(drop=True))
    return pd.concat(out,ignore_index=True).sort_values(['CODIGO','Periodo']).reset_index(drop=True)

def features(df):
    z=df.sort_values(['CODIGO','Periodo']).copy(); grp=z.groupby('CODIGO',sort=False).VENTA
    for l in LAGS: z[f'lag{l}']=grp.shift(l)
    for w in VENTANAS:
        z[f'mean{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).mean())
        z[f'std{w}']=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z['sin_mes']=np.sin(2*np.pi*z.MES_NUM/12); z['cos_mes']=np.cos(2*np.pi*z.MES_NUM/12)
    return z

def entrenar(df,cfg):
    f=features(df).dropna(subset=['lag1','lag2','VENTA']).copy()
    med=f[FEATURES].median().fillna(0); X=f[FEATURES].fillna(med).astype('float32'); y=np.log1p(f.VENTA.clip(lower=0))
    # Peso moderado. Para el modelo final sí existen Nov-Dic 2024/2025 en entrenamiento.
    sw=np.where(f.MES_NUM.isin([11,12]) & f.AÑO.isin([2024,2025]),cfg['peso'],1.0)
    m=ExtraTreesRegressor(n_estimators=cfg['arboles'],max_depth=cfg['profundidad'],min_samples_leaf=cfg['hoja'],
                          max_features=cfg['variables'],n_jobs=-1,random_state=42)
    m.fit(X,y,sample_weight=sw); return m,med

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
    return {'WAPE':100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),'MAE':mean_absolute_error(y,p),
            'RMSE':mean_squared_error(y,p)**.5,'R2':r2_score(y,p)}

def backtest_ultimos_dos(df,cfg):
    sizes=df.groupby('CODIGO').size(); codes=sizes[sizes>=5].index; base=df[df.CODIGO.isin(codes)].copy()
    held=base.groupby('CODIGO',group_keys=False).tail(2); train=base.drop(index=held.index)
    modelo,med=entrenar(train,cfg); hist={c:g for c,g in train.groupby('CODIGO',sort=False)}; rows=[]
    for code,real in held.groupby('CODIGO',sort=False):
        if code not in hist: continue
        real=real.sort_values('Periodo'); ps=predecir(modelo,med,hist[code],real.Periodo.astype(int).tolist())
        for (_,r),p in zip(real.iterrows(),ps): rows.append({'CODIGO':code,'DESCRIPCION':r.get('DESCRIPCION',''),'Periodo':int(r.Periodo),'Actual':float(r.VENTA),'Prediccion':p})
    return pd.DataFrame(rows)

archivos=st.file_uploader('Sube ventas 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
with st.sidebar.form('cfg'):
    st.header('Configuracion recomendada')
    arboles=st.select_slider('Arboles',[100,150,200,250,300,400],value=250)
    profundidad=st.selectbox('Profundidad',[10,12,14,16,20,None],index=4,format_func=lambda x:'Sin limite' if x is None else str(x))
    hoja=st.select_slider('Minimo por hoja',[2,3,4,5],value=2)
    variables=st.select_slider('Variables por arbol',[0.7,0.8,0.9,1.0],value=1.0)
    peso=st.slider('Peso Nov-Dic 2024/2025',1.0,3.0,1.5,0.25)
    st.form_submit_button('Aplicar')

if archivos:
    try:
        df=cargar(archivos); cfg={'arboles':arboles,'profundidad':profundidad,'hoja':hoja,'variables':variables,'peso':peso}
        firma=(tuple((a.name,a.size) for a in archivos),tuple(cfg.items()),int(df.Periodo.max()))
        st.success(f"{df.CODIGO.nunique():,} productos | {len(df):,} registros | Ultimo periodo {int(df.Periodo.max())}")
        st.warning('Estas metricas son comparables con la V2: reservan los ultimos dos meses por producto. El 42.79% provenia de otro backtest, Nov-Dic historico, y no debe compararse directamente.')
        for k,v in [('modelo',None),('firma',None),('bt',None)]:
            if k not in st.session_state: st.session_state[k]=v
        c1,c2=st.columns(2)
        with c1:
            if st.button('Calcular metricas comparables',type='primary'):
                with st.spinner('Calculando backtest...'): st.session_state.bt=backtest_ultimos_dos(df,cfg)
        with c2:
            if st.button('Entrenar modelo final Nov-Dic 2026'):
                with st.spinner('Entrenando una sola vez...'):
                    st.session_state.modelo=entrenar(df,cfg); st.session_state.firma=firma
        if st.session_state.bt is not None and not st.session_state.bt.empty:
            bt=st.session_state.bt.copy(); m=metricas(bt); a,b,c,d=st.columns(4)
            a.metric('WAPE',f"{m['WAPE']:.2f}%"); b.metric('MAE',f"${m['MAE']:,.2f}"); c.metric('RMSE',f"${m['RMSE']:,.2f}"); d.metric('R²',f"{m['R2']:.3f}")
            bt['Error_Abs']=(bt.Actual-bt.Prediccion).abs(); bt['Aporte_WAPE_pp']=100*bt.Error_Abs/max(bt.Actual.abs().sum(),1)
            st.dataframe(bt.nlargest(15,'Error_Abs'),use_container_width=True,hide_index=True)
            st.download_button('Descargar backtest',bt.to_csv(index=False).encode('utf-8-sig'),'backtest_v5.csv','text/csv')
        if st.session_state.modelo is not None:
            if st.session_state.firma!=firma: st.warning('La configuracion cambio. Entrena nuevamente el modelo final.')
            else:
                modelo,med=st.session_state.modelo; hist={c:g.sort_values('Periodo') for c,g in df.groupby('CODIGO',sort=False)}
                ult=df.sort_values('Periodo').groupby('CODIGO').tail(1); ops={r.CODIGO:f"{r.CODIGO} - {r.get('DESCRIPCION','')}" for _,r in ult.iterrows()}
                code=st.selectbox('Producto',list(ops),format_func=lambda x:ops[x]); ultimo=int(df.Periodo.max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
                periodos=[]
                for i in [1,2]: f=fecha+pd.DateOffset(months=i); periodos.append(f.year*100+f.month)
                ps=predecir(modelo,med,hist[code],periodos); st.write(f"**{periodos[0]}:** ${ps[0]:,.2f} | **{periodos[1]}:** ${ps[1]:,.2f}")
                graf=pd.concat([hist[code][['Periodo','VENTA']].assign(Tipo='Real'),pd.DataFrame({'Periodo':periodos,'VENTA':ps,'Tipo':'Proyectado'})])
                st.altair_chart(alt.Chart(graf).mark_line(point=True).encode(x='Periodo:O',y='VENTA:Q',color='Tipo:N',tooltip=['Periodo','VENTA','Tipo']).properties(height=350),use_container_width=True)
                if st.button('Generar proyeccion masiva'):
                    rows=[]
                    with st.spinner('Prediciendo sin reentrenar...'):
                        for code,h in hist.items():
                            p=predecir(modelo,med,h,periodos); r=h.iloc[-1]
                            rows.append({'CODIGO':code,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),'Pred_Nov_2026':p[0],'Pred_Dic_2026':p[1]})
                    out=pd.DataFrame(rows); st.dataframe(out,use_container_width=True,hide_index=True)
                    st.download_button('Descargar CSV',out.to_csv(index=False).encode('utf-8-sig'),'prediccion_v5.csv','text/csv')
        else: st.info("Primero calcula las metricas y despues entrena el modelo final.")
    except Exception as e: st.exception(e)


