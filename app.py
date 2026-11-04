import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Prediccion ventas V2', page_icon='📦', layout='wide')
st.title('📦 Prediccion mensual V2: ensamble optimizado por WAPE')

MESES={'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,
       'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
LAGS=[1,2,3,4,5,6,9,12]
FEATURES=[f'lag{x}' for x in LAGS]+[f'{x}{w}' for w in (3,6,12) for x in ('mean','median','std','min','max')]+[
    'trend3','trend6','cero3','cero6','sin_mes','cos_mes']
META=['DESCRIPCION','CATEGORIA','FAMILIA','TIPO']

@st.cache_data(show_spinner='Leyendo archivos...')
def cargar(archivos):
    partes=[]
    for archivo in archivos:
        for _,h in pd.read_excel(archivo,sheet_name=None,engine='openpyxl').items():
            partes.append(h)
    d=pd.concat(partes,ignore_index=True)
    d.columns=[str(c).strip().upper() for c in d.columns]
    req=['CODIGO','VENTA','MES','AÑO']
    falt=[c for c in req if c not in d.columns]
    if falt: raise ValueError('Faltan columnas: '+', '.join(falt))
    d['MES_NUM']=d.MES.astype(str).str.strip().str.upper().map(MESES)
    d['AÑO']=pd.to_numeric(d['AÑO'],errors='coerce')
    d['VENTA']=pd.to_numeric(d.VENTA,errors='coerce').clip(lower=0)
    d['CODIGO']=d.CODIGO.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    d=d.dropna(subset=['MES_NUM','AÑO','VENTA','CODIGO'])
    d['Periodo']=(d['AÑO']*100+d.MES_NUM).astype(int)
    agg={'VENTA':'sum',**{c:'first' for c in META if c in d.columns}}
    g=d.groupby(['CODIGO','Periodo','MES_NUM','AÑO'],as_index=False).agg(agg)
    salida=[]
    for cod,x in g.groupby('CODIGO'):
        x=x.sort_values('Periodo').copy()
        fechas=pd.to_datetime(x.AÑO.astype(int).astype(str)+'-'+x.MES_NUM.astype(int).astype(str)+'-01')
        x.index=fechas
        x=x.reindex(pd.date_range(fechas.min(),fechas.max(),freq='MS'))
        x['CODIGO']=cod; x['VENTA']=x.VENTA.fillna(0.0)
        x['MES_NUM']=x.index.month; x['AÑO']=x.index.year
        x['Periodo']=x.index.year*100+x.index.month
        for c in META:
            if c in x: x[c]=x[c].ffill().bfill()
        salida.append(x.reset_index(drop=True))
    return pd.concat(salida,ignore_index=True).sort_values(['CODIGO','Periodo']).reset_index(drop=True)

def features(d):
    z=d.sort_values(['CODIGO','Periodo']).copy(); grp=z.groupby('CODIGO').VENTA
    for lag in LAGS: z[f'lag{lag}']=grp.shift(lag)
    for w in (3,6,12):
        roll=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).mean())
        z[f'mean{w}']=roll
        z[f'median{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).median())
        z[f'std{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
        z[f'min{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).min())
        z[f'max{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).max())
    z['trend3']=(z.lag1-z.lag3)/3
    z['trend6']=(z.lag1-z.lag6)/6
    z['cero3']=grp.transform(lambda s:s.shift(1).rolling(3,min_periods=2).apply(lambda a:(a==0).mean()))
    z['cero6']=grp.transform(lambda s:s.shift(1).rolling(6,min_periods=2).apply(lambda a:(a==0).mean()))
    z['sin_mes']=np.sin(2*np.pi*z.MES_NUM/12); z['cos_mes']=np.cos(2*np.pi*z.MES_NUM/12)
    return z

def modelos():
    raw=RandomForestRegressor(n_estimators=350,max_depth=16,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=42)
    log=ExtraTreesRegressor(n_estimators=450,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42)
    return raw,log

def entrenar(d):
    f=features(d).dropna(subset=['lag1','lag2','VENTA']).copy()
    med=f[FEATURES].median().fillna(0); X=f[FEATURES].fillna(med); y=f.VENTA
    raw,log=modelos(); raw.fit(X,y); log.fit(X,np.log1p(y))
    return raw,log,med

def vector(hist,mes,med):
    v=hist.sort_values('Periodo').VENTA.astype(float).tolist(); q={}
    for lag in LAGS: q[f'lag{lag}']=v[-lag] if len(v)>=lag else np.nan
    for w in (3,6,12):
        a=np.asarray(v[-w:],float)
        q[f'mean{w}']=a.mean() if len(a)>=2 else np.nan
        q[f'median{w}']=np.median(a) if len(a)>=2 else np.nan
        q[f'std{w}']=a.std(ddof=1) if len(a)>=2 else 0
        q[f'min{w}']=a.min() if len(a)>=2 else np.nan
        q[f'max{w}']=a.max() if len(a)>=2 else np.nan
    q['trend3']=(q['lag1']-q['lag3'])/3 if pd.notna(q['lag3']) else 0
    q['trend6']=(q['lag1']-q['lag6'])/6 if pd.notna(q['lag6']) else 0
    q['cero3']=np.mean(np.asarray(v[-3:])==0) if len(v)>=2 else 0
    q['cero6']=np.mean(np.asarray(v[-6:])==0) if len(v)>=2 else 0
    q['sin_mes']=np.sin(2*np.pi*mes/12); q['cos_mes']=np.cos(2*np.pi*mes/12)
    return pd.DataFrame([q],columns=FEATURES).fillna(med)

def predecir(raw,log,med,hist,periodos,alpha):
    h=hist.copy(); out=[]
    for per in periodos:
        X=vector(h,per%100,med)
        p_raw=max(0,float(raw.predict(X)[0]))
        p_log=max(0,float(np.expm1(log.predict(X)[0])))
        # alpha alto favorece log (menor WAPE); alpha bajo favorece raw (menor RMSE en volumen alto)
        p=alpha*p_log+(1-alpha)*p_raw
        out.append(p)
        r=h.iloc[-1].copy();r['Periodo']=per;r['AÑO']=per//100;r['MES_NUM']=per%100;r['VENTA']=p
        h=pd.concat([h,pd.DataFrame([r])],ignore_index=True)
    return out

def corte(d,n):
    valid=d.groupby('CODIGO').filter(lambda x:len(x)>=n+3)
    held=valid.groupby('CODIGO',group_keys=False).tail(n)
    return valid.drop(index=held.index),held

def pronosticar_holdout(train,held,alpha):
    raw,log,med=entrenar(train); filas=[]
    for cod,r in held.groupby('CODIGO'):
        r=r.sort_values('Periodo'); h=train[train.CODIGO==cod]
        ps=predecir(raw,log,med,h,r.Periodo.astype(int).tolist(),alpha)
        for i,((_,x),p) in enumerate(zip(r.iterrows(),ps),1):
            filas.append({'CODIGO':cod,'DESCRIPCION':x.get('DESCRIPCION',''),'CATEGORIA':x.get('CATEGORIA',''),
                          'Periodo_Real':int(x.Periodo),'Mes_Predecido':i,'Actual':float(x.VENTA),'Prediccion':p})
    return pd.DataFrame(filas)

def wape(df):
    return 100*(df.Actual-df.Prediccion).abs().sum()/max(df.Actual.abs().sum(),1)

def seleccionar_alpha(train_externo,n_inner=2):
    tr,va=corte(train_externo,n_inner)
    candidatos=np.arange(0,1.01,.1); mejor=(1e99,.7)
    # Se generan raw y log una sola vez usando alpha 0 y 1; luego se prueban mezclas sin reentrenar.
    a=pronosticar_holdout(tr,va,0.0); b=pronosticar_holdout(tr,va,1.0)
    for alpha in candidatos:
        t=a.copy();t['Prediccion']=alpha*b.Prediccion.values+(1-alpha)*a.Prediccion.values
        score=wape(t)
        if score<mejor[0]: mejor=(score,float(alpha))
    return mejor[1],mejor[0]

archivos=st.file_uploader('Sube los archivos 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
if archivos:
    try:
        d=cargar(archivos)
        st.success(f'{d.CODIGO.nunique():,} productos y {len(d):,} observaciones mensuales')
        n=st.slider('Meses reservados para backtest final',1,4,2)
        train,held=corte(d,n)
        with st.spinner('Seleccionando ensamble con validacion temporal interna...'):
            alpha,wape_inner=seleccionar_alpha(train,min(2,n))
            bt=pronosticar_holdout(train,held,alpha)
        y=bt.Actual.values;p=bt.Prediccion.values
        W=wape(bt);MAE=mean_absolute_error(y,p);RMSE=mean_squared_error(y,p)**.5;R2=r2_score(y,p)
        st.info(f'Peso log seleccionado: {alpha:.1f} | WAPE validacion interna: {wape_inner:.2f}%')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('WAPE',f'{W:.2f}%');c2.metric('MAE',f'${MAE:,.2f}')
        c3.metric('RMSE',f'${RMSE:,.2f}');c4.metric('R²',f'{R2:.3f}')
        bt['Error_Abs']=(bt.Actual-bt.Prediccion).abs()
        bt['Variacion_%']=np.where(bt.Actual>0,100*(bt.Prediccion-bt.Actual)/bt.Actual,np.nan)
        st.dataframe(bt.nlargest(20,'Error_Abs'),use_container_width=True,hide_index=True)

        raw,log,med=entrenar(d); ultimo=int(d.Periodo.max()); f=pd.Timestamp(ultimo//100,ultimo%100,1)
        periodos=[]
        for i in (1,2):
            x=f+pd.DateOffset(months=i);periodos.append(x.year*100+x.month)
        ult=d.groupby('CODIGO',group_keys=False).tail(1); filas=[]
        for _,r in ult.iterrows():
            h=d[d.CODIGO==r.CODIGO];ps=predecir(raw,log,med,h,periodos,alpha)
            filas.append({'CODIGO':r.CODIGO,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),
                          f'Pred {periodos[0]}':ps[0],f'Pred {periodos[1]}':ps[1]})
        res=pd.DataFrame(filas)
        st.subheader('Proyeccion masiva');st.dataframe(res,use_container_width=True,hide_index=True)
        st.download_button('Descargar CSV',res.to_csv(index=False).encode('utf-8-sig'),'proyeccion_v2.csv','text/csv')
    except Exception as e:
        st.exception(e)


