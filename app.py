import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title='Prediccion mensual de ventas', page_icon='📦', layout='wide')
st.title('📦 Simulador predictivo mensual optimizado')
st.caption('Modelo global Extra Trees sobre log(venta), estacionalidad, rezagos y promedios moviles.')

MESES = {'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,
         'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
FEATURES = ['lag1','lag2','lag3','lag4','lag6','lag12','mean3','std3','mean6','std6',
            'mean12','std12','sin_mes','cos_mes']


def completar_calendario(g):
    salida=[]
    meta=[c for c in ['DESCRIPCION','CATEGORIA','FAMILIA','TIPO'] if c in g.columns]
    for codigo,x in g.groupby('CODIGO'):
        x=x.sort_values('Periodo').copy()
        fechas=pd.to_datetime(x['AÑO'].astype(int).astype(str)+'-'+x['MES_NUM'].astype(int).astype(str)+'-01')
        x.index=fechas
        idx=pd.date_range(fechas.min(),fechas.max(),freq='MS')
        x=x.reindex(idx)
        x['CODIGO']=codigo
        # Si una hoja mensual existe pero el SKU no aparece, se interpreta como venta 0.
        x['VENTA']=x['VENTA'].fillna(0.0)
        x['MES_NUM']=x.index.month
        x['AÑO']=x.index.year
        x['Periodo']=x.index.year*100+x.index.month
        for c in meta: x[c]=x[c].ffill().bfill()
        salida.append(x.reset_index(drop=True))
    return pd.concat(salida,ignore_index=True)


def crear_features(df):
    z=df.sort_values(['CODIGO','Periodo']).copy()
    grp=z.groupby('CODIGO')['VENTA']
    for lag in [1,2,3,4,6,12]: z[f'lag{lag}']=grp.shift(lag)
    for w in [3,6,12]:
        z[f'mean{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).mean())
        z[f'std{w}']=grp.transform(lambda s:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z['sin_mes']=np.sin(2*np.pi*z['MES_NUM']/12)
    z['cos_mes']=np.cos(2*np.pi*z['MES_NUM']/12)
    return z


def nuevo_modelo():
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=1.0,
                               n_jobs=-1, random_state=42)


def entrenar(df):
    f=crear_features(df).dropna(subset=['lag1','lag2','VENTA']).copy()
    medianas=f[FEATURES].median().fillna(0)
    X=f[FEATURES].fillna(medianas)
    modelo=nuevo_modelo()
    # Log1p disminuye la dominancia de productos millonarios y mejora WAPE/MAE.
    modelo.fit(X,np.log1p(f['VENTA'].clip(lower=0)))
    return modelo,medianas


def vector_futuro(hist, mes_num, medianas):
    vals=hist.sort_values('Periodo')['VENTA'].astype(float).tolist()
    q={f'lag{lag}': vals[-lag] if len(vals)>=lag else np.nan for lag in [1,2,3,4,6,12]}
    for w in [3,6,12]:
        a=np.asarray(vals[-w:],dtype=float)
        q[f'mean{w}']=a.mean() if len(a)>=2 else np.nan
        q[f'std{w}']=a.std(ddof=1) if len(a)>=2 else 0.0
    q['sin_mes']=np.sin(2*np.pi*mes_num/12)
    q['cos_mes']=np.cos(2*np.pi*mes_num/12)
    return pd.DataFrame([q],columns=FEATURES).fillna(medianas)


def predecir_recursivo(modelo, medianas, historico, periodos):
    hist=historico.copy(); preds=[]
    for periodo in periodos:
        mes=periodo%100
        p=float(np.expm1(modelo.predict(vector_futuro(hist,mes,medianas))[0]))
        p=max(0.0,p); preds.append(p)
        base=hist.iloc[-1].copy(); base['Periodo']=periodo;base['AÑO']=periodo//100;base['MES_NUM']=mes;base['VENTA']=p
        hist=pd.concat([hist,pd.DataFrame([base])],ignore_index=True)
    return preds

@st.cache_data(show_spinner='Leyendo y normalizando archivos...')
def cargar(archivos):
    partes=[]
    for archivo in archivos:
        for _,h in pd.read_excel(archivo,sheet_name=None,engine='openpyxl').items(): partes.append(h)
    df=pd.concat(partes,ignore_index=True); df.columns=[str(c).strip().upper() for c in df.columns]
    oblig=['CODIGO','VENTA','MES','AÑO']
    falt=[c for c in oblig if c not in df.columns]
    if falt: raise ValueError('Faltan columnas: '+', '.join(falt))
    df['MES_NUM']=df['MES'].astype(str).str.strip().str.upper().map(MESES)
    df['AÑO']=pd.to_numeric(df['AÑO'],errors='coerce');df['VENTA']=pd.to_numeric(df['VENTA'],errors='coerce')
    # Mantener CODIGO como texto evita perdida de precision/notacion cientifica.
    df['CODIGO']=df['CODIGO'].astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    df=df.dropna(subset=['MES_NUM','AÑO','VENTA']);df['Periodo']=(df['AÑO']*100+df['MES_NUM']).astype(int)
    meta={c:'first' for c in ['DESCRIPCION','CATEGORIA','FAMILIA','TIPO'] if c in df.columns}
    g=df.groupby(['CODIGO','Periodo','MES_NUM','AÑO'],as_index=False).agg({'VENTA':'sum',**meta})
    return completar_calendario(g).sort_values(['CODIGO','Periodo']).reset_index(drop=True)


def backtest(df,n_holdout=2):
    tamanos=df.groupby('CODIGO').size(); codigos=tamanos[tamanos>=n_holdout+3].index
    base=df[df.CODIGO.isin(codigos)].copy()
    held=base.groupby('CODIGO',group_keys=False).tail(n_holdout)
    train=base.drop(index=held.index)
    modelo,medianas=entrenar(train)
    filas=[]
    for codigo,real in held.groupby('CODIGO'):
        real=real.sort_values('Periodo'); hist=train[train.CODIGO==codigo].sort_values('Periodo')
        pred=predecir_recursivo(modelo,medianas,hist,real.Periodo.astype(int).tolist())
        for paso,((_,r),p) in enumerate(zip(real.iterrows(),pred),1):
            filas.append({'CODIGO':codigo,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),
                          'Periodo_Real':int(r.Periodo),'Mes_Predecido':paso,'Prediccion':p,'Actual':float(r.VENTA)})
    return pd.DataFrame(filas)

archivos=st.file_uploader('Sube ventas 2024, 2025 y 2026',type=['xlsx','xlsm'],accept_multiple_files=True)
if archivos:
    try:
        df=cargar(archivos)
        st.success(f'{len(df):,} observaciones mensuales normalizadas | {df.CODIGO.nunique():,} productos')
        holdout=st.slider('Meses de backtest',1,4,2)
        with st.spinner('Entrenando y validando...'):
            bt=backtest(df,holdout)
            y=bt.Actual.to_numpy();p=bt.Prediccion.to_numpy()
            wape=100*np.abs(y-p).sum()/max(np.abs(y).sum(),1)
            c1,c2,c3,c4=st.columns(4)
            c1.metric('WAPE',f'{wape:.2f}%');c2.metric('MAE',f'${mean_absolute_error(y,p):,.2f}')
            c3.metric('RMSE',f'${mean_squared_error(y,p)**0.5:,.2f}');c4.metric('R²',f'{r2_score(y,p):.3f}')
            bt['Error_Abs']=(bt.Prediccion-bt.Actual).abs()
            bt['Variacion_%']=np.where(bt.Actual>0,100*(bt.Prediccion-bt.Actual)/bt.Actual,np.nan)
            st.dataframe(bt.nlargest(10,'Error_Abs'),use_container_width=True,hide_index=True)

        modelo,medianas=entrenar(df)
        ult=df.sort_values('Periodo').groupby('CODIGO').tail(1)
        opciones={r.CODIGO:f'{r.CODIGO} - {r.get("DESCRIPCION","")}' for _,r in ult.iterrows()}
        codigo=st.selectbox('Producto',list(opciones),format_func=lambda x:opciones[x])
        hist=df[df.CODIGO==codigo].sort_values('Periodo')
        ultimo=int(df.Periodo.max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
        periodos=[]
        for i in [1,2]:
            f=fecha+pd.DateOffset(months=i);periodos.append(f.year*100+f.month)
        pred=predecir_recursivo(modelo,medianas,hist,periodos)
        st.write(f'**{periodos[0]}:** ${pred[0]:,.2f} | **{periodos[1]}:** ${pred[1]:,.2f}')
        graf=pd.concat([hist[['Periodo','VENTA']].assign(Tipo='Real'),pd.DataFrame({'Periodo':periodos,'VENTA':pred,'Tipo':'Proyectado'})])
        ch=alt.Chart(graf).mark_line(point=True).encode(x=alt.X('Periodo:O'),y=alt.Y('VENTA:Q'),color='Tipo:N',tooltip=['Periodo','VENTA','Tipo']).properties(height=350)
        st.altair_chart(ch,use_container_width=True)

        filas=[]
        for _,r in ult.iterrows():
            h=df[df.CODIGO==r.CODIGO]; pr=predecir_recursivo(modelo,medianas,h,periodos)
            filas.append({'CODIGO':r.CODIGO,'DESCRIPCION':r.get('DESCRIPCION',''),'CATEGORIA':r.get('CATEGORIA',''),
                          f'Pred {periodos[0]}':pr[0],f'Pred {periodos[1]}':pr[1]})
        resultado=pd.DataFrame(filas)
        st.subheader('Proyeccion masiva');st.dataframe(resultado,use_container_width=True,hide_index=True)
        st.download_button('Descargar proyeccion CSV',resultado.to_csv(index=False).encode('utf-8-sig'),'proyeccion_ventas.csv','text/csv')
    except Exception as e:
        st.exception(e)

