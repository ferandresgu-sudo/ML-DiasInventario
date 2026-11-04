import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Prediccion V7 RMSE", page_icon="📦", layout="wide")
st.title("📦 Prediccion V7: optimizacion automatica de RMSE")
st.caption("Busqueda rapida de hiperparametros, segmentador de categoria, cache y entrenamiento bajo demanda.")

MESES={"ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
       "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12}
LAGS=[1,2,3,4,6,12]
VENTANAS=[3,6,12]
FEATURES=([f"lag{x}" for x in LAGS]+[f"mean{x}" for x in VENTANAS]
          +[f"median{x}" for x in VENTANAS]+[f"std{x}" for x in VENTANAS]
          +["sin_mes","cos_mes","ceros6","cv6"])

# Cuatro candidatos pequenos para que la seleccion automatica no tarde demasiado.
CANDIDATOS=[
    {"max_depth":12,"min_samples_leaf":4,"max_features":0.70},
    {"max_depth":14,"min_samples_leaf":4,"max_features":0.80},
    {"max_depth":16,"min_samples_leaf":3,"max_features":0.80},
    {"max_depth":20,"min_samples_leaf":2,"max_features":1.00},
]


def limpiar_columna(c):
    s=str(c).replace("\ufeff","").replace("\xa0"," ").strip().upper()
    s=unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+","_",s).strip("_")


def normalizar_columnas(df):
    df=df.copy(); df.columns=[limpiar_columna(c) for c in df.columns]
    alias={"COD_PRODUCTO":"CODIGO","CODIGO_PRODUCTO":"CODIGO","COD_ARTICULO":"CODIGO",
           "SKU":"CODIGO","MATERIAL":"CODIGO","VENTAS":"VENTA","VENTA_NETA":"VENTA",
           "MONTO_VENTA":"VENTA","IMPORTE":"VENTA","ANO":"AÑO","ANIO":"AÑO",
           "YEAR":"AÑO","MONTH":"MES","CATEGORY":"CATEGORIA","CAT":"CATEGORIA"}
    df=df.rename(columns={c:alias.get(c,c) for c in df.columns})
    if df.columns.duplicated().any():
        unido={}
        for c in dict.fromkeys(df.columns):
            unido[c]=df.loc[:,df.columns==c].bfill(axis=1).iloc[:,0]
        df=pd.DataFrame(unido,index=df.index)
    return df


def fila_encabezado(raw):
    for i in range(min(15,len(raw))):
        vals={limpiar_columna(v) for v in raw.iloc[i].dropna().tolist()}
        vals={"CODIGO" if v in {"SKU","MATERIAL","COD_PRODUCTO","CODIGO_PRODUCTO","COD_ARTICULO"} else v for v in vals}
        vals={"VENTA" if v in {"VENTAS","VENTA_NETA","MONTO_VENTA","IMPORTE"} else v for v in vals}
        if {"CODIGO","VENTA"}.issubset(vals): return i
    return 0


@st.cache_data(show_spinner="Leyendo archivos...")
def cargar(archivos):
    partes=[]; omitidas=[]
    for archivo in archivos:
        excel=pd.ExcelFile(archivo,engine="openpyxl")
        for hoja in excel.sheet_names:
            raw=pd.read_excel(archivo,sheet_name=hoja,header=None,engine="openpyxl")
            if raw.empty: continue
            i=fila_encabezado(raw); h=raw.iloc[i+1:].copy(); h.columns=raw.iloc[i].tolist()
            h=normalizar_columnas(h.dropna(how="all"))
            faltan=[c for c in ["CODIGO","VENTA","MES","AÑO","CATEGORIA"] if c not in h]
            if faltan:
                omitidas.append(f"{archivo.name}/{hoja}: faltan {faltan}")
                continue
            partes.append(h)
    if not partes: raise ValueError("No hay hojas validas con CODIGO, VENTA, MES, AÑO y CATEGORIA.")
    df=pd.concat(partes,ignore_index=True,sort=False)
    df["MES_NUM"]=df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["MES_NUM"]=df["MES_NUM"].fillna(pd.to_numeric(df["MES"],errors="coerce"))
    df["AÑO"]=pd.to_numeric(df["AÑO"],errors="coerce")
    df["VENTA"]=pd.to_numeric(df["VENTA"],errors="coerce").clip(lower=0)
    df["CODIGO"]=df["CODIGO"].astype(str).str.strip().str.replace(r"\.0$","",regex=True)
    df["CATEGORIA"]=df["CATEGORIA"].astype(str).str.strip()
    df=df.dropna(subset=["MES_NUM","AÑO","VENTA"])
    df=df[~df["CODIGO"].isin(["","nan","None"])]
    df=df[df["MES_NUM"].between(1,12)]
    df["Periodo"]=(df["AÑO"].astype(int)*100+df["MES_NUM"].astype(int)).astype(int)
    meta={c:"first" for c in ["DESCRIPCION","CATEGORIA","FAMILIA","TIPO"] if c in df}
    g=df.groupby(["CODIGO","Periodo","MES_NUM","AÑO"],as_index=False).agg({"VENTA":"sum",**meta})
    salida=[]
    for code,x in g.groupby("CODIGO",sort=False):
        x=x.sort_values("Periodo").copy()
        fechas=pd.to_datetime(x["AÑO"].astype(int).astype(str)+"-"+x["MES_NUM"].astype(int).astype(str)+"-01")
        x.index=fechas; x=x.reindex(pd.date_range(fechas.min(),fechas.max(),freq="MS"))
        x["CODIGO"]=code; x["VENTA"]=x["VENTA"].fillna(0.0)
        x["MES_NUM"]=x.index.month; x["AÑO"]=x.index.year; x["Periodo"]=x.index.year*100+x.index.month
        for c in meta: x[c]=x[c].ffill().bfill()
        salida.append(x.reset_index(drop=True))
    return pd.concat(salida,ignore_index=True).sort_values(["CODIGO","Periodo"]).reset_index(drop=True),omitidas


@st.cache_data(show_spinner=False)
def crear_features(df):
    z=df.sort_values(["CODIGO","Periodo"]).copy(); grp=z.groupby("CODIGO",sort=False)["VENTA"]
    for lag in LAGS: z[f"lag{lag}"]=grp.shift(lag)
    for w in VENTANAS:
        z[f"mean{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).mean())
        z[f"median{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).median())
        z[f"std{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z["sin_mes"]=np.sin(2*np.pi*z["MES_NUM"]/12); z["cos_mes"]=np.cos(2*np.pi*z["MES_NUM"]/12)
    z["ceros6"]=grp.transform(lambda s:s.shift(1).rolling(6,min_periods=2).apply(lambda a:(a==0).mean(),raw=True))
    z["cv6"]=(z["std6"]/z["mean6"].replace(0,np.nan)).clip(0,10)
    return z


def preparar(df):
    tamanos=df.groupby("CODIGO").size(); codigos=tamanos[tamanos>=5].index
    base=df[df["CODIGO"].isin(codigos)].copy()
    held=base.groupby("CODIGO",group_keys=False).tail(2); train=base.drop(index=held.index)
    f=crear_features(train).dropna(subset=["lag1","lag2","VENTA"]).copy()
    medianas=f[FEATURES].median().fillna(0)
    return train,held,f[FEATURES].fillna(medianas).astype("float32"),f["VENTA"].astype(float),medianas


def nuevo_modelo(config,n_arboles):
    return ExtraTreesRegressor(n_estimators=n_arboles,max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],max_features=config["max_features"],
        n_jobs=-1,random_state=42)


def vector(hist,mes,medianas):
    v=hist.sort_values("Periodo")["VENTA"].to_numpy(float); d={}
    for lag in LAGS: d[f"lag{lag}"]=v[-lag] if len(v)>=lag else np.nan
    for w in VENTANAS:
        a=v[-w:]; d[f"mean{w}"]=a.mean() if len(a)>=2 else np.nan
        d[f"median{w}"]=np.median(a) if len(a)>=2 else np.nan
        d[f"std{w}"]=a.std(ddof=1) if len(a)>=2 else 0
    d["sin_mes"]=np.sin(2*np.pi*mes/12); d["cos_mes"]=np.cos(2*np.pi*mes/12)
    a6=v[-6:]; d["ceros6"]=np.mean(a6==0) if len(a6)>=2 else 0
    media=a6.mean() if len(a6) else 0; desv=a6.std(ddof=1) if len(a6)>=2 else 0
    d["cv6"]=min(desv/media,10) if media>0 else np.nan
    return pd.DataFrame([d],columns=FEATURES).fillna(medianas).astype("float32")


def predecir(modelo,medianas,hist,periodos,mezcla_mediana=0.15):
    h=hist.copy(); pred=[]
    for periodo in periodos:
        p_ml=max(0,float(np.expm1(modelo.predict(vector(h,periodo%100,medianas))[0])))
        med6=float(h.sort_values("Periodo")["VENTA"].tail(6).median())
        p=(1-mezcla_mediana)*p_ml+mezcla_mediana*med6
        pred.append(max(0,p)); r=h.iloc[-1].copy(); r["Periodo"]=periodo; r["AÑO"]=periodo//100; r["MES_NUM"]=periodo%100; r["VENTA"]=p
        h=pd.concat([h,pd.DataFrame([r])],ignore_index=True)
    return pred


def evaluar(modelo,medianas,train,held,mezcla):
    historicos={c:g for c,g in train.groupby("CODIGO",sort=False)}; filas=[]
    for code,reales in held.groupby("CODIGO",sort=False):
        if code not in historicos: continue
        reales=reales.sort_values("Periodo"); ps=predecir(modelo,medianas,historicos[code],reales["Periodo"].astype(int).tolist(),mezcla)
        for (_,r),p in zip(reales.iterrows(),ps):
            filas.append({"CODIGO":code,"DESCRIPCION":r.get("DESCRIPCION",""),"CATEGORIA":r.get("CATEGORIA",""),
                          "Periodo":int(r["Periodo"]),"Actual":float(r["VENTA"]),"Prediccion":p})
    return pd.DataFrame(filas)


def metricas(r):
    y=r["Actual"].to_numpy(float); p=r["Prediccion"].to_numpy(float)
    return {"WAPE":100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),"MAE":mean_absolute_error(y,p),
            "RMSE":np.sqrt(mean_squared_error(y,p)),"R2":r2_score(y,p)}


def optimizar_rmse(df,arboles_busqueda):
    train,held,X,y,medianas=preparar(df); resultados=[]; mejor=None
    total=len(CANDIDATOS)*3; avance=st.progress(0); paso=0
    for config in CANDIDATOS:
        modelo=nuevo_modelo(config,arboles_busqueda); modelo.fit(X,np.log1p(y.clip(lower=0)))
        for mezcla in [0.0,0.10,0.20]:
            bt=evaluar(modelo,medianas,train,held,mezcla); m=metricas(bt)
            fila={**config,"mezcla_mediana":mezcla,**m}; resultados.append(fila)
            if mejor is None or m["RMSE"]<mejor[0]: mejor=(m["RMSE"],config.copy(),mezcla,bt)
            paso+=1; avance.progress(paso/total)
    avance.empty()
    return pd.DataFrame(resultados).sort_values("RMSE"),mejor,train,held


archivos=st.file_uploader("Sube ventas 2024, 2025 y 2026",type=["xlsx","xlsm"],accept_multiple_files=True)
with st.sidebar.form("opciones"):
    arboles_busqueda=st.select_slider("Arboles durante optimizacion",[50,75,100,125],value=75)
    arboles_final=st.select_slider("Arboles del modelo final",[100,150,200,250,300],value=180 if 180 in [100,150,200,250,300] else 200)
    st.form_submit_button("Aplicar")

if archivos:
    try:
        total,omitidas=cargar(archivos)
        categorias=sorted(total["CATEGORIA"].dropna().astype(str).unique())
        categoria=st.selectbox("CATEGORIA para optimizar y entrenar",["Todas"]+categorias)
        df=total.copy() if categoria=="Todas" else total[total["CATEGORIA"].astype(str)==categoria].copy()
        firma=(tuple((a.name,a.size) for a in archivos),categoria,arboles_busqueda,arboles_final,len(df))
        st.success(f"Categoria: {categoria} | {df['CODIGO'].nunique():,} productos | {len(df):,} registros")
        if omitidas:
            with st.expander("Hojas omitidas"): [st.warning(x) for x in omitidas]
        for k,v in [("v7_resultados",None),("v7_mejor",None),("v7_firma",None),("v7_modelo",None),("v7_firma_modelo",None)]:
            if k not in st.session_state: st.session_state[k]=v

        if st.button("Optimizar automaticamente RMSE",type="primary"):
            with st.spinner("Probando 12 combinaciones rapidas..."):
                resultados,mejor,_,_=optimizar_rmse(df,arboles_busqueda)
                st.session_state.v7_resultados=resultados; st.session_state.v7_mejor=mejor; st.session_state.v7_firma=firma

        if st.session_state.v7_resultados is not None and st.session_state.v7_firma==firma:
            resultados=st.session_state.v7_resultados; mejor=st.session_state.v7_mejor
            st.subheader("Resultados de optimizacion")
            st.dataframe(resultados.style.format({"WAPE":"{:.2f}%","MAE":"${:,.2f}","RMSE":"${:,.2f}","R2":"{:.3f}"}),use_container_width=True,hide_index=True)
            bt=mejor[3].copy(); m=metricas(bt); a,b,c,d=st.columns(4)
            a.metric("WAPE",f"{m['WAPE']:.2f}%"); b.metric("MAE",f"${m['MAE']:,.2f}"); c.metric("RMSE",f"${m['RMSE']:,.2f}"); d.metric("R²",f"{m['R2']:.3f}")
            st.info(f"Mejor configuracion RMSE: profundidad={mejor[1]['max_depth']}, hoja={mejor[1]['min_samples_leaf']}, variables={mejor[1]['max_features']}, mezcla mediana={mejor[2]:.0%}")
            bt["Error_Abs"]=(bt.Actual-bt.Prediccion).abs(); st.dataframe(bt.nlargest(15,"Error_Abs"),use_container_width=True,hide_index=True)

            if st.button("Entrenar modelo final con la mejor configuracion"):
                with st.spinner("Entrenando modelo final una sola vez..."):
                    f=crear_features(df).dropna(subset=["lag1","lag2","VENTA"]).copy(); med=f[FEATURES].median().fillna(0)
                    modelo=nuevo_modelo(mejor[1],arboles_final); modelo.fit(f[FEATURES].fillna(med).astype("float32"),np.log1p(f.VENTA.clip(lower=0)))
                    st.session_state.v7_modelo=(modelo,med,mejor[2]); st.session_state.v7_firma_modelo=firma

        if st.session_state.v7_modelo is not None and st.session_state.v7_firma_modelo==firma:
            modelo,med,mezcla=st.session_state.v7_modelo
            historicos={c:g.sort_values("Periodo") for c,g in df.groupby("CODIGO",sort=False)}
            ultimo=int(df.Periodo.max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
            periodos=[(fecha+pd.DateOffset(months=i)).year*100+(fecha+pd.DateOffset(months=i)).month for i in [1,2]]
            if st.button("Generar proyeccion masiva"):
                filas=[]
                with st.spinner("Prediciendo sin reentrenar..."):
                    for code,h in historicos.items():
                        ps=predecir(modelo,med,h,periodos,mezcla); r=h.iloc[-1]
                        filas.append({"CODIGO":code,"DESCRIPCION":r.get("DESCRIPCION",""),"CATEGORIA":r.get("CATEGORIA",""),
                                      f"Pred_{periodos[0]}":ps[0],f"Pred_{periodos[1]}":ps[1]})
                salida=pd.DataFrame(filas); st.dataframe(salida,use_container_width=True,hide_index=True)
                nombre="todas" if categoria=="Todas" else re.sub(r"[^A-Za-z0-9]+","_",categoria)
                st.download_button("Descargar proyeccion",salida.to_csv(index=False).encode("utf-8-sig"),f"prediccion_v7_{nombre}.csv","text/csv")
        elif st.session_state.v7_modelo is not None:
            st.warning("La categoria o configuracion cambio. Optimiza y entrena nuevamente.")
    except Exception as e:
        st.exception(e)



