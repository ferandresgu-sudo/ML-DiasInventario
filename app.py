import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title="Prediccion Nov-Dic 2026 V4", page_icon="📦", layout="wide")
st.title("📦 Prediccion optimizada Noviembre-Diciembre 2026")
st.caption("Modelos directos H+1/H+2 con peso estacional para noviembre y diciembre de 2024 y 2025.")

MESES = {"ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
         "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12}
LAGS = [1,2,3,4,6,12]
VENTANAS = [3,6,12]
FEATURES = ([f"lag{x}" for x in LAGS]
            + [f"mean{x}" for x in VENTANAS]
            + [f"median{x}" for x in VENTANAS]
            + [f"std{x}" for x in VENTANAS]
            + ["venta_mismo_mes_anio_anterior","ratio_lag1_mediana6","ceros6","cv6",
               "es_promocion","sin_mes_obj","cos_mes_obj"])


def detectar_promocion(df):
    desc=df.get("DESCRIPCION",pd.Series("",index=df.index)).fillna("").astype(str)
    tipo=df.get("TIPO",pd.Series("",index=df.index)).fillna("").astype(str)
    return (desc+" "+tipo).str.upper().str.contains(
        r"PACK|TRIPACK|COMBO|PROMO|GRATIS|BONUS|\+|IN ?& ?OUT|IN ?OUT",regex=True,na=False
    ).astype(np.int8)


@st.cache_data(show_spinner="Leyendo archivos una sola vez...")
def cargar_archivos(archivos):
    partes=[]
    for archivo in archivos:
        for hoja in pd.read_excel(archivo,sheet_name=None,engine="openpyxl").values():
            if hoja is not None and not hoja.empty: partes.append(hoja)
    if not partes: raise ValueError("No se encontraron datos.")
    df=pd.concat(partes,ignore_index=True)
    df.columns=[str(c).strip().upper() for c in df.columns]
    requeridas=["CODIGO","VENTA","MES","AÑO"]
    faltan=[c for c in requeridas if c not in df.columns]
    if faltan: raise ValueError("Faltan columnas: "+", ".join(faltan))
    df["MES_NUM"]=df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"]=pd.to_numeric(df["AÑO"],errors="coerce")
    df["VENTA"]=pd.to_numeric(df["VENTA"],errors="coerce")
    df["CODIGO"]=df["CODIGO"].astype(str).str.strip().str.replace(r"\.0$","",regex=True)
    df=df.dropna(subset=["MES_NUM","AÑO","VENTA"])
    df=df[df["CODIGO"].ne("") & df["CODIGO"].ne("nan")]
    df["Periodo"]=(df["AÑO"]*100+df["MES_NUM"]).astype(int)
    meta={c:"first" for c in ["DESCRIPCION","CATEGORIA","FAMILIA","TIPO"] if c in df.columns}
    g=df.groupby(["CODIGO","Periodo","MES_NUM","AÑO"],as_index=False).agg({"VENTA":"sum",**meta})
    salida=[]
    for codigo,x in g.groupby("CODIGO",sort=False):
        x=x.sort_values("Periodo").copy()
        fechas=pd.to_datetime(x["AÑO"].astype(int).astype(str)+"-"+x["MES_NUM"].astype(int).astype(str)+"-01")
        x.index=fechas
        x=x.reindex(pd.date_range(fechas.min(),fechas.max(),freq="MS"))
        x["CODIGO"]=codigo; x["VENTA"]=x["VENTA"].fillna(0.0)
        x["MES_NUM"]=x.index.month.astype(np.int8); x["AÑO"]=x.index.year.astype(np.int16)
        x["Periodo"]=(x.index.year*100+x.index.month).astype(np.int32)
        for c in meta: x[c]=x[c].ffill().bfill()
        salida.append(x.reset_index(drop=True))
    g=pd.concat(salida,ignore_index=True).sort_values(["CODIGO","Periodo"]).reset_index(drop=True)
    g["ES_PROMOCION"]=detectar_promocion(g)
    g["VENTA"]=g["VENTA"].astype(np.float32)
    return g


def construir_matriz(df,horizonte):
    z=df.sort_values(["CODIGO","Periodo"]).copy()
    grp=z.groupby("CODIGO",sort=False)["VENTA"]
    for lag in LAGS: z[f"lag{lag}"]=grp.shift(lag)
    for w in VENTANAS:
        z[f"mean{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).mean())
        z[f"median{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).median())
        z[f"std{w}"]=grp.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=2).std()).fillna(0)
    z["venta_mismo_mes_anio_anterior"]=z["lag12"]
    z["ratio_lag1_mediana6"]=(z["lag1"]/z["median6"].replace(0,np.nan)).clip(0,10)
    z["ceros6"]=grp.transform(lambda s:s.shift(1).rolling(6,min_periods=3).apply(lambda x:np.sum(x==0),raw=True))
    z["cv6"]=(z["std6"]/z["mean6"].replace(0,np.nan)).clip(0,10)
    z["es_promocion"]=z["ES_PROMOCION"].fillna(0)
    z["TARGET"]=grp.shift(-horizonte)
    z["MES_OBJ"]=z.groupby("CODIGO",sort=False)["MES_NUM"].shift(-horizonte)
    z["ANIO_OBJ"]=z.groupby("CODIGO",sort=False)["AÑO"].shift(-horizonte)
    z["sin_mes_obj"]=np.sin(2*np.pi*z["MES_OBJ"]/12)
    z["cos_mes_obj"]=np.cos(2*np.pi*z["MES_OBJ"]/12)
    return z


def peso_estacional(mes_obj,anio_obj,peso_nov_dic):
    # Se da mayor peso SOLO a los objetivos Nov-Dic 2024/2025, no a todas las filas del año.
    especial=mes_obj.isin([11,12]) & anio_obj.isin([2024,2025])
    return np.where(especial,peso_nov_dic,1.0).astype(np.float32)


def entrenar_modelo(df,horizonte,cfg):
    f=construir_matriz(df,horizonte).dropna(subset=["lag1","lag2","TARGET","MES_OBJ","ANIO_OBJ"]).copy()
    medianas=f[FEATURES].median().fillna(0)
    X=f[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(medianas).astype(np.float32)
    y=np.log1p(f["TARGET"].clip(lower=0).astype(np.float32))
    pesos=peso_estacional(f["MES_OBJ"],f["ANIO_OBJ"],cfg["peso_nov_dic"])
    modelo=ExtraTreesRegressor(n_estimators=cfg["arboles"],max_depth=cfg["profundidad"],
        min_samples_leaf=cfg["min_hoja"],max_features=cfg["max_features"],
        n_jobs=-1,random_state=42)
    modelo.fit(X,y,sample_weight=pesos)
    return modelo,medianas


def vector_rapido(hist,horizonte,medianas):
    hist=hist.sort_values("Periodo"); ventas=hist["VENTA"].to_numpy(dtype=np.float32)
    ultimo=hist.iloc[-1]; d={}
    for lag in LAGS: d[f"lag{lag}"]=float(ventas[-lag]) if len(ventas)>=lag else np.nan
    for w in VENTANAS:
        a=ventas[-w:]
        d[f"mean{w}"]=float(np.mean(a)) if len(a)>=2 else np.nan
        d[f"median{w}"]=float(np.median(a)) if len(a)>=2 else np.nan
        d[f"std{w}"]=float(np.std(a,ddof=1)) if len(a)>=2 else 0.0
    d["venta_mismo_mes_anio_anterior"]=float(ventas[-12]) if len(ventas)>=12 else np.nan
    med6=d["median6"]; mean6=d["mean6"]
    d["ratio_lag1_mediana6"]=min(float(ventas[-1])/med6,10) if pd.notna(med6) and med6>0 else np.nan
    a6=ventas[-6:]; d["ceros6"]=float(np.sum(a6==0))
    d["cv6"]=min(d["std6"]/mean6,10) if pd.notna(mean6) and mean6>0 else np.nan
    d["es_promocion"]=int(ultimo.get("ES_PROMOCION",0))
    mes_obj=((int(ultimo["MES_NUM"])-1+horizonte)%12)+1
    d["sin_mes_obj"]=np.sin(2*np.pi*mes_obj/12); d["cos_mes_obj"]=np.cos(2*np.pi*mes_obj/12)
    return pd.DataFrame([d],columns=FEATURES).replace([np.inf,-np.inf],np.nan).fillna(medianas).astype(np.float32)


def pred_producto(modelo,medianas,hist,horizonte,ajustar):
    p=max(0,float(np.expm1(modelo.predict(vector_rapido(hist,horizonte,medianas))[0])))
    ventas=hist.sort_values("Periodo")["VENTA"].tail(12).astype(float)
    # Productos intermitentes: pronostico conservador.
    if len(ventas)>=6 and int((ventas.tail(6)==0).sum())>=3:
        pos=ventas[ventas>0]
        return float((ventas>0).mean()*pos.median()) if not pos.empty else 0.0,"Intermitente"
    # Limite solamente para promocionales con pico reciente.
    if ajustar and int(hist.iloc[-1].get("ES_PROMOCION",0))==1 and len(ventas)>=6:
        med6=float(ventas.tail(6).median()); ultimo=float(ventas.iloc[-1])
        if med6>0 and ultimo>1.8*med6: p=min(p,max(1.4*med6,float(ventas.quantile(.90))))
    return p,"Modelo"


def metricas(r):
    y=r.Actual.to_numpy(float); p=r.Prediccion.to_numpy(float)
    return {"WAPE":100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),
            "MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**.5,
            "R2":r2_score(y,p) if len(y)>1 else np.nan}


def backtest_nov_dic(df,anio,cfg):
    # Simula exactamente el escenario real: entrena hasta octubre y predice Nov-Dic.
    corte=anio*100+10; train=df[df.Periodo<=corte].copy(); reales=df[df.Periodo.isin([anio*100+11,anio*100+12])]
    modelos={h:entrenar_modelo(train,h,cfg) for h in [1,2]}; hist={c:g for c,g in train.groupby("CODIGO",sort=False)}
    filas=[]
    for codigo,h in hist.items():
        for horizonte,periodo in [(1,anio*100+11),(2,anio*100+12)]:
            real=reales[(reales.CODIGO==codigo)&(reales.Periodo==periodo)]
            if real.empty: continue
            p,met=pred_producto(*modelos[horizonte],h,horizonte,cfg["ajuste"])
            r=real.iloc[0]; filas.append({"CODIGO":codigo,"DESCRIPCION":r.get("DESCRIPCION",""),
                "Periodo":periodo,"Actual":float(r.VENTA),"Prediccion":p,"Metodo":met})
    return pd.DataFrame(filas)

# --------------------------- INTERFAZ ---------------------------
archivos=st.file_uploader("Sube ventas 2024, 2025 y 2026",type=["xlsx","xlsm"],accept_multiple_files=True)
with st.sidebar.form("config"):
    st.header("Configuracion rapida")
    arboles=st.select_slider("Arboles",[100,150,200,250,300,400],value=250)
    profundidad=st.selectbox("Profundidad",[10,12,14,16,20,None],index=2,format_func=lambda x:"Sin limite" if x is None else str(x))
    min_hoja=st.select_slider("Minimo por hoja",[2,3,4,5,8],value=3)
    max_features=st.select_slider("Variables por arbol",[0.6,0.7,0.8,0.9,1.0],value=0.8)
    peso=st.slider("Peso Nov-Dic 2024/2025",1.0,5.0,2.5,0.25)
    ajuste=st.checkbox("Control sobreestimacion promocional",True)
    aplicar=st.form_submit_button("Aplicar configuracion")

if archivos:
    try:
        df=cargar_archivos(archivos)
        cfg={"arboles":arboles,"profundidad":profundidad,"min_hoja":min_hoja,
             "max_features":max_features,"peso_nov_dic":peso,"ajuste":ajuste}
        firma=(tuple((a.name,a.size) for a in archivos),tuple(cfg.items()),int(df.Periodo.max()))
        st.success(f"{df.CODIGO.nunique():,} productos | {len(df):,} registros | Ultimo periodo {int(df.Periodo.max())}")
        st.info("Recomendado: 250 arboles y peso estacional 2.5. Un peso excesivo puede sobreajustar solo cuatro meses.")

        if "modelos_v4" not in st.session_state: st.session_state.modelos_v4=None
        if "firma_v4" not in st.session_state: st.session_state.firma_v4=None
        if "backtest_v4" not in st.session_state: st.session_state.backtest_v4=None

        c1,c2=st.columns(2)
        with c1:
            if st.button("Calcular metricas WAPE, MAE, RMSE y R2", type="primary"):
                resultados=[]
                with st.spinner("Ejecutando backtest estacional..."):
                    for anio in [2024,2025]:
                        r=backtest_nov_dic(df,anio,cfg)
                        if not r.empty: resultados.append(r.assign(Anio=anio))
                st.session_state.backtest_v4=pd.concat(resultados,ignore_index=True) if resultados else None
        with c2:
            if st.button("Entrenar modelo final 2026",type="primary"):
                with st.spinner("Entrenando H+1 y H+2..."):
                    st.session_state.modelos_v4={h:entrenar_modelo(df,h,cfg) for h in [1,2]}
                    st.session_state.firma_v4=firma

        st.subheader("Metricas de validacion Noviembre-Diciembre")
        if st.session_state.backtest_v4 is not None and not st.session_state.backtest_v4.empty:
            bt=st.session_state.backtest_v4.copy()
            m=metricas(bt)
            a,b,c,d=st.columns(4)
            a.metric("WAPE global",f"{m['WAPE']:.2f}%",help="Error absoluto total dividido entre venta real total")
            b.metric("MAE",f"${m['MAE']:,.2f}",help="Error absoluto promedio por producto-mes")
            c.metric("RMSE",f"${m['RMSE']:,.2f}",help="Penaliza con mayor fuerza los errores grandes")
            d.metric("R²",f"{m['R2']:.3f}",help="Proporcion de variabilidad explicada por el modelo")

            resumen=[]
            for (anio,periodo),g in bt.groupby(["Anio","Periodo"]):
                mm=metricas(g)
                resumen.append({"Año":int(anio),"Mes":"Noviembre" if int(periodo)%100==11 else "Diciembre",
                                "WAPE":mm["WAPE"],"MAE":mm["MAE"],"RMSE":mm["RMSE"],"R²":mm["R2"],
                                "Observaciones":len(g)})
            resumen_df=pd.DataFrame(resumen)
            st.markdown("#### Metricas por año y mes")
            st.dataframe(resumen_df.style.format({"WAPE":"{:.2f}%","MAE":"${:,.2f}",
                                                  "RMSE":"${:,.2f}","R²":"{:.3f}"}),
                         use_container_width=True,hide_index=True)

            bt["Error_Abs"]=(bt["Actual"]-bt["Prediccion"]).abs()
            total_real=max(bt["Actual"].abs().sum(),1)
            bt["Aporte_WAPE_pp"]=100*bt["Error_Abs"]/total_real
            bt["Error_%"]=np.where(bt["Actual"]>0,100*(bt["Prediccion"]-bt["Actual"])/bt["Actual"],np.nan)
            st.markdown("#### 15 productos-mes con mayor error")
            columnas=["Anio","Periodo","CODIGO","DESCRIPCION","Actual","Prediccion",
                      "Error_Abs","Error_%","Aporte_WAPE_pp","Metodo"]
            st.dataframe(bt.nlargest(15,"Error_Abs")[columnas].style.format({
                "Actual":"${:,.2f}","Prediccion":"${:,.2f}","Error_Abs":"${:,.2f}",
                "Error_%":"{:+.1f}%","Aporte_WAPE_pp":"{:.3f}"
            }),use_container_width=True,hide_index=True)
            st.download_button("Descargar detalle de metricas",
                bt.to_csv(index=False).encode("utf-8-sig"),
                "metricas_backtest_nov_dic_v4_1.csv","text/csv")
        else:
            st.info("Presiona 'Calcular metricas WAPE, MAE, RMSE y R2'. El sistema entrenara hasta octubre de 2024 y 2025 y evaluara noviembre y diciembre de cada año.")

        if st.session_state.modelos_v4 is not None:
            if st.session_state.firma_v4!=firma:
                st.warning("La configuracion cambio. Entrena nuevamente el modelo final.")
            else:
                modelos=st.session_state.modelos_v4
                historicos={c:g.sort_values("Periodo") for c,g in df.groupby("CODIGO",sort=False)}
                ult=df.sort_values("Periodo").groupby("CODIGO").tail(1)
                opciones={r.CODIGO:f"{r.CODIGO} - {r.get('DESCRIPCION','')}" for _,r in ult.iterrows()}
                codigo=st.selectbox("Producto",list(opciones),format_func=lambda x:opciones[x])
                hist=historicos[codigo]; pred=[]
                ultimo=int(df.Periodo.max()); fecha=pd.Timestamp(ultimo//100,ultimo%100,1)
                for h in [1,2]:
                    p,met=pred_producto(*modelos[h],hist,h,ajuste); f=fecha+pd.DateOffset(months=h)
                    pred.append({"Periodo":f.year*100+f.month,"VENTA":p,"Tipo":"Proyectado","Metodo":met})
                st.write(f"**{pred[0]['Periodo']}:** ${pred[0]['VENTA']:,.2f} | **{pred[1]['Periodo']}:** ${pred[1]['VENTA']:,.2f}")
                graf=pd.concat([hist[["Periodo","VENTA"]].assign(Tipo="Real"),pd.DataFrame(pred)[["Periodo","VENTA","Tipo"]]])
                st.altair_chart(alt.Chart(graf).mark_line(point=True).encode(x="Periodo:O",y="VENTA:Q",color="Tipo:N",tooltip=["Periodo","VENTA","Tipo"]).properties(height=350),use_container_width=True)

                if st.button("Generar proyeccion masiva"):
                    filas=[]
                    with st.spinner("Prediciendo productos sin reentrenar..."):
                        for codigo,hist in historicos.items():
                            row=hist.iloc[-1]; ps=[]; mets=[]
                            for hor in [1,2]:
                                p,met=pred_producto(*modelos[hor],hist,hor,ajuste); ps.append(p); mets.append(met)
                            filas.append({"CODIGO":codigo,"DESCRIPCION":row.get("DESCRIPCION",""),"CATEGORIA":row.get("CATEGORIA",""),
                                "Pred_Nov_2026":ps[0],"Pred_Dic_2026":ps[1],"Metodo_Nov":mets[0],"Metodo_Dic":mets[1]})
                    salida=pd.DataFrame(filas); st.dataframe(salida,use_container_width=True,hide_index=True)
                    st.download_button("Descargar CSV",salida.to_csv(index=False).encode("utf-8-sig"),"prediccion_nov_dic_2026_v4.csv","text/csv")
        else:
            st.warning("Presiona 'Entrenar modelo final 2026' para habilitar predicciones.")
    except Exception as e:
        st.exception(e)


