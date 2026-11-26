import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast Nov-Dic 2026", layout="wide")
st.markdown("## Forecast de ventas | Noviembre y diciembre 2026")
st.caption("Histórico mensual comparable y pronóstico conectado, con filtros de categoría, combo y producto.")

with st.expander("¿Cómo funciona el modelo?", expanded=False):
    st.markdown(
        """
### Resumen

El forecast combina **tres modelos globales CatBoost** y una corrección estacional controlada. El objetivo es estimar las ventas de noviembre y diciembre de 2026 sin limitarse a copiar el cierre de 2025.

### 1. Información utilizada

El modelo aprende con datos mensuales de 2024, 2025 y 2026 hasta octubre:

- Venta monetaria.
- Unidades vendidas.
- Precio promedio, recalculado como `VENTA / UNIDADES`.
- Código, categoría, familia, tipo de producto y condición de combo.
- Rezagos de 1, 2, 3, 6, 12 y 24 meses.
- Promedios y variabilidad de los últimos 3, 6 y 12 meses.
- Mes del año y tendencia reciente.

### 2. Tres pronósticos CatBoost

Se entrenan tres modelos:

1. **Venta directa:** pronostica el importe monetario.
2. **Unidades:** pronostica cuántas unidades se venderán.
3. **Precio promedio:** pronostica el precio medio esperado.

La venta estructural se calcula así:

`Venta estructural = Unidades pronosticadas × Precio pronosticado`

La predicción base combina:

`Predicción base = 85% venta directa + 15% venta estructural`

### 3. Estacionalidad de noviembre y diciembre

El ajuste externo solo se activa cuando el SKU presenta un índice estacional mínimo de **1.20**. Esto significa que históricamente noviembre y diciembre deben vender al menos 20% más que el resto del año.

Cuando se activa, se construye una referencia con:

- 65% del mismo mes más reciente.
- 35% del mismo mes de hace dos años.
- Crecimiento enero-octubre del año objetivo frente al año anterior.

La señal queda combinada como:

`Predicción estacional = 85% predicción base + 15% referencia histórica YTD`

### 4. Piso parcial validado

La venta del mismo mes anterior no se usa como piso rígido. Solo funciona como protección ante una subestimación extrema.

- En **noviembre** se recupera 45% de la distancia hacia el piso parcial.
- En **diciembre** se recupera 60%, porque diciembre mostró mayor subestimación en el backtest.
- Si la predicción ya está por encima del piso, no se agrega corrección.

Esto permite que el forecast quede por encima o por debajo de 2025 según la tendencia, sin copiar exactamente ese valor.

### 5. Validación histórica

La configuración se validó entrenando hasta octubre de 2025 y pronosticando noviembre-diciembre de 2025. Sin combos, los resultados fueron:

| Métrica | Resultado |
|---|---:|
| WAPE | 18.42% |
| MAE | $39,321.84 |
| RMSE | $113,103.17 |
| R² | 0.9457 |
| Sesgo | -6.43% |

El sesgo negativo indica que el modelo sigue siendo conservador. Las métricas son una referencia histórica, no una garantía del error real de 2026.

### 6. Interpretación de la gráfica

- **Línea azul:** ventas reales hasta octubre de 2026.
- **Línea naranja punteada:** pronóstico de noviembre y diciembre de 2026.
- Los filtros permiten analizar categoría, productos sin combos, combos o un SKU individual.
- El histórico utiliza el mismo universo de SKU activos incluidos en el forecast.

### 7. Alcance

El resultado es adecuado como forecast operativo con revisión comercial. Los combos, lanzamientos, bajas, promociones futuras, faltantes y cambios extraordinarios de distribución requieren validación adicional porque esas decisiones no siempre están presentes en el histórico.
        """
    )

with st.expander("Fórmulas principales", expanded=False):
    st.code(
        """Venta estructural = Unidades predichas * Precio predicho

Predicción base = 0.85 * Venta directa + 0.15 * Venta estructural

Predicción estacional = 0.85 * Predicción base + 0.15 * Referencia histórica YTD

Si Predicción combinada < Piso parcial:
    Noviembre: Predicción final = combinada + 0.45 * (piso - combinada)
    Diciembre: Predicción final = combinada + 0.60 * (piso - combinada)
Si no:
    Predicción final = Predicción combinada""",
        language="text",
    )

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}

CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
NUM_FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [
        "mean3", "std3", "mean6", "std6", "mean12", "std12",
        "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12",
    ]
)
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20

# Calibración selectiva V2 validada sobre el backtest Nov-Dic 2025 sin combos.
CALIBRACION_CATEGORIAS_V2 = {
    "LACTEOS CULINARIOS": 1.00,  # Sin uplift: contracción comercial conocida.
    "CHOCOLATE MESA": 1.10,
    "NESTLE PROFESSIONAL": 1.10,
    "RTD": 1.08,
    "CHOCOLATE IMPULSO": 0.90,
    "CARNATION POLVO": 0.92,
    "CEREALES": 0.97,
    "AGUAS": 0.97,
}
FACTOR_CALIBRACION_MIN = 0.90
FACTOR_CALIBRACION_MAX = 1.10
APLICAR_CALIBRACION_A_COMBOS = False
# Ajuste comercial solicitado para CHOCOLATE MESA, solo noviembre 2026.
REDUCCION_NOV_CHOCOLATE_MESA = 0.094
FACTOR_NOV_CHOCOLATE_MESA = 1.0 - REDUCCION_NOV_CHOCOLATE_MESA

# Control comercial de diciembre para LACTEOS CULINARIOS.
# Diciembre se ancla al noviembre actualizado del escenario, no al forecast alto original.
APLICAR_TOPE_DICIEMBRE_LACTEOS = True
PISO_DIC_VS_NOV_LACTEOS = 0.95       # Diciembre no cae más de 5% vs noviembre actualizado.
CRECIMIENTO_BASE_DIC_LACTEOS = 0.05  # Escenario central: +5% vs noviembre actualizado.
TOPE_DIC_VS_NOV_LACTEOS = 1.18       # Máximo: +18% vs noviembre actualizado.
USAR_RELACION_HISTORICA_LACTEOS = True

# Control definitivo del forecast base de Lácteos Culinarios.
# Si el modelo supera este nivel, diciembre se ajusta a noviembre × 1.18.
APLICAR_CONTROL_BASE_DIC_LACTEOS = True
CRECIMIENTO_DIC_LACTEOS = 0.18
FACTOR_DIC_VS_NOV_LACTEOS = 1.0 + CRECIMIENTO_DIC_LACTEOS


def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _normalizar_encabezado(valor):
    """Normaliza encabezados de Excel sin depender de espacios, acentos o saltos."""
    import re
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("\n", " ").replace("\r", " ").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _buscar_columna(columnas, aliases, prefijos=None):
    """Busca encabezados exactos y también variantes como UNIDADES 2025."""
    mapa = {_normalizar_encabezado(c): c for c in columnas}
    for alias in aliases:
        alias_normalizado = _normalizar_encabezado(alias)
        if alias_normalizado in mapa:
            return mapa[alias_normalizado]
    for prefijo in prefijos or []:
        prefijo_normalizado = _normalizar_encabezado(prefijo)
        for encabezado_normalizado, encabezado_original in mapa.items():
            if encabezado_normalizado.startswith(prefijo_normalizado):
                return encabezado_original
    return None


def _inferir_ano(nombre_archivo):
    import re
    coincidencia = re.search(r"(?:19|20)\d{2}", str(nombre_archivo))
    return int(coincidencia.group()) if coincidencia else None


def _inferir_mes(nombre_hoja):
    nombre = _normalizar_encabezado(nombre_hoja)
    for mes, numero in MESES.items():
        if mes in nombre:
            return numero
    return None


def cargar_archivos(archivos):
    """Carga robusta de los tres Excel.

    El año se toma del nombre VENTAS 2024/2025/2026. La función acepta una hoja
    por mes o una tabla consolidada, y reconoce variaciones de encabezados.
    No usa caché para evitar reutilizar una lectura anterior incompleta.
    """
    partes = []
    diagnostico = []

    aliases = {
        "CODIGO": ["CODIGO", "CÓDIGO", "SKU", "COD PRODUCTO", "COD. PRODUCTO"],
        "VENTA": ["VENTA", "VENTAS", "MONTO VENTA", "VENTA NETA"],
        "UNIDADES": ["UNIDADES", "UNIDAD", "UNIDADES VENDIDAS", "CANTIDAD", "PIEZAS"],
        "MES": ["MES", "NOMBRE MES"],
        "DESCRIPCION": ["DESCRIPCION", "DESCRIPCIÓN", "PRODUCTO", "NOMBRE PRODUCTO"],
        "CATEGORIA": ["CATEGORIA", "CATEGORÍA"],
        "FAMILIA": ["FAMILIA"],
        "TIPO": ["TIPO"],
        "COMBOS": ["COMBOS", "COMBO", "ES COMBO"],
    }

    for archivo in archivos:
        nombre = archivo.name
        ano_archivo = _inferir_ano(nombre)
        if ano_archivo not in {2024, 2025, 2026}:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": "El nombre debe contener 2024, 2025 o 2026",
            })
            continue

        try:
            contenido = archivo.getvalue()
            libro = pd.ExcelFile(io.BytesIO(contenido), engine="openpyxl")
        except Exception as exc:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": f"No fue posible abrir el Excel: {exc}",
            })
            continue

        for hoja in libro.sheet_names:
            try:
                h = pd.read_excel(libro, sheet_name=hoja)
            except Exception as exc:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "ERROR",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": f"No fue posible leer la hoja: {exc}",
                })
                continue

            if h is None or h.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "VACÍA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "Hoja sin registros",
                })
                continue

            columnas = {
                destino: _buscar_columna(
                    h.columns,
                    opciones,
                    prefijos=["UNIDADES"] if destino == "UNIDADES" else None,
                )
                for destino, opciones in aliases.items()
            }
            obligatorias = ["CODIGO", "VENTA", "UNIDADES"]
            faltantes = [c for c in obligatorias if columnas[c] is None]
            mes_hoja = _inferir_mes(hoja)
            if columnas["MES"] is None and mes_hoja is None:
                faltantes.append("MES")

            if faltantes:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": len(h), "Año asignado": ano_archivo,
                    "Detalle": "Faltan columnas: " + ", ".join(faltantes),
                })
                continue

            temp = pd.DataFrame(index=h.index)
            temp["CODIGO"] = h[columnas["CODIGO"]]
            temp["VENTA"] = h[columnas["VENTA"]]
            temp["UNIDADES"] = h[columnas["UNIDADES"]]
            temp["AÑO"] = ano_archivo

            if columnas["MES"] is not None:
                mes_original = h[columnas["MES"]]
                mes_texto = mes_original.astype(str).map(_normalizar_encabezado)
                mes_numero = pd.to_numeric(mes_original, errors="coerce")
                temp["MES_NUM"] = mes_texto.map(MESES).fillna(mes_numero)
            else:
                temp["MES_NUM"] = mes_hoja

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = (
                    h[columnas[col]] if columnas[col] is not None else "SIN_DATO"
                )
            temp["COMBOS"] = (
                h[columnas["COMBOS"]] if columnas["COMBOS"] is not None else 0
            )

            temp["CODIGO"] = normalizar_codigo(temp["CODIGO"])
            temp["VENTA"] = pd.to_numeric(temp["VENTA"], errors="coerce")
            temp["UNIDADES"] = pd.to_numeric(temp["UNIDADES"], errors="coerce")
            temp["MES_NUM"] = pd.to_numeric(temp["MES_NUM"], errors="coerce")
            temp["COMBOS"] = pd.to_numeric(temp["COMBOS"], errors="coerce").fillna(0)

            temp = temp.dropna(subset=["CODIGO", "VENTA", "UNIDADES", "MES_NUM"])
            temp = temp[temp["MES_NUM"].between(1, 12)]
            temp["MES_NUM"] = temp["MES_NUM"].astype(int)
            temp["VENTA"] = temp["VENTA"].clip(lower=0)
            temp["UNIDADES"] = temp["UNIDADES"].clip(lower=0)
            temp["COMBOS"] = temp["COMBOS"].clip(0, 1).astype(int)
            temp["Periodo"] = temp["AÑO"] * 100 + temp["MES_NUM"]

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = temp[col].fillna("SIN_DATO").astype(str).str.strip()

            if temp.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "No quedaron filas válidas después de limpiar",
                })
                continue

            partes.append(temp)
            diagnostico.append({
                "Archivo": nombre, "Hoja": hoja, "Estado": "CARGADA",
                "Filas": len(temp), "Año asignado": ano_archivo,
                "Detalle": (
                    f"{temp['CODIGO'].nunique()} SKU | meses {sorted(temp['MES_NUM'].unique().tolist())} | "
                    f"unidades leídas desde: {columnas['UNIDADES']}"
                ),
            })

    if not partes:
        raise ValueError(
            "Ninguna hoja pudo cargarse. Abre el diagnóstico y revisa los encabezados."
        )

    df = pd.concat(partes, ignore_index=True)
    agrupado = (
        df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
        .agg(
            VENTA=("VENTA", "sum"),
            UNIDADES=("UNIDADES", "sum"),
            DESCRIPCION=("DESCRIPCION", "first"),
            CATEGORIA=("CATEGORIA", "first"),
            FAMILIA=("FAMILIA", "first"),
            TIPO=("TIPO", "first"),
            COMBOS=("COMBOS", "first"),
        )
    )
    agrupado["PRECIO_PROMEDIO"] = np.where(
        agrupado["UNIDADES"] > 0,
        agrupado["VENTA"] / agrupado["UNIDADES"],
        np.nan,
    )
    salida = completar_calendario(agrupado)
    return salida, pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            dict(year=x["AÑO"], month=x["MES_NUM"], day=1)
        )
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[col] = x[col].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(
            x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan
        )
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))

    z = pd.concat(salida, ignore_index=True)
    mediana_categoria = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = (
        z["PRECIO_PROMEDIO"]
        .fillna(mediana_categoria)
        .fillna(z["PRECIO_PROMEDIO"].median())
        .clip(lower=0)
    )
    z["COMBOS"] = pd.to_numeric(z["COMBOS"], errors="coerce").fillna(0).astype(int)
    return z


def crear_features(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")[objetivo]

    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)

    for ventana in [3, 6, 12]:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).mean()
        )
        z[f"std{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).std()
        ).fillna(0)

    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(
        z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0
    )

    for col in CAT_FEATURES:
        z[col] = z[col].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=10,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def entrenar_objetivo(df, objetivo):
    f = crear_features(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())

    modelo = nuevo_modelo()
    modelo.fit(
        f[FEATURES],
        np.log1p(f[objetivo].clip(lower=0)),
        cat_features=CAT_FEATURES,
        sample_weight=pesos,
    )
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df):
    return {
        objetivo: entrenar_objetivo(df, objetivo)
        for objetivo in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]
    }


def vector_futuro(historial, periodo, objetivo, medianas):
    hist = historial.sort_values("Periodo")
    valores = hist[objetivo].astype(float).tolist()
    fila = {
        f"lag{lag}": valores[-lag] if len(valores) >= lag else np.nan
        for lag in LAGS
    }

    for ventana in [3, 6, 12]:
        arreglo = np.asarray(valores[-ventana:], dtype=float)
        fila[f"mean{ventana}"] = arreglo.mean() if len(arreglo) >= 2 else np.nan
        fila[f"std{ventana}"] = arreglo.std(ddof=1) if len(arreglo) >= 2 else 0.0

    mes = periodo % 100
    ano = periodo // 100
    fila["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    fila["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    fila["MES_NUM"] = mes
    fila["ANO"] = ano
    fila["tendencia_3_12"] = (
        fila["mean3"] / fila["mean12"]
        if pd.notna(fila["mean12"]) and fila["mean12"] > 0
        else 1.0
    )

    ultimo = hist.iloc[-1]
    for col in CAT_FEATURES:
        fila[col] = str(ultimo.get(col, "SIN_DATO"))

    x = pd.DataFrame([fila], columns=FEATURES)
    x[NUM_FEATURES] = x[NUM_FEATURES].fillna(medianas)
    return x


def predecir_objetivo(modelo, medianas, historial, periodos, objetivo):
    hist = historial.copy()
    predicciones = []

    for periodo in periodos:
        pred = float(
            np.expm1(
                modelo.predict(vector_futuro(hist, periodo, objetivo, medianas))[0]
            )
        )
        pred = max(0.0, pred)
        predicciones.append(pred)

        nueva = hist.iloc[-1].copy()
        nueva["Periodo"] = periodo
        nueva["AÑO"] = periodo // 100
        nueva["MES_NUM"] = periodo % 100
        nueva[objetivo] = pred
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)

    return predicciones


def indice_estacional(historial, ano_objetivo):
    h = historial[historial["AÑO"] < ano_objetivo]
    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temporada) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temporada.mean() / resto.mean())


def crecimiento_ytd(historial, ano_objetivo):
    actual = historial[
        (historial["AÑO"] == ano_objetivo) & (historial["MES_NUM"] <= 10)
    ]
    previo = historial[
        (historial["AÑO"] == ano_objetivo - 1) & (historial["MES_NUM"] <= 10)
    ]
    meses = sorted(set(actual["MES_NUM"]) & set(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    return float(np.clip(va / vp, 0.80, 1.40)) if vp > 0 else 1.0


def aplicar_ajuste_estacional(pred_base, historial, periodo):
    """Ajuste estacional final validado: piso parcial 45% Nov / 60% Dic.

    No copia automáticamente el mismo mes anterior. Si la predicción queda por
    debajo del piso parcial, recupera solo una parte de la diferencia.
    """
    mes = periodo % 100
    ano = periodo // 100
    indice = indice_estacional(historial, ano)
    mismos = historial[
        (historial["MES_NUM"] == mes) & (historial["AÑO"] < ano)
    ].sort_values("AÑO").tail(2)

    diagnostico = {
        "Indice_estacional": float(indice),
        "Referencia_ultimo_ano": np.nan,
        "Crecimiento_YTD": 1.0,
        "Prediccion_estacional": float(pred_base),
        "Prediccion_combinada": float(pred_base),
        "Piso_parcial": np.nan,
        "Porcentaje_recuperacion": 0.0,
        "Correccion_piso": 0.0,
        "Ajuste_aplicado": False,
    }

    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return max(0.0, float(pred_base)), diagnostico

    valores = mismos["VENTA"].astype(float).to_numpy()
    ultimo_ano = float(valores[-1])
    referencia_historica = (
        ultimo_ano if len(valores) == 1
        else 0.65 * ultimo_ano + 0.35 * float(valores[-2])
    )
    crec = crecimiento_ytd(historial, ano)
    referencia_ytd = referencia_historica * crec

    prediccion_estacional = (
        PESO_FORECAST_ESTACIONAL * float(pred_base)
        + (1.0 - PESO_FORECAST_ESTACIONAL) * referencia_ytd
    )

    crecimiento_suavizado = float(np.clip(
        1.0 + 0.50 * (crec - 1.0), 0.90, 1.20
    ))
    referencia_suavizada = ultimo_ano * crecimiento_suavizado
    prediccion_combinada = (
        0.90 * prediccion_estacional + 0.10 * referencia_suavizada
    )

    factor_piso = float(np.clip(
        1.0 + 0.35 * (crec - 1.0), 0.94, 1.15
    ))
    piso_parcial = ultimo_ano * factor_piso

    porcentaje_recuperacion = 0.45 if mes == 11 else 0.60 if mes == 12 else 0.40
    if prediccion_combinada < piso_parcial:
        correccion_piso = porcentaje_recuperacion * (
            piso_parcial - prediccion_combinada
        )
        prediccion_final = prediccion_combinada + correccion_piso
    else:
        correccion_piso = 0.0
        prediccion_final = prediccion_combinada

    diagnostico.update({
        "Referencia_ultimo_ano": ultimo_ano,
        "Crecimiento_YTD": float(crec),
        "Prediccion_estacional": float(prediccion_estacional),
        "Prediccion_combinada": float(prediccion_combinada),
        "Piso_parcial": float(piso_parcial),
        "Porcentaje_recuperacion": float(porcentaje_recuperacion),
        "Correccion_piso": float(correccion_piso),
        "Ajuste_aplicado": True,
    })
    return max(0.0, float(prediccion_final)), diagnostico

def normalizar_categoria_calibracion(categoria):
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(categoria))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.upper().strip().split())


def aplicar_calibracion_categoria_v2(prediccion, categoria, es_combo=0):
    """Aplica factor V2 solo a categorías autorizadas y productos sin combo."""
    prediccion = max(0.0, float(prediccion))
    categoria_norm = normalizar_categoria_calibracion(categoria)
    if int(es_combo) == 1 and not APLICAR_CALIBRACION_A_COMBOS:
        return prediccion, 1.0, False, "Combo sin calibración"
    factor = float(np.clip(
        CALIBRACION_CATEGORIAS_V2.get(categoria_norm, 1.0),
        FACTOR_CALIBRACION_MIN,
        FACTOR_CALIBRACION_MAX,
    ))
    aplicada = not np.isclose(factor, 1.0)
    motivo = f"Factor V2 de {categoria_norm}" if aplicada else "Categoría sin corrección"
    return prediccion * factor, factor, aplicada, motivo


def extender_historial_hasta_octubre_2026(historial):
    """Completa meses faltantes hasta octubre 2026 para SKU inactivos.

    Los meses posteriores al último registro se consideran sin venta y sin
    unidades. El precio se conserva como última referencia disponible. Esto
    evita tratar julio como si fuera el mes inmediatamente anterior a noviembre.
    """
    hist = historial.sort_values("Periodo").copy()
    ultimo_periodo = int(hist["Periodo"].max())
    if ultimo_periodo >= 202610:
        return hist

    ultima_fecha = pd.Timestamp(
        year=ultimo_periodo // 100,
        month=ultimo_periodo % 100,
        day=1,
    )
    fechas_faltantes = pd.date_range(
        ultima_fecha + pd.offsets.MonthBegin(1),
        pd.Timestamp("2026-10-01"),
        freq="MS",
    )
    if len(fechas_faltantes) == 0:
        return hist

    meta = hist.iloc[-1].copy()
    nuevas = []
    for fecha in fechas_faltantes:
        fila = meta.copy()
        fila["AÑO"] = int(fecha.year)
        fila["MES_NUM"] = int(fecha.month)
        fila["Periodo"] = int(fecha.year * 100 + fecha.month)
        fila["VENTA"] = 0.0
        fila["UNIDADES"] = 0.0
        # El precio se conserva como referencia; no se inventa un precio cero.
        fila["PRECIO_PROMEDIO"] = float(meta.get("PRECIO_PROMEDIO", 0.0) or 0.0)
        nuevas.append(fila)
    return pd.concat([hist, pd.DataFrame(nuevas)], ignore_index=True)


def meses_entre_periodos(periodo_inicial, periodo_final=202610):
    ai, mi = divmod(int(periodo_inicial), 100)
    af, mf = divmod(int(periodo_final), 100)
    return max(0, (af - ai) * 12 + (mf - mi))


def ajustar_noviembre_chocolate_mesa(forecast):
    """Reduce 9.4% el forecast monetario de CHOCOLATE MESA en noviembre 2026.

    El factor se aplica proporcionalmente a todos los SKU de la categoria. Las
    unidades se recalculan con el precio reconciliado para mantener coherencia.
    Diciembre no se modifica.
    """
    out = forecast.copy()
    out["Prediccion_antes_ajuste_chocolate_mesa"] = out["Prediccion_final"]
    out["Factor_ajuste_nov_chocolate_mesa"] = 1.0
    out["Ajuste_nov_chocolate_mesa_aplicado"] = False
    if out.empty:
        return out
    cat = out["CATEGORIA"].map(normalizar_categoria_calibracion)
    mascara = cat.eq("CHOCOLATE MESA") & out["Periodo"].eq(202611)
    out.loc[mascara, "Prediccion_final"] *= FACTOR_NOV_CHOCOLATE_MESA
    out.loc[mascara, "Unidades_pred"] = np.where(
        out.loc[mascara, "Precio_pred"] > 0,
        out.loc[mascara, "Prediccion_final"] / out.loc[mascara, "Precio_pred"],
        0.0,
    )
    out.loc[mascara, "Venta_estructural"] = (
        out.loc[mascara, "Unidades_pred"] * out.loc[mascara, "Precio_pred"]
    )
    out.loc[mascara, "Factor_ajuste_nov_chocolate_mesa"] = FACTOR_NOV_CHOCOLATE_MESA
    out.loc[mascara, "Ajuste_nov_chocolate_mesa_aplicado"] = True
    return out

def ajustar_diciembre_lacteos_18(forecast):
    """Ajusta diciembre de Lácteos Culinarios a un máximo de +18% vs noviembre.

    La reducción se reparte proporcionalmente entre los SKU de diciembre. El mismo
    factor se aplica a unidades para conservar coherencia entre venta y volumen.
    """
    out = forecast.copy()
    out["Prediccion_antes_control_lacteos"] = out["Prediccion_final"]
    out["Unidades_antes_control_lacteos"] = out["Unidades_pred"]
    out["Factor_control_diciembre_lacteos"] = 1.0
    out["Control_diciembre_lacteos_aplicado"] = False
    if out.empty or not APLICAR_CONTROL_BASE_DIC_LACTEOS:
        return out
    cat = out["CATEGORIA"].map(normalizar_categoria_calibracion)
    lacteos = cat.eq("LACTEOS CULINARIOS")
    nov = lacteos & out["Periodo"].eq(202611)
    dic = lacteos & out["Periodo"].eq(202612)
    total_nov = float(out.loc[nov, "Prediccion_final"].sum())
    total_dic = float(out.loc[dic, "Prediccion_final"].sum())
    objetivo_dic = total_nov * FACTOR_DIC_VS_NOV_LACTEOS
    if total_nov > 0 and total_dic > objetivo_dic:
        factor = float(objetivo_dic / total_dic)
        out.loc[dic, "Prediccion_final"] *= factor
        out.loc[dic, "Unidades_pred"] = np.where(
            out.loc[dic, "Precio_pred"] > 0,
            out.loc[dic, "Prediccion_final"] / out.loc[dic, "Precio_pred"],
            0.0,
        )
        out.loc[dic, "Venta_estructural"] = out.loc[dic, "Unidades_pred"] * out.loc[dic, "Precio_pred"]
        out.loc[dic, "Factor_control_diciembre_lacteos"] = factor
        out.loc[dic, "Control_diciembre_lacteos_aplicado"] = True
    return out


PRECIOS_WEB_VERIFICADOS = {
    # Solo coincidencias exactas de UPC y presentacion. No se usan coincidencias por nombre.
    "7501059233980": {"precio": 65.10, "fuente": "WEB - Superdulces", "url": "https://superdulces.com/products/apolv", "fecha_consulta": "2026-11-24"},
    "7501059235441": {"precio": 14.00, "fuente": "WEB - Farmacia San Pablo", "url": "https://www.farmaciasanpablo.com.mx/alimentos-y-bebidas/bebidas/aguas-purificadas/sta-maria-agua-pura-de-manantial/p/000000000007790029", "fecha_consulta": "2026-11-24"},
    "7501059200050": {"precio": 14.00, "fuente": "WEB - Veggicano", "url": "https://veggicano.com.mx/products/agua-embotellada-santa-maria-1", "fecha_consulta": "2026-11-24"},
}


def cargar_sabana_precios(archivo):
    """Lee UPC y $ Precio Venta Sugerido de la hoja SABANA DE PRECIOS."""
    if archivo is None:
        return pd.DataFrame(columns=["CODIGO", "Precio_sugerido_sabana", "Descripcion_sabana"])
    contenido = archivo.getvalue()
    tabla = pd.read_excel(io.BytesIO(contenido), sheet_name="SABANA DE PRECIOS", header=2, engine="openpyxl")
    col_upc = _buscar_columna(tabla.columns, ["UPC"])
    col_precio = _buscar_columna(tabla.columns, ["$ PRECIO VENTA SUGERIDO", "PRECIO VENTA SUGERIDO"])
    col_desc = _buscar_columna(tabla.columns, ["DESCRIPCION", "DESCRIPCIÓN"])
    if col_upc is None or col_precio is None:
        raise ValueError("La sábana debe contener las columnas UPC y $ Precio Venta Sugerido en la fila 3.")
    out = pd.DataFrame({
        "CODIGO": normalizar_codigo(tabla[col_upc]),
        "Precio_sugerido_sabana": pd.to_numeric(tabla[col_precio], errors="coerce"),
        "Descripcion_sabana": tabla[col_desc].astype(str).str.strip() if col_desc is not None else "",
    })
    out = out.loc[out["CODIGO"].ne("") & out["Precio_sugerido_sabana"].gt(0)].copy()
    # Si un UPC se repite, se conserva la mediana para evitar duplicar el forecast.
    return out.groupby("CODIGO", as_index=False).agg(
        Precio_sugerido_sabana=("Precio_sugerido_sabana", "median"),
        Descripcion_sabana=("Descripcion_sabana", "first"),
    )


# Instantaneas de los precios publicados por producto en los Excel adjuntos.
# Los totales y precios no positivos no se incluyen. Sin cruces aproximados.
PRECIOS_NUEVO_ARCHIVO = {'***DESCONT***SALSA INGLESA CROSSE & BLACKWELL 50ML': [20.900000000000002],
 '1 COFFEE MATE 160G MAS LECHERA ORIGINAL 375G': [49.9],
 '1 COFFEE MATE 800G GRATIS MAGGI MICHE 145ML': [88.57605633802817],
 '1 COFFEE MATE ORIGINAL 900G GRATIS AZUCAR 900G': [127.97692307692309],
 '1 LECHERA 375 G MAS CARNATION 360 G PRECIO ESP': [49.9],
 '1 NESCAFE CLASICO 200G GRATIS 1 COFFEE MATE 530G': [100.9],
 '1 NESCAFE CLASICO 300G GRATIS AZUCAR 900G': [189.9],
 '1 NESCAFE DOLCA 170G MAS COFFEE MATE 160G': [98.9],
 '2 COFFEE MATE 530G MAS ESPRESSO BLACK PRECIO ESP': [69.9],
 '2 COFFEE MATE 530G MAS ESPRESSO DE VAINILLA PRECIO ESP': [69.9],
 '2 LECHERA 375G MAS 1 CARNATION 360G GRATIS HARINA PARA HOT CAKES': [74.9],
 '2 NESCAFE DOLCA 170G MAS COFFEE MATE 160G': [196.9],
 '3 PACK CARNATION LECHE EVAPORADA LATA 360G': [59.9],
 'AGUA DE MANANTIAL SANTA MARIA BOTELLA 4LT': [33.9],
 'AGUA DE MANANTIAL STA MARIA BOTELLA 1L': [9.553395083719273],
 'AGUA PUREZA VITAL NESTLE 1.5 LT': [13.711508810572687],
 'AGUA PUREZA VITAL NESTLE 1LT': [7.396983105390184],
 'AGUA PURIFICADA GERBER 4 L': [32.89497487437186],
 'AGUA SANTA MARIA NESTLE 1.5 LT': [14.670495342929721],
 'AGUA SANTA MARIA NESTLE 500 ML': [7.386059299191374],
 'BEBIDA LATTE NESCAFE 330ML': [26.9],
 'CAFE DE OLLA INSTANTANEO NESCAFE 170 G': [88.54601769911505],
 'CAFE DE OLLA INSTANTANEO NESCAFE 46 G': [35.87174657534246],
 'CAFE RESERVA MEXICANA CHIAPAS 180GR': [144.14178154825026],
 'CAFE RESERVA MEXICANA VERACRUZ 180 GR': [143.91415571284125],
 'CAFE SOLUBLE NESCAFE CLASICO 1 KG': [559.9000000000001],
 'CAFE SOLUBLE NESCAFE CLASICO 120 G': [109.82486851990984],
 'CAFE SOLUBLE NESCAFE CLASICO 170 G': [109.87441860465117],
 'CAFE SOLUBLE NESCAFE CLASICO 200 G': [134.53223140495868],
 'CAFE SOLUBLE NESCAFE CLASICO 225 G + 25% GRATIS IN & OUT': [134.49035444509005],
 'CAFE SOLUBLE NESCAFE CLASICO 2X120 G': [139.9],
 'CAFE SOLUBLE NESCAFE CLASICO 42 G': [47.62474738279472],
 'CAFE SOLUBLE NESCAFE CLASICO 60 G': [73.9],
 'CAFE SOLUBLE NESCAFE CLASICO 85 GR': [64.9],
 'CAFE SOLUBLE NESCAFE CLASICO BLACK 170 G': [109.9],
 'CAFE SOLUBLE NESCAFE CLASICO MAS TAZA 200G': [134.19945054945055],
 'CAFE SOLUBLE NESCAFE DECAF 120 G': [124.9],
 'CAFE SOLUBLE NESCAFE DECAF 40 G': [48.843180260452364],
 'CAFE SOLUBLE NESCAFE DOLCA 1 KG': [447.8629629629629],
 'CAFE SOLUBLE NESCAFE DOLCA 170 G': [98.9],
 'CAFE SOLUBLE NESCAFE DOLCA 300 G': [162.07730496453902],
 'CAFE SOLUBLE NESCAFE DOLCA 46 G': [38.890694171578254],
 'CAFE SOLUBLE NESCAFE DOLCA 80 G': [68.9],
 'CAFE SOLUBLE NESCAFE DOLCA DOYPACK DE 22 G': [23.900000000000002],
 'CAFE SOLUBLE NESCAFE TASTERS CHOICE DESCAFEINADO 100 G': [167.94123711340205],
 'CAFE SOLUBLE NESCAFE TASTERS CHOICE GOURMET 100 G': [166.26],
 'CALABAZA HALLOWEEN NESTLE': [55.462],
 'CARNATION LECHE EVAPORADA UHT 360GR': [23.698230088495574],
 'CARNATION ORIGINAL UHT 720G + LLCH LATA 375G IN&OUT': [59.9],
 'CEREAL CARLOS V NESTLE 590 G': [64.95885454545456],
 'CEREAL CHEERIOS 480 G + CEREAL NESQUICK 620 G': [113.4322513761468],
 'CEREAL CHEERIOS CON MIEL Y ALMENDRAS 480 G': [68.23253594306051],
 'CEREAL CHEERIOS HONEY NESTLE 230 G': [23.9004],
 'CEREAL CHEERIOS MANZANA CANELA NESTLE CAJA 480 G': [67.53163725690892],
 'CEREAL CHEERIOS MIEL 480 G MAS CF NESTLE 500 G GRATIS TRIX 430 GR': [138.8],
 'CEREAL CINNAMON TOAST CRUNCH NESTLE 340GR': [84.24574631828978],
 'CEREAL COOKIE CRISP NESTLE 480 G': [66.53592097428958],
 'CEREAL CORN FLAKES 500 G MAS TRIX 430 G NESTLE PRECIO ESPECIAL': [92.5],
 'CEREAL CORN FLAKES NESTLE 500 G': [59.64801455563332],
 'CEREAL INFLADO GERBER 42 G': [66.89999999999999],
 'CEREAL INFLADO JUNIOR GERBER 42': [68.9],
 'CEREAL INFLADO MORA AZUL GERBER 42 G': [68.89999999999999],
 'CEREAL INTEGRAL FITNESS ORIGINAL NESTLE 570 G': [74.30326171875001],
 'CEREAL LUCKY CHARMS NESTLE 297 G': [83.59008969359333],
 'CEREAL NESQUICK NESTLE 620 G': [75.80521529745043],
 'CEREAL NESQUIK 230G MAS CEREAL TRIX 230G': [49.89877105263158],
 'CEREAL NESQUIK 620 G MAS PAPEL KLEENEX 4 PZS': [69.35155844155844],
 'CEREAL NESQUIK 620 Y TRIX 430G MAS ZUCOSOS 650G': [116.92148594377511],
 'CEREAL NESQUIK 620G MAS ZUCOSOS 650G': [79.89999999999999],
 'CEREAL NESQUIK DE FRESA 310G': [35.47193142857143],
 'CEREAL NESQUIK FRESA 310G MAS CF 500G P ESPECIAL': [73.98],
 'CEREAL TRIGO INTEGRAL MIEL GERBER 270 G.': [58.404672897196264],
 'CEREAL TRIGO INTEGRAL Y MANZANA GERBER 270 GRS': [59.9],
 'CEREAL TRIX NESTLE 430 GR.': [74.81680896226416],
 'CEREAL TRIX SABOR MARSHMALLOW 230G': [36.53463898916967],
 'CEREAL ZUCOSOS NESTLE 650G': [59.26908387096775],
 'CEREALES NESQUIK 620G MAS ZUCOSOS 650G': [99.9],
 'CHOCO ABUELITA 540G Y CARNATION 360G': [120.27888888888889],
 'CHOCOLATE ABUELITA 270 GR (3 TABLETAS DE 90GR C/U)': [79.89990821052632],
 'CHOCOLATE CARLOS V DE 216G MAS UNO': [111.11],
 'CHOCOLATE CARLOS V NESTLE 18 G': [13.794964055996973],
 'CHOCOLATE CARLOS V STICK 20 PZAS DE 8 G C/U': [59.898685714285726],
 'CHOCOLATE CRUNCH 40 G': [19.11044237837838],
 'CHOCOLATE CRUNCH STICK NESTLE 20 PZAS': [45.423815789473686],
 'CHOCOLATE EN LATA NESQUICK 400 G': [71.88110816326531],
 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 357 G': [47.899896122896855],
 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 445G': [47.899956521739135],
 'CHOCOLATE EN TABLETA ABUELITA NESTLE 540 G': [139.82751778546714],
 'CHOCOLATE GRANULADO ABUELITA 320 G': [52.535926133184105],
 'CHOCOLATE KIT KAT 41.5 G': [18.158146820809247],
 'CHOCOLATE KIT KAT CAJA CON 9 PZAS DE 41.5 G C/U': [161.94125],
 'CHOCOLATE KIT KAT DARK NESTLE 41.5G': [19.7279538883806],
 'CHOCOLATE KITKAT CHUNKY NESTLE 40G': [18.170086646884275],
 'CHOCOLATE LARIN CERO AZUCAR 19 G': [19.879375977653634],
 'CHOCOLATE POLVO NESQUICK BOLSA 200 G': [32.80890906593407],
 'CHOCOLATE POLVO NESQUICK BOLSA 357 G': [56.05263378271214],
 'COFFEE MATE 311 G GRATIS THERMO': [57.9],
 'COFFEE MATE 311G GRATIS VASO NESTLE': [60.9],
 'COFFEE MATE DULCE DE LECHE 530G': [44.9],
 'COFFEE MATE EXTRA CREMOSO 530 G': [49.900000000000006],
 'COFFEE MATE LIQUIDO AVELLANA 769.7 G': [58.98227762803234],
 'COFFEE MATE LIQUIDO ORIGINAL 769.7 G': [59.22159406858202],
 'COFFEE MATE MAZAPAN NESTLE 530G': [19.9],
 'COFFEE MATE ORIGINAL 400G GRATIS AZUCAR DULY': [65.99271523178808],
 'COFFEE MATE ORIGINAL 800 G': [84.98863253200126],
 'COFFEE MATE PUMPKIN SPICE POLVO 350 G': [32.9],
 'COLADO DE DURAZNO ET 3 GERBER 170 G': [20.900000000000002],
 'COLADO GERBER 3AET FRUT.TROP-170 GRS': [20.9],
 'COLADO GERBER 3AET MANZANA 170G MX': [20.9],
 'COLADO GERBER 3AET PERA 170 GRS': [20.9],
 'COLADO GERBER 3AET VEG PAVO Y PASTA 170 G': [20.9],
 'COLADO GERBER 3AET VEGETALES Y PASTA 170 G': [20.9],
 'COLADO GERBER 3AET VERD/POLL/ARR 170GMX': [20.900000000000002],
 'COLADO GERBER 3RAET VER/RES/ARR 170 GRS': [20.9],
 'CONSOMATE EN POLVO 5 SOBRES DE 21 G': [14.9],
 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI 12 CUBOS 105 G': [17.9],
 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI CAJA 8 CUBOS 84 G': [18.900000000000002],
 'CREMA EN POLVO PARA CAFE COFFEE MATE BOLSA 210 G': [39.9],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 160 G': [37.852536997885835],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 311 G': [60.51339963833635],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 400 G': [65.62869565217392],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 520 G': [67.66975169300225],
 'CREMA LIQUIDA PARA CAFE BAILEYS 468 ML': [39.9],
 'CREMA LIQUIDA PARA CAFE CARLOS V 530 ML': [19.9],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE AVELLANA 530 G': [39.63835920177384],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE CANELA 530 ML': [39.9],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE ORIGINAL 530 ML': [39.77684729064039],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE VAINILLA 530 G': [39.62786772304513],
 'DULCE DE LECHE LA LECHERA POUCH 209 G': [25.83321548821549],
 'DULCE LECHE LECHERA SIRVE FACIL 325 GR': [43.428],
 'DUO PACK CARNATION UHT 360 G': [37.9],
 'DUO PACK COFFEE MATE 769.7 G': [99.89999999999999],
 'FRESKAS NESTLE 35 G': [16.406799874725966],
 'FRESKAS NESTLE 9 PZAS CON 35 G C/U': [111.02],
 'GERBER 2A ETAPA POLLO Y ARROZ 100 G': [12.899999999999999],
 'GERBER 2A ETAPA PURE DE MANZANA 100 G': [12.9],
 'GERBER 2AET CIRUELA 100G': [12.899999999999999],
 'GERBER 2AET DURAZNO 100GR': [12.899999999999999],
 'GERBER 2AET FRUTAS MIXTAS 100G': [12.899999999999999],
 'GERBER 2AET MANGO 100G': [12.9],
 'GERBER 2AET PERA 100G': [12.9],
 'GERBER 2AET PLATANO 100G': [12.899999999999999],
 'GERBER 2AET PURE FRU TROPICALES 113G': [16.9],
 'GERBER 2AET RES Y ARROZ 100G': [12.9],
 'GERBER ET JNR MANZANA POUCH LE 195G': [13.9],
 'GERBER ET JUNIO MANGO 95G': [13.9],
 'GERBER ET JUNIOR FRUTAS MIXTAS 95G': [13.9],
 'IN&OUT COFFEE MATE LIQUIDO VAINILLA 769.7G': [59.20309614340032],
 'IN&OUT NESCAFE ICE 170 G': [132.27724550898202],
 'JUGO GERBER JUNIOR DE MANGO 200 ML': [14.9],
 'JUGO GERBER JUNIOR FRUTAS MIXTAS 200 ML': [14.9],
 'JUGO GERBER JUNIOR MANZANA 200 ML': [14.899999999999999],
 'JUGO GERBER JUNIOR PERA 200 ML': [14.9],
 'JUGO SAZONADOR CLASICO MAGGI 100 ML': [43.83511988716502],
 'JUGO SAZONADOR MAGGI 125G + SALSA PICANTE VALENTINA 250ML': [39.9],
 'JUGO SAZONADOR MAGGI 45 ML': [20.9],
 'JUGO SAZONADOR MAGGI BOTELLA 200 ML': [72.89999999999999],
 'JUGOSO AL SARTEN AJO Y CEBOLLA MAGGI SOBRE 4 PZAS 23.2 G': [21.9],
 'JUGOSO AL SARTEN FINAS HIERBAS MAGGI SOBRE 4 PZAS 23.4 G': [21.900000000000002],
 'JUGOSO AL SARTEN MAGGI A LA LENA 24 G': [21.9],
 'LA LECHERA ORIGINAL BOLSA 209G': [16.238235294117647],
 'LA LECHERA ORIGINAL CARTON 430 G': [32.344308445532434],
 'LA LECHERA SIRVE FACIL 335 GRS': [47.900000000000006],
 'LA LECHERA TRES LECHES 1 KG': [71.83333333333333],
 'LECHE CARLOS V CHOCOLATE NESTLE 185 ML': [9.898305084745763],
 'LECHE CARLOS V CHOCOLATE NESTLE 240 ML': [12.9],
 'LECHE CARNATION 1 LTR': [62.80529073114565],
 'LECHE CARNATION POLVO DOY PACK 460 GRS': [63.798400752587014],
 'LECHE CLAVEL DESLACTOSADA 360 GRS.': [24.01264367816092],
 'LECHE CONDENSADA LA LECHERA ORIGINAL NESTLE 375 G': [29.632214620797498],
 'LECHE EN POLVO CARNATION DOY PACK 520 G': [67.9],
 'LECHE EVAPORADA CARNATION 360 G': [21.728692257836435],
 'LECHE EVAPORADA EN POLVO CARNATION 120GR': [19.466433566433565],
 'LECHE EVAPORADA NESTLE CARNATION CLAVEL 720 G': [42.90500500500501],
 'LECHE NAN 1 OPTIPRO 1.2KG': [624.9],
 'LECHE NAN 2 OPTIPRO 720GR': [349.9],
 'LECHE NESQUIK CHOCOLATE 185 ML': [9.763246268656717],
 'LECHE NESTLE NUTRI RINDES 460 G GRATIS DOLCA 22 G': [65.9],
 'LECHE NESTLE NUTRI RINDES 460 GRS.': [65.89999999999999],
 'LECHE NIDO CLASICA FORTI 720G': [141.9],
 'LECHE NIDO FORTICRECE 1.56 KG': [269.9],
 'LECHE NIDO KINDER 1.4 KG': [257.9152671755725],
 'LECHE NIDO KINDER 360 GR.': [69.05299334811531],
 'LECHE NIDO KINDER 800 GR.': [169.9],
 'LECHE SABORIZADA ABUELITA SABOR CHOCOLATE 930 ML': [44.900000000000006],
 'LECHERA NESTLE 100 GR.': [12.881836528758829],
 'MALTEADA SABOR FRESA NESQUIK 185 ML': [9.57387580299786],
 'MALTEADA SABOR VAINILLA NESQUIK 185 ML': [9.362349676225717],
 'MALTEADA VAINILLA NESQUICK BOLSA 357 G': [59.9],
 'MEDIA CREMA NESTLE 1 LTR': [73.93434227330779],
 'MEDIA CREMA NESTLE 190 GR': [17.57597298284045],
 'MEDIA CREMA NESTLE 225 G': [19.99738618274435],
 'MEDIA CREMA NESTLE 500ML': [35.913400335008376],
 'NAN 3 OPTIPRO 720GR': [217.9281690140845],
 'NESCAFE CAFE DE OLLA 170+46 G': [88.9],
 'NESCAFE CAFE DE OLLA FRASCO 260 G': [121.72263329706203],
 'NESCAFE CAFE DE OLLA FRASCO 85 GR.': [55.85703125],
 'NESCAFE CAPPUCCINO KITKAT 6X25G': [32.89977551020409],
 'NESCAFE CAPPUCCINO MOKA 6 SOBRES X 22 GR': [64.72032258064516],
 'NESCAFE CAPPUCCINO ORIGINAL 6 SOBRES X 20 GR': [69.89999999999999],
 'NESCAFE CAPPUCCINO VAINILLA 6X22G': [69.9],
 'NESCAFE CLASICO 120 G GRATIS CAJA SPLENDA 50 PZ': [113.31463414634146],
 'NESCAFE CLASICO 120 G GRATIS TAZA': [110.9],
 'NESCAFE CLASICO 200 G GRATIS ARROZ SCHETTINO': [134.6],
 'NESCAFE CLASICO 200G MAS NESCAFE DECAF 40G': [142.86],
 'NESCAFE CLASICO 200G MAS NESCAFE ICE 40G': [134.2169398907104],
 'NESCAFE CLASICO DAWN JAR 300G': [180.46643726839596],
 'NESCAFE DECAF 170+40 G': [141.8261658031088],
 'NESCAFE DECAF 170G': [151.22905982905982],
 'NESCAFE DOLCA 170 G GRATIS ARROZ SCHETTINO': [98.9],
 'NESCAFE DOLCA 80 G GRATIS 46 G': [99.9],
 'NESCAFE DOLCA 80 G Y COFFEE MATE 160 G PRECIO ESPECIAL': [99.9],
 'NESCAFE DOLCA JIRAFA 170 G + 30% GRATIS': [98.9],
 'NESCAFE ICE 40 G': [47.9],
 "NESCAFE TASTER'S CHOICE, CAFE TOSTADO Y MOLIDO 300 G": [189.9],
 'NESQUIK CHOCOLATE LPB TETRA 240 ML.': [12.9],
 'NESQUIK MALTEADA FRESA BOLSA 357 G': [59.9],
 'NESQUIK POLVO FRESA BOLSA 200G': [36.815],
 'NESTLE MEDIA CREMA 225G PR 20%FREE MX': [20.08796992481203],
 'NESTUM 2AET CEREAL 4 CER. ECO.270GRS': [55.9],
 'NIDAL 1 FORMULA INFANTIL 330G': [61.9],
 'NIDAL 1 FORMULA INFANTIL 900GR': [309.522641509434],
 'NIDAL 2 FORMULA INFANTIL 120 G.': [37.9],
 'PUFFS GERBER CALABACITA 35 G': [34.9],
 'PUFFS GERBER PLATANO 42 G': [68.9],
 'SALSA DE SOYA MAGGI 140ML': [24.900000000000002],
 'SALSA DE SOYA MAGGI 290ML MAS140ML': [49.900000000000006],
 'SALSA INGL 50ML MAS JUGO MAGGI 45ML': [39.900000000000006],
 'SALSA INGLESA C&B 290 ML': [78.89999999999999],
 'SALSA INGLESA C&B. 145 ML.': [43.73233532934132],
 'SALSA MAGGI MICHE 145 ML': [19.900000000000002],
 'SALSA SOYA MAGGI 290 ML.': [38.9],
 'SOPA INSTANTANEA BBQ PICANTE MAGGI 67G': [19.9],
 'SOPA INSTANTANEA ESPECIES MAGGI 67G': [19.9],
 'SOPA INSTANTANEA TERIYAKI MAGGI 70G': [19.900000000000002],
 'STARBUCKS CAFE FRAPPUCCINO CARAMELO 280 ML': [36.764667393675026],
 'SUSTITUTO DE CREMA EXTRA CREMOSO COFFE MATE 365 GR.': [69.86222222222223],
 'SUSTITUTO DE CREMA PARA CAFE COFFEE MATE ORIGINAL 900 GR.': [130.7263464566929],
 'TRIPACK CARNATION LECHE EVAPORADA 360 G': [59.9],
 'TRIPACK CARNATION LECHE EVAPORADA 360G': [59.9],
 'TRIPACK CARNATION LECHE EVAPORADA UHT 360G': [59.9],
 'TRIPACK COFFEE MATE VARIOS SABORES 530 G': [109.9],
 'TRIPACK LA LECHERA 375 G': [86.27867647058824],
 'TRIPACK LA LECHERA NESTLE 375G': [69.16182965299684],
 'TRIPACK MEDIA CREMA NESTLE 225G': [49.883974358974356],
 'YOGUR GERBER FRUTAS MIXTAS 95 G': [22.900000000000002],
 'YOGUR GERBER MANGO 95 G': [22.900000000000002],
 'YOGUR GERBER MORA AZUL 95 G': [22.9],
 'YOGUR GERBER PLATANO MANZANA Y FRESA 95 G': [22.9]}
PRECIOS_PREDICCIONES_2MESES = {'***DESCONT***SALSA INGLESA CROSSE & BLACKWELL 50ML': [20.900000000000002],
 '1 COFFEE MATE 160G MAS LECHERA ORIGINAL 375G': [49.9],
 '1 COFFEE MATE 800G GRATIS MAGGI MICHE 145ML': [88.57605633802817],
 '1 COFFEE MATE ORIGINAL 900G GRATIS AZUCAR 900G': [127.97692307692309],
 '1 LECHERA 375 G MAS CARNATION 360 G PRECIO ESP': [49.9],
 '1 NESCAFE CLASICO 200G GRATIS 1 COFFEE MATE 530G': [100.9],
 '1 NESCAFE CLASICO 300G GRATIS AZUCAR 900G': [189.9],
 '1 NESCAFE DOLCA 170G MAS COFFEE MATE 160G': [98.9],
 '2 COFFEE MATE 530G MAS ESPRESSO BLACK PRECIO ESP': [69.9],
 '2 COFFEE MATE 530G MAS ESPRESSO DE VAINILLA PRECIO ESP': [69.9],
 '2 LECHERA 375G MAS 1 CARNATION 360G GRATIS HARINA PARA HOT CAKES': [74.9],
 '2 NESCAFE DOLCA 170G MAS COFFEE MATE 160G': [196.9],
 '3 PACK CARNATION LECHE EVAPORADA LATA 360G': [59.9],
 'AGUA DE MANANTIAL SANTA MARIA BOTELLA 4LT': [33.9],
 'AGUA DE MANANTIAL STA MARIA BOTELLA 1L': [9.553395083719273],
 'AGUA PUREZA VITAL NESTLE 1.5 LT': [13.711508810572687],
 'AGUA PUREZA VITAL NESTLE 1LT': [7.396983105390184],
 'AGUA PURIFICADA GERBER 4 L': [32.89497487437186],
 'AGUA SANTA MARIA NESTLE 1.5 LT': [14.670495342929721],
 'AGUA SANTA MARIA NESTLE 500 ML': [7.386059299191374],
 'BEBIDA LATTE NESCAFE 330ML': [26.9],
 'CAFE DE OLLA INSTANTANEO NESCAFE 170 G': [88.54601769911505],
 'CAFE DE OLLA INSTANTANEO NESCAFE 46 G': [35.87174657534246],
 'CAFE RESERVA MEXICANA CHIAPAS 180GR': [144.14178154825026],
 'CAFE RESERVA MEXICANA VERACRUZ 180 GR': [143.91415571284125],
 'CAFE SOLUBLE NESCAFE CLASICO 1 KG': [559.9000000000001],
 'CAFE SOLUBLE NESCAFE CLASICO 120 G': [109.82486851990984],
 'CAFE SOLUBLE NESCAFE CLASICO 170 G': [109.87441860465117],
 'CAFE SOLUBLE NESCAFE CLASICO 200 G': [134.53223140495868],
 'CAFE SOLUBLE NESCAFE CLASICO 225 G + 25% GRATIS IN & OUT': [134.49035444509005],
 'CAFE SOLUBLE NESCAFE CLASICO 2X120 G': [139.9],
 'CAFE SOLUBLE NESCAFE CLASICO 42 G': [47.62474738279472],
 'CAFE SOLUBLE NESCAFE CLASICO 60 G': [73.9],
 'CAFE SOLUBLE NESCAFE CLASICO 85 GR': [64.9],
 'CAFE SOLUBLE NESCAFE CLASICO BLACK 170 G': [109.9],
 'CAFE SOLUBLE NESCAFE CLASICO MAS TAZA 200G': [134.19945054945055],
 'CAFE SOLUBLE NESCAFE DECAF 120 G': [124.9],
 'CAFE SOLUBLE NESCAFE DECAF 40 G': [48.843180260452364],
 'CAFE SOLUBLE NESCAFE DOLCA 1 KG': [447.8629629629629],
 'CAFE SOLUBLE NESCAFE DOLCA 170 G': [98.9],
 'CAFE SOLUBLE NESCAFE DOLCA 300 G': [162.07730496453902],
 'CAFE SOLUBLE NESCAFE DOLCA 46 G': [38.890694171578254],
 'CAFE SOLUBLE NESCAFE DOLCA 80 G': [68.9],
 'CAFE SOLUBLE NESCAFE DOLCA DOYPACK DE 22 G': [23.900000000000002],
 'CAFE SOLUBLE NESCAFE TASTERS CHOICE DESCAFEINADO 100 G': [167.94123711340205],
 'CAFE SOLUBLE NESCAFE TASTERS CHOICE GOURMET 100 G': [166.26],
 'CALABAZA HALLOWEEN NESTLE': [55.462],
 'CARNATION LECHE EVAPORADA UHT 360GR': [23.698230088495574],
 'CARNATION ORIGINAL UHT 720G + LLCH LATA 375G IN&OUT': [59.9],
 'CEREAL CARLOS V NESTLE 590 G': [64.95885454545456],
 'CEREAL CHEERIOS 480 G + CEREAL NESQUICK 620 G': [113.4322513761468],
 'CEREAL CHEERIOS CON MIEL Y ALMENDRAS 480 G': [68.23253594306051],
 'CEREAL CHEERIOS HONEY NESTLE 230 G': [23.9004],
 'CEREAL CHEERIOS MANZANA CANELA NESTLE CAJA 480 G': [67.53163725690892],
 'CEREAL CHEERIOS MIEL 480 G MAS CF NESTLE 500 G GRATIS TRIX 430 GR': [138.8],
 'CEREAL CINNAMON TOAST CRUNCH NESTLE 340GR': [84.24574631828978],
 'CEREAL COOKIE CRISP NESTLE 480 G': [66.53592097428958],
 'CEREAL CORN FLAKES 500 G MAS TRIX 430 G NESTLE PRECIO ESPECIAL': [92.5],
 'CEREAL CORN FLAKES NESTLE 500 G': [59.64801455563332],
 'CEREAL INFLADO GERBER 42 G': [66.89999999999999],
 'CEREAL INFLADO JUNIOR GERBER 42': [68.9],
 'CEREAL INFLADO MORA AZUL GERBER 42 G': [68.89999999999999],
 'CEREAL INTEGRAL FITNESS ORIGINAL NESTLE 570 G': [74.30326171875001],
 'CEREAL LUCKY CHARMS NESTLE 297 G': [83.59008969359333],
 'CEREAL NESQUICK NESTLE 620 G': [75.80521529745043],
 'CEREAL NESQUIK 230G MAS CEREAL TRIX 230G': [49.89877105263158],
 'CEREAL NESQUIK 620 G MAS PAPEL KLEENEX 4 PZS': [69.35155844155844],
 'CEREAL NESQUIK 620 Y TRIX 430G MAS ZUCOSOS 650G': [116.92148594377511],
 'CEREAL NESQUIK 620G MAS ZUCOSOS 650G': [79.89999999999999],
 'CEREAL NESQUIK DE FRESA 310G': [35.47193142857143],
 'CEREAL NESQUIK FRESA 310G MAS CF 500G P ESPECIAL': [73.98],
 'CEREAL TRIGO INTEGRAL MIEL GERBER 270 G.': [58.404672897196264],
 'CEREAL TRIGO INTEGRAL Y MANZANA GERBER 270 GRS': [59.9],
 'CEREAL TRIX NESTLE 430 GR.': [74.81680896226416],
 'CEREAL TRIX SABOR MARSHMALLOW 230G': [36.53463898916967],
 'CEREAL ZUCOSOS NESTLE 650G': [59.26908387096775],
 'CEREALES NESQUIK 620G MAS ZUCOSOS 650G': [99.9],
 'CHOCO ABUELITA 540G Y CARNATION 360G': [120.27888888888889],
 'CHOCOLATE ABUELITA 270 GR (3 TABLETAS DE 90GR C/U)': [79.89990821052632],
 'CHOCOLATE CARLOS V DE 216G MAS UNO': [111.11],
 'CHOCOLATE CARLOS V NESTLE 18 G': [13.794964055996973],
 'CHOCOLATE CARLOS V STICK 20 PZAS DE 8 G C/U': [59.898685714285726],
 'CHOCOLATE CRUNCH 40 G': [19.11044237837838],
 'CHOCOLATE CRUNCH STICK NESTLE 20 PZAS': [45.423815789473686],
 'CHOCOLATE EN LATA NESQUICK 400 G': [71.88110816326531],
 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 357 G': [47.899896122896855],
 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 445G': [47.899956521739135],
 'CHOCOLATE EN TABLETA ABUELITA NESTLE 540 G': [139.82751778546714],
 'CHOCOLATE GRANULADO ABUELITA 320 G': [52.535926133184105],
 'CHOCOLATE KIT KAT 41.5 G': [18.158146820809247],
 'CHOCOLATE KIT KAT CAJA CON 9 PZAS DE 41.5 G C/U': [161.94125],
 'CHOCOLATE KIT KAT DARK NESTLE 41.5G': [19.7279538883806],
 'CHOCOLATE KITKAT CHUNKY NESTLE 40G': [18.170086646884275],
 'CHOCOLATE LARIN CERO AZUCAR 19 G': [19.879375977653634],
 'CHOCOLATE POLVO NESQUICK BOLSA 200 G': [32.80890906593407],
 'CHOCOLATE POLVO NESQUICK BOLSA 357 G': [56.05263378271214],
 'COFFEE MATE 311 G GRATIS THERMO': [57.9],
 'COFFEE MATE 311G GRATIS VASO NESTLE': [60.9],
 'COFFEE MATE DULCE DE LECHE 530G': [44.9],
 'COFFEE MATE EXTRA CREMOSO 530 G': [49.900000000000006],
 'COFFEE MATE LIQUIDO AVELLANA 769.7 G': [58.98227762803234],
 'COFFEE MATE LIQUIDO ORIGINAL 769.7 G': [59.22159406858202],
 'COFFEE MATE MAZAPAN NESTLE 530G': [19.9],
 'COFFEE MATE ORIGINAL 400G GRATIS AZUCAR DULY': [65.99271523178808],
 'COFFEE MATE ORIGINAL 800 G': [84.98863253200126],
 'COFFEE MATE PUMPKIN SPICE POLVO 350 G': [32.9],
 'COLADO DE DURAZNO ET 3 GERBER 170 G': [20.900000000000002],
 'COLADO GERBER 3AET FRUT.TROP-170 GRS': [20.9],
 'COLADO GERBER 3AET MANZANA 170G MX': [20.9],
 'COLADO GERBER 3AET PERA 170 GRS': [20.9],
 'COLADO GERBER 3AET VEG PAVO Y PASTA 170 G': [20.9],
 'COLADO GERBER 3AET VEGETALES Y PASTA 170 G': [20.9],
 'COLADO GERBER 3AET VERD/POLL/ARR 170GMX': [20.900000000000002],
 'COLADO GERBER 3RAET VER/RES/ARR 170 GRS': [20.9],
 'CONSOMATE EN POLVO 5 SOBRES DE 21 G': [14.9],
 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI 12 CUBOS 105 G': [17.9],
 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI CAJA 8 CUBOS 84 G': [18.900000000000002],
 'CREMA EN POLVO PARA CAFE COFFEE MATE BOLSA 210 G': [39.9],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 160 G': [37.852536997885835],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 311 G': [60.51339963833635],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 400 G': [65.62869565217392],
 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 520 G': [67.66975169300225],
 'CREMA LIQUIDA PARA CAFE BAILEYS 468 ML': [39.9],
 'CREMA LIQUIDA PARA CAFE CARLOS V 530 ML': [19.9],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE AVELLANA 530 G': [39.63835920177384],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE CANELA 530 ML': [39.9],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE ORIGINAL 530 ML': [39.77684729064039],
 'CREMA LIQUIDA PARA CAFE COFFEE MATE VAINILLA 530 G': [39.62786772304513],
 'DULCE DE LECHE LA LECHERA POUCH 209 G': [25.83321548821549],
 'DULCE LECHE LECHERA SIRVE FACIL 325 GR': [43.428],
 'DUO PACK CARNATION UHT 360 G': [37.9],
 'DUO PACK COFFEE MATE 769.7 G': [99.89999999999999],
 'FRESKAS NESTLE 35 G': [16.406799874725966],
 'FRESKAS NESTLE 9 PZAS CON 35 G C/U': [111.02],
 'GERBER 2A ETAPA POLLO Y ARROZ 100 G': [12.899999999999999],
 'GERBER 2A ETAPA PURE DE MANZANA 100 G': [12.9],
 'GERBER 2AET CIRUELA 100G': [12.899999999999999],
 'GERBER 2AET DURAZNO 100GR': [12.899999999999999],
 'GERBER 2AET FRUTAS MIXTAS 100G': [12.899999999999999],
 'GERBER 2AET MANGO 100G': [12.9],
 'GERBER 2AET PERA 100G': [12.9],
 'GERBER 2AET PLATANO 100G': [12.899999999999999],
 'GERBER 2AET PURE FRU TROPICALES 113G': [16.9],
 'GERBER 2AET RES Y ARROZ 100G': [12.9],
 'GERBER ET JNR MANZANA POUCH LE 195G': [13.9],
 'GERBER ET JUNIO MANGO 95G': [13.9],
 'GERBER ET JUNIOR FRUTAS MIXTAS 95G': [13.9],
 'IN&OUT COFFEE MATE LIQUIDO VAINILLA 769.7G': [59.20309614340032],
 'IN&OUT NESCAFE ICE 170 G': [132.27724550898202],
 'JUGO GERBER JUNIOR DE MANGO 200 ML': [14.9],
 'JUGO GERBER JUNIOR FRUTAS MIXTAS 200 ML': [14.9],
 'JUGO GERBER JUNIOR MANZANA 200 ML': [14.899999999999999],
 'JUGO GERBER JUNIOR PERA 200 ML': [14.9],
 'JUGO SAZONADOR CLASICO MAGGI 100 ML': [43.83511988716502],
 'JUGO SAZONADOR MAGGI 125G + SALSA PICANTE VALENTINA 250ML': [39.9],
 'JUGO SAZONADOR MAGGI 45 ML': [20.9],
 'JUGO SAZONADOR MAGGI BOTELLA 200 ML': [72.89999999999999],
 'JUGOSO AL SARTEN AJO Y CEBOLLA MAGGI SOBRE 4 PZAS 23.2 G': [21.9],
 'JUGOSO AL SARTEN FINAS HIERBAS MAGGI SOBRE 4 PZAS 23.4 G': [21.900000000000002],
 'JUGOSO AL SARTEN MAGGI A LA LENA 24 G': [21.9],
 'LA LECHERA ORIGINAL BOLSA 209G': [16.238235294117647],
 'LA LECHERA ORIGINAL CARTON 430 G': [32.344308445532434],
 'LA LECHERA SIRVE FACIL 335 GRS': [47.900000000000006],
 'LA LECHERA TRES LECHES 1 KG': [71.83333333333333],
 'LECHE CARLOS V CHOCOLATE NESTLE 185 ML': [9.898305084745763],
 'LECHE CARLOS V CHOCOLATE NESTLE 240 ML': [12.9],
 'LECHE CARNATION 1 LTR': [62.80529073114565],
 'LECHE CARNATION POLVO DOY PACK 460 GRS': [63.798400752587014],
 'LECHE CLAVEL DESLACTOSADA 360 GRS.': [24.01264367816092],
 'LECHE CONDENSADA LA LECHERA ORIGINAL NESTLE 375 G': [29.632214620797498],
 'LECHE EN POLVO CARNATION DOY PACK 520 G': [67.9],
 'LECHE EVAPORADA CARNATION 360 G': [21.728692257836435],
 'LECHE EVAPORADA EN POLVO CARNATION 120GR': [19.466433566433565],
 'LECHE EVAPORADA NESTLE CARNATION CLAVEL 720 G': [42.90500500500501],
 'LECHE NAN 1 OPTIPRO 1.2KG': [624.9],
 'LECHE NAN 2 OPTIPRO 720GR': [349.9],
 'LECHE NESQUIK CHOCOLATE 185 ML': [9.763246268656717],
 'LECHE NESTLE NUTRI RINDES 460 G GRATIS DOLCA 22 G': [65.9],
 'LECHE NESTLE NUTRI RINDES 460 GRS.': [65.89999999999999],
 'LECHE NIDO CLASICA FORTI 720G': [141.9],
 'LECHE NIDO FORTICRECE 1.56 KG': [269.9],
 'LECHE NIDO KINDER 1.4 KG': [257.9152671755725],
 'LECHE NIDO KINDER 360 GR.': [69.05299334811531],
 'LECHE NIDO KINDER 800 GR.': [169.9],
 'LECHE SABORIZADA ABUELITA SABOR CHOCOLATE 930 ML': [44.900000000000006],
 'LECHERA NESTLE 100 GR.': [12.881836528758829],
 'MALTEADA SABOR FRESA NESQUIK 185 ML': [9.57387580299786],
 'MALTEADA SABOR VAINILLA NESQUIK 185 ML': [9.362349676225717],
 'MALTEADA VAINILLA NESQUICK BOLSA 357 G': [59.9],
 'MEDIA CREMA NESTLE 1 LTR': [73.93434227330779],
 'MEDIA CREMA NESTLE 190 GR': [17.57597298284045],
 'MEDIA CREMA NESTLE 225 G': [19.99738618274435],
 'MEDIA CREMA NESTLE 500ML': [35.913400335008376],
 'NAN 3 OPTIPRO 720GR': [217.9281690140845],
 'NESCAFE CAFE DE OLLA 170+46 G': [88.9],
 'NESCAFE CAFE DE OLLA FRASCO 260 G': [121.72263329706203],
 'NESCAFE CAFE DE OLLA FRASCO 85 GR.': [55.85703125],
 'NESCAFE CAPPUCCINO KITKAT 6X25G': [32.89977551020409],
 'NESCAFE CAPPUCCINO MOKA 6 SOBRES X 22 GR': [64.72032258064516],
 'NESCAFE CAPPUCCINO ORIGINAL 6 SOBRES X 20 GR': [69.89999999999999],
 'NESCAFE CAPPUCCINO VAINILLA 6X22G': [69.9],
 'NESCAFE CLASICO 120 G GRATIS CAJA SPLENDA 50 PZ': [113.31463414634146],
 'NESCAFE CLASICO 120 G GRATIS TAZA': [110.9],
 'NESCAFE CLASICO 200 G GRATIS ARROZ SCHETTINO': [134.6],
 'NESCAFE CLASICO 200G MAS NESCAFE DECAF 40G': [142.86],
 'NESCAFE CLASICO 200G MAS NESCAFE ICE 40G': [134.2169398907104],
 'NESCAFE CLASICO DAWN JAR 300G': [180.46643726839596],
 'NESCAFE DECAF 170+40 G': [141.8261658031088],
 'NESCAFE DECAF 170G': [151.22905982905982],
 'NESCAFE DOLCA 170 G GRATIS ARROZ SCHETTINO': [98.9],
 'NESCAFE DOLCA 80 G GRATIS 46 G': [99.9],
 'NESCAFE DOLCA 80 G Y COFFEE MATE 160 G PRECIO ESPECIAL': [99.9],
 'NESCAFE DOLCA JIRAFA 170 G + 30% GRATIS': [98.9],
 'NESCAFE ICE 40 G': [47.9],
 "NESCAFE TASTER'S CHOICE, CAFE TOSTADO Y MOLIDO 300 G": [189.9],
 'NESQUIK CHOCOLATE LPB TETRA 240 ML.': [12.9],
 'NESQUIK MALTEADA FRESA BOLSA 357 G': [59.9],
 'NESQUIK POLVO FRESA BOLSA 200G': [36.815],
 'NESTLE MEDIA CREMA 225G PR 20%FREE MX': [20.08796992481203],
 'NESTUM 2AET CEREAL 4 CER. ECO.270GRS': [55.9],
 'NIDAL 1 FORMULA INFANTIL 330G': [61.9],
 'NIDAL 1 FORMULA INFANTIL 900GR': [309.522641509434],
 'NIDAL 2 FORMULA INFANTIL 120 G.': [37.9],
 'PUFFS GERBER CALABACITA 35 G': [34.9],
 'PUFFS GERBER PLATANO 42 G': [68.9],
 'SALSA DE SOYA MAGGI 140ML': [24.900000000000002],
 'SALSA DE SOYA MAGGI 290ML MAS140ML': [49.900000000000006],
 'SALSA INGL 50ML MAS JUGO MAGGI 45ML': [39.900000000000006],
 'SALSA INGLESA C&B 290 ML': [78.89999999999999],
 'SALSA INGLESA C&B. 145 ML.': [43.73233532934132],
 'SALSA MAGGI MICHE 145 ML': [19.900000000000002],
 'SALSA SOYA MAGGI 290 ML.': [38.9],
 'SOPA INSTANTANEA BBQ PICANTE MAGGI 67G': [19.9],
 'SOPA INSTANTANEA ESPECIES MAGGI 67G': [19.9],
 'SOPA INSTANTANEA TERIYAKI MAGGI 70G': [19.900000000000002],
 'STARBUCKS CAFE FRAPPUCCINO CARAMELO 280 ML': [36.764667393675026],
 'SUSTITUTO DE CREMA EXTRA CREMOSO COFFE MATE 365 GR.': [69.86222222222223],
 'SUSTITUTO DE CREMA PARA CAFE COFFEE MATE ORIGINAL 900 GR.': [130.7263464566929],
 'TRIPACK CARNATION LECHE EVAPORADA 360 G': [59.9],
 'TRIPACK CARNATION LECHE EVAPORADA 360G': [59.9],
 'TRIPACK CARNATION LECHE EVAPORADA UHT 360G': [59.9],
 'TRIPACK COFFEE MATE VARIOS SABORES 530 G': [109.9],
 'TRIPACK LA LECHERA 375 G': [86.27867647058824],
 'TRIPACK LA LECHERA NESTLE 375G': [69.16182965299684],
 'TRIPACK MEDIA CREMA NESTLE 225G': [49.883974358974356],
 'YOGUR GERBER FRUTAS MIXTAS 95 G': [22.900000000000002],
 'YOGUR GERBER MANGO 95 G': [22.900000000000002],
 'YOGUR GERBER MORA AZUL 95 G': [22.9],
 'YOGUR GERBER PLATANO MANZANA Y FRESA 95 G': [22.9]}
ALIAS_PRECIOS_POR_CODIGO = [('12198', 'CALABAZA HALLOWEEN NESTLE'),
 ('13123', '3 PACK CARNATION LECHE EVAPORADA LATA 360G'),
 ('13134', 'TRIPACK COFFEE MATE VARIOS SABORES 530 G'),
 ('13229', 'DUO PACK COFFEE MATE 769.7 G'),
 ('13232', 'NESCAFE DOLCA 80 G Y COFFEE MATE 160 G PRECIO ESPECIAL'),
 ('13436', 'CEREAL CORN FLAKES 500 G MAS TRIX 430 G NESTLE PRECIO ESPECIAL'),
 ('13439', 'NESCAFE CLASICO 120 G GRATIS TAZA'),
 ('13458', 'NESCAFE CLASICO 120 G GRATIS CAJA SPLENDA 50 PZ'),
 ('13464', '1 NESCAFE CLASICO 200G GRATIS 1 COFFEE MATE 530G'),
 ('13467', '1 NESCAFE CLASICO 200G GRATIS 1 COFFEE MATE 530G'),
 ('13479', '1 NESCAFE CLASICO 300G GRATIS AZUCAR 900G'),
 ('13481', '2 NESCAFE DOLCA 170G MAS COFFEE MATE 160G'),
 ('13482', '1 NESCAFE DOLCA 170G MAS COFFEE MATE 160G'),
 ('13487', '1 COFFEE MATE 160G MAS LECHERA ORIGINAL 375G'),
 ('13489', 'CEREAL NESQUIK FRESA 310G MAS CF 500G P ESPECIAL'),
 ('13492', 'CEREAL CHEERIOS MIEL 480 G MAS CF NESTLE 500 G GRATIS TRIX 430 GR'),
 ('13494', '1 COFFEE MATE ORIGINAL 900G GRATIS AZUCAR 900G'),
 ('13495', 'CEREAL NESQUIK 620 Y TRIX 430G MAS ZUCOSOS 650G'),
 ('13496', 'CEREAL NESQUIK 620 G MAS PAPEL KLEENEX 4 PZS'),
 ('13512', '2 LECHERA 375G MAS 1 CARNATION 360G GRATIS HARINA PARA HOT CAKES'),
 ('13513', '1 COFFEE MATE 800G GRATIS MAGGI MICHE 145ML'),
 ('13522', '2 COFFEE MATE 530G MAS ESPRESSO BLACK PRECIO ESP'),
 ('13523', '2 COFFEE MATE 530G MAS ESPRESSO DE VAINILLA PRECIO ESP'),
 ('13524', 'COFFEE MATE ORIGINAL 400G GRATIS AZUCAR DULY'),
 ('13531', 'NESCAFE CLASICO 200 G GRATIS ARROZ SCHETTINO'),
 ('13532', 'NESCAFE DOLCA 170 G GRATIS ARROZ SCHETTINO'),
 ('13533', 'CEREAL NESQUIK 620G MAS ZUCOSOS 650G'),
 ('13736', 'CHOCO ABUELITA 540G Y CARNATION 360G'),
 ('13737', '1 LECHERA 375 G MAS CARNATION 360 G PRECIO ESP'),
 ('13789', 'TRIPACK LA LECHERA 375 G'),
 ('13799', 'COFFEE MATE 311 G GRATIS THERMO'),
 ('13803', 'COFFEE MATE 311G GRATIS VASO NESTLE'),
 ('160', 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 311 G'),
 ('16000135437', 'CEREAL LUCKY CHARMS NESTLE 297 G'),
 ('16000221284', 'CEREAL CINNAMON TOAST CRUNCH NESTLE 340GR'),
 ('17501001604315', 'SALSA INGLESA C&B. 145 ML.'),
 ('17501001619043', 'LECHE CARNATION 1 LTR'),
 ('3800020423547', 'CHOCOLATE KITKAT CHUNKY NESTLE 40G'),
 ('45000011304', 'CHOCOLATE CRUNCH STICK NESTLE 20 PZAS'),
 ('75000011', 'NESQUIK CHOCOLATE LPB TETRA 240 ML.'),
 ('75003456', 'GERBER 2AET PURE FRU TROPICALES 113G'),
 ('75004705', 'COLADO GERBER 3AET MANZANA 170G MX'),
 ('75004712', 'COLADO DE DURAZNO ET 3 GERBER 170 G'),
 ('75004729', 'COLADO GERBER 3AET FRUT.TROP-170 GRS'),
 ('75004767', 'COLADO GERBER 3AET PERA 170 GRS'),
 ('7501000906246', 'COLADO GERBER 3AET VERD/POLL/ARR 170GMX'),
 ('7501000906253', 'COLADO GERBER 3RAET VER/RES/ARR 170 GRS'),
 ('7501000906284', 'COLADO GERBER 3AET VEG PAVO Y PASTA 170 G'),
 ('7501000906680', 'COLADO GERBER 3AET VEGETALES Y PASTA 170 G'),
 ('7501000909568', 'CEREAL INFLADO JUNIOR GERBER 42'),
 ('7501000912605', 'CAFE SOLUBLE NESCAFE DOLCA DOYPACK DE 22 G'),
 ('7501000912803', 'CAFE SOLUBLE NESCAFE CLASICO 42 G'),
 ('7501000913367', 'LECHE CLAVEL DESLACTOSADA 360 GRS.'),
 ('7501000984770', 'AGUA PURIFICADA GERBER 4 L'),
 ('7501001600426', 'MEDIA CREMA NESTLE 225 G'),
 ('7501001604103', 'JUGO SAZONADOR CLASICO MAGGI 100 ML'),
 ('7501001604110', 'JUGO SAZONADOR MAGGI BOTELLA 200 ML'),
 ('7501001604325', 'SALSA INGLESA C&B 290 ML'),
 ('7501001614027', 'MEDIA CREMA NESTLE 1 LTR'),
 ('7501058610942', 'CAFE SOLUBLE NESCAFE CLASICO 170 G'),
 ('7501058610959', 'BEBIDA LATTE NESCAFE 330ML'),
 ('7501058611062', 'LECHE EVAPORADA CARNATION 360 G'),
 ('7501058615596', 'LA LECHERA TRES LECHES 1 KG'),
 ('7501058616470', 'NESCAFE DECAF 170G'),
 ('7501058616548', 'CAFE SOLUBLE NESCAFE DOLCA 170 G'),
 ('7501058618597', 'CHOCOLATE POLVO NESQUICK BOLSA 200 G'),
 ('7501058618917', 'CAFE DE OLLA INSTANTANEO NESCAFE 170 G'),
 ('7501058618924', 'CAFE DE OLLA INSTANTANEO NESCAFE 46 G'),
 ('7501058618931', 'NESCAFE CAFE DE OLLA FRASCO 85 GR.'),
 ('7501058619211', 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 400 G'),
 ('7501058619228', 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 160 G'),
 ('7501058619235', 'CREMA EN POLVO PARA CAFE COFFEE MATE BOLSA 210 G'),
 ('7501058620002', 'CAFE SOLUBLE NESCAFE DOLCA 46 G'),
 ('7501058620019', 'CAFE SOLUBLE NESCAFE DOLCA 80 G'),
 ('7501058620095', 'CAFE SOLUBLE NESCAFE CLASICO 60 G'),
 ('7501058620101', 'CAFE SOLUBLE NESCAFE CLASICO 120 G'),
 ('7501058622099', 'NESCAFE DOLCA JIRAFA 170 G + 30% GRATIS'),
 ('7501058623256', 'LECHE NAN 1 OPTIPRO 1.2KG'),
 ('7501058623348', 'CHOCOLATE CARLOS V STICK 20 PZAS DE 8 G C/U'),
 ('7501058624130', 'CAFE RESERVA MEXICANA CHIAPAS 180GR'),
 ('7501058624147', 'CAFE RESERVA MEXICANA VERACRUZ 180 GR'),
 ('7501058624635', 'NESCAFE CLASICO DAWN JAR 300G'),
 ('7501058624666', 'CHOCOLATE ABUELITA 270 GR (3 TABLETAS DE 90GR C/U)'),
 ('7501058625229', 'LECHE NAN 2 OPTIPRO 720GR'),
 ('7501058625236', 'NAN 3 OPTIPRO 720GR'),
 ('7501058626226', 'CEREAL INFLADO MORA AZUL GERBER 42 G'),
 ('7501058628466', 'CAFE SOLUBLE NESCAFE DECAF 40 G'),
 ('7501058628473', 'CAFE SOLUBLE NESCAFE DECAF 120 G'),
 ('7501058628503', 'SALSA DE SOYA MAGGI 140ML'),
 ('7501058628664', '***DESCONT***SALSA INGLESA CROSSE & BLACKWELL 50ML'),
 ('7501058628718', 'LECHE CARLOS V CHOCOLATE NESTLE 240 ML'),
 ('7501058628831', 'LECHE EVAPORADA NESTLE CARNATION CLAVEL 720 G'),
 ('7501058629913', 'CHOCOLATE KIT KAT CAJA CON 9 PZAS DE 41.5 G C/U'),
 ('7501058631961', 'LA LECHERA ORIGINAL BOLSA 209G'),
 ('7501058632227', 'CREMA EN POLVO PARA CAFE COFFEE MATE ORIGINAL 520 G'),
 ('7501058637659', 'JUGO GERBER JUNIOR DE MANGO 200 ML'),
 ('7501058637734', 'CEREAL CHEERIOS HONEY NESTLE 230 G'),
 ('7501058638076', 'MEDIA CREMA NESTLE 500ML'),
 ('7501058638083', 'CHOCOLATE CARLOS V NESTLE 18 G'),
 ('7501058639493', 'PUFFS GERBER PLATANO 42 G'),
 ('7501058642127', 'CREMA LIQUIDA PARA CAFE COFFEE MATE ORIGINAL 530 ML'),
 ('7501058642134', 'CREMA LIQUIDA PARA CAFE COFFEE MATE VAINILLA 530 G'),
 ('7501058642141', 'CREMA LIQUIDA PARA CAFE COFFEE MATE AVELLANA 530 G'),
 ('7501058642165', 'CREMA LIQUIDA PARA CAFE COFFEE MATE CANELA 530 ML'),
 ('7501058642592', 'CAFE SOLUBLE NESCAFE TASTERS CHOICE DESCAFEINADO 100 G'),
 ('7501058642608', 'CAFE SOLUBLE NESCAFE TASTERS CHOICE GOURMET 100 G'),
 ('7501058644824', 'LECHE CARNATION POLVO DOY PACK 460 GRS'),
 ('7501058644848', 'LECHE EN POLVO CARNATION DOY PACK 520 G'),
 ('7501058644862', 'LECHE EVAPORADA EN POLVO CARNATION 120GR'),
 ('7501058650665', 'SALSA INGL 50ML MAS JUGO MAGGI 45ML'),
 ('7501058651129', 'GERBER ET JUNIOR FRUTAS MIXTAS 95G'),
 ('7501058651136', 'GERBER ET JUNIO MANGO 95G'),
 ('7501058651273', 'CEREAL CARLOS V NESTLE 590 G'),
 ('7501058652652', 'CEREAL INTEGRAL FITNESS ORIGINAL NESTLE 570 G'),
 ('7501058652683', 'CEREAL CORN FLAKES NESTLE 500 G'),
 ('7501058652690', 'CEREAL NESQUICK NESTLE 620 G'),
 ('7501058654205', 'SUSTITUTO DE CREMA PARA CAFE COFFEE MATE ORIGINAL 900 GR.'),
 ('7501058654793', "NESCAFE TASTER'S CHOICE, CAFE TOSTADO Y MOLIDO 300 G"),
 ('7501058655172', 'SUSTITUTO DE CREMA EXTRA CREMOSO COFFE MATE 365 GR.'),
 ('7501058655219', 'CREMA LIQUIDA PARA CAFE CARLOS V 530 ML'),
 ('7501058656247', 'CHOCOLATE LARIN CERO AZUCAR 19 G'),
 ('7501059200012', 'AGUA SANTA MARIA NESTLE 1.5 LT'),
 ('7501059200050', 'AGUA SANTA MARIA NESTLE 500 ML'),
 ('7501059200777', 'AGUA PUREZA VITAL NESTLE 1.5 LT'),
 ('75010592096330', 'CAFE SOLUBLE NESCAFE CLASICO 1 KG'),
 ('7501059209657', 'CAFE SOLUBLE NESCAFE DOLCA 1 KG'),
 ('7501059211209', 'LECHERA NESTLE 100 GR.'),
 ('7501059214385', 'CHOCOLATE EN LATA NESQUICK 400 G'),
 ('7501059214590', 'CHOCOLATE POLVO NESQUICK BOLSA 357 G'),
 ('7501059215764', 'AGUA PUREZA VITAL NESTLE 1LT'),
 ('7501059223905', 'CHOCOLATE CRUNCH 40 G'),
 ('7501059224827', 'CAFE SOLUBLE NESCAFE CLASICO 200 G'),
 ('7501059225350', 'LECHE NIDO KINDER 800 GR.'),
 ('7501059225411', 'LECHE NIDO KINDER 360 GR.'),
 ('7501059229211', 'NESQUIK MALTEADA FRESA BOLSA 357 G'),
 ('7501059229228', 'MALTEADA VAINILLA NESQUICK BOLSA 357 G'),
 ('7501059231962', 'CAFE SOLUBLE NESCAFE DOLCA 300 G'),
 ('7501059233980', 'CHOCOLATE GRANULADO ABUELITA 320 G'),
 ('7501059234390', 'LECHE NESTLE NUTRI RINDES 460 GRS.'),
 ('7501059235441', 'AGUA DE MANANTIAL STA MARIA BOTELLA 1L'),
 ('7501059236776', 'AGUA DE MANANTIAL SANTA MARIA BOTELLA 4LT'),
 ('7501059237131', 'JUGOSO AL SARTEN AJO Y CEBOLLA MAGGI SOBRE 4 PZAS 23.2 G'),
 ('7501059239630', 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 357 G'),
 ('7501059240742', 'SALSA SOYA MAGGI 290 ML.'),
 ('7501059241060', 'CEREAL COOKIE CRISP NESTLE 480 G'),
 ('7501059243873', 'CEREAL CHEERIOS CON MIEL Y ALMENDRAS 480 G'),
 ('7501059243880', 'CEREAL CHEERIOS MANZANA CANELA NESTLE CAJA 480 G'),
 ('7501059278691', 'NESTUM 2AET CEREAL 4 CER. ECO.270GRS'),
 ('7501059278868', 'LA LECHERA SIRVE FACIL 335 GRS'),
 ('7501059281165', 'FRESKAS NESTLE 35 G'),
 ('7501059281172', 'FRESKAS NESTLE 9 PZAS CON 35 G C/U'),
 ('7501059281660', 'DULCE LECHE LECHERA SIRVE FACIL 325 GR'),
 ('7501059287938', 'NESTLE MEDIA CREMA 225G PR 20%FREE MX'),
 ('7501059297586', 'JUGO SAZONADOR MAGGI 45 ML'),
 ('7501073411173', 'CHOCOLATE EN TABLETA ABUELITA NESTLE 540 G'),
 ('7506475100250', 'CARNATION LECHE EVAPORADA UHT 360GR'),
 ('7506475101264', 'CREMA LIQUIDA PARA CAFE BAILEYS 468 ML'),
 ('7506475101981', 'COFFEE MATE EXTRA CREMOSO 530 G'),
 ('7506475102353', 'CAFE SOLUBLE NESCAFE CLASICO BLACK 170 G'),
 ('7506475102421', 'GERBER 2A ETAPA PURE DE MANZANA 100 G'),
 ('7506475102452', 'GERBER 2AET PERA 100G'),
 ('7506475102469', 'GERBER 2AET MANGO 100G'),
 ('7506475102476', 'GERBER 2AET DURAZNO 100GR'),
 ('7506475102483', 'GERBER 2AET CIRUELA 100G'),
 ('7506475102490', 'GERBER 2AET FRUTAS MIXTAS 100G'),
 ('7506475102506', 'GERBER 2AET PLATANO 100G'),
 ('7506475102520', 'GERBER 2AET RES Y ARROZ 100G'),
 ('7506475102537', 'GERBER 2A ETAPA POLLO Y ARROZ 100 G'),
 ('7506475102834', 'LECHE SABORIZADA ABUELITA SABOR CHOCOLATE 930 ML'),
 ('7506475103053', 'NIDAL 1 FORMULA INFANTIL 330G'),
 ('7506475103213', 'CEREAL TRIGO INTEGRAL MIEL GERBER 270 G.'),
 ('7506475103220', 'CEREAL TRIGO INTEGRAL Y MANZANA GERBER 270 GRS'),
 ('7506475103244', 'LECHE NIDO KINDER 1.4 KG'),
 ('7506475103855', 'NESCAFE CAFE DE OLLA FRASCO 260 G'),
 ('7506475104722', 'LECHE CONDENSADA LA LECHERA ORIGINAL NESTLE 375 G'),
 ('7506475106153', 'NIDAL 1 FORMULA INFANTIL 900GR'),
 ('7506475106818', 'NIDAL 2 FORMULA INFANTIL 120 G.'),
 ('7506475108492', 'CEREAL INFLADO GERBER 42 G'),
 ('7506475108829', 'IN&OUT COFFEE MATE LIQUIDO VAINILLA 769.7G'),
 ('7506475108935', 'JUGO SAZONADOR MAGGI 125G + SALSA PICANTE VALENTINA 250ML'),
 ('7506475110112', 'CAFE SOLUBLE NESCAFE CLASICO 85 GR'),
 ('7506475110983', 'NESCAFE CAPPUCCINO ORIGINAL 6 SOBRES X 20 GR'),
 ('7506475111003', 'NESCAFE CAPPUCCINO VAINILLA 6X22G'),
 ('7506475111027', 'NESCAFE CAPPUCCINO MOKA 6 SOBRES X 22 GR'),
 ('7506475111546', 'CEREAL TRIX NESTLE 430 GR.'),
 ('7506475111690', 'DUO PACK CARNATION UHT 360 G'),
 ('7506475111812', 'CEREAL CHEERIOS 480 G + CEREAL NESQUICK 620 G'),
 ('7506475112093', 'DULCE DE LECHE LA LECHERA POUCH 209 G'),
 ('7506475112956', 'LECHE NIDO CLASICA FORTI 720G'),
 ('7506475112963', 'LECHE NIDO FORTICRECE 1.56 KG'),
 ('7506475112970', 'COFFEE MATE LIQUIDO AVELLANA 769.7 G'),
 ('7506475113168', 'CHOCOLATE EN POLVO MORELIA PRESIDENCIAL 445G'),
 ('7506475113564', 'STARBUCKS CAFE FRAPPUCCINO CARAMELO 280 ML'),
 ('7506475113861', 'LA LECHERA ORIGINAL CARTON 430 G'),
 ('7506475113946', 'CAFE SOLUBLE NESCAFE CLASICO 225 G + 25% GRATIS IN & OUT'),
 ('7506475114073', 'GERBER ET JNR MANZANA POUCH LE 195G'),
 ('7506475114356', 'SALSA MAGGI MICHE 145 ML'),
 ('7506475114936', 'NESCAFE CAFE DE OLLA 170+46 G'),
 ('7506475114998', 'IN&OUT NESCAFE ICE 170 G'),
 ('7506475115254', 'NESCAFE DECAF 170+40 G'),
 ('7506475115421', 'YOGUR GERBER MORA AZUL 95 G'),
 ('7506475115438', 'YOGUR GERBER FRUTAS MIXTAS 95 G'),
 ('7506475116367', 'CARNATION ORIGINAL UHT 720G + LLCH LATA 375G IN&OUT'),
 ('7506475116893', 'CEREAL TRIX SABOR MARSHMALLOW 230G'),
 ('7506475117081', 'JUGO GERBER JUNIOR FRUTAS MIXTAS 200 ML'),
 ('7506475117364', 'COFFEE MATE ORIGINAL 800 G'),
 ('7506475117876', 'LECHE NESQUIK CHOCOLATE 185 ML'),
 ('7506475117883', 'LECHE CARLOS V CHOCOLATE NESTLE 185 ML'),
 ('7506475118446', 'SALSA DE SOYA MAGGI 290ML MAS140ML'),
 ('7506475119665', 'YOGUR GERBER PLATANO MANZANA Y FRESA 95 G'),
 ('7506475119672', 'YOGUR GERBER MANGO 95 G'),
 ('7506475121156', 'COFFEE MATE PUMPKIN SPICE POLVO 350 G'),
 ('7506475121231', 'TRIPACK CARNATION LECHE EVAPORADA 360 G'),
 ('7506475122115', 'JUGO GERBER JUNIOR MANZANA 200 ML'),
 ('7506475122139', 'JUGO GERBER JUNIOR PERA 200 ML'),
 ('7506475122764', 'COFFEE MATE LIQUIDO ORIGINAL 769.7 G'),
 ('7506475123051', 'CAFE SOLUBLE NESCAFE CLASICO 2X120 G'),
 ('7506475123792', 'NESCAFE DOLCA 80 G GRATIS 46 G'),
 ('7506475125253', 'CAFE SOLUBLE NESCAFE CLASICO MAS TAZA 200G'),
 ('7506475125543', 'NESCAFE ICE 40 G'),
 ('7506475125673', 'MALTEADA SABOR FRESA NESQUIK 185 ML'),
 ('7506475125680', 'MALTEADA SABOR VAINILLA NESQUIK 185 ML'),
 ('7506475126090', 'LECHE NESTLE NUTRI RINDES 460 G GRATIS DOLCA 22 G'),
 ('7506475126977', 'CEREAL NESQUIK 230G MAS CEREAL TRIX 230G'),
 ('7506475127158', 'COFFEE MATE MAZAPAN NESTLE 530G'),
 ('7506475128117', 'CEREAL ZUCOSOS NESTLE 650G'),
 ('7506475128162', 'NESCAFE CLASICO 200G MAS NESCAFE ICE 40G'),
 ('7506475128179', 'TRIPACK CARNATION LECHE EVAPORADA 360 G'),
 ('7506475128179', 'TRIPACK CARNATION LECHE EVAPORADA 360G'),
 ('7506475128285', 'CEREAL NESQUIK DE FRESA 310G'),
 ('7506475128292', 'NESQUIK POLVO FRESA BOLSA 200G'),
 ('7506475128339', 'CHOCOLATE CARLOS V DE 216G MAS UNO'),
 ('7506475128346', 'TRIPACK LA LECHERA 375 G'),
 ('7506475128346', 'TRIPACK LA LECHERA NESTLE 375G'),
 ('7506475128858', 'CEREALES NESQUIK 620G MAS ZUCOSOS 650G'),
 ('7506475130134', 'NESCAFE CAPPUCCINO KITKAT 6X25G'),
 ('7506475130295', 'COFFEE MATE DULCE DE LECHE 530G'),
 ('7506475130318', 'SOPA INSTANTANEA ESPECIES MAGGI 67G'),
 ('7506475130325', 'SOPA INSTANTANEA TERIYAKI MAGGI 70G'),
 ('7506475130332', 'SOPA INSTANTANEA BBQ PICANTE MAGGI 67G'),
 ('7506475130707', 'NESCAFE CLASICO 200G MAS NESCAFE DECAF 40G'),
 ('7506475131179', 'TRIPACK CARNATION LECHE EVAPORADA UHT 360G'),
 ('7506475131629', 'TRIPACK MEDIA CREMA NESTLE 225G'),
 ('7613030695295', 'MEDIA CREMA NESTLE 190 GR'),
 ('7613037012637', 'JUGOSO AL SARTEN MAGGI A LA LENA 24 G'),
 ('7613287216458', 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI CAJA 8 CUBOS 84 G'),
 ('7613287218162', 'CONSOME POLLO Y TOMATE CONSOMATE MAGGI 12 CUBOS 105 G'),
 ('7891000248362', 'CHOCOLATE KIT KAT 41.5 G'),
 ('7891000277119', 'CHOCOLATE KIT KAT DARK NESTLE 41.5G'),
 ('7891000395745', 'PUFFS GERBER CALABACITA 35 G'),
 ('8445290925299', 'CONSOMATE EN POLVO 5 SOBRES DE 21 G'),
 ('88169052054', 'JUGOSO AL SARTEN FINAS HIERBAS MAGGI SOBRE 4 PZAS 23.4 G')]

def clave_descripcion_precio(valor):
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(valor).upper())
    texto = "".join(c for c in texto if not unicodedata.combining(c) and unicodedata.category(c) != "Cf")
    return " ".join(texto.split())

def conciliar_precios_adjuntos(df, referencias):
    """Aplica precios por codigo solo cuando su vinculacion es inequivoca."""
    codigos_por_nombre = {}
    def registrar(codigo, nombre):
        if codigo and nombre and nombre not in {"NAN", "NONE", "SIN_DATO"}:
            codigos_por_nombre.setdefault(nombre, set()).add(codigo)
    for codigo, nombre in ALIAS_PRECIOS_POR_CODIGO:
        registrar(codigo, nombre)
    for codigo, descripcion in df[["CODIGO", "DESCRIPCION"]].dropna().drop_duplicates().itertuples(index=False, name=None):
        registrar(str(codigo), clave_descripcion_precio(descripcion))

    def indexar(catalogo):
        candidatos, ambiguos = {}, set()
        for nombre, precios in catalogo.items():
            codigos = codigos_por_nombre.get(nombre, set())
            if len(codigos) != 1:
                ambiguos.update(codigos)
                continue
            codigo = next(iter(codigos))
            candidatos.setdefault(codigo, set()).update(precios)
        ambiguos.update(codigo for codigo, valores in candidatos.items() if len(valores) != 1)
        # Una coincidencia inequívoca puede resolver otro nombre compartido.
        validos = {codigo: next(iter(valores)) for codigo, valores in candidatos.items() if len(valores) == 1}
        return validos, ambiguos - validos.keys()

    nuevos, amb_nuevos = indexar(PRECIOS_NUEVO_ARCHIVO)
    estaticos, amb_estaticos = indexar(PRECIOS_PREDICCIONES_2MESES)
    for codigo, ref in referencias.items():
        anterior = ref.get("Precio_referencia_SKU", np.nan)
        fuente_anterior = ref.get("Fuente_precio", "SIN PRECIO")
        nuevo, estatico = nuevos.get(codigo, np.nan), estaticos.get(codigo, np.nan)
        ref.update(Precio_nuevo_archivo=nuevo, Precio_estatico_predicciones=estatico,
                   Precio_referencia_anterior=anterior, Fuente_precio_anterior=fuente_anterior,
                   Revision_cruce_precio=("REVISAR PRECIO AMBIGUO" if codigo in amb_nuevos or codigo in amb_estaticos else ""))
        if np.isfinite(nuevo) and nuevo > 0:
            precio, fuente = nuevo, "NUEVO PRECIO - PRECIO ESTATICO"
        elif np.isfinite(anterior) and anterior > 0:
            continue
        elif np.isfinite(estatico) and estatico > 0:
            precio, fuente = estatico, "PREDICCIONES 2 MESES - PRECIO ESTATICO"
        else:
            continue
        ref.update(Precio_referencia_SKU=precio, Fuente_precio=fuente,
                   Precio_min_auditoria=precio * 0.8, Precio_max_auditoria=precio * 1.2,
                   Calidad_referencia_precio=fuente)
    return referencias

def construir_referencias_precio(df, sabana, corte=202610):
    """Prioridad: sábana por UPC, web exacta por UPC, último mes válido del mismo UPC."""
    h = df.loc[df["Periodo"].le(corte), ["CODIGO", "Periodo", "VENTA", "UNIDADES"]].copy()
    h["CODIGO"] = normalizar_codigo(h["CODIGO"])
    h["VENTA"] = pd.to_numeric(h["VENTA"], errors="coerce")
    h["UNIDADES"] = pd.to_numeric(h["UNIDADES"], errors="coerce")
    h = h.loc[h["VENTA"].gt(0) & h["UNIDADES"].gt(0)].copy()
    h = h.groupby(["CODIGO", "Periodo"], as_index=False)[["VENTA", "UNIDADES"]].sum()
    h["Precio_ultimo_mes"] = h["VENTA"] / h["UNIDADES"]
    ultimo = h.sort_values(["CODIGO", "Periodo"]).groupby("CODIGO", as_index=False).tail(1)
    ultimo = ultimo[["CODIGO", "Periodo", "Precio_ultimo_mes"]].rename(columns={"Periodo": "Periodo_ultimo_precio"})
    base = pd.DataFrame({"CODIGO": normalizar_codigo(df["CODIGO"]).drop_duplicates()})
    base = base.merge(ultimo, on="CODIGO", how="left", validate="one_to_one")
    if sabana is not None and not sabana.empty:
        s = sabana.copy(); s["CODIGO"] = normalizar_codigo(s["CODIGO"])
        base = base.merge(s, on="CODIGO", how="left", validate="one_to_one")
    else:
        base["Precio_sugerido_sabana"] = np.nan; base["Descripcion_sabana"] = np.nan
    web = pd.DataFrame([{"CODIGO": k, "Precio_web": v["precio"], "Fuente_web": v["fuente"],
                         "URL_precio_web": v["url"], "Fecha_precio_web": v["fecha_consulta"]}
                        for k, v in PRECIOS_WEB_VERIFICADOS.items()])
    base = base.merge(web, on="CODIGO", how="left", validate="one_to_one")
    base["Precio_referencia_SKU"] = base["Precio_sugerido_sabana"].combine_first(base["Precio_web"]).combine_first(base["Precio_ultimo_mes"])
    base["Fuente_precio"] = np.select(
        [base["Precio_sugerido_sabana"].notna(), base["Precio_web"].notna(), base["Precio_ultimo_mes"].notna()],
        ["SABANA - PRECIO VENTA SUGERIDO", "WEB - UPC EXACTO", "HISTORICO - ULTIMO MES"], default="SIN PRECIO")
    base["Coincidencia_codigo_sabana"] = base["Precio_sugerido_sabana"].notna()
    base["Precio_min_auditoria"] = base["Precio_referencia_SKU"] * 0.80
    base["Precio_max_auditoria"] = base["Precio_referencia_SKU"] * 1.20
    base["Calidad_referencia_precio"] = np.select(
        [base["Fuente_precio"].eq("SABANA - PRECIO VENTA SUGERIDO"), base["Fuente_precio"].eq("WEB - UPC EXACTO"),
         base["Fuente_precio"].eq("HISTORICO - ULTIMO MES")],
        ["PRECIO SUGERIDO POR UPC", "PRECIO WEB VERIFICADO POR UPC", "ULTIMO MES DEL SKU"], default="SIN REFERENCIA SKU")
    return conciliar_precios_adjuntos(df, base.set_index("CODIGO").to_dict("index"))


def referencias_precio_observado(df, corte):
    """Precios por SKU a partir de pares observados, sin precios imputados."""
    h = df.loc[df["Periodo"].le(corte), ["CODIGO", "Periodo", "VENTA", "UNIDADES"]].copy()
    for col in ["VENTA", "UNIDADES"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h = h.loc[np.isfinite(h["VENTA"]) & np.isfinite(h["UNIDADES"])]
    h = h.loc[(h["VENTA"] > 0) & (h["UNIDADES"] > 0)].copy()
    h = h.groupby(["CODIGO", "Periodo"], as_index=False)[["VENTA", "UNIDADES"]].sum()
    h["Precio"] = h["VENTA"] / h["UNIDADES"]
    h = h.loc[np.isfinite(h["Precio"]) & h["Precio"].gt(0)].sort_values("Periodo")
    corte_mes = (corte // 100) * 12 + corte % 100
    referencias = {}
    for codigo, serie in h.groupby("CODIGO", sort=False):
        edad = corte_mes - (serie["Periodo"] // 100 * 12 + serie["Periodo"] % 100)
        ventana = serie.loc[edad.lt(12)]
        antigua = ventana.empty
        if antigua:
            ventana = serie.tail(12)
        precios = ventana["Precio"].to_numpy()
        logp = np.log(precios)
        centro = np.median(logp)
        mad = np.median(np.abs(logp - centro))
        # Tolerancia de deteccion, no conversion de piezas/cajas ni precio de mercado.
        filtro = np.abs(logp - centro) <= max(np.log(1.5), 3.5 * 1.4826 * mad)
        limpios = ventana.loc[filtro, "Precio"]
        precio = float(limpios.tail(6).median())
        inf, sup = limpios.quantile([0.10, 0.90]).tolist()
        referencias[str(codigo)] = {
            "Precio_referencia_SKU": precio,
            "Precio_min_auditoria": min(float(inf), precio / 1.25),
            "Precio_max_auditoria": max(float(sup), precio * 1.25),
            "Meses_precio_validos": len(limpios),
            "Ultimo_periodo_precio": int(ventana.loc[filtro, "Periodo"].iloc[-1]),
            "Meses_precio_atipicos": int((~filtro).sum()),
            "Calidad_referencia_precio": (
                "HISTORICO ANTIGUO" if antigua else
                "POCOS MESES" if len(limpios) < 3 else "HISTORICO RECIENTE"
            ),
        }
    return referencias


def auditar_y_reconciliar_unidades(forecast, referencias, columna_unidades):
    """Conserva el importe; deja sin estimar el volumen sin referencia del SKU."""
    out = forecast.copy()
    if out.empty:
        return out
    ref = pd.DataFrame([referencias.get(str(codigo), {}) for codigo in out["CODIGO"]], index=out.index)
    for col in ["Precio_referencia_SKU", "Precio_min_auditoria", "Precio_max_auditoria",
                "Precio_sugerido_sabana", "Precio_web", "Precio_ultimo_mes", "Periodo_ultimo_precio",
                "Fuente_precio", "Coincidencia_codigo_sabana", "Fuente_web", "URL_precio_web", "Fecha_precio_web",
                "Precio_nuevo_archivo", "Precio_estatico_predicciones", "Precio_referencia_anterior",
                "Fuente_precio_anterior", "Revision_cruce_precio"]:
        out[col] = ref.get(col, np.nan)
    out["Calidad_referencia_precio"] = ref.get("Calidad_referencia_precio", pd.Series(index=out.index, dtype=object)).fillna("SIN REFERENCIA SKU")
    out["Meses_precio_validos"] = np.where(out["Precio_ultimo_mes"].notna(), 1, 0)
    out["Ultimo_periodo_precio"] = out["Periodo_ultimo_precio"]
    out["Meses_precio_atipicos"] = 0
    venta = out["Prediccion_final"]
    anterior = pd.to_numeric(out[columna_unidades], errors="coerce")
    anterior = anterior.where(np.isfinite(anterior))
    precio = out["Precio_referencia_SKU"]
    implicito = venta.div(anterior.where(anterior.gt(0)))
    out["Unidades_antes_validacion"] = anterior
    out["Precio_implicito_anterior"] = implicito
    out["Alerta_precio_anterior"] = np.select(
        [venta.eq(0) & anterior.gt(0), venta.gt(0) & precio.isna(),
         venta.gt(0) & (anterior.isna() | anterior.le(0)),
         venta.gt(0) & implicito.lt(out["Precio_min_auditoria"]),
         venta.gt(0) & implicito.gt(out["Precio_max_auditoria"])],
        ["UNIDADES SIN VENTA", "SIN REFERENCIA SKU", "VENTA SIN UNIDADES VALIDAS",
         "PRECIO IMPLICITO BAJO", "PRECIO IMPLICITO ALTO"], default="SIN ALERTA")
    relacion_precio = implicito.div(precio.where(precio.gt(0)))
    out["Factor_inconsistencia_precio"] = np.maximum(
        relacion_precio, 1.0 / relacion_precio.where(relacion_precio.gt(0))
    ).replace([np.inf, -np.inf], np.nan)
    out["Prioridad_correccion_precio"] = np.select(
        [out["Alerta_precio_anterior"].isin(["UNIDADES SIN VENTA", "VENTA SIN UNIDADES VALIDAS", "SIN REFERENCIA SKU"]),
         out["Factor_inconsistencia_precio"].ge(3.0),
         out["Factor_inconsistencia_precio"].ge(1.5),
         out["Alerta_precio_anterior"].ne("SIN ALERTA")],
        ["P1 - CRITICA", "P1 - CRITICA", "P2 - ALTA", "P3 - MEDIA"],
        default="P4 - SIN ALERTA"
    )
    out["Precio_pred"] = precio
    out["Unidades_pred"] = venta.div(precio.where(precio.gt(0))).mask(venta.eq(0), 0.0)
    out["Venta_estructural"] = (out["Unidades_pred"] * precio).mask(venta.eq(0), 0.0)
    out["Diferencia_unidades"] = out["Unidades_pred"] - anterior
    return out


@st.cache_data(show_spinner=False)
def generar_forecast(df, sabana_precios, version_modelo="forecast-precio-estatico-v9"):
    """Genera forecast para todos los SKU presentes en ventas 2026.

    Los SKU activos desde agosto 2026 forman el universo base. Los SKU presentes
    en 2026, pero sin actividad desde agosto, se pronostican y se etiquetan para
    que el usuario decida mediante segmentador si afectan o no los totales.
    """
    modelos = entrenar_modelos(df)
    filas = []
    periodos = [202611, 202612]
    codigos_2026 = set(
        df.loc[df["AÑO"] == 2026, "CODIGO"].astype(str).unique().tolist()
    )

    for codigo, historial in df.groupby("CODIGO", sort=False):
        if str(codigo) not in codigos_2026:
            continue
        hist_original = historial.sort_values("Periodo").copy()
        if hist_original.empty:
            continue

        ultimo_periodo_disponible = int(hist_original["Periodo"].max())
        incluido_base = ultimo_periodo_disponible >= 202608
        estado_forecast = "INCLUIDO_BASE" if incluido_base else "NO_CONTADO_BASE"
        meses_sin_actividad = meses_entre_periodos(ultimo_periodo_disponible, 202610)

        # Para SKU no contados en el forecast base, se completan con ceros hasta
        # octubre. Se mantienen separados y no afectan el total salvo elección.
        hist = (
            hist_original
            if incluido_base
            else extender_historial_hasta_octubre_2026(hist_original)
        )
        meta = hist.iloc[-1]

        venta_directa = predecir_objetivo(
            *modelos["VENTA"], hist, periodos, "VENTA"
        )
        unidades = predecir_objetivo(
            *modelos["UNIDADES"], hist, periodos, "UNIDADES"
        )
        precio = predecir_objetivo(
            *modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO"
        )

        for periodo, vd, unid, precio_prom in zip(
            periodos, venta_directa, unidades, precio
        ):
            estructural = unid * precio_prom
            base = PESO_VENTA_DIRECTA * vd + PESO_ESTRUCTURAL * estructural
            final_antes_calibracion, diag = aplicar_ajuste_estacional(
                base, hist, periodo
            )
            final, factor_categoria, calibracion_aplicada, motivo_calibracion = (
                aplicar_calibracion_categoria_v2(
                    final_antes_calibracion,
                    meta.get("CATEGORIA", "SIN_DATO"),
                    meta.get("COMBOS", 0),
                )
            )
            # La salida independiente se guarda para auditar antes de reconciliar.
            unidades_modelo = float(unid)
            filas.append({
                "CODIGO": str(codigo),
                "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)),
                "Periodo": periodo,
                "Estado_forecast": estado_forecast,
                "Incluido_forecast_base": bool(incluido_base),
                "Ultimo_periodo_disponible": ultimo_periodo_disponible,
                "Meses_sin_actividad_a_octubre": int(meses_sin_actividad),
                "Venta_directa": float(vd),
                "Unidades_modelo_sin_reconciliar": unidades_modelo,
                "Unidades_pred": unidades_modelo,
                "Precio_modelo": float(precio_prom),
                "Precio_pred": float(precio_prom),
                "Venta_estructural_modelo": float(estructural),
                "Venta_estructural": float(estructural),
                "Prediccion_base": float(base),
                "Referencia_ultimo_ano": diag["Referencia_ultimo_ano"],
                "Crecimiento_YTD": diag["Crecimiento_YTD"],
                "Prediccion_estacional": diag["Prediccion_estacional"],
                "Prediccion_combinada": diag["Prediccion_combinada"],
                "Piso_parcial": diag["Piso_parcial"],
                "Porcentaje_recuperacion": diag["Porcentaje_recuperacion"],
                "Correccion_piso": diag["Correccion_piso"],
                "Indice_estacional": diag["Indice_estacional"],
                "Ajuste_aplicado": diag["Ajuste_aplicado"],
                "Prediccion_antes_calibracion": float(final_antes_calibracion),
                "Factor_calibracion_categoria": float(factor_categoria),
                "Calibracion_categoria_aplicada": bool(calibracion_aplicada),
                "Motivo_calibracion": motivo_calibracion,
                "Prediccion_final": float(final),
            })
    resultado = pd.DataFrame(filas)
    if resultado.empty:
        return resultado
    resultado = ajustar_noviembre_chocolate_mesa(resultado)
    resultado = ajustar_diciembre_lacteos_18(resultado)
    return auditar_y_reconciliar_unidades(
        resultado, construir_referencias_precio(df, sabana_precios, 202610), "Unidades_modelo_sin_reconciliar"
    )


def cargar_venta_semanal_noviembre(archivo):
    """Lee la tabla dinámica por categoría de la hoja PREDICCIONES NOVIEMBRE.

    Devuelve solo categorías válidas. La fila Total general se conserva aparte
    mediante la suma de categorías para evitar duplicar importes.
    """
    if archivo is None:
        return pd.DataFrame(columns=["CATEGORIA", "Venta_Semana_1", "Unidades_Semana_1"])
    contenido = archivo.getvalue()
    tabla = pd.read_excel(
        io.BytesIO(contenido), sheet_name="PREDICCIONES NOVIEMBRE",
        usecols="A:D", engine="openpyxl",
    )
    renombres = {}
    for col in tabla.columns:
        nombre = _normalizar_encabezado(col)
        if nombre == "CATEGORIA": renombres[col] = "CATEGORIA"
        elif nombre == "VENTA SEMANAL 1": renombres[col] = "Venta_Semana_1"
        elif nombre == "UNIDADES VENDIDAS": renombres[col] = "Unidades_Semana_1"
    tabla = tabla.rename(columns=renombres)
    requeridas = {"CATEGORIA", "Venta_Semana_1"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError(
            "La hoja PREDICCIONES NOVIEMBRE debe contener CATEGORIA y Venta Semanal 1."
        )
    if "Unidades_Semana_1" not in tabla.columns:
        tabla["Unidades_Semana_1"] = 0.0
    tabla["CATEGORIA"] = tabla["CATEGORIA"].astype(str).str.strip()
    tabla["Categoria_normalizada"] = tabla["CATEGORIA"].map(normalizar_categoria_calibracion)
    tabla["Venta_Semana_1"] = pd.to_numeric(tabla["Venta_Semana_1"], errors="coerce")
    tabla["Unidades_Semana_1"] = pd.to_numeric(tabla["Unidades_Semana_1"], errors="coerce")
    tabla = tabla[
        tabla["Venta_Semana_1"].notna()
        & ~tabla["Categoria_normalizada"].isin(["TOTAL GENERAL", "NAN", "NONE", ""])
    ].copy()
    return (tabla.groupby(["CATEGORIA", "Categoria_normalizada"], as_index=False)
            .agg(Venta_Semana_1=("Venta_Semana_1", "sum"),
                 Unidades_Semana_1=("Unidades_Semana_1", "sum")))


def construir_comparativo_semanal(forecast_filtrado, venta_semanal, peso_real=0.30):
    """Compara forecast original, ritmo lineal y nowcast ponderado de noviembre."""
    nov = forecast_filtrado[forecast_filtrado["Periodo"] == 202611].copy()
    if nov.empty or venta_semanal.empty:
        return pd.DataFrame()
    base = (nov.assign(Categoria_normalizada=nov["CATEGORIA"].map(normalizar_categoria_calibracion))
              .groupby("Categoria_normalizada", as_index=False)
              .agg(Prediccion_Original=("Prediccion_final", "sum")))
    comp = venta_semanal.merge(base, on="Categoria_normalizada", how="inner")
    dias_transcurridos = 7.0
    dias_mes = 30.0
    comp["Avance_Calendario"] = dias_transcurridos / dias_mes
    comp["Cumplimiento_Forecast"] = np.where(
        comp["Prediccion_Original"] > 0,
        comp["Venta_Semana_1"] / comp["Prediccion_Original"], np.nan,
    )
    comp["Proyeccion_Ritmo"] = comp["Venta_Semana_1"] * dias_mes / dias_transcurridos

    # Conservamos el valor del modelo solo para auditoría. La columna visible
    # Prediccion_Original se corrige directamente, sin crear una barra adicional.
    comp["Prediccion_Modelo_Sin_Ajuste"] = comp["Prediccion_Original"]
    comp["Peso_Correccion_Ritmo"] = 0.0
    comp["Tope_Correccion"] = np.nan

    lacteos = comp["Categoria_normalizada"].eq("LACTEOS CULINARIOS")
    cereales = comp["Categoria_normalizada"].eq("CEREALES")

    # Lácteos: forecast y ritmo están próximos. Se recupera 90% de la brecha.
    comp.loc[lacteos, "Peso_Correccion_Ritmo"] = 0.90

    # Cereales: la semana 1 está muy por encima del modelo. Se recupera 50%
    # de la brecha, pero con tope de 1.60x para no extrapolar una posible promoción.
    comp.loc[cereales, "Peso_Correccion_Ritmo"] = 0.50
    comp.loc[cereales, "Tope_Correccion"] = 1.60

    comp["Prediccion_Original"] = (
        comp["Prediccion_Modelo_Sin_Ajuste"]
        + comp["Peso_Correccion_Ritmo"]
        * (comp["Proyeccion_Ritmo"] - comp["Prediccion_Modelo_Sin_Ajuste"])
    )
    comp.loc[cereales, "Prediccion_Original"] = np.minimum(
        comp.loc[cereales, "Prediccion_Original"],
        comp.loc[cereales, "Prediccion_Modelo_Sin_Ajuste"] * 1.60,
    )
    comp["Factor_Correccion_Aplicado"] = np.where(
        comp["Prediccion_Modelo_Sin_Ajuste"] > 0,
        comp["Prediccion_Original"] / comp["Prediccion_Modelo_Sin_Ajuste"],
        1.0,
    )
    comp["Correccion_Aplicada"] = np.select(
        [lacteos, cereales],
        ["Lácteos: 90% de brecha", "Cereales: 50% de brecha; tope 1.60x"],
        default="Sin corrección",
    )

    # El nowcast se recalcula desde el forecast ya corregido.
    comp["Cumplimiento_Forecast"] = np.where(
        comp["Prediccion_Original"] > 0,
        comp["Venta_Semana_1"] / comp["Prediccion_Original"], np.nan,
    )
    comp["Nowcast_Ponderado"] = (
        (1.0 - peso_real) * comp["Prediccion_Original"]
        + peso_real * comp["Proyeccion_Ritmo"]
    )
    comp["Brecha_Ritmo_pp"] = (
        comp["Cumplimiento_Forecast"] - comp["Avance_Calendario"]
    ) * 100
    comp["Estado"] = np.select(
        [comp["Brecha_Ritmo_pp"] >= 5, comp["Brecha_Ritmo_pp"] <= -5],
        ["ADELANTADA", "REZAGADA"], default="EN RANGO",
    )
    return comp


def preparar_grafica(df_historico, forecast, categoria, combo, producto):
    """Prepara histórico comparable y una línea forecast conectada a octubre 2026."""
    futuro = forecast.copy()

    if categoria != "TODAS":
        futuro = futuro[futuro["CATEGORIA"] == categoria]
    if combo != "TODOS":
        valor_combo = 1 if combo == "COMBOS" else 0
        futuro = futuro[futuro["COMBOS"] == valor_combo]
    if producto != "TODOS":
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]

    # El histórico se restringe al mismo universo de SKU que sí tienen forecast.
    codigos_activos = futuro["CODIGO"].astype(str).unique().tolist()
    hist = df_historico[df_historico["CODIGO"].astype(str).isin(codigos_activos)].copy()
    if hist.empty or futuro.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(dict(
        year=historico["AÑO"].astype(int),
        month=historico["MES_NUM"].astype(int), day=1
    ))
    historico = historico[historico["Fecha"] <= pd.Timestamp("2026-10-01")]
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"

    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(dict(
        year=(previsto["Periodo"] // 100).astype(int),
        month=(previsto["Periodo"] % 100).astype(int), day=1
    ))
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast"

    # Punto puente: octubre real se agrega al forecast para conectar las líneas.
    octubre = historico[historico["Fecha"] == pd.Timestamp("2026-10-01")][["Fecha", "Valor"]].copy()
    if octubre.empty:
        ultima = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()
        puente = ultima
    else:
        puente = octubre
    puente["Serie"] = "Forecast"
    forecast_conectado = pd.concat(
        [puente[["Fecha", "Valor", "Serie"]], previsto[["Fecha", "Valor", "Serie"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return (
        historico[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
        forecast_conectado,
        previsto[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
    )


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)
archivo_semanal = st.file_uploader(
    "Carga PREDICCIONES NOVIEMBRE.xlsx para comparar la venta real de la semana 1",
    type=["xlsx"],
    accept_multiple_files=False,
    key="archivo_venta_semanal_noviembre",
    help="Se lee la tabla por categoría de la hoja PREDICCIONES NOVIEMBRE.",
)
archivo_sabana = st.file_uploader(
    "Carga la SÁBANA DE PRECIOS para conciliar unidades por UPC",
    type=["xlsx"], accept_multiple_files=False, key="archivo_sabana_precios",
    help="La sábana sigue siendo opcional. Los precios nuevos adjuntos ya están incorporados y tienen prioridad.",
)

if archivos:
    try:
        # La carga no usa caché: cada ejecución procesa exactamente los archivos visibles.
        df, diagnostico_carga = cargar_archivos(archivos)
        resumen_anual = (
            df.groupby("AÑO", as_index=False)
            .agg(SKU=("CODIGO", "nunique"), Registros=("Periodo", "size"),
                 Desde=("Periodo", "min"), Hasta=("Periodo", "max"))
        )
        anos_cargados = set(resumen_anual["AÑO"].astype(int).tolist())
        st.success("Los archivos fueron leídos nuevamente sin utilizar caché.")
        st.dataframe(resumen_anual, use_container_width=True, hide_index=True)
        with st.expander("Diagnóstico detallado de cada hoja"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        sabana_precios = cargar_sabana_precios(archivo_sabana)
        faltantes = sorted({2024, 2025, 2026} - anos_cargados)
        if faltantes:
            st.error(
                f"No se cargaron datos válidos de {faltantes}. Revisa en el diagnóstico "
                "qué columna falta en las hojas correspondientes."
            )
            st.stop()

        col_boton, _ = st.columns([1, 2])
        with col_boton:
            if st.button("Limpiar forecast anterior"):
                st.session_state.pop("forecast_2026_precio_estatico_v9", None)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        if "forecast_2026_precio_estatico_v9" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast final Nov-Dic 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026_precio_estatico_v9"] = generar_forecast(df, sabana_precios)
            st.rerun()

        if "forecast_2026_precio_estatico_v9" in st.session_state:
            forecast = st.session_state["forecast_2026_precio_estatico_v9"]
            if forecast.empty:
                st.error("No se encontraron SKU activos con datos recientes de 2026.")
                st.stop()

            st.markdown("### Filtros")
            st.markdown("#### Universo de SKU 2026")
            universo_sku = st.segmented_control(
                "¿Qué SKU deben afectar la gráfica y los totales?",
                options=[
                    "SOLO INCLUIDOS BASE",
                    "INCLUIR NO CONTADOS",
                    "SOLO NO CONTADOS",
                ],
                default="SOLO INCLUIDOS BASE",
                help=(
                    "Incluidos base: SKU con datos desde agosto 2026. "
                    "Incluir no contados: suma también SKU presentes en 2026 sin actividad reciente. "
                    "Solo no contados: permite revisarlos por separado sin mezclar el resto."
                ),
            )
            if universo_sku is None:
                universo_sku = "SOLO INCLUIDOS BASE"

            if universo_sku == "SOLO INCLUIDOS BASE":
                forecast_universo = forecast[forecast["Incluido_forecast_base"]].copy()
            elif universo_sku == "SOLO NO CONTADOS":
                forecast_universo = forecast[~forecast["Incluido_forecast_base"]].copy()
            else:
                forecast_universo = forecast.copy()

            conteo_base = forecast.loc[forecast["Incluido_forecast_base"], "CODIGO"].nunique()
            conteo_no_contado = forecast.loc[~forecast["Incluido_forecast_base"], "CODIGO"].nunique()
            u1, u2, u3 = st.columns(3)
            u1.metric("SKU incluidos base", f"{conteo_base:,}")
            u2.metric("SKU 2026 no contados", f"{conteo_no_contado:,}")
            u3.metric("SKU en universo seleccionado", f"{forecast_universo['CODIGO'].nunique():,}")

            if universo_sku != "SOLO INCLUIDOS BASE":
                st.warning(
                    "Los SKU no contados tienen menor evidencia reciente. Sus meses faltantes hasta "
                    "octubre se completan con venta/unidades cero. Úsalos como escenario de sensibilidad."
                )

            categorias = ["TODAS"] + sorted(
                forecast_universo["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
            )
            c1, c2, c3 = st.columns([2, 1.25, 3])
            with c1:
                categoria = st.selectbox("Categoría", categorias)
            with c2:
                combo = st.segmented_control(
                    "Tipo de producto",
                    options=["TODOS", "SIN COMBOS", "COMBOS"],
                    default="SIN COMBOS",
                )
                if combo is None:
                    combo = "SIN COMBOS"

            universo = forecast_universo.copy()
            if categoria != "TODAS":
                universo = universo[universo["CATEGORIA"] == categoria]
            if combo != "TODOS":
                universo = universo[
                    universo["COMBOS"] == (1 if combo == "COMBOS" else 0)
                ]

            productos = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
            productos["ETIQUETA"] = (
                productos["CODIGO"].astype(str) + " | "
                + productos["DESCRIPCION"].astype(str)
            )
            mapa_productos = dict(
                zip(productos["ETIQUETA"], productos["CODIGO"].astype(str))
            )
            with c3:
                seleccion_producto = st.selectbox(
                    "Producto", ["TODOS"] + sorted(mapa_productos)
                )
            producto = "TODOS" if seleccion_producto == "TODOS" else mapa_productos[seleccion_producto]

            historico, forecast_linea, forecast_puntos = preparar_grafica(
                df, forecast_universo, categoria, combo, producto
            )
            if historico.empty or forecast_puntos.empty:
                st.warning("No hay datos comparables para la selección actual.")
                st.stop()

            nov = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-11-01"), "Valor"
            ].sum()
            dic = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-12-01"), "Valor"
            ].sum()
            oct_real = historico.loc[
                historico["Fecha"] == pd.Timestamp("2026-10-01"), "Valor"
            ].sum()
            crecimiento_nov = (nov / oct_real - 1) * 100 if oct_real > 0 else np.nan
            crecimiento_dic = (dic / nov - 1) * 100 if nov > 0 else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Octubre 2026 real", f"${oct_real:,.0f}")
            k2.metric("Noviembre 2026", f"${nov:,.0f}",
                      f"{crecimiento_nov:+.1f}% vs octubre" if pd.notna(crecimiento_nov) else None)
            k3.metric("Diciembre 2026", f"${dic:,.0f}",
                      f"{crecimiento_dic:+.1f}% vs noviembre" if pd.notna(crecimiento_dic) else None)
            sku_mostrados = (
                universo["CODIGO"].astype(str).nunique()
                if producto == "TODOS" else 1
            )
            k4.metric("SKU incluidos", f"{sku_mostrados:,}")

            fecha_min = pd.Timestamp("2024-01-01")
            fecha_max = pd.Timestamp("2026-12-01")
            eje_x = alt.X(
                "Fecha:T", title=None,
                scale=alt.Scale(domain=[fecha_min, fecha_max]),
                axis=alt.Axis(
                    format="%b %Y", labelAngle=-35,
                    tickCount={"interval": "month", "step": 2},
                    grid=False, labelPadding=10,
                ),
            )
            eje_y = alt.Y(
                "Valor:Q", title="Venta mensual",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(format="$,.2s", grid=True, gridOpacity=0.14),
            )
            tooltip = [
                alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
            ]

            linea_real = alt.Chart(historico).mark_line(
                color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_reales = alt.Chart(historico).mark_circle(
                color="#3B82F6", size=44, opacity=0.9
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            linea_forecast = alt.Chart(forecast_linea).mark_line(
                color="#F59E0B", strokeWidth=3.5,
                strokeDash=[8, 5], interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_forecast = alt.Chart(forecast_puntos).mark_circle(
                color="#F59E0B", size=125, stroke="white", strokeWidth=1.5
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            limite = pd.DataFrame({"Fecha": [pd.Timestamp("2026-11-01")]})
            regla = alt.Chart(limite).mark_rule(
                color="#94A3B8", strokeDash=[4, 4], opacity=0.75
            ).encode(x="Fecha:T")
            etiqueta = alt.Chart(pd.DataFrame({
                "Fecha": [pd.Timestamp("2026-11-01")],
                "Texto": ["Inicio forecast"]
            })).mark_text(
                align="left", dx=7, dy=-205,
                color="#94A3B8", fontSize=12
            ).encode(x="Fecha:T", text="Texto:N")

            chart = alt.layer(
                linea_real, puntos_reales,
                linea_forecast, puntos_forecast,
                regla, etiqueta,
            ).properties(
                height=520,
                title=alt.TitleParams(
                    text="Evolución mensual y forecast Nov-Dic 2026",
                    subtitle="Línea azul: ventas reales | Línea naranja punteada: pronóstico",
                    anchor="start", fontSize=18, subtitleFontSize=12,
                    offset=16,
                ),
            ).configure_view(strokeWidth=0).configure_axis(
                labelFontSize=12, titleFontSize=13
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "El histórico utiliza exactamente los mismos SKU activos incluidos en el forecast. "
                "Los productos sin actividad reciente en 2026 se excluyen para evitar saltos artificiales."
            )

            # --------------------------------------------------------------
            # Seguimiento de noviembre con venta real de la primera semana
            # --------------------------------------------------------------
            # Valor predeterminado disponible también para la tabla SKU.
            peso_real = 0.30
            st.markdown("### Seguimiento semanal de noviembre")
            if archivo_semanal is None:
                st.info(
                    "Carga PREDICCIONES NOVIEMBRE.xlsx arriba para comparar el forecast "
                    "con la venta real acumulada de la semana 1."
                )
            else:
                venta_semanal = cargar_venta_semanal_noviembre(archivo_semanal)
                peso_real = st.slider(
                    "Peso de la venta real semanal en el nowcast",
                    min_value=0, max_value=100, value=30, step=5,
                    format="%d%%",
                    help=(
                        "30% conserva estabilidad del forecast y permite que el ritmo real "
                        "actualice gradualmente la proyección mensual."
                    ),
                ) / 100.0
                comparativo = construir_comparativo_semanal(
                    forecast_universo, venta_semanal, peso_real
                )
                st.info(
                    "La barra naranja ya incluye la corrección selectiva: Lácteos recupera 90% "
                    "de la brecha hacia ritmo; Cereales recupera 50% con máximo de 1.60x. "
                    "Las demás categorías conservan el forecast del modelo."
                )
                if categoria != "TODAS":
                    cat_norm = normalizar_categoria_calibracion(categoria)
                    comparativo = comparativo[
                        comparativo["Categoria_normalizada"] == cat_norm
                    ].copy()

                categorias_disponibles = sorted(comparativo["CATEGORIA"].unique().tolist())
                categorias_seleccionadas = st.multiselect(
                    "Segmentador de categorías para comparar proyecciones",
                    options=categorias_disponibles,
                    default=categorias_disponibles,
                    help="Lácteos Culinarios y Cereales ya incorporan la corrección selectiva en la barra naranja.",
                )
                comparativo = comparativo[
                    comparativo["CATEGORIA"].isin(categorias_seleccionadas)
                ].copy()

                escenarios = st.segmented_control(
                    "Escenarios visibles",
                    options=["TODOS", "FORECAST VS NOWCAST", "FORECAST VS RITMO"],
                    default="TODOS",
                ) or "TODOS"

                if comparativo.empty:
                    st.warning("No hay categorías coincidentes entre el forecast y la tabla semanal.")
                else:
                    total_semana = comparativo["Venta_Semana_1"].sum()
                    total_original = comparativo["Prediccion_Original"].sum()
                    total_ritmo = comparativo["Proyeccion_Ritmo"].sum()
                    total_nowcast = comparativo["Nowcast_Ponderado"].sum()
                    avance = total_semana / total_original if total_original > 0 else np.nan
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Venta real semana 1", f"${total_semana:,.0f}")
                    s2.metric("Forecast original Nov", f"${total_original:,.0f}")
                    s3.metric("Proyección por ritmo", f"${total_ritmo:,.0f}",
                              f"{(total_ritmo/total_original-1)*100:+.1f}% vs forecast" if total_original else None)
                    s4.metric("Nowcast ponderado", f"${total_nowcast:,.0f}",
                              f"{(total_nowcast/total_original-1)*100:+.1f}% vs forecast" if total_original else None)
                    st.caption(
                        f"Avance acumulado: {avance:.1%}. Avance calendario de referencia: 23.3% (7 de 30 días)."
                    )

                    columnas_escenario = {
                        "TODOS": ["Prediccion_Original", "Proyeccion_Ritmo", "Nowcast_Ponderado"],
                        "FORECAST VS NOWCAST": ["Prediccion_Original", "Nowcast_Ponderado"],
                        "FORECAST VS RITMO": ["Prediccion_Original", "Proyeccion_Ritmo"],
                    }[escenarios]
                    etiquetas = {
                        "Prediccion_Original": "Forecast original corregido",
                        "Proyeccion_Ritmo": "Proyección por ritmo",
                        "Nowcast_Ponderado": "Nowcast ponderado",
                    }
                    grafica_comp = comparativo.melt(
                        id_vars=["CATEGORIA"], value_vars=columnas_escenario,
                        var_name="Escenario", value_name="Valor",
                    )
                    grafica_comp["Escenario"] = grafica_comp["Escenario"].map(etiquetas)
                    barras = alt.Chart(grafica_comp).mark_bar().encode(
                        x=alt.X("CATEGORIA:N", title=None, sort="-y",
                                axis=alt.Axis(labelAngle=-35, labelLimit=180)),
                        xOffset="Escenario:N",
                        y=alt.Y("Valor:Q", title="Venta proyectada", axis=alt.Axis(format="$,.2s")),
                        color=alt.Color(
                            "Escenario:N", title="Escenario",
                            scale=alt.Scale(
                                domain=["Forecast original corregido", "Proyección por ritmo", "Nowcast ponderado"],
                                range=["#F59E0B", "#EF4444", "#10B981"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("CATEGORIA:N", title="Categoría"),
                            alt.Tooltip("Escenario:N", title="Escenario"),
                            alt.Tooltip("Valor:Q", title="Valor", format="$,.2f"),
                        ],
                    ).properties(height=430, title="Comparación de proyecciones de noviembre por categoría")
                    # Dos gráficas adicionales con el mismo histórico mensual.
                    # Noviembre se actualiza con la venta semanal. Para evitar una caída
                    # artificial en diciembre, se propaga parcialmente el ajuste relativo
                    # de noviembre. Configuración prudente para semana 1: 40%, con límite ±15%.
                    st.markdown("#### Ajuste de continuidad hacia diciembre")
                    col_prop, col_lim = st.columns(2)
                    with col_prop:
                        propagacion_dic = st.slider(
                            "Propagación del ajuste de noviembre a diciembre",
                            min_value=0, max_value=100, value=40, step=5,
                            format="%d%%",
                            help=(
                                "Semana 1 recomendada: 40%. Traslada solo una parte de la "
                                "desviación relativa de noviembre hacia diciembre."
                            ),
                        ) / 100.0
                    with col_lim:
                        limite_ajuste_dic = st.slider(
                            "Límite máximo del ajuste de diciembre",
                            min_value=5, max_value=30, value=15, step=5,
                            format="±%d%%",
                            help="Protege diciembre contra una primera semana atípica.",
                        ) / 100.0

                    def relacion_historica_dic_nov_lacteos():
                        """Mediana robusta Dic/Nov 2024-2025 para Lacteos Culinarios."""
                        cat_obj = "LACTEOS CULINARIOS"
                        hist_cat = df[
                            df["CATEGORIA"].map(normalizar_categoria_calibracion).eq(cat_obj)
                            & df["AÑO"].isin([2024, 2025])
                            & df["MES_NUM"].isin([11, 12])
                        ].groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
                        ratios = []
                        for ano in [2024, 2025]:
                            nov_h = hist_cat.loc[
                                (hist_cat["AÑO"] == ano) & (hist_cat["MES_NUM"] == 11), "VENTA"
                            ].sum()
                            dic_h = hist_cat.loc[
                                (hist_cat["AÑO"] == ano) & (hist_cat["MES_NUM"] == 12), "VENTA"
                            ].sum()
                            if nov_h > 0 and dic_h > 0:
                                ratios.append(dic_h / nov_h)
                        if not ratios:
                            return 1.0 + CRECIMIENTO_BASE_DIC_LACTEOS, []
                        return float(np.median(ratios)), ratios

                    categoria_actual_norm = (
                        normalizar_categoria_calibracion(categoria)
                        if categoria != "TODAS" else "TODAS"
                    )
                    seleccion_exclusiva_lacteos = categoria_actual_norm == "LACTEOS CULINARIOS"
                    relacion_hist_lacteos, ratios_hist_lacteos = relacion_historica_dic_nov_lacteos()

                    def ajustar_diciembre(valor_noviembre_escenario):
                        if total_original <= 0 or dic <= 0:
                            return float(dic), 1.0, 0.0, "Sin ajuste"
                        factor_noviembre = float(valor_noviembre_escenario) / float(total_original)

                        # Regla especial solo cuando el filtro muestra exclusivamente Lacteos Culinarios.
                        # Usa noviembre actualizado como ancla y limita diciembre a -5% / +10%.
                        if APLICAR_TOPE_DICIEMBRE_LACTEOS and seleccion_exclusiva_lacteos:
                            ratio_base = (
                                relacion_hist_lacteos
                                if USAR_RELACION_HISTORICA_LACTEOS
                                else 1.0 + CRECIMIENTO_BASE_DIC_LACTEOS
                            )
                            ratio_controlado = float(np.clip(
                                ratio_base,
                                PISO_DIC_VS_NOV_LACTEOS,
                                TOPE_DIC_VS_NOV_LACTEOS,
                            ))
                            dic_controlado = float(valor_noviembre_escenario) * ratio_controlado
                            ajuste_vs_dic_original = (
                                dic_controlado / float(dic) - 1.0 if dic > 0 else 0.0
                            )
                            metodo = (
                                f"Lácteos: Nov escenario × {ratio_controlado:.3f} "
                                f"(relación histórica limitada entre "
                                f"{PISO_DIC_VS_NOV_LACTEOS:.2f} y {TOPE_DIC_VS_NOV_LACTEOS:.2f})"
                            )
                            return dic_controlado, factor_noviembre, ajuste_vs_dic_original, metodo

                        # Para el resto de categorías conserva la propagación parcial existente.
                        ajuste_sin_limite = (factor_noviembre - 1.0) * propagacion_dic
                        ajuste_limitado = float(np.clip(
                            ajuste_sin_limite, -limite_ajuste_dic, limite_ajuste_dic
                        ))
                        metodo = "Propagación parcial sobre diciembre original"
                        return (
                            float(dic) * (1.0 + ajuste_limitado),
                            factor_noviembre,
                            ajuste_limitado,
                            metodo,
                        )

                    dic_ritmo, factor_nov_ritmo, ajuste_dic_ritmo, metodo_dic_ritmo = ajustar_diciembre(total_ritmo)
                    dic_nowcast, factor_nov_nowcast, ajuste_dic_nowcast, metodo_dic_nowcast = ajustar_diciembre(total_nowcast)

                    d1, d2 = st.columns(2)
                    d1.metric(
                        "Diciembre ajustado por ritmo", f"${dic_ritmo:,.0f}",
                        f"{ajuste_dic_ritmo:+.1%} vs diciembre original",
                    )
                    d2.metric(
                        "Diciembre ajustado por nowcast", f"${dic_nowcast:,.0f}",
                        f"{ajuste_dic_nowcast:+.1%} vs diciembre original",
                    )
                    if seleccion_exclusiva_lacteos:
                        ratios_texto = ", ".join(f"{r:.3f}" for r in ratios_hist_lacteos) or "sin historial válido"
                        st.success(
                            "Control especial activo para Lácteos Culinarios: diciembre se calcula "
                            "desde noviembre actualizado y queda limitado entre -5% y +18%."
                        )
                        st.caption(
                            f"Relaciones históricas Dic/Nov disponibles: {ratios_texto}. "
                            f"Método nowcast: {metodo_dic_nowcast}."
                        )
                    else:
                        st.caption(
                            "Fórmula general: Dic. ajustado = Dic. original × [1 + límite((Nov. escenario / "
                            "Nov. original − 1) × propagación)]."
                        )

                    octubre_puente = historico[
                        historico["Fecha"] == pd.Timestamp("2026-10-01")
                    ][["Fecha", "Valor"]].copy()
                    if octubre_puente.empty:
                        octubre_puente = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()

                    def crear_grafica_escenario(valor_noviembre, valor_diciembre, nombre, color, subtitulo):
                        puntos = pd.DataFrame({
                            "Fecha": [pd.Timestamp("2026-11-01"), pd.Timestamp("2026-12-01")],
                            "Valor": [float(valor_noviembre), float(valor_diciembre)],
                            "Serie": [nombre, nombre],
                        })
                        puente = octubre_puente.copy()
                        puente["Serie"] = nombre
                        linea = pd.concat(
                            [puente[["Fecha", "Valor", "Serie"]], puntos],
                            ignore_index=True,
                        ).sort_values("Fecha")
                        tooltip_escenario = [
                            alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                            alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
                            alt.Tooltip("Serie:N", title="Escenario"),
                        ]
                        linea_real_esc = alt.Chart(historico).mark_line(
                            color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
                        puntos_reales_esc = alt.Chart(historico).mark_circle(
                            color="#3B82F6", size=44, opacity=0.9
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
                        linea_esc = alt.Chart(linea).mark_line(
                            color=color, strokeWidth=3.5, strokeDash=[8, 5],
                            interpolate="monotone",
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip_escenario)
                        puntos_esc = alt.Chart(puntos).mark_circle(
                            color=color, size=125, stroke="white", strokeWidth=1.5
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip_escenario)
                        return alt.layer(
                            linea_real_esc, puntos_reales_esc, linea_esc, puntos_esc,
                            regla, etiqueta,
                        ).properties(
                            height=520,
                            title=alt.TitleParams(
                                text=f"Evolución mensual y {nombre.lower()} Nov-Dic 2026",
                                subtitle=subtitulo,
                                anchor="start", fontSize=18, subtitleFontSize=12,
                                offset=16,
                            ),
                        ).configure_view(strokeWidth=0).configure_axis(
                            labelFontSize=12, titleFontSize=13
                        )

                    st.markdown("### Evolución mensual con proyección por ritmo")
                    chart_ritmo = crear_grafica_escenario(
                        total_ritmo,
                        dic_ritmo,
                        "Proyección por ritmo",
                        "#EF4444",
                        (f"Azul: ventas reales | Rojo: ritmo; diciembre anclado a noviembre con tope +18%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Rojo: ritmo; diciembre propaga {propagacion_dic:.0%} del ajuste, limitado a ±{limite_ajuste_dic:.0%}"),
                    )
                    st.altair_chart(chart_ritmo, use_container_width=True)
                    st.caption(
                        "Noviembre extrapola la venta de la semana 1 a 30 días. Diciembre recibe solo "
                        f"{propagacion_dic:.0%} del cambio relativo, con una protección máxima de ±{limite_ajuste_dic:.0%}."
                    )

                    st.markdown("### Evolución mensual con nowcast ponderado")
                    chart_nowcast = crear_grafica_escenario(
                        total_nowcast,
                        dic_nowcast,
                        "Nowcast ponderado",
                        "#10B981",
                        (f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%}; diciembre anclado a noviembre con tope +18%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%} de peso semanal; diciembre ajustado con propagación parcial"),
                    )
                    st.altair_chart(chart_nowcast, use_container_width=True)
                    st.caption(
                        "El nowcast combina el forecast original de noviembre con la proyección por ritmo. "
                        "Diciembre mantiene la estructura del forecast, pero incorpora una fracción controlada del cambio de noviembre."
                    )

                    st.markdown("### Comparación de proyecciones por categoría")
                    st.altair_chart(barras, use_container_width=True)

                    resumen_escenarios_dic = pd.DataFrame({
                        "Escenario": ["Forecast original", "Proyección por ritmo", "Nowcast ponderado"],
                        "Noviembre": [total_original, total_ritmo, total_nowcast],
                        "Diciembre": [dic, dic_ritmo, dic_nowcast],
                        "Factor_Noviembre_vs_original": [1.0, factor_nov_ritmo, factor_nov_nowcast],
                        "Ajuste_propagado_Diciembre": [0.0, ajuste_dic_ritmo, ajuste_dic_nowcast],
                        "Metodo_Diciembre": ["Forecast original", metodo_dic_ritmo, metodo_dic_nowcast],
                    })
                    st.dataframe(
                        resumen_escenarios_dic.style.format({
                            "Noviembre": "${:,.2f}", "Diciembre": "${:,.2f}",
                            "Factor_Noviembre_vs_original": "{:.3f}",
                            "Ajuste_propagado_Diciembre": "{:+.1%}",
                        }), use_container_width=True, hide_index=True,
                    )

                    tabla_comp = comparativo[[
                        "CATEGORIA", "Venta_Semana_1", "Prediccion_Modelo_Sin_Ajuste",
                        "Prediccion_Original", "Factor_Correccion_Aplicado", "Correccion_Aplicada",
                        "Cumplimiento_Forecast", "Proyeccion_Ritmo", "Nowcast_Ponderado",
                        "Brecha_Ritmo_pp", "Estado",
                    ]].sort_values("Brecha_Ritmo_pp")
                    st.dataframe(
                        tabla_comp.style.format({
                            "Venta_Semana_1": "${:,.2f}",
                            "Prediccion_Modelo_Sin_Ajuste": "${:,.2f}",
                            "Prediccion_Original": "${:,.2f}",
                            "Factor_Correccion_Aplicado": "{:.3f}",
                            "Cumplimiento_Forecast": "{:.1%}",
                            "Proyeccion_Ritmo": "${:,.2f}",
                            "Nowcast_Ponderado": "${:,.2f}",
                            "Brecha_Ritmo_pp": "{:+.1f} pp",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.download_button(
                        "Descargar comparativo semanal CSV",
                        tabla_comp.to_csv(index=False).encode("utf-8-sig"),
                        "comparativo_semana_1_vs_forecast_noviembre.csv",
                        "text/csv",
                    )

            # ------------------------------------------------------------------
            # Tabla resumen: Categoría, Código, Descripción, Nov 2026, Dic 2026
            # ------------------------------------------------------------------
            universo_tabla = universo if producto == "TODOS" else universo.loc[universo["CODIGO"].astype(str).eq(producto)]
            with st.expander("Auditoría de equivalencia: ventas, unidades y precio"):
                st.caption("Precio implícito anterior = venta final / unidades originales de CatBoost. "
                           "Se prioriza NUEVO PRECIO; si falta, se conserva la referencia anterior. Si tampoco existe, se usa PRECIO ESTATICO de PREDICCIONES 2 MESES. "
                           "Las referencias antiguas o con pocos meses requieren revisión.")
                cols_aud = ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo", "Prediccion_final",
                            "Unidades_antes_validacion", "Precio_implicito_anterior", "Precio_referencia_SKU",
                            "Precio_nuevo_archivo", "Precio_estatico_predicciones", "Precio_referencia_anterior",
                            "Fuente_precio_anterior", "Revision_cruce_precio",
                            "Precio_sugerido_sabana", "Precio_web", "Precio_ultimo_mes", "Fuente_precio",
                            "Coincidencia_codigo_sabana", "Fuente_web", "URL_precio_web",
                            "Unidades_pred", "Diferencia_unidades", "Factor_inconsistencia_precio",
                            "Prioridad_correccion_precio", "Alerta_precio_anterior",
                            "Calidad_referencia_precio", "Meses_precio_validos", "Ultimo_periodo_precio",
                            "Meses_precio_atipicos"]
                aud = universo_tabla[cols_aud].copy()
                prioridad_orden = {"P1 - CRITICA": 1, "P2 - ALTA": 2, "P3 - MEDIA": 3, "P4 - SIN ALERTA": 4}
                aud["Orden_prioridad"] = aud["Prioridad_correccion_precio"].map(prioridad_orden).fillna(5)
                aud = aud.sort_values(["Orden_prioridad", "Factor_inconsistencia_precio"], ascending=[True, False])
                resumen_cat = aud.groupby(["CATEGORIA", "Periodo"], as_index=False).agg(
                    Venta_MXN=("Prediccion_final", "sum"),
                    Unidades_antes=("Unidades_antes_validacion", "sum"),
                    Unidades_corregidas=("Unidades_pred", "sum"),
                    Precio_implicito_antes=("Precio_implicito_anterior", "median"),
                    Precio_referencia_mediana=("Precio_referencia_SKU", "median"),
                    Factor_inconsistencia_mediano=("Factor_inconsistencia_precio", "median"),
                    Factor_inconsistencia_maximo=("Factor_inconsistencia_precio", "max"),
                    SKU_analizados=("CODIGO", "nunique"),
                    Casos_P1=("Prioridad_correccion_precio", lambda s: int((s == "P1 - CRITICA").sum())),
                    Casos_P2=("Prioridad_correccion_precio", lambda s: int((s == "P2 - ALTA").sum())),
                )
                resumen_cat["Precio_categoria_antes"] = resumen_cat["Venta_MXN"].div(
                    resumen_cat["Unidades_antes"].where(resumen_cat["Unidades_antes"].gt(0))
                )
                resumen_cat["Precio_categoria_corregido"] = resumen_cat["Venta_MXN"].div(
                    resumen_cat["Unidades_corregidas"].where(resumen_cat["Unidades_corregidas"].gt(0))
                )
                resumen_cat["Prioridad_categoria"] = np.select(
                    [resumen_cat["Casos_P1"].gt(0) | resumen_cat["Factor_inconsistencia_mediano"].ge(3.0),
                     resumen_cat["Casos_P2"].gt(0) | resumen_cat["Factor_inconsistencia_mediano"].ge(1.5)],
                    ["P1 - CRITICA", "P2 - ALTA"], default="P3 - MONITOREO"
                )
                resumen_cat["Orden"] = resumen_cat["Prioridad_categoria"].map(
                    {"P1 - CRITICA": 1, "P2 - ALTA": 2, "P3 - MONITOREO": 3}
                )
                resumen_cat = resumen_cat.sort_values(
                    ["Orden", "Casos_P1", "Factor_inconsistencia_mediano", "Venta_MXN"],
                    ascending=[True, False, False, False]
                ).drop(columns="Orden")
                st.markdown("#### Prioridad por categoría")
                st.dataframe(resumen_cat, hide_index=True, use_container_width=True)
                st.download_button(
                    "Descargar prioridad de corrección por categoría CSV",
                    resumen_cat.to_csv(index=False).encode("utf-8-sig"),
                    "prioridad_correccion_precio_por_categoria.csv", "text/csv"
                )
                alertas = aud["Alerta_precio_anterior"].ne("SIN ALERTA") | aud["Calidad_referencia_precio"].ne("HISTORICO RECIENTE")
                solo = st.checkbox("Mostrar solo casos para revisión", value=True, key="solo_alertas_equivalencia")
                st.dataframe(aud.loc[alertas] if solo else aud, hide_index=True, use_container_width=True)
                aud_descarga = aud.drop(columns="Orden_prioridad")
                st.download_button("Descargar auditoría de equivalencia CSV", aud_descarga.to_csv(index=False).encode("utf-8-sig"),
                                   "auditoria_equivalencia_forecast.csv", "text/csv")
            if universo_tabla["Unidades_pred"].isna().any():
                st.warning("Hay SKU sin precio histórico válido. El total de unidades del periodo queda sin estimar; el importe sí incluye todos los SKU seleccionados.")
            st.markdown("### Tabla resumen: predicción por SKU (Nov-Dic 2026)")
            st.caption(
                "Unidades = venta monetaria final / precio conciliado por UPC. Prioridad: NUEVO PRECIO, referencia anterior y PRECIO ESTATICO de PREDICCIONES 2 MESES. "
                "La venta semanal está agregada por categoría; los escenarios por SKU se distribuyen "
                "según la participación monetaria de cada SKU."
            )

            tabla_sku = (
                universo.groupby(
                    ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo"],
                    as_index=False,
                )["Prediccion_final"].sum()
                .pivot(
                    index=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                    columns="Periodo",
                    values="Prediccion_final",
                )
                .reset_index()
            )
            tabla_sku.columns.name = None
            tabla_sku = tabla_sku.rename(columns={
                202611: "Prediccion_Nov_2026",
                202612: "Prediccion_Dic_2026",
            })

            # Agrega unidades reconciliadas con precio observado del SKU.
            tabla_unidades_sku = (
                universo.groupby(
                    ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo"],
                    as_index=False,
                )["Unidades_pred"].sum(min_count=1)
                .pivot(
                    index=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                    columns="Periodo",
                    values="Unidades_pred",
                )
                .reset_index()
            )
            tabla_unidades_sku.columns.name = None
            tabla_unidades_sku = tabla_unidades_sku.rename(columns={
                202611: "Unidades_Pred_Nov_2026",
                202612: "Unidades_Pred_Dic_2026",
            })
            tabla_sku = tabla_sku.merge(
                tabla_unidades_sku,
                on=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                how="left",
                validate="one_to_one",
            )
            tabla_sku = tabla_sku.merge(
                universo[["CODIGO", "Precio_referencia_SKU", "Fuente_precio"]].drop_duplicates(),
                on="CODIGO", how="left", validate="one_to_one",
            )
            for col in [
                "Prediccion_Nov_2026", "Prediccion_Dic_2026",
                "Unidades_Pred_Nov_2026", "Unidades_Pred_Dic_2026",
            ]:
                if col not in tabla_sku.columns:
                    tabla_sku[col] = np.nan
                tabla_sku[col] = pd.to_numeric(
                    tabla_sku[col], errors="coerce"
                )
                if col.startswith("Prediccion_"):
                    tabla_sku[col] = tabla_sku[col].fillna(0.0)

            tabla_sku["Precio_Implicito_Nov_2026"] = tabla_sku["Prediccion_Nov_2026"].div(
                tabla_sku["Unidades_Pred_Nov_2026"].where(tabla_sku["Unidades_Pred_Nov_2026"].gt(0))
            )
            tabla_sku["Precio_Implicito_Dic_2026"] = tabla_sku["Prediccion_Dic_2026"].div(
                tabla_sku["Unidades_Pred_Dic_2026"].where(tabla_sku["Unidades_Pred_Dic_2026"].gt(0))
            )
            tabla_sku["Total_NovDic"] = (
                tabla_sku["Prediccion_Nov_2026"] + tabla_sku["Prediccion_Dic_2026"]
            )
            tabla_sku["Total_Unidades_NovDic"] = (
                tabla_sku["Unidades_Pred_Nov_2026"]
                + tabla_sku["Unidades_Pred_Dic_2026"]
            )

            # --------------------------------------------------------------
            # Escenarios de noviembre por SKU
            # --------------------------------------------------------------
            # La tabla semanal disponible está agregada por categoría. Por ello,
            # la venta de semana 1 se distribuye entre SKU según la participación
            # de cada SKU en el forecast original de noviembre de su categoría.
            # Es una estimación por SKU, no una venta real observada por código.
            tabla_sku["Categoria_normalizada"] = tabla_sku["CATEGORIA"].map(
                normalizar_categoria_calibracion
            )
            tabla_sku["Participacion_SKU_Categoria"] = np.where(
                tabla_sku.groupby("Categoria_normalizada")["Prediccion_Nov_2026"]
                         .transform("sum") > 0,
                tabla_sku["Prediccion_Nov_2026"] /
                tabla_sku.groupby("Categoria_normalizada")["Prediccion_Nov_2026"]
                         .transform("sum"),
                0.0,
            )
            tabla_sku["Participacion_Unidades_SKU_Categoria"] = np.where(
                tabla_sku.groupby("Categoria_normalizada")["Unidades_Pred_Nov_2026"].transform("sum") > 0,
                tabla_sku["Unidades_Pred_Nov_2026"] / tabla_sku.groupby("Categoria_normalizada")["Unidades_Pred_Nov_2026"].transform("sum"),
                0.0,
            )
            tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] = np.nan
            tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"] = np.nan
            tabla_sku["Proyeccion_Ritmo_Nov_2026"] = np.nan
            tabla_sku["Nowcast_Ponderado_Nov_2026"] = np.nan

            if archivo_semanal is not None:
                venta_semanal_sku = cargar_venta_semanal_noviembre(archivo_semanal)[
                    ["Categoria_normalizada", "Venta_Semana_1", "Unidades_Semana_1"]
                ].copy()
                tabla_sku = tabla_sku.merge(
                    venta_semanal_sku,
                    on="Categoria_normalizada",
                    how="left",
                )
                tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] = (
                    tabla_sku["Venta_Semana_1"] * tabla_sku["Participacion_SKU_Categoria"]
                )
                tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"] = (
                    tabla_sku["Unidades_Semana_1"] * tabla_sku["Participacion_Unidades_SKU_Categoria"]
                )
                tabla_sku["Proyeccion_Ritmo_Nov_2026"] = (
                    tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] * 30.0 / 7.0
                )
                tabla_sku["Nowcast_Ponderado_Nov_2026"] = (
                    (1.0 - peso_real) * tabla_sku["Prediccion_Nov_2026"]
                    + peso_real * tabla_sku["Proyeccion_Ritmo_Nov_2026"]
                )
                tabla_sku = tabla_sku.drop(columns=["Venta_Semana_1", "Unidades_Semana_1"])
            else:
                st.info(
                    "Para mostrar ritmo y nowcast por SKU, carga PREDICCIONES NOVIEMBRE.xlsx. "
                    "Como la venta semanal está por categoría, el reparto por SKU se estima "
                    "con la participación del forecast original de noviembre."
                )

            tabla_sku = tabla_sku.drop(columns=["Categoria_normalizada"])
            tabla_sku = tabla_sku.sort_values(
                ["CATEGORIA", "CODIGO"]
            ).reset_index(drop=True)

            if producto != "TODOS":
                tabla_sku = tabla_sku.loc[tabla_sku["CODIGO"].astype(str).eq(producto)].copy()

            # Fila de totales al final
            fila_total = pd.DataFrame([{
                "CATEGORIA": "TOTAL",
                "CODIGO": "",
                "DESCRIPCION": "",
                "Prediccion_Nov_2026": tabla_sku["Prediccion_Nov_2026"].sum(),
                "Unidades_Pred_Nov_2026": tabla_sku["Unidades_Pred_Nov_2026"].sum(skipna=False),
                "Monto_Venta_Semana_Prediccion_SKU_Estimado": tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"].sum(min_count=1),
                "Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas": tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"].sum(min_count=1),
                "Proyeccion_Ritmo_Nov_2026": tabla_sku["Proyeccion_Ritmo_Nov_2026"].sum(min_count=1),
                "Nowcast_Ponderado_Nov_2026": tabla_sku["Nowcast_Ponderado_Nov_2026"].sum(min_count=1),
                "Prediccion_Dic_2026": tabla_sku["Prediccion_Dic_2026"].sum(),
                "Unidades_Pred_Dic_2026": tabla_sku["Unidades_Pred_Dic_2026"].sum(skipna=False),
                "Total_NovDic": tabla_sku["Total_NovDic"].sum(),
                "Total_Unidades_NovDic": tabla_sku["Total_Unidades_NovDic"].sum(skipna=False),
                "Participacion_SKU_Categoria": np.nan,
                "Participacion_Unidades_SKU_Categoria": np.nan,
            }])
            for mes in ("Nov", "Dic"):
                unidades_total = fila_total[f"Unidades_Pred_{mes}_2026"]
                fila_total[f"Precio_Implicito_{mes}_2026"] = fila_total[f"Prediccion_{mes}_2026"].div(
                    unidades_total.where(unidades_total.gt(0))
                )
            tabla_sku_mostrar = pd.concat([tabla_sku, fila_total], ignore_index=True)

            st.dataframe(
                tabla_sku_mostrar.style.format({
                    "Prediccion_Nov_2026": "${:,.2f}",
                    "Precio_Implicito_Nov_2026": "${:,.2f}",
                    "Precio_referencia_SKU": "${:,.2f}",
                    "Unidades_Pred_Nov_2026": "{:,.0f}",
                    "Monto_Venta_Semana_Prediccion_SKU_Estimado": "${:,.2f}",
                    "Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas": "{:,.0f}",
                    "Proyeccion_Ritmo_Nov_2026": "${:,.2f}",
                    "Nowcast_Ponderado_Nov_2026": "${:,.2f}",
                    "Participacion_SKU_Categoria": "{:.2%}",
                    "Participacion_Unidades_SKU_Categoria": "{:.2%}",
                    "Prediccion_Dic_2026": "${:,.2f}",
                    "Precio_Implicito_Dic_2026": "${:,.2f}",
                    "Unidades_Pred_Dic_2026": "{:,.0f}",
                    "Total_NovDic": "${:,.2f}",
                    "Total_Unidades_NovDic": "{:,.0f}",
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
                height=440,
            )
            st.download_button(
                "Descargar tabla SKU con semana de predicción, unidades, ritmo y nowcast CSV",
                tabla_sku.to_csv(index=False).encode("utf-8-sig"),
                "tabla_sku_forecast_ventas_unidades_ritmo_nowcast_nov_dic_2026.csv",
                "text/csv",
            )

            st.download_button(
                "Descargar forecast filtrado CSV",
                universo_tabla.to_csv(index=False).encode("utf-8-sig"),
                "forecast_nov_dic_2026_universo_seleccionado.csv",
                "text/csv",
            )
            with st.expander("Diagnóstico del forecast y universo SKU"):
                st.caption(
                    "Predicción base muestra la salida 85% venta directa + 15% unidades por precio. "
                    "Corrección piso indica cuánto añadió la protección estacional; cero significa que el modelo ya estaba por encima del piso."
                )
                diag = universo[[
                    "CODIGO", "DESCRIPCION", "CATEGORIA", "COMBOS", "Periodo",
                    "Prediccion_base", "Referencia_ultimo_ano", "Crecimiento_YTD",
                    "Prediccion_estacional", "Prediccion_combinada", "Piso_parcial",
                    "Porcentaje_recuperacion", "Correccion_piso",
                    "Ajuste_aplicado", "Prediccion_antes_calibracion",
                    "Factor_calibracion_categoria", "Calibracion_categoria_aplicada",
                    "Motivo_calibracion", "Prediccion_antes_control_lacteos",
                    "Unidades_antes_control_lacteos", "Factor_control_diciembre_lacteos",
                    "Control_diciembre_lacteos_aplicado", "Prediccion_antes_ajuste_chocolate_mesa", "Factor_ajuste_nov_chocolate_mesa", "Ajuste_nov_chocolate_mesa_aplicado", "Prediccion_final", "Unidades_modelo_sin_reconciliar", "Precio_modelo", "Precio_pred", "Unidades_pred",
                ]].copy()
                resumen_categoria_v2 = (
                    diag.groupby("CATEGORIA", as_index=False)
                    .agg(
                        SKU=("CODIGO", "nunique"),
                        Antes=("Prediccion_antes_calibracion", "sum"),
                        Despues=("Prediccion_final", "sum"),
                        Factor=("Factor_calibracion_categoria", "max"),
                    )
                )
                resumen_categoria_v2["Impacto"] = resumen_categoria_v2["Despues"] - resumen_categoria_v2["Antes"]
                st.markdown("#### Resumen por categoría")
                st.dataframe(
                    resumen_categoria_v2.style.format({
                        "Antes": "${:,.2f}", "Despues": "${:,.2f}",
                        "Factor": "{:.2f}", "Impacto": "${:+,.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                no_contados = forecast[~forecast["Incluido_forecast_base"]].copy()
                if not no_contados.empty:
                    resumen_no_contados = (
                        no_contados.groupby(
                            ["CODIGO", "DESCRIPCION", "CATEGORIA", "COMBOS",
                             "Ultimo_periodo_disponible", "Meses_sin_actividad_a_octubre"],
                            as_index=False,
                        )
                        .agg(Forecast_NovDic=("Prediccion_final", "sum"))
                        .sort_values("Forecast_NovDic", ascending=False)
                    )
                    st.markdown("#### SKU presentes en 2026 no contados en el universo base")
                    st.dataframe(
                        resumen_no_contados.style.format({"Forecast_NovDic": "${:,.2f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Descargar SKU 2026 no contados",
                        resumen_no_contados.to_csv(index=False).encode("utf-8-sig"),
                        "sku_2026_no_contados_forecast_base.csv",
                        "text/csv",
                    )
                st.markdown("#### Detalle por SKU")
                st.dataframe(diag, use_container_width=True, hide_index=True)


    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")
