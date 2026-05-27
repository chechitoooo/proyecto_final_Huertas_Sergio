# Proyecto Final - Machine Learning con PySpark y Docker

**Estudiante:** Sergio Huertas  
**Curso:** Machine Learning con PySpark y Docker | 2026-I  
**Repositorio:** `proyecto_final_Huertas_Sergio`

---

## Descripcion del problema

Este proyecto aborda dos problemas analiticos complementarios sobre datos colombianos:

**Bloque 1 y 2 — Analisis del uso del suelo en Bogota.** A partir del dataset TUSO del catastro distrital, que contiene mas de un millon de predios con su tipo de uso y area, se aplican tecnicas de analisis exploratorio (EDA), aprendizaje no supervisado (PCA + K-Means) y clasificacion supervisada (Regresion Logistica + Random Forest). El objetivo es identificar patrones de zonificacion urbana, segmentar los predios en clusters significativos y construir un modelo predictivo que determine si un predio es residencial (68% de los casos) a partir de su area, sector geografico y otras caracteristicas. Este tipo de analisis tiene aplicacion directa en avaluos catastrales, planeacion urbana y politicas de vivienda.

**Bloque 3 — Analisis de sentimiento sobre opinion publica institucional colombiana.** Sobre un corpus de 204 comentarios y tweets acerca de alcaldias, ministerios, TransMilenio, policia y servicios publicos, se compara el rendimiento de un pipeline clasico de TF-IDF + Regresion Logistica contra un modelo pre-entrenado de Hugging Face (robertuito-sentiment-analysis) mediante knowledge distillation. El objetivo es determinar que enfoque conviene para un sistema de monitoreo automatico de opinion ciudadana en produccion, evaluando precision, interpretabilidad, manejo de sarcasmo y costo computacional.

---

## Datasets utilizados

| Bloque | Dataset | Filas | Columnas | Fuente |
|---|---|---|---|---|
| 1 y 2 | `Data/TUSO.csv` | 1,060,001 | USOCLOTE, USOTUSO, USOAREA | [Datos Abiertos Bogota / IDECA](https://datosabiertos.bogota.gov.co/dataset/) - Catastro Distrital - Uso del Suelo |
| 3 | `bloque3_nlp/corpus_colombia.csv` | 204 | id, texto | Corpus sintetico elaborado a partir de patrones reales de opinion publica en redes sociales colombianas |

---

## Como ejecutar los notebooks

### Opcion 1: Docker (recomendado)

```bash
# Construir y levantar el contenedor
docker-compose up --build

# Abrir en el navegador
http://localhost:8890
```

El contenedor incluye:
- **Python 3.10** con Jupyter Notebook
- **Apache Spark 3.5.0** (PySpark)
- Todas las dependencias listadas en `requirements.txt`
- Los notebooks se ejecutan desde `/home/jovyan/work`

### Opcion 2: Entorno local

```bash
# Requisitos: Python >= 3.10, Java >= 11
pip install -r requirements.txt

# Orden de ejecucion:
# 1. bloque1_eda/bloque1_eda_Huertas.ipynb   (EDA con PySpark)
# 2. bloque2_ml/bloque2_ml_huertas.ipynb      (ML con Spark ML)
# 3. bloque3_nlp/bloque3_nlp_huertas.ipynb     (NLP con HF + PySpark)
```

### Dependencias principales

| Paquete | Version | Uso |
|---|---|---|
| `pyspark` | >= 3.5.0 | Motor de procesamiento distribuido |
| `pandas` | >= 2.0.0 | Manipulacion de datos en local |
| `scikit-learn` | >= 1.3.0 | Metricas y complementos |
| `matplotlib` / `seaborn` | >= 3.7.0 / >= 0.12.0 | Visualizaciones |
| `transformers` | (Bloque 3) | Modelo pre-entrenado Hugging Face |
| `sentence-transformers` | (Bloque 3) | Embeddings semanticos |

---

## Estructura del repositorio

```
proyecto_final_Huertas_Sergio/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Data/
│   └── TUSO.csv
├── bloque1_eda/
│   └── bloque1_eda_Huertas.ipynb
├── bloque2_ml/
│   └── bloque2_ml_huertas.ipynb
└── bloque3_nlp/
    ├── corpus_colombia.csv
    └── bloque3_nlp_huertas.ipynb
```

---

## Conclusion integrada

El proyecto recorre el flujo completo de un caso real de ciencia de datos aplicado al contexto colombiano. En el Bloque 1 se comprobo que el 68% de los predios de Bogota corresponden a uso residencial (codigo 1) con un area promedio de 187 m², y que existen valores atipicos extremos que distorsionan la media general. El Bloque 2 demostro que el area del predio —tanto en escala original como logaritmica— explica mas del 95% de la capacidad predictiva para clasificar el uso residencial (AUC 0.79 con Random Forest), mientras que el sector geografico apenas aporta un 4.3% de importancia. El Bloque 3 evidencio que, para tareas de analisis de sentimiento sobre opinion publica institucional, el modelo pre-entrenado de Hugging Face (robertuito) supera ampliamente al pipeline TF-IDF + Regresion Logistica en el manejo de sarcasmo, negaciones y ambiguedad, aunque este ultimo ofrece total interpretabilidad a traves de sus coeficientes. La integracion de los tres bloques muestra que la eleccion del modelo depende del contexto: cuando la explicabilidad y la eficiencia computacional son prioritarias (catastro, avaluos), los modelos clasicos de Spark ML son la opcion correcta; cuando la precision semantica es critica (monitoreo de redes sociales), la transferencia de aprendizaje con modelos pre-entrenados es indispensable.
