# Proyecto Final - Machine Learning con PySpark y Docker

**Estudiante:** Sergio Huertas  
**Curso:** Machine Learning con PySpark y Docker | 2026-I  
**Repositorio:** `proyecto_final_Huertas_Sergio`

---

## Descripción del problema

Este proyecto aborda dos problemas analiticos complementarios sobre datos colombianos:

**Bloque 1 y 2 — Análisis del uso del suelo en Bogota.** A partir del dataset TUSO del catastro distrital, que contiene mas de un millón de predios con su tipo de uso y área, se aplican técnicas de análisis exploratorio (EDA), aprendizaje no supervisado (PCA + K-Means) y clasificación supervisada (Regresion Logística + Random Forest).

El objetivo es identificar patrones de zonificación urbana, segmentar los predios en clusters significativos y construir un modelo predictivo que determine si un predio es residencial (68% de los casos) a partir de su área, sector geográfico y otras caracteristicas. Este tipo de analisis tiene aplicación directa en avaluos catastrales, planeación urbana y politicas de vivienda.

**Bloque 3 — Análisis de sentimiento sobre opinion pública institucional colombiana.** Sobre un corpus de 204 comentarios y tweets acerca de alcaldias, ministerios, TransMilenio, policia y servicios publicos, se compara el rendimiento de un pipeline clásico de TF-IDF + Regresión Logistica contra un modelo pre-entrenado de Hugging Face (robertuito-sentiment-analysis) mediante knowledge distillation. El objetivo es determinar que enfoque conviene para un sistema de monitoreo automatico de opinión ciudadana en producción, evaluando precisión, interpretabilidad, manejo de sarcasmo y costo computacional.

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

### Opción 2: Entorno local

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
| `pandas` | >= 2.0.0 | Manipulación de datos en local |
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

## Conclusión integrada

El proyecto recorre el flujo completo de un caso real de ciencia de datos aplicado al contexto colombiano. 

En el Bloque 1 se comprobo que el 68% de los predios de Bogotá corresponden a uso residencial (codigo 1) con un area promedio de 187 m², y que existen valores atipicos extremos que distorsionan la media general.

El Bloque 2 demostro que el area del predio tanto en escala original como logaritmica, explica más del 95% de la capacidad predictiva para clasificar el uso residencial (AUC 0.79 con Random Forest), mientras que el sector geografico apenas aporta un 4.3% de importancia.

El Bloque 3 evidencio que, para tareas de análisis de sentimiento sobre opinión publica institucional, el modelo pre-entrenado de Hugging Face (robertuito) supera ampliamente al pipeline TF-IDF + Regresión Logistica en el manejo de sarcasmo, negaciones y ambiguedad, aunque este ultimo ofrece total interpretabilidad a través de sus coeficientes. La integración de los tres bloques muestra que la elección del modelo depende del contexto: cuando la explicabilidad y la eficiencia computacional son prioritarias (catastro, avaluos), los modelos clasicos de Spark ML son la opcion correcta; cuando la precision semantica es critica (monitoreo de redes sociales), la transferencia de aprendizaje con modelos pre-entrenados es indispensable.
