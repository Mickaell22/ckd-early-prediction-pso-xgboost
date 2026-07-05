# Predicción Temprana de Enfermedad Renal Crónica con PSO-XGBoost

**Idioma / Language:** Español | [English](README.en.md)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-EB5424)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-En%20desarrollo-yellow)

Predicción temprana de **Enfermedad Renal Crónica (ERC)** en pacientes con **Diabetes Mellitus tipo 2 (DM2)** usando **XGBoost** con hiperparámetros optimizados mediante **Particle Swarm Optimization (PSO)**, interpretado con **SHAP**.

Trabajo académico para la asignatura de Aprendizaje Automático — Ingeniería en Software, Universidad de Guayaquil.

---

## Descripción

La ERC es una de las complicaciones más frecuentes y silenciosas de la diabetes tipo 2: cuando los síntomas aparecen, el daño renal suele ser irreversible. Este proyecto construye un modelo de clasificación que detecta ERC de forma temprana a partir de variables clínicas y de laboratorio rutinarias.

El enfoque central es **PSO-XGBoost**: un clasificador XGBoost cuyos seis hiperparámetros principales se ajustan con un enjambre de partículas que maximiza el AUC-ROC en validación cruzada estratificada de 5 folds. El modelo se compara contra cuatro baselines y sus predicciones se explican con valores SHAP.

### Pipeline

```
Dataset UCI CKD (400 instancias, 25 atributos)
        │
        ├── Filtrado: subgrupo diabético (dm = 'yes')
        ├── Limpieza y tipado ('?' → NaN)
        ├── Partición estratificada 80/20 (seed = 42)
        │
        ├── Preprocesamiento (ajustado SOLO con train)
        │     ├── Imputación KNN (k=5) para numéricas
        │     ├── Imputación por moda para categóricas
        │     ├── Codificación binaria y one-hot
        │     └── MinMaxScaler
        │
        ├── SMOTE (solo sobre entrenamiento)
        │
        ├── PSO (30 partículas, 50 iteraciones)
        │     └── Fitness: AUC-ROC promedio en 5-fold CV
        │
        ├── XGBoost final (5 corridas con seeds 42-46)
        │
        └── Evaluación + interpretabilidad SHAP
```

### Modelos comparados

| Modelo | Configuración |
|---|---|
| Regresión Logística | `max_iter=1000` |
| SVM | kernel RBF, `probability=True` |
| Random Forest | 100 árboles |
| XGBoost | hiperparámetros por defecto |
| **PSO-XGBoost** | **hiperparámetros optimizados por PSO** |

### Espacio de búsqueda del PSO

| Hiperparámetro | Rango | Tipo |
|---|---|---|
| `learning_rate` | [0.01, 0.30] | continuo |
| `n_estimators` | [50, 500] | entero |
| `max_depth` | [3, 10] | entero |
| `min_child_weight` | [1, 10] | entero |
| `subsample` | [0.5, 1.0] | continuo |
| `colsample_bytree` | [0.5, 1.0] | continuo |

Configuración del enjambre: 30 partículas, 50 iteraciones, `c1 = c2 = 2.0`, inercia decreciente linealmente de 0.9 a 0.4, velocidad máxima al 20% del rango de cada dimensión (`pyswarms.single.GlobalBestPSO`).

### Reproducibilidad

- Semilla fija `42` para particiones, SMOTE, PSO y XGBoost.
- SMOTE se aplica únicamente al conjunto de entrenamiento.
- Cada experimento final se repite 5 veces (seeds 42–46) y se reporta media ± desviación estándar.
- Preprocesadores ajustados solo con datos de entrenamiento y persistidos con `joblib`.

---

## Estructura del proyecto

```
ckd-early-prediction-pso-xgboost/
├── data/
│   ├── raw/                    # Dataset UCI (no versionado)
│   └── processed/              # Particiones train/test procesadas
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   ├── 02_preprocessing.ipynb  # Pipeline de preprocesamiento
│   ├── 03_baselines.ipynb      # Modelos base
│   ├── 04_pso_xgboost.ipynb    # Optimización PSO
│   └── 05_shap_analysis.ipynb  # Interpretabilidad
├── src/
│   ├── config.py               # Constantes y rutas
│   ├── data_loader.py          # Carga y partición del dataset
│   ├── preprocessing.py        # CKDPreprocessor + SMOTE
│   ├── pso_optimizer.py        # PSOXGBoostOptimizer
│   ├── models.py               # Baselines y entrenamiento
│   ├── evaluation.py           # Métricas, ROC, multi-seed
│   └── interpretability.py     # Análisis SHAP
├── scripts/
│   ├── train_baselines.py      # Entrena los 4 modelos base
│   ├── train_pso_xgboost.py    # Optimiza y entrena PSO-XGBoost
│   └── generate_report.py      # Tablas y figuras finales
├── results/
│   ├── figures/                # Figuras para el paper
│   └── models/                 # Modelos serializados
├── tests/                      # Tests unitarios (pytest)
└── paper/                      # Paper IEEE de referencia
```

---

## Instalación

### Requisitos

- Python 3.11 (o Docker, ver mas abajo)
- Git
- ~500 MB de espacio en disco

### Pasos

```bash
git clone https://github.com/Mickaell22/ckd-early-prediction-pso-xgboost.git
cd ckd-early-prediction-pso-xgboost
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Alternativa con Docker

Si no tienes Python 3.11 instalado, la imagen Docker resuelve el entorno completo:

```bash
docker build -t ckd-pso-xgboost .

# Correr cualquier script montando data/ y results/ como volumenes
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  ckd-pso-xgboost python scripts/train_baselines.py

# Tests
docker run --rm ckd-pso-xgboost pytest tests/

# Jupyter (disponible en http://localhost:8888)
docker run --rm -p 8888:8888 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  ckd-pso-xgboost jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root notebooks/
```

### Dataset

Descargar el [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) y colocar el CSV en `data/raw/ckd_uci.csv`. Ver `data/raw/README.md` para instrucciones detalladas.

---

## Uso

Reproducir los resultados completos:

```bash
# 1. Entrenar los baselines (5 seeds)
python scripts/train_baselines.py

# 2. Optimizar hiperparámetros con PSO y entrenar el modelo final
python scripts/train_pso_xgboost.py

# 3. Consolidar métricas y generar figuras para el paper
python scripts/generate_report.py
```

Las métricas quedan en `results/` y las figuras en `results/figures/`.

Para exploración interactiva:

```bash
jupyter notebook notebooks/
```

Empezar por `01_eda.ipynb` y seguir el orden numérico.

### Tests

```bash
pytest tests/
```

---

## Resultados obtenidos

Media sobre 5 semillas (42–46), partición de prueba estratificada del 20%. Los
valores pueden variar levemente entre corridas por la naturaleza estocástica del PSO.

| Modelo | Accuracy | Precisión | Sensibilidad | Especificidad | F1 | AUC |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.950 | 1.000 | 0.920 | 1.000 | 0.958 | 1.000 |
| SVM (RBF) | 0.988 | 1.000 | 0.980 | 1.000 | 0.990 | 1.000 |
| Random Forest | 0.988 | 1.000 | 0.980 | 1.000 | 0.990 | 1.000 |
| XGBoost (default) | 0.963 | 1.000 | 0.940 | 1.000 | 0.969 | 0.999 |
| **PSO-XGBoost** | **0.970** | **1.000** | **0.952** | **1.000** | **0.975** | **0.999** |

Mejores hiperparámetros hallados por el PSO (AUC-ROC en CV = 0.9992): `learning_rate=0.153`,
`n_estimators=271`, `max_depth=5`, `min_child_weight=1`, `subsample=0.859`, `colsample_bytree=0.800`.

El dataset UCI CKD es altamente separable, por lo que todos los modelos alcanzan un
rendimiento cercano al techo; el PSO-XGBoost logra el mejor AUC en validación cruzada y
precisión/especificidad perfectas. Según SHAP, las variables más influyentes son
hemoglobina (`hemo`), gravedad específica de la orina (`sg`), creatinina sérica (`sc`),
conteo de glóbulos rojos (`rc`) y albuminuria (`al`), consistente con la fisiopatología de la ERC.

### Figuras

Todas las figuras se generan en `results/figures/`: convergencia del PSO, curvas ROC
comparativas, matriz de confusión, importancia de variables (XGBoost) y análisis SHAP
(summary, importancia global y waterfall de un caso individual).

### Nota sobre el alcance

El paper planteaba el análisis primario sobre el subgrupo de pacientes diabéticos
(`dm == 'yes'`). En el dataset UCI **los 137 pacientes diabéticos tienen todos ERC**
(una sola clase), por lo que ese subgrupo no permite entrenar un clasificador. El
modelado se realiza sobre el dataset completo (400 instancias: 250 con ERC, 150 sin ERC)
y el ángulo diabético se documenta como observación de prevalencia en el EDA
(`notebooks/01_eda.ipynb`).

---

## Autores

- Mickaell Morán — Universidad de Guayaquil

Trabajo final de la asignatura de Aprendizaje Automático, semestre 2026.

## Cita

```bibtex
@techreport{mickaell2026psoxgboost,
  title  = {Optimización de Hiperparámetros de XGBoost mediante Algoritmo de Enjambre de Partículas para la Predicción Temprana de Enfermedad Renal Crónica en Pacientes con Diabetes Mellitus Tipo 2},
  author = {Mickaell Morán},
  year   = {2026},
  institution = {Universidad de Guayaquil},
  type   = {Trabajo académico}
}
```

## Licencia

MIT License. Ver [LICENSE](LICENSE) para el texto completo.

## Reconocimientos

Dataset UCI Chronic Kidney Disease donado por L. Rubini, P. Soundarapandian y P. Eswaran (2015). Disponible en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) bajo licencia CC BY 4.0.
