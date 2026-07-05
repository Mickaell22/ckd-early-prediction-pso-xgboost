# Early Prediction of Chronic Kidney Disease with PSO-XGBoost

**Language / Idioma:** [Español](README.md) | English

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-EB5424)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20development-yellow)

Early prediction of **Chronic Kidney Disease (CKD)** in patients with **Type 2 Diabetes Mellitus (T2DM)** using **XGBoost** with hyperparameters optimized via **Particle Swarm Optimization (PSO)**, interpreted with **SHAP**.

Academic project for the Machine Learning course — Software Engineering, Universidad de Guayaquil.

---

## Description

CKD is one of the most frequent and silent complications of type 2 diabetes: by the time symptoms appear, kidney damage is usually irreversible. This project builds a classification model that detects CKD early from routine clinical and laboratory variables.

The core approach is **PSO-XGBoost**: an XGBoost classifier whose six main hyperparameters are tuned by a particle swarm that maximizes the AUC-ROC in stratified 5-fold cross-validation. The model is compared against four baselines and its predictions are explained with SHAP values.

### Pipeline

```
UCI CKD Dataset (400 instances, 25 attributes)
        │
        ├── Filtering: diabetic subgroup (dm = 'yes')
        ├── Cleaning and typing ('?' → NaN)
        ├── Stratified 80/20 split (seed = 42)
        │
        ├── Preprocessing (fitted ONLY on train)
        │     ├── KNN imputation (k=5) for numerical features
        │     ├── Mode imputation for categorical features
        │     ├── Binary and one-hot encoding
        │     └── MinMaxScaler
        │
        ├── SMOTE (training set only)
        │
        ├── PSO (30 particles, 50 iterations)
        │     └── Fitness: mean AUC-ROC over 5-fold CV
        │
        ├── Final XGBoost (5 runs with seeds 42-46)
        │
        └── Evaluation + SHAP interpretability
```

### Compared models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| SVM | RBF kernel, `probability=True` |
| Random Forest | 100 trees |
| XGBoost | default hyperparameters |
| **PSO-XGBoost** | **PSO-optimized hyperparameters** |

### PSO search space

| Hyperparameter | Range | Type |
|---|---|---|
| `learning_rate` | [0.01, 0.30] | continuous |
| `n_estimators` | [50, 500] | integer |
| `max_depth` | [3, 10] | integer |
| `min_child_weight` | [1, 10] | integer |
| `subsample` | [0.5, 1.0] | continuous |
| `colsample_bytree` | [0.5, 1.0] | continuous |

Swarm configuration: 30 particles, 50 iterations, `c1 = c2 = 2.0`, inertia decreasing linearly from 0.9 to 0.4, maximum velocity at 20% of each dimension's range (`pyswarms.single.GlobalBestPSO`).

### Reproducibility

- Fixed seed `42` for splits, SMOTE, PSO and XGBoost.
- SMOTE is applied to the training set only.
- Each final experiment is repeated 5 times (seeds 42–46) and mean ± standard deviation is reported.
- Preprocessors are fitted on training data only and persisted with `joblib`.

---

## Project structure

```
ckd-early-prediction-pso-xgboost/
├── data/
│   ├── raw/                    # UCI dataset (not versioned)
│   └── processed/              # Processed train/test splits
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Preprocessing pipeline
│   ├── 03_baselines.ipynb      # Baseline models
│   ├── 04_pso_xgboost.ipynb    # PSO optimization
│   └── 05_shap_analysis.ipynb  # Interpretability
├── src/
│   ├── config.py               # Constants and paths
│   ├── data_loader.py          # Dataset loading and splitting
│   ├── preprocessing.py        # CKDPreprocessor + SMOTE
│   ├── pso_optimizer.py        # PSOXGBoostOptimizer
│   ├── models.py               # Baselines and training
│   ├── evaluation.py           # Metrics, ROC, multi-seed
│   └── interpretability.py     # SHAP analysis
├── scripts/
│   ├── train_baselines.py      # Trains the 4 baseline models
│   ├── train_pso_xgboost.py    # Optimizes and trains PSO-XGBoost
│   └── generate_report.py      # Final tables and figures
├── results/
│   ├── figures/                # Figures for the paper
│   └── models/                 # Serialized models
├── tests/                      # Unit tests (pytest)
└── paper/                      # Reference IEEE paper
```

---

## Installation

### Requirements

- Python 3.11 (or Docker, see below)
- Git
- ~500 MB of disk space

### Steps

```bash
git clone https://github.com/Mickaell22/ckd-early-prediction-pso-xgboost.git
cd ckd-early-prediction-pso-xgboost
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Docker alternative

If you don't have Python 3.11 installed, the Docker image provides the full environment:

```bash
docker build -t ckd-pso-xgboost .

# Run any script mounting data/ and results/ as volumes
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  ckd-pso-xgboost python scripts/train_baselines.py

# Tests
docker run --rm ckd-pso-xgboost pytest tests/

# Jupyter (available at http://localhost:8888)
docker run --rm -p 8888:8888 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  ckd-pso-xgboost jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root notebooks/
```

### Dataset

Download the [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) and place the CSV at `data/raw/ckd_uci.csv`. See `data/raw/README.md` for detailed instructions.

---

## Usage

Reproduce the full results:

```bash
# 1. Train the baselines (5 seeds)
python scripts/train_baselines.py

# 2. Optimize hyperparameters with PSO and train the final model
python scripts/train_pso_xgboost.py

# 3. Consolidate metrics and generate figures for the paper
python scripts/generate_report.py
```

Metrics are written to `results/` and figures to `results/figures/`.

For interactive exploration:

```bash
jupyter notebook notebooks/
```

Start with `01_eda.ipynb` and follow the numerical order.

### Tests

```bash
pytest tests/
```

---

## Expected results

Exact values may vary slightly between runs, but the model should reach equivalent levels:

| Model | Accuracy | Precision | Sensitivity | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.913 | 0.900 | 0.940 | 0.920 | 0.918 |
| SVM (RBF) | 0.938 | 0.926 | 0.960 | 0.943 | 0.945 |
| Random Forest | 0.963 | 0.962 | 0.980 | 0.971 | 0.972 |
| XGBoost (default) | 0.975 | 0.980 | 0.980 | 0.980 | 0.964 |
| **PSO-XGBoost** | **0.992** | **0.984** | **1.000** | **0.992** | **0.997** |

According to SHAP, the most influential variables should include serum creatinine, hemoglobin, urine specific gravity and albuminuria.

---

## Authors

- Mickaell Morán — Universidad de Guayaquil

Final project for the Machine Learning course, 2026 semester.

## Citation

```bibtex
@techreport{mickaell2026psoxgboost,
  title  = {Optimización de Hiperparámetros de XGBoost mediante Algoritmo de Enjambre de Partículas para la Predicción Temprana de Enfermedad Renal Crónica en Pacientes con Diabetes Mellitus Tipo 2},
  author = {Mickaell Morán},
  year   = {2026},
  institution = {Universidad de Guayaquil},
  type   = {Trabajo académico}
}
```

## License

MIT License. See [LICENSE](LICENSE) for the full text.

## Acknowledgements

UCI Chronic Kidney Disease dataset donated by L. Rubini, P. Soundarapandian and P. Eswaran (2015). Available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) under a CC BY 4.0 license.
