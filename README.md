# Project 10: Feature Selection & Dimensionality Reduction (PCA, Information Gain, Chi-Square)

## Overview & Problem Statement
High-dimensional datasets frequently suffer from the "curse of dimensionality": correlated features inflate computational overhead, degrade model interpretability, and risk severe overfitting. Effective feature space optimization requires understanding when to compress feature representations via **unsupervised dimensionality reduction** versus when to filter features via **supervised statistical tests**.

This project investigates the empirical trade-offs between full feature sets, Principal Component Analysis (PCA), Information Gain (Mutual Information), and Chi-Square ($\chi^2$) feature selection, evaluated against a baseline Decision Tree on the Wisconsin Breast Cancer diagnostic dataset.

---

## Dataset Description
- **Source**: Wisconsin Diagnostic Breast Cancer (WDBC) Database (via `sklearn.datasets.load_breast_cancer`).
- **Volume**: 569 instances, 30 continuous dimensional attributes.
- **Target Variable**: Binary diagnosis (`0` = Malignant, `1` = Benign).
- **Features**: 30 quantitative measurements computed from digitized images of fine needle aspirate (FNA) biopsies:
  - Mean, standard error, and "worst" (largest) values for: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension`.

---

## Methodology & Optimization Strategies

### 1. Baseline Model Architecture
- Partitioned dataset into 80% train and 20% holdout test splits (`random_state=42`).
- Trained a benchmark `DecisionTreeClassifier(random_state=42)` across all 30 raw features to establish target performance metrics.

### 2. Unsupervised Dimensionality Reduction: PCA
- Computed **Principal Component Analysis (PCA)** projecting 30 correlated features onto 3 principal orthogonal components capturing maximal variance.
- Retrained the Decision Tree classifier strictly on the 3-dimensional latent manifold (`x_train_pca`).

### 3. Supervised Feature Selection: Information Gain (Mutual Information)
- Applied non-parametric mutual information ranking via `SelectKBest(score_func=mutual_info_classif, k=4)`.
- Extracted the top 4 non-linearly dependent features relating to tumor malignancy and retrained the classifier (`x_train_gain`).

### 4. Supervised Feature Selection: Chi-Square ($\chi^2$) Test
- Applied statistical hypothesis testing of independence using `SelectKBest(score_func=chi2, k=1)` and $k=4$.
- Selected top features exhibiting maximal statistical dependence with the target diagnostic class (`x_train_chi2`).

---

## Performance Comparison & Empirical Findings

Evaluated on the independent test set ($N = 114$):

| Optimization Strategy | Method Type | Feature Space Size | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Features)** | None (Raw 30 Features) | 30 | **95.0%** | 0.94 | 0.94 | **0.94** |
| **PCA** | Unsupervised Projection | **3 (90% reduction)** | **94.0%** | 0.94 | 0.93 | **0.93** |
| **Information Gain** | Supervised Filter | **4 (87% reduction)** | **94.0%** | 0.94 | 0.93 | **0.93** |
| **Chi-Square ($\chi^2$)** | Supervised Statistical Filter | **4 (87% reduction)** | **95.0%** | 0.94 | 0.94 | **0.94** |

### Key Engineering Insights:
- **Chi-Square Feature Selection matches baseline performance (95.0% Accuracy, 0.94 F1)** while eliminating **87% of redundant input dimensions**.
- **PCA compresses 30 features into just 3 components** with only a negligible 1% accuracy decrease, drastically reducing training latency and memory consumption.
- **Dimensionality reduction proves essential** for latency-sensitive embedded deployments where capturing 30 complex biomarkers is cost-prohibitive.

---

## Quickstart Guide

### 1. Environment Setup
Navigate to the project folder:
```bash
cd 10-feature-selection-dimensionality-reduction
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Running the Notebook
Launch Jupyter:
```bash
jupyter notebook "FS&DR.ipynb"
```
Run all cells to execute the baseline tree, compute PCA components, extract Information Gain and Chi-Square feature subsets, and compare classification reports.
