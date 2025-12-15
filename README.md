# CS229 Options Mispricing Detection Project

**Discovering Nonlinear Latent Drivers of Option Mispricing via Kernel PCA, Support Vector Machines, and Enhanced Feature Engineering**

By Juli Huang, Jake Cheng, and Rupert Lu  
Stanford University CS229 - Fall 2025

---

## Project Overview

This project uses machine learning to detect mispriced options in real financial markets. We combine Kernel PCA for dimensionality reduction with Support Vector Machines and ensemble methods (Gradient Boosting, Random Forest) to classify options as underpriced, fairly priced, or overpriced relative to Black-Scholes theoretical values.

**Key Results:**
- Best Model: Gradient Boosting achieves **93.8% accuracy** and **78.1% F1-macro** 
- Dataset: 66,207 real AAPL option contracts (September-October 2025)
- Methods: Kernel PCA (RBF, Polynomial, Sigmoid, Linear) + SVMs, plus enhanced feature engineering with tree-based ensembles

---

## Getting the Data

**You don't need to download any data files** - the code automatically fetches real options data from Yahoo Finance.

### Automatic Data Collection

The `frontend/get_data.py` script fetches live AAPL options data:

```bash
cd frontend
python get_data.py
```

This creates `frontend/aapl_options.csv` with:
- Real-time option quotes (calls and puts)
- Computed Greeks (delta, gamma, theta, vega)
- Implied volatility calculations
- VIX market volatility index
- Black-Scholes theoretical prices
- Classification labels (underpriced/fair/overpriced)

**Data will be saved to:** `frontend/aapl_options.csv`

For other tickers (TSLA, MU), modify the script or run:
```bash
python get_data.py TSLA --output tsla_options.csv
python get_data.py MU --output mu_options.csv
```

**Note:** You need an internet connection to fetch data from Yahoo Finance. The script uses the `yfinance` library.

---

## Setup Instructions

### 1. Install Dependencies

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate cs229-quantml
```

**Or using pip:**
```bash
python -m venv cs229_env
source cs229_env/bin/activate  # On Windows: cs229_env\Scripts\activate
pip install numpy pandas scikit-learn scipy matplotlib seaborn yfinance py_vollib
```

### 2. Verify Installation
```bash
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

You should see scikit-learn 1.5.0 or higher.

---

## Reproducing Paper Results

### Main Experiments (Table 1 in Paper)

Run the primary training script to generate all model results from the paper:

```bash
python train_real_market_models.py
```

This will:
- Load real AAPL options data (66,207 contracts)
- Train all models: Gradient Boosting, Random Forest, Voting Ensemble, MLP, Logistic Regression, and KPCA+SVM variants
- Output accuracy, F1-macro, and cross-validation scores for each model
- Results match Table 1 in the paper

**Expected output:**
```
Gradient Boosting: 93.8% accuracy, 78.1% F1-macro
Random Forest: 93.0% accuracy, 72.5% F1-macro
MLP (Deep Learning): 93.1% accuracy, 73.7% F1-macro
...
```

### Generate Figures

**Figure 1 (Performance Comparison):**
```bash
python generate_visualizations.py
```
Creates `performance_comparison.png` - 4-panel Kernel PCA analysis showing accuracy vs F1, performance ranking, AUC scores, and hyperparameter analysis.

**Figure 2 (Confusion Matrix):**
```bash
python generate_confusion_matrix.py
```
Creates `confusion_matrix_gradient_boosting.png` - confusion matrix heatmap for the best model showing per-class performance.

### Validation Scripts

**Verify Data Quality:**
```bash
python validate_real_market_data.py
```
Checks data completeness, class distribution, temporal structure, and feature statistics.

**Calculate F1 Scores:**
```bash
python calculate_f1_scores.py
```
Computes detailed F1 scores and classification reports for all models.

---

## Project Structure

```
CS229_QuantML_Project/
│
├── Main Training Scripts
│   ├── train_real_market_models.py          # Main training (reproduces Table 1)
│   ├── validate_real_market_data.py         # Data validation
│   ├── generate_visualizations.py           # Figure 1 generation
│   ├── generate_confusion_matrix.py         # Figure 2 generation
│   └── calculate_f1_scores.py               # F1 score verification
│
├── Core ML Modules
│   ├── pca/
│   │   ├── kpca.py                          # Kernel PCA implementation
│   │   ├── utils.py                         # Variance plotting utilities
│   │   └── visualization.py                 # Latent space visualization
│   │
│   ├── svm/
│   │   ├── advanced_feature_engineering.py  # 12 engineered features
│   │   └── train_svm_on_embeddings.py       # SVM training on KPCA
│   │
│   └── frontend/
│       └── get_data.py                      # Yahoo Finance data fetcher
│
├── Configuration
│   ├── environment.yml                      # Conda environment
│   └── evaluation/configs/
│       └── experiment_config.yaml           # Hyperparameters
│
└── Documentation
    ├── README.md                            # This file
    ├── ENVIRONMENT_SETUP.md                 # Detailed setup guide
    └── finalpaper.tex                       # LaTeX paper source
```

---

## Team Contributions

### Jake Cheng - Data & Feature Engineering
- Extracted 66,207 real AAPL option contracts from Yahoo Finance
- Implemented Black-Scholes pricing model and Greeks computation
- Engineered base financial features (moneyness, implied volatility, delta, gamma, theta, vega)
- Integrated VIX data for market regime analysis
- Validated data quality and temporal structure

### Rupert Lu - Kernel PCA & Dimensionality Reduction  
- Implemented Kernel PCA with multiple kernel variants (Linear, RBF, Polynomial, Sigmoid)
- Conducted hyperparameter grid search (C, gamma) using 5-fold cross-validation
- Generated latent factor visualizations and variance spectra
- Analyzed component variance to interpret learned representations

### Juli Huang - Model Training & Comprehensive Evaluation
- Led project design and research direction
- Implemented enhanced feature engineering pipeline (Greek interactions, moneyness transformations)
- Trained ensemble methods and deep learning baselines
- Conducted TimeSeriesSplit cross-validation on real market data
- Created all performance visualizations and analyzed error patterns
- Generated confusion matrices and statistical analysis

---

## Key Features

### Advanced Dimensionality Reduction
- **Kernel PCA** with 4 kernel types capturing nonlinear patterns
- **5 principal components** balancing information retention with complexity
- **Variance analysis** showing explained variance by component

### Enhanced Feature Engineering
- **Base features (8):** moneyness, time to maturity, implied volatility, Greeks (delta, gamma, theta, vega), VIX
- **Engineered features (12):** Greek interactions, polynomial transformations, volatility ratios
- **Domain knowledge:** Financial theory-driven feature construction

### Robust Evaluation
- **TimeSeriesSplit cross-validation** preventing look-ahead bias
- **Temporal test split** at October 24, 2025 (80% train, 20% test)
- **Multiple metrics:** Accuracy, F1-macro, AUC-macro
- **Class imbalance handling:** 49.3% underpriced, 48.4% overpriced, 2.3% fair

### Model Diversity
- **Kernel Methods:** SVMs with RBF, Polynomial, Linear, Sigmoid kernels
- **Tree Ensembles:** Gradient Boosting, Random Forest, Voting Classifier
- **Deep Learning:** Multi-layer Perceptron baseline
- **Linear Baseline:** Logistic Regression

---

## Results Summary

### Model Performance (Real AAPL Data)

| Method | Test Accuracy | CV Accuracy | Test F1 | CV F1 |
|--------|--------------|-------------|---------|--------|
| **Gradient Boosting** | **93.8%** | 98.9% ± 0.5% | **78.1%** | 86.0% ± 4.1% |
| MLP (Deep Learning) | 93.1% | 98.6% ± 0.6% | 73.7% | 81.6% ± 5.8% |
| Random Forest | 93.0% | 99.0% ± 0.4% | 72.5% | 86.6% ± 5.3% |
| Voting Ensemble | 93.1% | 98.7% ± 0.5% | 72.4% | 83.7% ± 5.6% |
| Logistic Regression | 92.0% | 97.6% ± 0.7% | 69.0% | 79.4% ± 7.8% |
| Sigmoid KPCA + SVM | 72.3% | 75.4% ± 1.5% | 69.4% | 73.3% ± 2.0% |

### Key Findings

1. **Tree-based ensembles outperform kernel methods** - Gradient Boosting achieves 21.5% higher accuracy than best KPCA+SVM approach
2. **Enhanced features add value** - Domain-driven feature engineering (Greek interactions, moneyness²) improves F1 by 0.5% over base features
3. **Class imbalance is challenging** - Model excels at detecting clear mispricings (99% recall) but struggles with rare "fairly priced" class (18% recall)
4. **Deep learning competitive but not superior** - MLP achieves 93.1% accuracy but tree methods offer better interpretability

---

## What's Included in This Submission

### Source Code Files (15 files, ~88 KB)
-  All Python scripts to reproduce paper results
-  Core ML modules (Kernel PCA, feature engineering, SVM training)
-  Visualization generation scripts
-  Configuration files (YAML)
-  Documentation (README, environment setup)

### NOT Included (Per CS229 Guidelines)
-  Data files (too large, can be fetched via code)
-  Trained model files (.pkl files)
-  Generated outputs (CSVs, plots, results)
-  Python cache files (__pycache__)
-  Additional libraries (install via environment.yml)

**Total submission size: 31.9 KB (compressed) / 85.8 KB (uncompressed)**  
Well under the 5 MB limit!



---

## Citation

If you use this code or methodology, please cite:

```
@misc{huang2025options,
  title={Discovering Nonlinear Latent Drivers of Option Mispricing via Kernel PCA, 
         Support Vector Machines, and Enhanced Feature Engineering},
  author={Huang, Juli and Cheng, Jake and Lu, Rupert},
  year={2025},
  institution={Stanford University},
  note={CS229 Final Project}
}
```

---

## Questions or Issues?

Contact us at:
- Juli Huang: julih@stanford.edu
- Jake Cheng: jiajunc4@stanford.edu  
- Rupert Lu: rupertlu@stanford.edu


