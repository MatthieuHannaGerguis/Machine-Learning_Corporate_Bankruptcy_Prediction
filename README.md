# Machine Learning Project – Corporate Bankruptcy Prediction

Predicting corporate bankruptcy from financial ratios with imbalanced-learning methods.


## Academic Context
**Program:** A4 IF3
**Course:** Machine Learning
**School:** ESILV
**Year:** 2025–2026


## Authors
- Matthieu HANNA GERGUIS  
- Renaud DE L'EPINE  
- Ilian SEGOIN  

---

## Project Overview
This project addresses the problem of **corporate bankruptcy prediction** using machine learning techniques on **highly imbalanced financial data**.  
The objective is to build a robust and interpretable prediction pipeline while prioritizing the detection of bankrupt firms, which is a critical challenge in financial risk management.
**Keywords:** imbalanced classification, recall, SMOTE, ensembles, PCA, interpretability
- maximize recall on bankrupt firms
- keep pipeline interpretable
- ensure reproducible evaluation

---

## Dataset
The dataset used in this project is the Company Bankruptcy Prediction dataset, built from financial statements collected by the Taiwan Economic Journal (TEJ).

📊 Source (Kaggle)
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

Dataset Overview:

- 6,819 companies  
- 95 financial features  
- Binary target variable: `Bankrupt?`  
- Strong class imbalance (~3% bankrupt firms)

Target Variable
- Bankrupt? = 1 → Company went bankrupt
- Bankrupt? = 0 → Company did not go bankrupt
The dataset exhibits a severe class imbalance, which makes standard accuracy metrics unreliable and motivates the use of recall-focused evaluation.

Class Imbalance Considerations
- Bankruptcy events are rare (~3%)
- Models trained without imbalance handling tend to favor the majority class
- Specialized techniques are required to improve detection of bankrupt firms

Preprocessing Notes
- Low-variance features are removed
- Highly correlated features are filtered
- Scaling is applied where required (PCA, distance-based models, neural networks)
- Resampling methods are applied only on training data to avoid data leakage

Dataset Files
- data.csv
Original dataset as provided on Kaggle
- data_clean.csv
Cleaned version after preprocessing and feature filtering, used for modeling

Dataset Credit

Dataset originally compiled from Taiwan Economic Journal (TEJ) financial data and distributed on Kaggle by fedesoriano.

---

## Repository Structure
This repository contains the following key resources:

- `Notebook_Project.ipynb`  
  Main notebook containing data exploration, preprocessing, feature selection, modeling, and evaluation.

- `data_clean.csv`  
  Cleaned dataset after preprocessing and filtering steps.

- `data.csv`  
  Original dataset used as input.

- `dnn_tuning/`  
  Scripts and configurations related to deep neural network tuning.

- `Final_Report.pdf`  
  Complete academic report detailing methodology, experiments, and analysis.

- `Final_Video.mp4`  
  Project presentation video.

- `Report_Pre_Project_*.pdf`  
  Intermediate reports produced during the project.

---

## Methodology (High-Level)

1. Data Exploration
- Initial inspection of feature distributions
- Identification of class imbalance in the target variable
- Basic statistical analysis to detect anomalies and constant features

This step aims to understand the structure of the data before any transformation or modeling.

2. Data Cleaning and Preprocessing
- Removal of zero-variance and near-zero-variance features
- Filtering of highly correlated variables to reduce redundancy
- Verification of numerical consistency across all features

These operations help stabilize model training and reduce noise.


3. Feature Selection and Dimensionality Reduction
- Evaluation of Principal Component Analysis (PCA) as a dimensionality reduction technique
- Analysis of the trade-off between explained variance and interpretability
- Comparison of models trained with and without PCA

PCA is primarily used to mitigate multicollinearity and improve model robustness.

---

## Results & Discussion
The repository and the report provide a detailed comparison of the tested approaches and their performance.  
Rather than highlighting a single conclusion here, we **invite the reader to consult the notebook and the final report** for a complete and transparent discussion of the results.

Within the scope of this academic project, the analysis was limited to the methods required by the course. Additional approaches (e.g. cost-sensitive learning, temporal modeling, or alternative architectures) could be explored in future work but were beyond the requested framework.

---

## Notes
This project was developed for educational purposes and aims to demonstrate rigorous methodology, clear structure, and reproducibility rather than production deployment.
