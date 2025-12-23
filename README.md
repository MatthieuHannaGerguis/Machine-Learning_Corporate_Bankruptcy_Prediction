# Machine Learning Project – Corporate Bankruptcy Prediction

Predicting corporate bankruptcy from financial ratios with imbalanced-learning methods.

	•	Academic Context
	•	Authors
	•	Project Overview
	•	Dataset
	•	Repository Structure
	•	Methodology
	•	Results
	•	Reproducibility
	•	License / Notes

## Academic Context
**Programme:** A4 IF3  
**Course:** Machine Learning  
**Institution:** ESILV  
**Academic Year:** 2025–2026  

## Authors
- Matthieu HANNA GERGUIS  
- Renaud DE L'EPINE  
- Ilian SEGOIN  

---

## Project Overview
This project addresses the problem of **corporate bankruptcy prediction** using machine learning techniques on **highly imbalanced financial data**.  
The objective is to build a robust and interpretable prediction pipeline while prioritizing the detection of bankrupt firms, which is a critical challenge in financial risk management.

---

## Dataset
The data used in this project come from the **Company Bankruptcy Prediction** dataset, derived from financial statements collected by the Taiwan Economic Journal (TEJ).

📊 **Source (Kaggle):**  
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

- 6,819 companies  
- 95 financial features  
- Binary target variable: `Bankrupt?`  
- Strong class imbalance (~3% bankrupt firms)

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
The project follows a complete machine learning pipeline:
- Data exploration and cleaning (zero analysis, variance and correlation filtering)
- Feature selection and dimensionality reduction (PCA)
- Handling class imbalance (undersampling, random oversampling, SMOTE)
- Comparison of baseline models, ensemble methods, and deep neural networks
- Evaluation using recall-focused metrics adapted to imbalanced classification

---

## Results & Discussion
The repository and the report provide a detailed comparison of the tested approaches and their performance.  
Rather than highlighting a single conclusion here, we **invite the reader to consult the notebook and the final report** for a complete and transparent discussion of the results.

Within the scope of this academic project, the analysis was limited to the methods required by the course. Additional approaches (e.g. cost-sensitive learning, temporal modeling, or alternative architectures) could be explored in future work but were beyond the requested framework.

---

## Notes
This project was developed for educational purposes and aims to demonstrate rigorous methodology, clear structure, and reproducibility rather than production deployment.
