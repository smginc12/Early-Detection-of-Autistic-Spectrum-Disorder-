# Early Detection of Autism Spectrum Disorder Across Age Groups Using Machine Learning

This project uses machine learning to support early screening for Autism Spectrum Disorder (ASD) across age groups, with a focus on toddlers (2 to 4 years) and children (5 to 18 years). We train and compare several classifiers on public ASD screening data, find the screening questions that matter most, and evaluate how well each model separates ASD and non-ASD cases.

Group project for **MSBD 540**, School of Applied Computational Sciences, Meharry Medical College.

> **Note:** This is an academic project built on screening questionnaire data. It is not a diagnostic tool and should not be used for clinical decisions.

---

## Table of Contents

- [Background](#background)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Background

ASD affects communication, social interaction, and behavior. Symptoms often appear between 12 and 18 months of age, and early intervention can greatly improve outcomes. There is no standard medical test for ASD, so diagnosis depends on observation and behavioral assessments. Machine learning can help flag higher-risk individuals faster using short screening questionnaires.

---

## Data

We used the **Autistic Spectrum Disorder Screening Data** from the UCI Machine Learning Repository and Kaggle. The data was collected through the ASDTests mobile app using the **Q-CHAT-10** (toddlers) and **AQ-10** (children, adolescents, adults) screening tools. Each person is scored from 0 to 10, and a score of 6 or higher suggests ASD.

| Dataset | Instances |
|---------|-----------|
| Toddlers | 1,054 |
| Children | 292 |
| Adolescents | 104 |
| Adults | 704 |

- **Type:** Multivariate
- **Attributes:** 21 (categorical and integer)
- **Task:** Binary classification (ASD vs. non-ASD)
- **Features:** 10 screening questions (A1 to A10), age, gender, ethnicity, country, born with jaundice, family member with PDD, used screening app before

Data sources:
- [UCI: ASD Screening Data for Children](https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children)
- [UCI: ASD Screening Data for Adolescents](https://archive.ics.uci.edu/dataset/420/autistic+spectrum+disorder+screening+data+for+adolescent)
- [UCI: Autism Screening Adult](https://archive.ics.uci.edu/dataset/426/autism+screening+adult)
- [Kaggle: Autism Screening for Toddlers](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers)

---

## Methods

### Data Preparation
1. Loaded `.arff` files with `scipy.io.arff` and converted them to pandas DataFrames
2. Renamed columns for readability (e.g., `Q1_Score` to `Q10_Score`, `born_with_jaundice`)
3. Filled missing ethnicity values using a country-based mapping
4. Imputed missing ages with a Random Forest model
5. Binary encoded categorical fields (gender, jaundice, family history, prior app use)
6. Standardized continuous features
7. Dropped `screening_score` because it is the sum of Q1 to Q10 and almost perfectly correlated with the label

### Exploratory Data Analysis
- ASD distribution by age, gender, country, and ethnicity
- Correlation heatmaps between features and the ASD label

### Modeling
- **Train/test split:** 70:30
- **Class imbalance:** Random Over Sampling
- **Models:** Logistic Regression (baseline), Random Forest, Gradient Boosting, Gaussian Naive Bayes, Decision Tree, XGBoost, LightGBM
- **Metrics:** Accuracy, Precision, Recall, F1 Score, AUC-ROC, Confusion Matrix
- **Feature importance:** from Random Forest and Gradient Boosting

---

## Results

### Children

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Random Forest | 95.89% | 92.59% | 96.15% | 94.34% | 0.96 |
| **Gradient Boosting** | **98.63%** | **96.30%** | **100%** | **98.11%** | **0.99** |
| Naive Bayes | 86.30% | 83.33% | 76.92% | 80.00% | 0.84 |

### Adults

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Random Forest | 93.75% | 93.02% | 83.33% | 87.91% | 0.90 |
| Gradient Boosting | 95.45% | 93.48% | 89.58% | 91.49% | 0.95 |
| **Naive Bayes** | **96.02%** | 91.84% | **93.75%** | **92.78%** | 0.84 |

---

## Key Findings

- **Gradient Boosting** was the best model for children, catching every ASD case in the test set (100% recall).
- For adults, **Gradient Boosting and Naive Bayes** both performed well. Gradient Boosting had the highest AUC and Naive Bayes had the highest recall.
- **Q4_Score** (shared interest / joint attention) was the most important feature in both Random Forest (~0.4) and Gradient Boosting (~0.5).
- **Q10_Score** ranked second, followed by Q3, Q7, and Q1.
- Demographic features (age, gender, ethnicity) were much weaker predictors than the screening questions.

---

## Repository Structure

```
Early-Detection-of-Autistic-Spectrum-Disorder/
├── README.md
├── data/                # raw and cleaned datasets
├── notebooks/           # EDA and modeling notebooks
├── figures/             # plots, confusion matrices, ROC curves
├── report/              # final IEEE-format paper
└── requirements.txt
```

---

## Getting Started

### Requirements
- Python 3.9+
- pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, imbalanced-learn, matplotlib, seaborn

### Setup

```bash
git clone https://github.com/YOUR-USERNAME/Early-Detection-of-Autistic-Spectrum-Disorder.git
cd Early-Detection-of-Autistic-Spectrum-Disorder
pip install -r requirements.txt
```

Then open the notebooks in the `notebooks/` folder and run them in order.

---

## Future Work

- Complete the toddler and adolescent analyses
- Add KNN, SVM, and neural network models once more data is available
- Build a stacking ensemble that combines the strongest classifiers
- Test feature scaling methods for each age group
- Combine behavioral screening data with genetic or environmental data

---

## Authors

- Malini Senthilkumar
- Mikaela Nixon
- Sade M. Graves

School of Applied Computational Sciences, Meharry Medical College, Nashville, TN

---

## Acknowledgments

We thank Dr. Gupta for the opportunity and support on this project.

---

## References

1. Das, A., et al. (2024). Early Autism Spectrum Disorder Screening in Toddlers: A Comprehensive Stacked Machine Learning Approach.
2. Farooq, M. S., Tehseen, R., Sabir, M., & Atal, Z. (2023). Detection of autism spectrum disorder (ASD) in children and adults using machine learning.
3. Kamel, S., & Al-harbi, R. (2021). Newly proposed technique for autism spectrum disorder-based machine learning.
4. Vakadkar, K., Purkayastha, D., & Krishnan, D. (2021). Detection of autism spectrum disorder in children using machine learning techniques.
5. Khudhur, D. D., & Khudhur, S. D. (2023). The classification of autism spectrum disorder by machine learning methods on multiple datasets for four age groups.

See the full reference list in the report.
