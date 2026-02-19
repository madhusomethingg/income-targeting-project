### Income Targeting Project

This project builds a machine learning system to predict whether a person earns more than $50,000 using U.S. Census data.

The goal is to help a retail client identify high-income individuals and apply more effective marketing strategies.

The dataset used is the 1994–95 U.S. Census Bureau Current Population Survey.

## Introduction

This project has two main parts:

1. Income Classification Model

An XGBoost model that predicts whether someone earns more than $50K.

- Census sample weights are used so results reflect the real population

- Because the dataset is highly imbalanced (about 15:1), we evaluate using PR-AUC and F1 instead of accuracy

- The classification threshold is chosen using profit simulation instead of the default 0.5

2. Segmentation Model

After predicting who likely earns >$50K, we group people into segments using K-Means clustering.

We build:

- Global segmentation (6 clusters) — groups across the full population

- Premium segmentation (3 clusters) — groups within the predicted high-income population

- This allows the client to:

    - Decide who to contact (classification)

    - Decide how to market to them (segmentation)

## Project Structure

income-targeting-project/
│
├── data/
│   └── raw/
│       ├── census-bureau.data
│       └── census-bureau.columns
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── segmentation/
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_diagnostics.ipynb
│   └── 03_segmentation_analysis.ipynb
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── config.yaml
├── requirements.txt
└── README.md


## How to Run the Project

Make sure you are in the project root directory.

- Step 1 — Train models

python -m src.models.train

This trains Logistic Regression, Random Forest, and XGBoost.

- Step 2 — Evaluate models

python -m src.evaluation.metrics

This calculates weighted ROC-AUC, PR-AUC, F1, and Brier score.

- Step 3 — Run profit simulation

python -m src.models.profit_simulation

This finds the best classification threshold based on expected profit.

- Step 4 — Run segmentation

python -m src.segmentation.cluster

This builds both global and premium customer segments.

## Notebooks

| Notebook                    | Purpose                                        |
| --------------------------- | ---------------------------------------------- |
| exploratory_analysis.ipynb  | Data exploration and preprocessing decisions   |
| model_diagnostics.ipynb     | Model comparison, SHAP, threshold selection    |
| segmentation_analysis.ipynb | Segment profiles and marketing recommendations |

## Key Design Decisions

- Census weights are used because the dataset is a stratified sample

- Accuracy is not used as the main metric due to class imbalance

- Threshold is selected using profit simulation

- Log transformation is applied to highly skewed features

- A migration flag is created because missingness was informative

- TruncatedSVD is used before clustering due to high-dimensional one-hot encoding

- Two-stage modeling separates targeting and messaging decisions

## Results Summary

**Best Model: XGBoost**

- ROC-AUC: 0.9475

- PR-AUC: 0.6488

- F1 Score: 0.5464

- Brier Score: 0.0352

- Optimal Threshold (validation profit): 0.1922 (~0.19)

- Population Contacted at Optimal Threshold: ~9.3%

- Profit Improvement vs Default 0.5 Threshold: +31.6%

## Configuration

All hyperparameters, file paths, and business assumptions are stored in:

config.yaml

Nothing important is hardcoded.

## Final Note

This project reflects how I would approach a real business problem end-to-end.

Instead of just training a model and reporting accuracy, I focused on understanding the data first, especially the strong class imbalance and the presence of census sample weights. All evaluation metrics were weighted so that results reflect the actual U.S. population rather than just the raw sample.

I also avoided using the default 0.5 classification threshold. Instead, I selected a threshold based on estimated business profit, which makes the solution more practical and aligned with how companies actually make decisions.

Finally, I extended the project beyond binary prediction by adding a segmentation layer. This allows the client not only to identify who to contact, but also how to tailor messaging for different groups within the population.
