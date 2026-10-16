# Financial Market Prediction

## Overview

This project explores the use of Machine Learning techniques to predict financial market signals using the **Jane Street Real-Time Market Data Forecasting** dataset.

Financial markets generate large amounts of noisy, high-dimensional, and non-stationary data. Extracting predictive information from these data is challenging because relationships observed in the past may weaken, disappear, or change over time.

The objective of this project is not simply to obtain the highest possible Kaggle score, but to study how different Machine Learning approaches behave when applied to real-world financial market data.

The main research question is:

> **Can Machine Learning models extract a predictive signal from noisy and non-stationary financial market data?**

---

## Dataset

The dataset comes from the Kaggle competition:

**Jane Street Real-Time Market Data Forecasting**

The observations correspond to anonymized financial instruments observed at different points in time.

Each observation contains:

- `date_id`: anonymized trading day;
- `time_id`: anonymized intraday timestamp;
- `symbol_id`: anonymized financial instrument;
- `feature_00` to `feature_78`: 79 anonymized market features;
- `responder_0` to `responder_8`: anonymized financial response variables;
- `weight`: importance associated with each observation.

The main prediction target is:

```text
responder_6
```

The Machine Learning problem can therefore be written as:

$$
X_t \longrightarrow \widehat{Y}_t
$$

where:

$$
X_t =
(\text{feature}_{00}, \ldots, \text{feature}_{78})
$$

and:

$$
Y_t = \text{responder}_6
$$

The problem is a **supervised regression task with a strong temporal component**.

---

## Main Challenges

Financial market prediction differs significantly from standard tabular Machine Learning problems.

### 1. High level of noise

Financial data usually contain much more noise than predictive signal.

Conceptually:

$$
\text{Observed Market Data}
=
\text{Predictive Signal}
+
\text{Noise}
$$

Even a small amount of predictive information may therefore be meaningful.

---

### 2. Non-stationarity

Relationships between variables may change over time.

A relationship that is useful during one market period may become weaker or disappear during another period.

Therefore:

$$
P(Y \mid X, t_1)
\neq
P(Y \mid X, t_2)
$$

This makes generalization to future market conditions particularly challenging.

---

### 3. Temporal dependence

Observations are ordered in time.

The model must only use information that would have been available at the moment of prediction.

Randomly mixing past and future observations during model validation could create **data leakage** and produce unrealistic performance estimates.

For this reason, the project will use chronological validation rather than a classical randomized train/test split.

---

### 4. High dimensionality

The dataset contains 79 market features and millions of observations.

The goal is to identify useful predictive information while limiting overfitting and computational complexity.

---

## Project Methodology

The project will follow a progressive approach, starting with simple models before introducing more complex methods.

### 1. Data Exploration

The first stage will focus on understanding the structure of the dataset.

The analysis will include:

- dataset dimensions;
- missing values;
- feature distributions;
- distribution of `responder_6`;
- correlations between variables;
- differences between financial instruments;
- evolution of variables through time;
- potential regime changes.

The purpose of this stage is to understand the statistical structure of the data before applying Machine Learning models.

---

### 2. Baseline Model

A simple benchmark will first be established.

Because the competition uses a zero-centered weighted $R^2$, a natural baseline is:

$$
\widehat{Y} = 0
$$

This model provides a reference against which all more advanced models can be compared.

A predictive model should ideally achieve:

$$
R^2 > 0
$$

---

### 3. Linear Models

The first statistical models will include approaches such as:

- Linear Regression;
- Ridge Regression;
- possibly Lasso or Elastic Net.

These models will provide a simple and interpretable benchmark.

They will also help evaluate whether the relationship between the market features and the target contains a meaningful linear component.

---

### 4. Tree-Based Machine Learning Models

The next stage will study nonlinear models such as:

- Random Forest;
- XGBoost;
- LightGBM.

Gradient Boosting models are particularly interesting because they can capture complex nonlinear interactions between features while remaining well suited to large tabular datasets.

The models will be compared according to their out-of-sample predictive performance.

---

### 5. Temporal Feature Engineering

The project will then introduce information from previous observations.

Possible features include:

- lagged variables;
- lagged responders when available;
- rolling means;
- rolling standard deviations;
- rolling volatility estimates;
- recent changes in features;
- statistics computed for each `symbol_id`.

For example:

$$
X_t,\ X_{t-1},\ X_{t-2},\ldots
$$

may contain more information than using $X_t$ alone.

This stage will investigate whether recent market history improves predictive performance.

---

### 6. Sequential Models

If computational resources allow it, recurrent neural networks such as **LSTM** or **GRU** will be investigated.

Instead of observing only one market state:

$$
X_t
$$

the model can observe a sequence:

$$
X_{t-k}, \ldots, X_{t-1}, X_t
$$

The objective will be to determine whether explicitly modeling temporal dependencies improves prediction compared with traditional tabular models.

A central comparison of the project will therefore be:

$$
\text{Gradient Boosting}
\quad \text{vs} \quad
\text{Sequential Neural Networks}
$$

---

## Temporal Validation

A major part of the project will focus on designing a realistic validation strategy.

A random train/test split will not be used.

Instead, the data will be divided chronologically:

```text
Past                                              Future

|---------------- TRAIN ----------------|--- VALIDATION ---|--- TEST ---|
```

This setup simulates the real-world situation in which a model is trained using historical information and evaluated on future observations.

A possible extension will be **walk-forward validation**, where the training and validation periods move progressively through time.

This will allow us to study whether model performance remains stable across different market periods.

---

## Evaluation Metric

The main evaluation metric is the weighted zero-mean $R^2$:

$$
R^2 =
1 -
\frac{
\sum_i w_i(y_i-\hat{y}_i)^2
}{
\sum_i w_i y_i^2
}
$$

where:

- $y_i$ is the true value of `responder_6`;
- $\hat{y}_i$ is the model prediction;
- $w_i$ is the observation weight.

A score of:

$$
R^2 = 0
$$

corresponds approximately to always predicting zero.

Therefore:

$$
R^2 > 0
$$

indicates that the model extracts some predictive information from the data.

---

## Model Comparison

The project will progressively compare several families of models:

| Model | Purpose |
|---|---|
| Zero Prediction | Baseline |
| Linear Regression | Simple linear benchmark |
| Ridge Regression | Regularized linear model |
| Random Forest | Nonlinear tree-based model |
| XGBoost / LightGBM | Gradient Boosting |
| LSTM / GRU | Temporal sequence modeling |

The objective is not necessarily to identify a universally superior algorithm, but rather to understand how each method behaves on financial market data.

---

## Additional Analysis

Beyond overall predictive performance, the project will investigate several questions.

### Stability Over Time

Does the model perform consistently across different periods?

$$
R^2_{t_1}
\stackrel{?}{\approx}
R^2_{t_2}
$$

---

### Performance Across Instruments

Are some financial instruments easier to predict than others?

---

### Feature Importance

Which anonymized features contain the strongest predictive information?

Possible methods include:

- XGBoost feature importance;
- permutation importance;
- SHAP values.

---

### Model Robustness

How quickly does model performance deteriorate when market conditions change?

This question is particularly important in financial applications because models are often trained in one market regime and deployed in another.

---

## Expected Outcome

The final objective is to build a complete Machine Learning pipeline for financial market prediction:

```text
Data Exploration
        ↓
Temporal Validation
        ↓
Baseline
        ↓
Machine Learning
        ↓
Feature Engineering
        ↓
Sequential Models
        ↓
Performance Analysis
```

The project aims to demonstrate both Machine Learning skills and an understanding of the specific challenges associated with financial data, including:

- temporal dependence;
- non-stationarity;
- noisy signals;
- overfitting;
- data leakage;
- model robustness;
- out-of-sample evaluation.

---

## Tools

The project will mainly use Python and the following libraries:

```text
Python
NumPy
Pandas / Polars
Matplotlib
Scikit-learn
XGBoost
LightGBM
PyTorch
```

Additional libraries may be introduced during the project.

---

## Project Structure

The repository will progressively follow a structure similar to:

```text
financial-market-prediction/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_gradient_boosting.ipynb
│   ├── 04_temporal_features.ipynb
│   └── 05_lstm.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── results/
│
├── requirements.txt
│
└── README.md
```

Large Kaggle datasets will not be stored directly in the Git repository.

---

## Long-Term Objective

This project is designed as an introduction to **Machine Learning for financial markets**.

Beyond obtaining predictive performance, the goal is to understand how statistical learning methods can be used to extract weak signals from noisy market data and how those models must be evaluated when observations are ordered through time.

The project therefore lies at the intersection of:

$$
\boxed{
\text{Machine Learning}
+
\text{Time Series}
+
\text{Statistics}
+
\text{Financial Markets}
}
$$

and provides a practical exploration of problems encountered in quantitative and systematic finance.
