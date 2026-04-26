# From-Scratch Regression and Data Analysis in Python

This repository contains a machine learning project centered on regression modeling and data analysis in Python.

The project includes:

- an implementation of ordinary least squares linear regression from scratch
- an extension to polynomial regression
- two applied workflows on real datasets with preprocessing, feature engineering, visualization, and error analysis

## Project structure

```text
.
|-- linear_regression.py             # OLS regressor solved with pseudo-inverse
|-- polynomial_fitting.py           # Polynomial regression via Vandermonde features
|-- house_price_prediction.py       # Housing-price preprocessing, feature analysis, and learning curve
|-- city_temperature_prediction.py  # Temperature exploration and polynomial model evaluation
|-- house_prices.csv                # Housing dataset
|-- city_temperature.csv            # Daily temperature dataset
|-- feature_plots/                  # Per-feature correlation visualizations for house prices
|-- learning_curve.png
|-- israel_temp_scatter.png
|-- israel_monthly_std.png
|-- country_monthly_avg_temp.png
|-- polynomial_test_errors_israel.png
|-- israel_model_generalization.png
```

## Overview

### 1. Linear regression

[`linear_regression.py`](./linear_regression.py) implements ordinary least squares using the Moore-Penrose pseudo-inverse.

The implementation includes:

- optional intercept handling
- `fit`, `predict`, and mean squared error `loss`
- a minimal NumPy-based implementation of the regression pipeline

### 2. Polynomial regression

[`polynomial_fitting.py`](./polynomial_fitting.py) extends the linear model with a Vandermonde-based polynomial transformation.

This part includes:

- univariate polynomial expansion up to degree `k`
- inheritance from the base linear regression implementation
- evaluation of test error across different polynomial degrees

### 3. House price prediction

[`house_price_prediction.py`](./house_price_prediction.py) applies linear regression to a tabular housing dataset.

The workflow includes:

- feature selection and data cleaning
- removal of invalid or incomplete samples
- feature engineering, including:
  - `house_age`
  - `was_renovated`
  - `years_since_renovation`
  - `living_to_lot_ratio`
  - `above_ground_ratio`
  - `basement_ratio`
  - `bath_bed_ratio`
- Pearson-correlation visualizations for each feature
- train/test splitting
- learning-curve analysis across different training-set sizes

### 4. City temperature modeling

[`city_temperature_prediction.py`](./city_temperature_prediction.py) applies polynomial regression to daily temperature data.

The workflow includes:

- data cleaning and day-of-year feature extraction
- exploratory analysis for Israel
- cross-country comparison of monthly average temperature and variability
- evaluation of polynomial degrees `k = 1..10`
- testing how a model fitted on Israel generalizes to other countries

## Output

The repository includes generated visualizations such as:

- feature correlation plots for the housing dataset
- `learning_curve.png`
- `israel_temp_scatter.png`
- `israel_monthly_std.png`
- `country_monthly_avg_temp.png`
- `polynomial_test_errors_israel.png`
- `israel_model_generalization.png`

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the housing-price workflow:

```bash
python house_price_prediction.py
```

Run the temperature workflow:

```bash
python city_temperature_prediction.py
```

## Dependencies

- NumPy
- pandas
- Matplotlib
