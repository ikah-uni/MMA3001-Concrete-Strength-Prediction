# Concrete Compressive Strength Prediction

## Project Overview

This project investigates the use of machine learning regression methods to predict the compressive strength of concrete based on its mixture composition and curing age.

The project was developed for MMA3001 and compares several regression methods to determine which provides the best balance between prediction accuracy and computational efficiency.

## Dataset

The project uses the UCI Concrete Compressive Strength Dataset, containing 1,030 observations with eight input variables and one target variable.

The input variables are cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate and curing age.

The target variable is concrete compressive strength, measured in MPa.

During preprocessing, 25 duplicate observations were removed, leaving 1,005 observations for analysis.

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

## Methods

Four regression approaches were evaluated:

- Mean prediction baseline
- Linear Regression
- Decision Tree Regression
- Support Vector Regression (SVR) with an RBF kernel

The dataset was divided into training (70%), validation (15%) and testing (15%) subsets.

Model parameters were investigated using validation data, followed by five-fold cross-validation on the combined development dataset.

Final model performance was evaluated using the independent test dataset.

## Results

| Model | Test MAE (MPa) | Test RMSE (MPa) | Test R² |
|---|---:|---:|---:|
| Baseline | 12.907 | 15.855 | -0.001 |
| Linear Regression | 8.938 | 11.147 | 0.505 |
| Decision Tree | 4.393 | 6.383 | 0.838 |
| SVR | 4.032 | 5.605 | 0.875 |

SVR achieved the lowest prediction error, with a test RMSE of 5.605 MPa and R² of 0.875.

Although SVR required more computational time than the other methods, its improved prediction accuracy justified the additional cost for this dataset.

## How to Run

1. Open the project notebook in Google Colab.
2. Run the notebook cells in order.
3. The notebook installs the required Python libraries and loads the dataset directly from GitHub.
4. Run the remaining cells to reproduce the preprocessing, model development, evaluation and visualisations.

The required libraries are listed in `requirements.txt`.

## Limitations

The model was developed using a relatively small dataset and may not generalise to concrete mixtures outside the range of the training data.

Prediction errors were also observed for some higher-strength concrete samples. Therefore, model predictions should not replace experimental testing for engineering applications without further validation.

## Acknowledgements

The dataset was obtained from the UCI Machine Learning Repository.

AI assistance was used during code development, debugging and documentation. The modelling methodology and results were reviewed and verified by the project author.
