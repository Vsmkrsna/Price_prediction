# Indian House Price Prediction

A supervised machine-learning project that explores and predicts residential property prices in India using property characteristics such as location, size, number of bedrooms, floor information, amenities, and nearby facilities.

The project is implemented in a Jupyter Notebook using Python and scikit-learn. It covers data exploration, feature engineering, data cleaning checks, preprocessing, regression model training, evaluation, and model interpretation.

## Project contents

- `indian house prices prediction(3).ipynb` — the main notebook containing the analysis and modelling workflow.
- `india_housing_prices.csv` — the dataset required to run the notebook. 

## Workflow

### 1. Data loading and overview
- Load the housing dataset using pandas.
- Remove duplicate records.
- Examine numerical and categorical variables using descriptive statistics.

### 2. Feature engineering and exploratory data analysis
- Expand the `Amenities` column into binary indicator features.
- Explore correlations between numerical variables.
- Plot relationships between property characteristics and the target, `Price_in_Lakhs`.
- Examine the distributions and skewness of price-related and numerical features.

### 3. Feature reduction and data checks
- Remove `Year_Built` because it is perfectly negatively correlated with `Age_of_Property` in the dataset.
- Identify records where `Floor_No` is greater than `Total_Floors`.
- Inspect missing values and potential outliers using the interquartile range (IQR) rule.

### 4. Preprocessing
- Separate the target (`Price_in_Lakhs`) from the predictor variables.
- Split the data into training and test sets.
- Standardise numerical features using `StandardScaler`.
- One-hot encode categorical features using `OneHotEncoder`.
- Fit the preprocessing transformations on the training data and apply them to the test data.

### 5. Regression models
The notebook trains and compares:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regression
- Random Forest Regression

Ridge and Lasso regularisation strengths are explored over a range of alpha values.

### 6. Model evaluation and interpretation
Models are compared using:
- **R² score** — measures the proportion of target variance explained by the model.
- **Mean Absolute Error (MAE)** — average absolute prediction error.
- **Root Mean Squared Error (RMSE)** — gives greater weight to larger prediction errors.
- **Execution time** — records the time taken for model fitting and prediction in the notebook.

The notebook also includes:
- Actual-versus-predicted price plots.
- Residual analysis.
- Random Forest feature-importance analysis.
- Regression-equation extraction for Linear, Ridge, and Lasso Regression.
- Numerical verification of the extracted regression equations.
- Visualisations of a Decision Tree and the first tree in the Random Forest.

## Reported result in the notebook

The notebook's final model-selection section reports Random Forest Regression as the best-performing model:

| Metric | Reported value |
|---|---:|
| R² score | 0.9956 |
| MAE | 7.21 lakhs |
| RMSE | 9.38 lakhs |



## Requirements

The notebook uses the following Python libraries:

- Python 3
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- Jupyter Notebook or Jupyter support in Visual Studio Code

Install the main dependencies with:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## How to run

1. Clone or download this project.
2. Obtain the dataset file, `india_housing_prices.csv`, from the appropriate dataset source.
3. Place the CSV file in the project directory.
4. Open the notebook in Jupyter Notebook or Visual Studio Code.
5. In the data-loading cell, change the current machine-specific file path:

   ```python
   df = pd.read_csv(r'C:\Users\vsmk3\ds_first_git\india_housing_prices.csv')
   ```

   to a path appropriate for your setup. If the CSV is in the same directory as the notebook, use:

   ```python
   df = pd.read_csv("india_housing_prices.csv")
   ```

6. Run all cells from top to bottom so that preprocessing, model training, evaluation, and plots are generated in the correct order.

## Notes and limitations 

- **Target leakage identified:** The analysis indicates that `Price_per_SqFt` is derived from information involving the target `Price_in_Lakhs`. It should therefore be excluded from the predictor variables. Retrain and reevaluate the models without this feature; until then, the reported performance metrics are likely optimistic and should be treated with caution.
- **Outlier handling:** The notebook currently investigates outliers but does not execute that removal step, as outliers should not be removed solely because they lie outside the typical statistical range. Their validity and possible influence on the analysis should first be analysed.

## Future improvements

- Add a reusable prediction function that accepts details of a new property.
- Use cross-validation or a separate validation set for hyperparameter selection, keeping the test set for final evaluation.
- Reassess feature importance and model performance after removing any target-derived predictors.
- Explore separate models for different cities or classify properties into price categories.

