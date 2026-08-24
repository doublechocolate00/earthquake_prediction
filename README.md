# Earthquake Prediction Using Machine Learning

A machine learning project for predicting **time to failure** from acoustic earthquake sensor data using the LANL Earthquake Prediction dataset.

## Overview

Earthquake prediction is a challenging machine learning problem because earthquake signals contain complex patterns and large amounts of noisy sensor data.

This project uses acoustic data from the **LANL Earthquake Prediction** dataset to predict the remaining **time to failure** before an earthquake event. The notebook follows a typical machine learning workflow:

1. Load and explore the dataset
2. Split the data into training and testing sets
3. Extract statistical features from acoustic-data segments
4. Standardize the extracted features
5. Train linear and non-linear regression models
6. Evaluate model performance using regression metrics
7. Compare the models and discuss their strengths and limitations

## Dataset

The project uses the `train.csv` file from the LANL Earthquake Prediction dataset.

The dataset contains two main columns:

* `acoustic_data` — acoustic sensor measurements
* `time_to_failure` — the target value representing the time remaining until the next failure/earthquake event

The notebook loads the data using:

```python
train = pd.read_csv(
    '../input/LANL-Earthquake-Prediction/train.csv',
    dtype={
        'acoustic_data': np.int16,
        'time_to_failure': np.float64
    }
)
```

The dataset contains approximately **629 million observations**, making direct modelling computationally expensive.

## Train-Test Split

The data is divided chronologically into:

* **70% training data**
* **30% testing data**

The split is performed without shuffling:

```python
data_train, data_test = train_test_split(
    train,
    test_size=0.3,
    shuffle=False
)
```

Keeping the original ordering is important because the observations represent a time-dependent acoustic signal.

## Exploratory Data Analysis

The notebook explores the relationship between acoustic measurements and time to failure.

A sampled version of the data is visualized to make the extremely large dataset easier to inspect. The acoustic signal and time-to-failure values are plotted together to investigate how changes in the acoustic signal relate to earthquake events.

The analysis suggests that peaks and changes in the acoustic signal are associated with changes in the time-to-failure signal.

## Feature Engineering

Because the raw dataset is extremely large, the acoustic signal is divided into segments of **150,000 observations**.

For every segment, statistical features are extracted from `acoustic_data`.

### Extracted features

The following features are calculated:

* Mean
* Standard deviation
* Maximum
* Minimum
* Kurtosis
* Skewness
* Rolling mean statistics
* Rolling standard-deviation statistics

A rolling window of **100 observations** is also used to calculate additional local statistical information.

The final feature representation allows the machine learning models to work with a much smaller dataset instead of the hundreds of millions of individual acoustic measurements.

## Feature Scaling

The extracted features are standardized using `StandardScaler` from scikit-learn.

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```

The same preprocessing approach is then applied to the test features.

## Machine Learning Models

Two regression approaches are investigated.

### 1. Linear Regression

`LinearRegression` is used as the linear baseline model.

The model assumes that the relationship between the extracted acoustic features and time to failure can be represented approximately as a linear combination of the features.

The model is evaluated using:

* R² score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

### 2. Random Forest Regression

`RandomForestRegressor` is used as the non-linear model.

Random Forest can model more complex relationships between acoustic features and time to failure and does not require the relationship between the input features and target to be linear.

The Random Forest model is evaluated using the same regression metrics as the Linear Regression model.

## Evaluation Metrics

The notebook uses three main metrics:

### R² Score

Measures how much of the variation in the target variable is explained by the model. Higher values generally indicate better predictive performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual time-to-failure values.

Lower MAE indicates better performance.

### Mean Squared Error (MSE)

Measures the average squared prediction error. Because the errors are squared, larger errors have a greater influence on the metric.

Lower MSE indicates better performance.

## Results

The notebook compares the Linear Regression and Random Forest Regression models on training and testing data.

The results show that the **non-linear Random Forest model performs substantially better than the Linear Regression model** for this application.

The notebook's conclusion notes that the Linear Regression model has a relatively low score and considerably higher MSE, while the Random Forest model provides better overall performance.

This indicates that the relationship between the extracted acoustic features and time to failure is likely too complex to be represented effectively using a simple linear model.

## Discussion

A major challenge in this project is the size of the original dataset. Segmenting the acoustic measurements into groups of 150,000 observations makes feature extraction and model training more practical.

The statistical feature extraction also reduces the raw acoustic signal to a smaller set of meaningful numerical characteristics.

The main strengths of the approach are:

* Handles a very large dataset through segmentation
* Uses statistical features to summarize acoustic signals
* Compares both linear and non-linear approaches
* Uses established scikit-learn algorithms
* Evaluates models using multiple regression metrics

### Limitations

There are also several limitations:

* Only a relatively small set of statistical features is extracted from each segment.
* The approach does not capture all temporal patterns present in the raw acoustic signal.
* Random Forest performance can depend strongly on its hyperparameters.
* The notebook does not perform extensive hyperparameter optimization.
* More advanced time-series or deep-learning approaches could potentially capture more complex signal patterns.

## Possible Future Improvements

Future versions of the project could investigate:

* More acoustic and time-series features
* Frequency-domain features using FFT
* More systematic hyperparameter tuning
* Cross-validation designed for time-series data
* Gradient boosting models
* Support Vector Regression
* Neural networks
* Recurrent or convolutional neural networks for directly modelling acoustic sequences

## Project Structure

```text
.
├── earthquake_prediction.ipynb
└── README.md
```

The notebook contains the complete data exploration, feature engineering, model training, evaluation, discussion, and submission workflow.

## Requirements

The notebook uses Python and common machine learning/data-science libraries including:

* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Running the Project

The notebook was designed for a Kaggle environment and expects the LANL dataset to be available under:

```text
../input/LANL-Earthquake-Prediction/
```

After downloading the dataset, open:

```text
earthquake_prediction.ipynb
```

and run the notebook cells sequentially.

## References

The notebook references several Kaggle notebooks used during development and exploration:

* LANL Earthquake EDA and Prediction
* Basic Feature Benchmark
* ML23 Week 8
* LANL Earthquake Prediction

## Disclaimer

This project is an educational machine learning exercise. The model predicts **time to failure from acoustic sensor data** and should not be interpreted as a reliable real-world earthquake prediction system.
