# Hitters Salary Prediction
This repository contains a project aimed at predicting baseball players' salaries based on various performance metrics using different machine learning techniques. The data used in this project is the Hitters dataset, which includes various statistics for baseball players such as their batting average, number of home runs, and years of experience.

Table of Contents
- Project Overview
- Dataset Information
- Project Structure
- Installation
- Usage
- Models Used
- Results
- Contributing
- License

## Project Overview
In this project, we explore the Hitters dataset to predict the salary of baseball players. The primary goal is to build a machine learning model that accurately predicts the salary based on a variety of input features such as players' performance statistics and other factors.

The project includes:

- Data preprocessing (handling missing data, feature encoding, etc.)
- Exploratory Data Analysis (EDA)
- Training different machine learning models for salary prediction
- Model evaluation and comparison

## Dataset Information
The Hitters dataset is a well-known dataset in the field of machine learning and statistics, used for regression analysis. It contains baseball player statistics from the 1986 and 1987 seasons, including player-specific information, their performance metrics, and their salaries.

- Number of observations: 322
- Number of features: 20 (excluding the target variable 'Salary')
- Target variable: Salary (annual salary of the players in 1987)

Features:
- AtBat: Number of times at bat in 1986
- Hits: Number of hits in 1986
- HmRun: Number of home runs in 1986
- Runs: Number of runs in 1986
- RBI: Number of runs batted in 1986
- Walks: Number of walks in 1986
- Years: Number of years in the major leagues
- CAtBat: Number of times at bat during a player's career
- CHits: Number of hits during a player's career
- CHmRun: Number of home runs during a player's career
- CRuns: Number of runs during a player's career
- CRBI: Number of runs batted in during a player's career
- CWalks: Number of walks during a player's career
- League: A factor with levels A and N indicating player's league at the end of 1986
- Division: A factor with levels E and W indicating player's division at the end of 1986
- PutOuts: Number of put outs in 1986
- Assists: Number of assists in 1986
- Errors: Number of errors in 1986
- Salary: 1987 annual salary on opening day in thousands of dollars (target variable)
- NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

More details about the dataset can be found [here](https://islp.readthedocs.io/en/latest/datasets/Hitters.html).

## Project Structure
The project is structured as follows:
```
├── data/               # Contains dataset files
├── notebooks/          # Jupyter notebooks for data analysis and modeling
├── models/             # Trained models
├── src/                # Python scripts for data processing and model building
├── app.py              # main python file to run project
├── README.md           # Project overview and instructions
└── requirements.txt    # Required Python packages
```

## Installation
To run this project locally, follow these steps:

Clone this repository:
```
git clone https://github.com/ibrahimYldzz/Hitters-Salary-Prediction.git
```
Navigate to the project directory:
```
cd Hitters-Salary-Prediction
```
Install the required Python packages:
```
pip install -r requirements.txt
```

## Usage
To use this project for salary prediction, you can follow these steps:

1. Data Preprocessing:
- Run the data preprocessing script located in the src/ folder to handle missing values, feature scaling, and encoding.

2. Training the Model:
- Use the provided Jupyter notebooks in the notebooks/ folder to train different machine learning models such as Linear Regression, Random Forest, and XGBoost.

3. Evaluating the Model:
- Evaluate the models using metrics like Mean Squared Error (MSE) and R-squared to compare their performance.

4. Prediction:
- Use the trained models to predict the salary of new baseball players based on their statistics.

## Models Used
The following machine learning models are trained and evaluated in this project:

- Linear Regression: A simple regression model that assumes a linear relationship between input features and the target variable.
- Random Forest: An ensemble method that builds multiple decision trees and averages their predictions to improve accuracy and prevent overfitting.
- XGBoost: An advanced ensemble model based on gradient boosting, known for its performance in regression tasks.

## Results
The model performance is evaluated based on the following metrics:

Mean Squared Error (MSE)
R-squared (R²)
Results for different models are recorded, and the best-performing model is selected based on the lowest MSE and highest R².

## Contributing
Contributions are welcome! If you'd like to contribute to this project, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.

