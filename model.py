# Importing necessary libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.api as sm
import pickle
from warnings import filterwarnings
filterwarnings('ignore')

# File path for the dataset
data = "data/Hitters.csv"

# Reading the dataset
df = pd.read_csv(data)

# Creating new features based on the career variables by calculating averages over years
df["OrtCAtBat"] = df["CAtBat"] / df["Years"]
df["OrtCHits"] = df["CHits"] / df["Years"]
df["OrtCHmRun"] = df["CHmRun"] / df["Years"]
df["OrtCruns"] = df["CRuns"] / df["Years"]
df["OrtCRBI"] = df["CRBI"] / df["Years"]
df["OrtCWalks"] = cwalks = df["CWalks"] / df["Years"]

# Dropping variables that do not contribute to the model based on trial and correlation results
df = df.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists','Errors',"PutOuts",'League','NewLeague', 'Division'], axis=1)

# Filling in missing values using the K-Nearest Neighbors (KNN) imputation method
imputer = KNNImputer(n_neighbors = 4)
df_filled = imputer.fit_transform(df)
df = pd.DataFrame(df_filled,columns = df.columns)

# Eliminating outliers in the Salary column using the IQR (Interquartile Range) method
Q1 = df.Salary.quantile(0.25)
Q3 = df.Salary.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Salary"] > upper,"Salary"] = upper

# Detecting and removing outliers using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
threshold = np.sort(df_scores)[7]
outlier = df_scores > threshold
df = df[outlier]

# Defining target variable (y) and feature set (X)
y = df["Salary"]
X = df.drop("Salary",axis=1)

cols = X.columns

# Backward Elimination for feature selection
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

# Selected features after Backward Elimination
selected_features_BE = cols
X = df[selected_features_BE]
X = df[["CRuns","OrtCWalks","CWalks"]]

# Scaling the features using Standard Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

# Best parameters for the Gradient Boosting Model
gbm_cv_model_best_params_ ={'learning_rate': 0.01,'loss': 'absolute_error','max_depth': 5,'n_estimators': 500,'subsample': 0.5}

# Training the tuned Gradient Boosting model
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model_best_params_).fit(X_train,y_train)

# Making predictions on the test set
y_pred = gbm_tuned.predict(X_test)

# Calculating the Root Mean Squared Error (RMSE) for model evaluation
gbm_final = np.sqrt(mean_squared_error(y_test, y_pred))
print(gbm_final)

# Saving the trained model to a file using pickle
pickle.dump(gbm_tuned, open('regression_model.pkl','wb'))

print("Model has been created.")
