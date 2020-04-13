
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer

#models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#validation libraries
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.display import display
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
#Data overview
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
(train_df.isnull().sum()/len(train_df)).sort_values(ascending=False)[:20]
train_df.shape
train_df.head(1)
test_df.shape
# ## Data Cleaning and Data transform
#fill the null value
train_df.fillna(0, inplace=True)
#Select the variavle needs for the model fitting.
numeric_cols = [x for x in train_df.columns if ('Area' in x) | ('SF' in x)] + ['SalePrice','LotFrontage','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']
for col in numeric_cols:
    train_df[col] = train_df[col].astype(float)
numeric_cols
#data transform
# data transform on highly skewed variable into a more normalized.
train_df['LogSalePrice'] = train_df['SalePrice'].map(lambda x : np.log(x))
train_df['LogSalePrice'] = np.log(train_df['SalePrice'])
#Select certain rang of variable to use.
train_df['above_200k'] = train_df['SalePrice'].map(lambda x : 1 if x > 200000 else 0)
train_df['above_200k'] = train_df['above_200k'].astype('category')
#combine all area into one variable.
train_df['LivArea_Total'] = train_df['GrLivArea'] + train_df['GarageArea'] + train_df['PoolArea']
train_df[['LivArea_Total','GrLivArea','GarageArea','PoolArea']].head()
## concatenating two different fields together in the same row
train_df['Lot_desc'] = train_df.apply(lambda val : val['MSZoning'] + val['LotShape'], axis=1)
train_df[['Lot_desc','MSZoning','LotShape']].head()
train_df['LotArea_norm'] = train_df['LotArea']
ss = StandardScaler()
mas = MaxAbsScaler()
qs = QuantileTransformer()
train_df['LotArea_norm'] = ss.fit_transform(train_df[['LotArea']])
train_df['LotArea_mas'] = mas.fit_transform(train_df[['LotArea']])
train_df['LotArea_qs'] = qs.fit_transform(train_df[['LotArea']])
#covert the catrgorical variable into numerical variable
train_df = pd.get_dummies(train_df)
## Model Training
## split train test data
features = [col for col in train_df.columns if 'Price' not in col]
target = 'LogSalePrice'
X_train, X_test, Y_train, Y_test = train_test_split(train_df[features],
                                                    train_df[target],
                                                    test_size = 0.3)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#Fititng the gradientboostingregressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
Y_train_pred = model.predict(X_train)
from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(Y_pred, Y_test)
test_mse
train_mse = mean_squared_error(Y_train_pred, Y_train)
train_mse
#Fitting by randomforest
clf=RandomForestRegressor()
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
rsm = mean_squared_error(Y_test, Y_pred)
from math import sqrt
sqrt(rsm)

#compare models by k-fold cross validation
from sklearn.model_selection import KFold
scores = []
best_gbm = GradientBoostingRegressor()
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(train_df):
    X_train = train_df.iloc[train_index][features]
    X_test = train_df.iloc[test_index][features]
    y_train = train_df.iloc[train_index][target]
    y_test = train_df.iloc[test_index][target]

    best_gbm.fit(X_train, y_train)

    scores.append(best_gbm.score(X_test, y_test))

scores
np.mean(scores)
plt.plot(range(len(scores)), scores)

from sklearn.model_selection import KFold
scores = []
best_forest = RandomForestRegressor()

cv = KFold(n_splits=10, random_state=42, shuffle=False)

for train_index, test_index in cv.split(train_df):

    X_train = train_df.iloc[train_index][features]
    X_test = train_df.iloc[test_index][features]
    y_train = train_df.iloc[train_index][target]
    y_test = train_df.iloc[test_index][target]

    best_forest.fit(X_train, y_train)

    scores.append(best_forest.score(X_test, y_test))
scores
np.mean(scores)
