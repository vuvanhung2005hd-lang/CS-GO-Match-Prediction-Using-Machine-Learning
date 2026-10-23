import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data = pd.read_csv("csgo.csv")
data = data.drop(columns=['team_a_rounds', 'team_b_rounds', 'date', 'wait_time_s', 'result'])

# print(data.head())
# print(data.tail())
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# print(data.corr(numeric_only=True))


numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sn.heatmap(corr_matrix, annot=True)
plt.title("(Features Correlation)")
plt.show()

target = "points"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(len(x_train), len(x_test), len(y_train), len(y_test))

numerical_features = ['day', 'month', 'year', 'match_time_s', 'ping', 'kills', 'assists', 'deaths', 'mvps',
                      'hs_percent']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('nom', nominal_transformer, ['map'])
])

reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestRegressor(random_state = 42))
])

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)


for i,j in zip(y_test,y_pred):
    print("Actual:{},Predicted:{}".format(i, j))

print("MAE {}".format(mean_absolute_error(y_test,y_pred)))
print("MSE {}".format((mean_squared_error(y_test,y_pred))))
print("R2 {}".format(r2_score(y_test,y_pred)))

plt.figure(figsize=(7, 6))
sn.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Points")
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")
plt.savefig("csgo_regression_result.jpg")
plt.show()