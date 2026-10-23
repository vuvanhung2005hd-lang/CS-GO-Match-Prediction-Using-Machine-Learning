import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("csgo.csv")
data = data.drop(columns=['team_a_rounds', 'team_b_rounds', "date","wait_time_s"])

# print(data.head())
# print(data.tail())
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# print(data.corr(numeric_only=True))


target = "result"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(len(x_train), len(x_test), len(y_train), len(y_test))

numerical_features = ['day', 'month', 'year', 'match_time_s', 'ping', 'kills', 'assists', 'deaths', 'mvps',
                      'hs_percent', 'points']

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

cls = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state = 42))
])

param = {
    "classifier__n_estimators": [50, 100, 200, 300],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [None, 10, 20],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced"]
}

grid_search = GridSearchCV(
    estimator=cls,             #
    param_grid=param,
    scoring="f1_macro",
    cv=5,
    verbose=1,
    n_jobs=-1
)


grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)


# for i,j in zip(y_test,y_pred):
#     print("Actual:{},Predicted:{}".format(i, j))

print(classification_report(y_test,y_pred))

labels = ["Win", "Lost", "Tie"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
confusion = pd.DataFrame(cm, index=labels, columns=labels)
sn.heatmap(confusion, annot=True, fmt="g", cmap="Blues")
plt.savefig("csgo_classification.jpg")
plt.show()
