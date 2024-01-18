import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")
print(df)

y = df['logS']
print(y)

X = df.drop("logS", axis=1)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)

print(X_train)
print(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

print(y_lr_train_pred)
print(y_lr_test_pred)
print(y_train)
print(y_lr_test_pred)


lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)