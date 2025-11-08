import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

model = RandomForestClassifier(n_estimators=100, random_state=33)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
print('Model created successfully!')