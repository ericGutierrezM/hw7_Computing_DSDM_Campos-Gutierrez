import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load and clean data #

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def updated_train_test_split(data, y_column_name):
    X = data.drop(y_column_name, axis=1)
    y = data[y_column_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )
    return X_train, X_test, y_train, y_test

def remove_nan(x_data, y_data, columns_list):
   mask = x_data[columns_list].notna().all(axis=1)
   new_X = x_data[mask].copy()
   new_y = y_data[mask].copy()

   return new_X, new_y

def fill_nan_mean(data, columns_list):
    data[columns_list] = data[columns_list].fillna(data[columns_list].mean())
    return data

def generate_dummies(data, columns_list):
    return pd.get_dummies(data, columns=columns_list, dtype=int)

def binary_encoding_gender(data):
    data['gender'] = data["gender"].map({"M": 1, "F": 0})
    return data


# Train model #

def fit_model(X_train, X_test, y_train):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std  = sc.transform(X_test)
    modelLogistic = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=102)
    modelLogistic.fit(X_train_std, y_train)
    train_y_pred = modelLogistic.predict_proba(X_train_std)[:, 1]
    test_y_pred = modelLogistic.predict_proba(X_test_std)[:, 1]
    updated_X_train = X_train
    updated_X_train['prediction'] = train_y_pred
    updated_X_test = X_test
    updated_X_test['prediction'] = test_y_pred
    return updated_X_train, updated_X_test


# Compute metrics #

def updated_roc_auc_score(updated_X_train, updated_X_test, y_train, y_test):
    train_roc_auc = roc_auc_score(y_train, updated_X_train['prediction'])
    test_roc_auc = roc_auc_score(y_test, updated_X_test['prediction'])
    return train_roc_auc, test_roc_auc