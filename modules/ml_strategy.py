import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def create_ml_dataset(ml_series, feature_window, prediction_horizon):
    lags = [ml_series.shift(i) for i in range(1, feature_window+1)]
    X = pd.concat(lags, axis=1)
    X.columns = [f"lag_{i}" for i in range(1, feature_window+1)]
    y = ml_series.shift(-prediction_horizon)
    df_ml = pd.concat([X, y.rename("target")], axis=1).dropna()
    return df_ml

def split_dataset(df_ml, train_split):
    train_size = int(len(df_ml) * (train_split / 100))
    train = df_ml.iloc[:train_size]
    test = df_ml.iloc[train_size:]
    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test = test.drop("target", axis=1)
    y_test = test["target"]
    return X_train, y_train, X_test, y_test

def train_and_predict(model_name, X_train, y_train, X_test):
    # Instantiate model based on name
    if model_name == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=42)
    elif model_name == "Neural Network":
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def compute_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
