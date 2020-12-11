# performs random forest regression using estimators (range: 1-50) with transformed dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Random_Regressor_Selective:
    def __init__(self):
        self.estimators = [num for num in range(1, 50)]

    def rmse(self, score):
        rmse = np.sqrt(-score)
        print(f'rmse= {"{:.2f}".format(rmse)}')

    def random_forest(self, selected, drop1, drop2, current):
        selected_data = pd.read_csv("./resources/transformed_samples_by_features.csv").drop(columns=["Unnamed: 0"])
        selected_features_df = pd.DataFrame()
        for feature in selected:
            selected_features_df[feature] = selected_data[feature]

        pred = pd.read_csv("/content/drive/MyDrive/Oxide_glass_1_5_02142020.csv").drop(["Index", "Code", "Glass #", "Author", "Year", "Trademark", "Glass_composition", drop1, drop2], axis=1)
        X = selected_features_df.iloc[:, 0:len(selected)]
        Y = pred[current]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

        rmse_test = []
        rmse_train = []
        r2_test = []
        r2_train = []
        for i in self.estimators:
            regressor = RandomForestRegressor(n_estimators=i, random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            rmse_test.append(math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            r2_test.append(metrics.r2_score(y_test, y_pred))
        for i in self.estimators:
            regressor = RandomForestRegressor(n_estimators=i, random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_train)
            rmse_train.append(math.sqrt(metrics.mean_squared_error(y_train, y_pred)))
            r2_train.append(metrics.r2_score(y_train, y_pred))
        return rmse_train, rmse_test, r2_train, r2_test

    def plot_rmse_estimators(self, rmse_train, rmse_test, name):
        plt.rcParams["figure.figsize"] = (5, 5)
        plt.plot(self.estimators, rmse_test, "b", label="Test Set")
        plt.plot(self.estimators, rmse_train, "r", label="Train Set")
        plt.title("RMSE vs. Estimators for Random Forest with Feature Selection " + name)
        plt.xlabel("Estimators")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show()

    def plot_r2(self, r2_train, r2_test, name):
        plt.plot(self.estimators, r2_test, "g", label="Test Set R2")
        plt.plot(self.estimators, r2_train, "m", label="Train Set R2")
        plt.title("R2 vs. Estimators for Random Forest with Feature Selection " + name)
        plt.xlabel("Estimators")
        plt.ylabel("R2")
        plt.legend()
        plt.show()