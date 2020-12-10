from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Random_Regressor_Nonselective:
    def __init__(self, compound_names):
        self.estimators = [num for num in range(1, 50)]
        self.compound_names = compound_names

    def rmse(self, score):
        rmse = np.sqrt(-score)
        print(f'rmse= {"{:.2f}".format(rmse)}')

    def random_regress(self, drop1, drop2):
        pred = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv").drop(["Index", "Code", "Glass #", "Author", "Year", "Trademark", "Glass_composition", drop1, drop2], axis=1)

        X = pred.iloc[:, 0:len(self.compound_names)]
        Y = pred.iloc[:, len(self.compound_names)]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

        rmse_test = []
        rmse_train = []
        for i in self.estimators:
            regressor = RandomForestRegressor(n_estimators=i, random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            rmse_test.append(math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        for i in self.estimators:
            regressor = RandomForestRegressor(n_estimators=i, random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_train)
            rmse_train.append(math.sqrt(metrics.mean_squared_error(y_train, y_pred)))  # add R-2 using r2_score method
        return rmse_train, rmse_test

    def plot_rmse_estimators(self, rmse_train, rmse_test, name):
        plt.rcParams["figure.figsize"] = (5, 5)
        plt.plot(self.estimators, rmse_test, "b", label="Test Set")
        plt.plot(self.estimators, rmse_train, "r", label="Train Set")
        plt.title("RMSE vs. Estimators for Random Forest " + name)
        plt.xlabel("Estimators")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show()