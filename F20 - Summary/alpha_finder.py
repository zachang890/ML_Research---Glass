from sklearn.linear_model import Lasso
from sklearn import metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict

class Alpha_Finder:
    def __init__(self, transformed_data):
        self.all_transformed_data = pd.read_csv(transformed_data).drop(columns=["Unnamed: 0"]).values
        self.all_feature_names = pd.read_csv(transformed_data).drop(columns=["Unnamed: 0"]).columns
        self.youngs_data = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv")["Young's modulus E (GPa)"].values
        self.shear_data = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv")["Shear modulus G (GPa)"].values
        self.poissons_data = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv")["Poisson's ratio v"].values
        self.alpha_powers = [0, 1, 1.5, 2, 2.5, 4, 5, 6, 7]
        self.alpha_vals = []
        for i in range(len(self.alpha_powers)):
            self.alpha_vals.append(10 ** (-self.alpha_powers[i]))
        self.states = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71]

    def calculate_alpha_rmse(self, y_data):  # CHOOSE FROM "YOUNGS", "SHEAR", "POISSONS"
        if y_data == "YOUNGS":
            y_data = self.youngs_data
        elif y_data == "SHEAR":
            y_data = self.shear_data
        elif y_data == "POISSONS":
            y_data = self.poissons_data
        alpha_state_to_rmse_test = {}
        alpha_state_to_rmse_train = {}

        for alpha_val in self.alpha_vals:
            for state in self.states:
                X_train, X_test, y_train, y_test = train_test_split(self.all_transformed_data, y_data, test_size=0.2,
                                                                    random_state=state)
                lasso = Lasso(alpha=alpha_val, max_iter=200000)
                lasso.fit(X_train, y_train)

                y_pred_test = lasso.predict(X_test)
                y_pred_train = lasso.predict(X_train)
                alpha_state_to_rmse_test[(alpha_val, state)] = math.sqrt(
                    metrics.mean_squared_error(y_test, y_pred_test))
                alpha_state_to_rmse_train[(alpha_val, state)] = math.sqrt(
                    metrics.mean_squared_error(y_train, y_pred_train))
        return alpha_state_to_rmse_train, alpha_state_to_rmse_test

    def average_rmse(self, train, test):
        train_rmse = defaultdict(float)  # alpha, sum RMSE
        for key in train.keys():
            train_rmse[key[0]] += train[key]

        test_rmse = defaultdict(float)
        for key in test.keys():
            test_rmse[key[0]] += test[key]

        train_rmse_averaged = []
        for key in train_rmse:
            train_rmse_averaged.append(train_rmse[key] / len(self.states))

        test_rmse_averaged = []
        for key in test_rmse:
            test_rmse_averaged.append(test_rmse[key] / len(self.states))
        return train_rmse_averaged, test_rmse_averaged

    def plot_rmse_alpha(self, train, test, name, color):
        plt.rcParams["figure.figsize"] = (5, 5)
        plt.plot(self.alpha_powers, train, color + "--", label="Train for " + name)
        plt.plot(self.alpha_powers, test, color, label="Test for " + name)
        plt.title("Averaged RMSE vs. -log(alpha) of Lasso for 15 Random States " + name)
        plt.ylabel("Averaged RMSE")
        plt.xlabel("-log(alpha)")
        plt.legend()
        plt.show()

    def lasso_coeffs(self, alpha_val, column, name):
        lasso = Lasso(alpha=alpha_val)

        X = self.all_transformed_data
        Y = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv")[column].values
        plt.rcParams["figure.figsize"] = (80, 15)
        lasso_coeff = lasso.fit(X, Y).coef_
        plotting = plt.plot(range(len(self.all_feature_names)), lasso_coeff)
        plotting = plt.xticks(range(len(self.all_feature_names)), self.all_feature_names, rotation=60)
        plotting = plt.ylabel("Coefficients")
        plotting = plt.title("Coefficients for " + name)
        plt.show()
        return lasso_coeff

    def coeffs_to_dict(self, coeffs):
        feature_names_to_coeff = {}
        for i in range(len(self.all_feature_names)):
            feature_names_to_coeff[self.all_feature_names[i]] = coeffs[i]
        return feature_names_to_coeff

    def filter_coeff_dicts(self, d):
        filtered_names_to_coeff = {}
        for feature in d:
            if d[feature] != 0.0 and d[feature] != -0.0:
                filtered_names_to_coeff[feature] = abs(d[feature])
        return filtered_names_to_coeff

    def rank_coeffs(self, filtered):
        ranked_names = []
        for name in filtered:
            ranked_names.append((name, filtered[name]))
        ranked_names.sort(key=lambda x: x[1], reverse=True)
        return ranked_names

    def select_features(self, names):
        selected_features = [name for name, val in names]
        return selected_features