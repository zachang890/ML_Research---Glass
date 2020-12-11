from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class Graph_Train_Test:
    def __init__(self, names):
        self.names = names

    def optimal_regress(self, estimators): #Predicting Poisson's Ratio
        pred = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv").drop(
            ["Index", "Code", "Glass #", "Author", "Year", "Trademark", "Glass_composition", "Young's modulus E (GPa)",
             "Shear modulus G (GPa)"], axis=1)
        X = pred.iloc[:, 0:len(self.names)]
        Y = pred.iloc[:, len(self.names)]

        X_train, X_test, y_actual_train, y_actual_test = train_test_split(X, Y, test_size=0.2, random_state=30)
        regressor = RandomForestRegressor(n_estimators=estimators, random_state=0)
        regressor.fit(X_train, y_actual_train)
        y_pred_test = regressor.predict(X_test)

        regressor = RandomForestRegressor(n_estimators=estimators, random_state=0)
        regressor.fit(X_train, y_actual_train)
        y_pred_train = regressor.predict(X_train)

        return y_actual_test, y_pred_test, y_actual_train, y_pred_train

    def plot_expectedvactual(self, y_actual_test, y_pred_test, y_actual_train, y_pred_train, title):
        r2_score_test = r2_score(y_actual_test, y_pred_test)
        label_test = "Test, R-Squared: " + str(r2_score_test)
        r2_score_train = r2_score(y_actual_train, y_pred_train)
        label_train = "Train, R-Squared: " + str(r2_score_train)

        plt.scatter(y_actual_test, y_pred_test, facecolors='none', edgecolors='b', label=label_test)
        plt.scatter(y_actual_train, y_pred_train, facecolors='none', edgecolors='r', label=label_train)
        plt.xlim([0.15, 0.35])
        plt.ylim([0.15, 0.35])
        plt.title(title)
        plt.xlabel("Actual Poisson's Ratio")
        plt.ylabel("Predicted Poisson's Ratio")

        x = [0, 0.345] #Linear for reference
        y = [0, 0.345]
        plt.plot(x, y)
        plt.legend()
        plt.show()

    def optimal_regress_selective(self, estimators, selected): # GRAPH 2 and 3
        selected_data = pd.read_csv("./resources/transformed_samples_by_features.csv").drop(columns=["Unnamed: 0"])
        selected_features_df = pd.DataFrame()
        for feature in selected:
            selected_features_df[feature] = selected_data[feature]
        selected_features_df

        pred = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv").drop(
            ["Index", "Code", "Glass #", "Author", "Year", "Trademark", "Glass_composition", "Young's modulus E (GPa)",
             "Shear modulus G (GPa)"], axis=1)
        X = selected_features_df.iloc[:, 0:len(selected)]
        Y = pred["Poisson's ratio v"]

        X_train, X_test, y_actual_train, y_actual_test = train_test_split(X, Y, test_size=0.2, random_state=30)
        regressor = RandomForestRegressor(n_estimators=estimators, random_state=0)
        regressor.fit(X_train, y_actual_train)
        y_pred_test = regressor.predict(X_test)

        regressor = RandomForestRegressor(n_estimators=estimators, random_state=0)
        regressor.fit(X_train, y_actual_train)
        y_pred_train = regressor.predict(X_train)

        return y_actual_test, y_pred_test, y_actual_train, y_pred_train