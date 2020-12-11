# SOURCE FILE FOR DEMONSTRATING CLOSENESS BETWEEN ACTUAL AND PREDICTED POISSON'S RATIOS

from graph_train_test import Graph_Train_Test
from atomic_mol_percentage import Atomic_Mol_Percentage
from alpha_finder import Alpha_Finder

if __name__ == "__main__":

    # 1. #################### GRAPH 1: Regression Using Glass Compositions (Chemical)

    atomic_obj = Atomic_Mol_Percentage()
    grapher = Graph_Train_Test(atomic_obj.retrieve_compound_names())
    y_actual_test, y_pred_test, y_actual_train, y_pred_train = grapher.optimal_regress(15)
    grapher.plot_expectedvactual(y_actual_test, y_pred_test, y_actual_train, y_pred_train, "Predicted Poisson's Ratio vs. Actual Poisson's Ratio - GRAPH 1 (chemical comp.)")

    # 2. #################### GRAPH 2: Regression Using First-Level Transform & No Feature Selection

    alpha_obj = Alpha_Finder("./resources/transformed_samples_by_features.csv")
    y_actual_test, y_pred_test, y_actual_train, y_pred_train = grapher.optimal_regress_selective(15, alpha_obj.all_feature_names)
    grapher.plot_expectedvactual(y_actual_test, y_pred_test, y_actual_train, y_pred_train,
                                 "Predicted Poisson's Ratio vs. Actual Poisson's Ratio - GRAPH 2 (w/ transform & no selection)")

    # 3. #################### GRAPH 3: Regression Using First-Level Transform & Feature Selection

    #EXCERPT FROM MAIN2.py
    poisson_train, poisson_test = alpha_obj.calculate_alpha_rmse("POISSONS")
    poisson_average_rmse_train, poisson_average_rmse_test = alpha_obj.average_rmse(poisson_train, poisson_test)
    alpha_obj.plot_rmse_alpha(poisson_average_rmse_train, poisson_average_rmse_test, "Poisson's Ratio", "g")
    poisson_coeffs = alpha_obj.lasso_coeffs(10 ** (-6), "Poisson's ratio v", "Poisson's Ratio")

    poisson_coeffs_dict = alpha_obj.coeffs_to_dict(poisson_coeffs)
    poisson_coeffs_dict = alpha_obj.filter_coeff_dicts(poisson_coeffs_dict)
    poisson_ranked = alpha_obj.rank_coeffs(poisson_coeffs_dict)
    poisson_selected = alpha_obj.select_features(poisson_ranked)

    poisson_selected = poisson_selected[:50]
    y_actual_test, y_pred_test, y_actual_train, y_pred_train = grapher.optimal_regress_selective(15, poisson_selected)
    grapher.plot_expectedvactual(y_actual_test, y_pred_test, y_actual_train, y_pred_train,
                                 "Predicted Poisson's Ratio vs. Actual Poisson's Ratio - GRAPH 3 (w/ transform & w/ selection)")




