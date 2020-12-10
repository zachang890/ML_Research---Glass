from alpha_finder import Alpha_Finder
from random_regress_with_selection import Random_Regressor_Selective

if __name__ == "__main__":

    # 1. ################# FIND OPTIMAL ALPHA VALUE FOR LASSO

    alpha_obj = Alpha_Finder("./resources/transformed_samples_by_features.csv")
    poisson_train, poisson_test = alpha_obj.calculate_alpha_rmse("POISSONS")
    poisson_average_rmse_train, poisson_average_rmse_test = alpha_obj.average_rmse(poisson_train, poisson_test)
    alpha_obj.plot_rmse_alpha(poisson_average_rmse_train, poisson_average_rmse_test, "Poisson's Ratio", "g")
    poisson_coeffs = alpha_obj.lasso_coeffs(10 ** (-6), "Poisson's ratio v", "Poisson's Ratio")

    # 2. ################# RANKING FEATURES

    poisson_coeffs_dict = alpha_obj.coeffs_to_dict(poisson_coeffs)
    poisson_coeffs_dict = alpha_obj.filter_coeff_dicts(poisson_coeffs_dict)
    poisson_ranked = alpha_obj.rank_coeffs(poisson_coeffs_dict)
    poisson_selected = alpha_obj.select_features(poisson_ranked)

    # 3. ################# RANDOM FOREST USING FEATURES (First-level transform, no selection) - GRAPH 2

    regressor = Random_Regressor_Selective() #use selective model but simply include all features to implement no selection
    poisson_rmse_train, poisson_rmse_test, poisson_r2_train, poisson_r2_test = regressor.random_forest(poisson_selected, "Young's modulus E (GPa)", "Shear modulus G (GPa)", "Poisson's ratio v")
    regressor.plot_rmse_estimators(poisson_rmse_train, poisson_rmse_test, "Poisson's Ratio")
    regressor.plot_r2(poisson_r2_train, poisson_r2_test, "Poisson's Ratio")

    # 4. ################# RANDOM FOREST USING FEATURES (First-level transform, with selection) - GRAPH 3

    poisson_selected = poisson_selected[:50]  # change this number based on number of features desired
    poisson_rmse_train, poisson_rmse_test, poisson_r2_train, poisson_r2_test = regressor.random_forest(poisson_selected, "Young's modulus E (GPa)", "Shear modulus G (GPa)", "Poisson's ratio v")
    regressor.plot_rmse_estimators(poisson_rmse_train, poisson_rmse_test, "Poisson's Ratio")
    regressor.plot_r2(poisson_r2_train, poisson_r2_test, "Poisson's Ratio")