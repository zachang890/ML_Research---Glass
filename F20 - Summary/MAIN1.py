from atomic_mol_percentage import Atomic_Mol_Percentage
from feature_onboard import Feature_Onboard
from feature_transformations import Feature_Transformations
from random_regress_no_selection import Random_Regressor_Nonselective

if __name__ == "__main__":

    # 1. ################ CALCULATING ATOMIC MOLE PERCENTAGES

    atomic_obj = Atomic_Mol_Percentage()
    full_data = atomic_obj.retrieve_compounds_data()
    compound_names = atomic_obj.retrieve_compound_names()
    compound_makeups_dict = atomic_obj.retrieve_compound_makeups()
    all_percentage_weights = atomic_obj.calc_atomic_mol_per()

    # 2. ################ MATRIX MULTIPLICATION, COMPOSITION x FEATURES

    feature_obj = Feature_Onboard()
    feature_names = feature_obj.retrieve_feature_names()
    glass_percentage_weights = feature_obj.convert_atom_mol_df(all_percentage_weights)
    samples_by_features = feature_obj.dot_compounds_features(glass_percentage_weights)
    transformed_samples_by_features = feature_obj.normalize_features(samples_by_features, feature_names)

    # 3. ################ FIRST-LEVEL MATHEMATICAL TRANSFORMATIONS

    transformation_obj = Feature_Transformations(feature_names, transformed_samples_by_features)
    transformation_obj.perform_transforms()
    print(transformation_obj.transformed_samples_by_features) ####### CHECKPOINT

    # 4. ############### FIRST-LEVEL NORMALIZATION

    first_level_names = transformation_obj.new_feature_names
    first_transformed_normalize = feature_obj.normalize_features(transformation_obj.transformed_samples_by_features, first_level_names)

    # 5. ############### RANDOM FOREST USING COMPOSITION (No transform, no selection) - GRAPH 1 (NOTE: THIS SECTION DOES NOT REQUIRE THE ABOVE CODE)

    regressor = Random_Regressor_Nonselective(compound_names)
    poisson_rmse_train, poisson_rmse_test = regressor.random_regress("Young's modulus E (GPa)", "Shear modulus G (GPa)")
    regressor.plot_rmse_estimators(poisson_rmse_train, poisson_rmse_test, "Poisson's Ratio")




