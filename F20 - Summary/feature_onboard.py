import pandas as pd

class Feature_Onboard:
    def __init__(self):
        self.feature_data = pd.read_csv("./resources/element_Descriptor_table_oxide_series_4.csv").drop(["Unnamed: 0"],
                                                                                              axis=1).dropna().astype(
            'float64')
        self.feature_data.index -= 1

    def retrieve_feature_names(self):  # RETRIEVE ALL FEATURE NAMES
        return [col for col in self.feature_data.columns]

    def convert_atom_mol_df(self, all_percentage_weights):  # CONVERT ATOM MOL PERCENTAGES TO DATAFRAME
        list_all_percentage_weights = []
        for i in all_percentage_weights:
            current_sample_weights = []
            for j in i.keys():
                current_sample_weights.append(float(i[j]))
            list_all_percentage_weights.append(current_sample_weights)
        return pd.DataFrame(list_all_percentage_weights)

    def dot_compounds_features(self, glass_percentage_weights):  # MATRIX MULTIPLY MOL PERCENTAGES AND FEATURES
        return glass_percentage_weights.dot(self.feature_data)

    def normalize_features(self, samples_by_features, names):  # NORMALIZE ALL FEATURES BETWEEN 0 AND 1
        #Names is for the NEW features that need to be normalized since the previous features have already been transformed
        add_end = [] #For previous features
        to_normalize = [] #Features to normalize
        for name in samples_by_features:
            if name not in names:
                add_end.append(name)
            else:
                to_normalize.append(name)
        transformed_samples_by_features = pd.DataFrame()
        for name in add_end:
            transformed_samples_by_features[name] = samples_by_features[name]
        for i in range(len(to_normalize)):
            for j in range(len(samples_by_features)):
                current_value = samples_by_features.at[j, names[i]]
                min_value = samples_by_features[names[i]].min()
                max_value = samples_by_features[names[i]].max()
                xx = (current_value - min_value) / (max_value - min_value)
                transformed_samples_by_features.at[j, names[i]] = xx
        return transformed_samples_by_features