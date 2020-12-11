# performs mathematical transformations

import numpy as np
import math

class Feature_Transformations:
    def __init__(self, feature_names, transformed_samples_by_features):
        self.feature_names = feature_names
        self.transformed_samples_by_features = transformed_samples_by_features
        self.feature_count = 10
        self.new_feature_names = []

    def square(self, value):
        return (value * value), "Square"

    def square_root(self, value):
        return (value ** 0.5), "Square Root"

    def exponential(self, value):
        return (math.exp(value)), "Exponential"

    def logarithmic(self, value):
        return (np.log(1 + value)), "Logarithmic"

    def inverse(self, value):
        return (1 / (1 + value)), "Inverse"

    def inverse_squared(self, value):
        return (1 / (1 + (value ** 2))), "Inverse Squared"

    def cubic(self, value):
        return (value ** 3), "Cubed"

    def inverse_sqrt(self, value):
        return (1 / (1 + value ** 0.5)), "Inversed Square Root"

    def inverse_exponential(self, value):
        return (1 / (math.exp(value))), "Exponential Inversed"

    def inverse_log(self, value):
        if value <= 0.0:
            return 0.5, "Inversed Log"
        return (1 / (1 + np.log(value))), "Inversed Log"

    def transform_function(self, func):
        for i in range(len(self.feature_names)):
            new_row = []
            word = ""
            for j in range(len(self.transformed_samples_by_features)):
                current_value = self.transformed_samples_by_features.at[j, self.feature_names[i]]
                transformed, word = func(current_value)
                new_row.append(transformed)
            name = self.feature_names[i] + " " + word
            self.transformed_samples_by_features[name] = new_row
            self.new_feature_names.append(name)

    def perform_transforms(self):
        self.transform_function(self.square)
        self.transform_function(self.square_root)
        self.transform_function(self.exponential)
        self.transform_function(self.logarithmic)
        self.transform_function(self.inverse)
        self.transform_function(self.inverse_squared)
        self.transform_function(self.cubic)
        self.transform_function(self.inverse_sqrt)
        self.transform_function(self.inverse_exponential)
        self.transform_function(self.inverse_log)

    def convert_to_csv(self, df):
        df.to_csv(r"./resources/transformed_samples_by_features.csv")