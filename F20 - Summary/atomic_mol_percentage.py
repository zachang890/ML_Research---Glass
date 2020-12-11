# calculate atomic mole percentages for glass
import pandas as pd
from collections import defaultdict
from collections import OrderedDict

class Atomic_Mol_Percentage:
    def __init__(self):
        self.full_data = pd.read_csv("./resources/Oxide_glass_1_5_02142020.csv").drop(
            ["Index", "Code", "Glass #", "Author", "Year", "Trademark", "Glass_composition", "Young's modulus E (GPa)",
             "Shear modulus G (GPa)", "Poisson's ratio v"], axis=1)
        self.compound_makeups = pd.read_csv("./resources/atomicquant_Descriptor_table_oxide_series_4.csv").drop(["Unnamed: 0"], axis=1)

    def retrieve_compounds_data(self):  # RETRIEVE RAW COMPOUND COMPOSITION DATA FOR EACH GLASS SAMPLE
        return self.full_data

    def retrieve_compound_names(self):  # RETRIEVE LIST OF COMPOUND NAMES
        return [col for col in self.full_data.columns]

    def retrieve_compound_makeups(self):  # RETRIEVE DICTIONARY RELATING COMPOUNDS AND THEIR COMPOSITIONS
        compound_names = self.retrieve_compound_names()
        element_names = [col for col in self.compound_makeups.columns]
        compound_makeups_dict = defaultdict(dict)
        for i in range(len(self.compound_makeups)):
            for j in range(len(self.compound_makeups.columns)):
                if self.compound_makeups.iloc[i, j] > 0:
                    compound_makeups_dict[compound_names[i]][element_names[j]] = self.compound_makeups.iloc[i, j]
        return compound_makeups_dict

    def calc_atomic_mol_per(self):  # RETRIEVE ATOMIC MOL PERCENTAGE FOR EACH GLASS SAMPLE
        all_percentage_weights = []
        compound_names = self.retrieve_compound_names()
        compound_makeups_dict = self.retrieve_compound_makeups()
        for i in range(len(self.full_data)):
            sample_0 = self.full_data.iloc[i, :]
            compound_moleper = defaultdict(float)
            for j in range(24):
                compound_moleper[compound_names[j]] = sample_0[j] / 100
            weighted_elements = defaultdict(float)
            for i in compound_moleper.keys():
                for j in compound_makeups_dict[i].keys():
                    weighted_elements[j] += compound_makeups_dict[i][j] * compound_moleper[i]
            s = 0
            for i in weighted_elements.keys():
                s += weighted_elements[i]
            percentage_weights = defaultdict(float)
            for i in weighted_elements.keys():
                percentage_weights[i] = (weighted_elements[i] / s)
            all_percentage_weights.append(OrderedDict(sorted(percentage_weights.items())))
        return all_percentage_weights