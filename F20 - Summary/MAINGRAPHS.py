# SOURCE FILE FOR DEMONSTRATING CLOSENESS BETWEEN ACTUAL AND PREDICTED POISSON'S RATIOS

from graph_train_test import Graph_Train_Test
from atomic_mol_percentage import Atomic_Mol_Percentage

if __name__ == "__main__":

    # 1. #################### GRAPH 1: Regression Using Glass Compositions (Chemical)

    atomic_obj = Atomic_Mol_Percentage()
    grapher = Graph_Train_Test(atomic_obj.retrieve_compound_names())
    y_actual_test, y_pred_test, y_actual_train, y_pred_train = grapher.optimal_regress(15)
    grapher.plot_expectedvactual(y_actual_test, y_pred_test, y_actual_train, y_pred_train, "Predicted Poisson's Ratio vs. Actual Poisson's Ratio - GRAPH 1 (chemical comp.)")