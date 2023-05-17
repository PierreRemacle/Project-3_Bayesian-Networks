import re
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

# A class for representing the CPT of a variable
class CPT:
    def __init__(self, head, parents):
        self.head = head  # The variable this CPT belongs to (object)
        self.parents = parents  # Parent variables (objects), in order
        self.entries = {}
        # Entries in the CPT. The key of the dictionnary is an
        # assignment to the parents; the associated value is a dictionnary
        # itself, reflecting one row in the CPT.
        # For a variable that has no parents, the key is the empty tuple.
    def copy(self, head):
        
        new_cpt = CPT(head, self.parents.copy())
        new_cpt.entries = deepcopy(self.entries)
        return new_cpt

    # String representation of the CPT according to the BIF format
    def __str__(self):
        comma = ", "
        if len(self.parents) == 0:
            return (
                f"probability ( {self.head.name} ) {{" + "\n"
                f"  table {comma.join ( map(str,self.entries[tuple()].values () ))};"
                + "\n"
                f"}}" + "\n"
            )
        else:
            return (
                f"probability ( {self.head.name} | {comma.join ( [p.name for p in self.parents ] )} ) {{"
                + "\n"
                + "\n".join(
                    [
                        f"  ({comma.join(names)}) {comma.join(map(str,values.values ()))};"
                        for names, values in self.entries.items()
                    ]
                )
                + "\n}\n"
            )


# A class for representing a variable
class Variable:
    def __init__(self, name, values):
        self.name = name  # Name of the variable
        self.values = values
        # The domain of the variable: names of the values
        self.cpt = None  # No CPT initially
    def copy(self):
        new_variable = Variable(self.name, self.values.copy())
        new_variable.cpt = self.cpt.copy(new_variable)
        return new_variable

    # String representation of the variable according to the BIF format
    def __str__(self):
        comma = ", "
        return (
            f"variable {self.name} {{"
            + "\n"
            + f"  type discrete [ {len(self.values)} ] {{ {(comma.join(self.values))} }};"
            + "\n"
            + f"}}"
            + "\n"
        )

    def valuestoint(self, int):
        return self.values.index(int)


class BayesianNetwork:
    # Method for reading a Bayesian Network from a BIF file;
    # fills a dictionary 'variables' with variable names mapped to Variable
    # objects having CPT objects.
    def __init__(self, input_file = None):
        self.progress=[]
        if (input_file == None):
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_csv(input_file, sep=",")
        self.variables = {}
        for columns in self.df.columns:
            sorted = np.sort(self.df[columns].unique())
            variable_values = [str(x) for x in sorted]

            variable = Variable(columns, variable_values)
            variable.cpt = CPT(variable, [])
            self.variables[columns] = variable

        self.computeCPT_init()

    def copy(self):
        new_network = BayesianNetwork()
        new_network.variables = {}
        new_network.df = self.df.copy()

        for var in self.variables:
            new_network.variables[var] = self.variables[var].copy()


        """for variable_name, variable in self.variables.items():
            new_variable = Variable(variable.name, variable.values.copy())
            new_variable.cpt = variable.cpt.copy(new_variable)
            new_network.variables[variable_name] = new_variable"""

        return new_network

    def score(self):
        score = 0

        col = self.df.columns.tolist()
        for _, rows in self.df.iterrows():
            prob = 1
            dico = {}

            for columns in self.df.columns:
                dico[columns] = str(rows[columns])
            for col in dico:
                prob *= self.P_Yisy_given_parents(col, dico[col], dico)
            score += np.log(prob)

        return score
    def score_BIC(self):
        score = 0
        num_parameters = 0  # Number of free parameters
        nbr_values = {var.name: len(var.values) for var in self.variables.values()}
        #print(nbr_values)

        col = self.df.columns.tolist()
        num_data_points = len(self.df)

        for _, rows in self.df.iterrows():
            prob = 1
            dico = {}

            for columns in self.df.columns:
                dico[columns] = str(rows[columns])

            for col in dico:
                prob = self.P_Yisy_given_parents(col, dico[col], dico)
            score += np.log(prob)

        # Calculate the number of free parameters
        for variable in self.variables.values():
            """num_parents = len(variable.cpt.parents)
            num_values = len(variable.values)
            num_parameters += (num_values - 1)*(num_parents + 1)"""
            mult_parent = 1
            for parent in variable.cpt.parents:
                mult_parent *= nbr_values[parent.name]

            num_parameters += (nbr_values[variable.name]-1) * mult_parent

        bic = score - (0.5 * num_parameters * np.log(num_data_points))
        #bic = score - num_parameters
        return bic


    def load(self, input_file):
        with open(input_file) as f:
            lines = f.readlines()

        self.variables = {}
        # The dictionary of variables, allowing quick lookup from a variable
        # name.
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace("/", "-")

        # Parsing all the variable definitions
        i = 0
        while not lines[i].startswith("probability"):
            if lines[i].startswith("variable"):
                variable_name = lines[i].rstrip().split(" ")[1]
                i += 1
                variable_def = lines[i].rstrip().split(" ")
                # only discrete BN are supported
                assert variable_def[1] == "discrete"
                variable_values = [x for x in variable_def[6:-1]]
                for j in range(len(variable_values)):
                    variable_values[j] = re.sub("\(|\)|,", "", variable_values[j])
                variable = Variable(variable_name, variable_values)
                self.variables[variable_name] = variable
            i += 1

        # Parsing all the CPT definitions
        while i < len(lines):
            split = lines[i].split(" ")
            target_variable_name = split[2]
            variable = self.variables[target_variable_name]

            parents = [
                self.variables[x.rstrip().lstrip().replace(",", "")]
                for x in split[4:-2]
            ]

            assert variable.name == split[2]

            cpt = CPT(variable, parents)
            i += 1

            nb_lines = 1
            for p in parents:
                nb_lines *= len(p.values)
            for lid in range(nb_lines):
                cpt_line = lines[i].split(" ")
                # parent_values = [parents[j].values[re.sub('\(|\)|,', '', cpt_line[j])] for j in range(len(parents))]
                parent_values = tuple(
                    [re.sub("\(|\)|,", "", cpt_line[j]) for j in range(len(parents))]
                )
                probabilities = re.findall("\d\.\d+(?:e-\d\d)?", lines[i])
                cpt.entries[parent_values] = {
                    v: float(p) for v, p in zip(variable.values, probabilities)
                }
                i += 1
            variable.cpt = cpt
            i += 1

    # Method for writing a Bayesian Network to an output file
    def write(self, filename):
        with open(filename, "w") as file:
            for var in self.variables.values():
                file.write(str(var))
            for var in self.variables.values():
                file.write(str(var.cpt))

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and x an assignment to its parents in the network, specified
    # in the correct order of parents.
    def P_Yisy_given_parents_x(self, Y, y, x=tuple()):

        return self.variables[Y].cpt.entries[x][y]

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and pa a dictionnary of assignments to the parents,
    # with for every parent variable its associated assignment.
    def P_Yisy_given_parents(self, Y, y, pa={}):
        x = tuple([pa[parent.name] for parent in self.variables[Y].cpt.parents])

        return self.P_Yisy_given_parents_x(Y, y, x)

    def joint_distrib_simple(self, Y, pa={}):
        # Find parent of Y

        # all the possible value of Y
        values = self.variables[Y].values

        num = {}
        denominator = 0
        for value in values:
            prob = 1
            new_dict = pa.copy()
            new_dict[Y] = value
            for x in new_dict:
                prob *= self.P_Yisy_given_parents(x, new_dict[x], new_dict)

            num[value] = prob
            denominator += prob

        for value in values:
            if denominator != 0:
                num[value] /= denominator

        return num

    def joint_distrib_double(self, Y, pa={}):
        values_0 = self.variables[Y[0]].values
        values_1 = self.variables[Y[1]].values

        num = {}
        denominator = 0

        for value_0 in values_0:
            for value_1 in values_1:
                prob = 1
                new_dict = pa.copy()
                new_dict[Y[0]] = value_0
                new_dict[Y[1]] = value_1
                for x in new_dict:
                    prob *= self.P_Yisy_given_parents(x, new_dict[x], new_dict)
                num[(value_0, value_1)] = prob
                denominator += prob

        for value in num:
            if denominator != 0:
                num[value] /= denominator

        return num

    # calls computeCPT for every column in the data file
    def computeCPT_init(self, alpha=0, K=0):
        for columns in self.df.columns:
            if self.variables[columns].cpt.entries == {}:
                self.computeCPT(columns, alpha, K)

    # Method for computing the CPT of a columns, given a data file and the bayesian network
    def computeCPT(self, column, alpha=0, K=0):
        retour = {}
        # if no parents
        if len(self.variables[column].cpt.parents) == 0:
            a = self.df[column].value_counts(normalize=True)
            a = a.to_dict()
            a = {str(key): value for key, value in a.items()}

            retour[()] = a
            self.variables[column].cpt.entries = retour

        # if parents

        else:
            here = []
            parents_name = [x.name for x in self.variables[column].cpt.parents]

            # get the possible value for each parents and put them in a list
            for parent in parents_name:
                unique = list(self.variables[parent].values)

                here.append(unique)

            combins = list(itertools.product(*here))

            # for each combination, calculate the probability of each value of the column
            for combin in combins:
                tmp = self.df.copy()
                todrop = []
                for row in tmp.index:                   
                    for i in range(len(combin)):
                        par =parents_name[i]
                        a_1 = tmp[par]
                        a= a_1[row]
                        b= combin[i]
                        if str(a) != b:
                            todrop.append(row)
                          
                tmp.drop(todrop, inplace=True)
                denom = len(tmp) + alpha * K

                values = {}
                sorted = np.sort(self.df[column].unique())
                for value in sorted:
                    num = len(tmp[tmp[column] == value]) + alpha
                    if denom != 0:
                        prob = num / denom
                    else:
                        prob = 0
                    values[str(value)] = prob

                retour[combin] = values

            self.variables[column].cpt.entries = retour

    def check_cylcle(self, variable_name, new_parent_name):

        parents = [parent.name for parent in self.variables[new_parent_name].cpt.parents]

        if parents == []:
            return False

        if variable_name in parents:
            return True
        else:
            for p in parents:
                if self.check_cylcle(variable_name, p):
                    return True
        return False
    
    def plot(self):
        G = nx.Graph()
        for var in self.variables:
            G.add_node(var)
        for child in self.variables:
            for parent in self.variables[child].cpt.parents:
                G.add_edge(parent.name, child)
        # Visualize the directed graph with arrows
        pos = nx.spring_layout(G)  # Compute node positions

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="red", node_size=50)

        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, arrows=True,arrowstyle="->", arrowsize=10, width=2)

        # Draw node labels
        nx.draw_networkx_labels(G, pos)

        # Show the plot
        plt.axis("off")
        plt.savefig("net_asia2.pdf", bbox_inches='tight')

    def plot_progression(self):
        plt.figure()
        plt.plot(self.progress)
        plt.savefig("progression.pdf", bbox_inches='tight')

    
def local_movev1(bn, var_isolated):
    """Simple local such that an isolated node can only be linked to another isolated node. Recusriv way"""
    score = bn.score()
    
    if len(var_isolated) < 2:
        return bn, score
    else:
        max_imp = 0
        max_comp = []
        for var1 in var_isolated:
            for var2 in var_isolated:
                if var1 != var2:
                    bn.variables[var1].cpt.parents.append(bn.variables[var2])
                    bn.computeCPT(var1)
                    tmp = bn.score()
                    if tmp > score:
                        imp = tmp - score
                        if imp > max_imp:
                            max_imp = imp
                            max_comp = [var1, var2]

                    bn.variables[var1].cpt.parents.remove(bn.variables[var2])
                    bn.computeCPT(var1)
        
        if max_imp > 0:
            bn.variables[max_comp[0]].cpt.parents.append(bn.variables[max_comp[1]])
            bn.computeCPT(max_comp[0])
            var_isolated.remove(max_comp[0])
            var_isolated.remove(max_comp[1])
            bn, score = local_movev1(bn, var_isolated)

            
    return bn, bn.score()

def local_movev2(bn, var_isolated, vars):
    """Simple local such that an isolated node can be linked to any other node. Recusriv way"""
    score = bn.score()
    # best_score = max_score
    if len(var_isolated) == 0:
        return bn, score
    else:
        max_imp = 0
        max_comp = []
        for var1 in var_isolated:
            for var2 in vars:
                if var1 != var2:
                    bn.variables[var1].cpt.parents.append(bn.variables[var2])
                    bn.computeCPT(var1)
                    tmp = bn.score()
                    if tmp > score:
                        imp = tmp - score
                        if imp > max_imp:
                            max_imp = imp
                            max_comp = [var1, var2]

                    bn.variables[var1].cpt.parents.remove(bn.variables[var2])
                    bn.computeCPT(var1)
        
        if max_imp > 0:
            bn.variables[max_comp[0]].cpt.parents.append(bn.variables[max_comp[1]])
            bn.computeCPT(max_comp[0])
            var_isolated.remove(max_comp[0])
            if max_comp[1] in var_isolated:
                var_isolated.remove(max_comp[1])
            bn, score = local_movev2(bn, var_isolated, vars)

            
    return bn, bn.score()

def local_movev3(bn, var_isolated):
    """Simple local such that an isolated node can be linked to any other node. Iterative way"""
    vars = [var for var in var_isolated]
    score = bn.score()

    max_score = score
    matrix_score = np.zeros((len(var_isolated),len(var_isolated)))

    for i,var1 in enumerate(var_isolated):
        print(i)

        for j,var2 in enumerate(var_isolated):
            if var1 != var2:
                bn.variables[var1].cpt.parents.append(bn.variables[var2])
                bn.computeCPT(var1)
                matrix_score[i,j] = bn.score()
                bn.variables[var1].cpt.parents.remove(bn.variables[var2])
                bn.computeCPT(var1)

    matrix_score -= max_score
    for i in range(len(var_isolated)):
        matrix_score[i,i] = -1

    while(len(var_isolated) > 0):
                
        #max_value = np.max(matrix_score)
        i, j = np.unravel_index(matrix_score.argmax(), matrix_score.shape)     

        if matrix_score[i,j] <= 0:
            break
        else:
            var1 = var_isolated[i]
            var2 = vars[j]
            bn.variables[var1].cpt.parents.append(bn.variables[var2])
            bn.computeCPT(var1)
            # delete the row and column of the max score
            matrix_score = np.delete(matrix_score, i, 0)
            #matrix_score = np.delete(matrix_score, i, 1)
            var_isolated.remove(var1)

            if var2 in var_isolated:
                j = var_isolated.index(var2)
                matrix_score = np.delete(matrix_score, j, 0)
                #matrix_score = np.delete(matrix_score, j, 1)
                var_isolated.remove(var2)

            
    return bn, bn.score()


def local_movev4(bn, vars, score_function=""):
    """Complex local move using Stochastic Greedy Search"""
    if score_function == "BIC":
        best_score = bn.score_BIC()
    else:
        best_score = bn.score()

    without_improvement = 0
    nbr_iter = 0
    while(nbr_iter<50):
        nbr_iter += 1
        x = random.choice(vars) # String

        score_improvmement = best_score
        action = ""
        
        parents_tested = []
        for i in range(1):
            """count = np.round(0.3 * len(vars))
            
            parents = random.sample(vars, int(count)) # String []"""
            parents = [var for var in vars if var != x]

            """if (x in parents):
                parents.remove(x)"""


            true_parents = [parent.name for parent in bn.variables[x].cpt.parents] # String []

            for parent in parents:
                if parent not in parents_tested:
                    parents_tested.append(parent)
                    if parent not in true_parents:
                        # check cycle
                        check = bn.check_cylcle(x, parent)
                        if not check:
                            # add
                            bn.variables[x].cpt.parents.append(bn.variables[parent])
                            # compute cpt
                            bn.computeCPT(x)
                            # score improvmement
                            if score_function == "BIC":
                                tmp_score = bn.score_BIC()
                            else:
                                tmp_score = bn.score()                            

                            # if score improvmement > last: Store the action and last = score improvmement
                            if tmp_score > score_improvmement:
                                score_improvmement = tmp_score
                                action = "add_" + x + "_" + parent

                            # remove
                            bn.variables[x].cpt.parents.remove(bn.variables[parent])
                            # compute cpt
                            bn.computeCPT(x)
                        
                    else:
                        
                        # if(Remove)
                        # Remove
                        bn.variables[x].cpt.parents.remove(bn.variables[parent])
                        # Compute cpt
                        bn.computeCPT(x)
                        # score improvmement
                        if score_function == "BIC":
                            tmp_score = bn.score_BIC()
                        else:
                            tmp_score = bn.score()


                        # if score improvmement > last: Store the action and last = score improvmement
                        if tmp_score > score_improvmement:
                            score_improvmement = tmp_score
                            action = "remove_" + x + "_" + parent

                        # add
                        bn.variables[x].cpt.parents.append(bn.variables[parent])
                        # compute cpt
                        bn.computeCPT(x)


                        # if(Reverse)
                        # Reverse = Remove + add
                        # Remove
                        bn.variables[x].cpt.parents.remove(bn.variables[parent])
                        bn.computeCPT(x)
                        # add
                        check = bn.check_cylcle(parent, x)
                        if not check:
                            bn.variables[parent].cpt.parents.append(bn.variables[x])
                            # compute cpt
                            bn.computeCPT(parent)
                            # score improvmement
                            if score_function == "BIC":
                                tmp_score = bn.score_BIC()
                            else:
                                tmp_score = bn.score()

                            # if score improvmement > last: Store the action and last = score improvmement
                            if tmp_score > score_improvmement:
                                score_improvmement = tmp_score
                                action = "reverse_" + x + "_" + parent
                            
                            # remove
                            bn.variables[parent].cpt.parents.remove(bn.variables[x])
                            # compute cpt
                            bn.computeCPT(parent)
                        #add
                        bn.variables[x].cpt.parents.append(bn.variables[parent])
                        # compute cpt
                        bn.computeCPT(x)                        
        print(best_score)
        bn.progress.append(best_score)
        # Apply the action and update the score if score_improvmement > 0
        if score_improvmement > best_score:
            best_score = score_improvmement
            if action.startswith("add"):
                bn.variables[x].cpt.parents.append(bn.variables[action.split("_")[2]])
                bn.computeCPT(x)
            elif action.startswith("remove"):
                bn.variables[x].cpt.parents.remove(bn.variables[action.split("_")[2]])
                bn.computeCPT(x)
            elif action.startswith("reverse"):
                bn.variables[x].cpt.parents.remove(bn.variables[action.split("_")[2]])
                bn.variables[action.split("_")[2]].cpt.parents.append(bn.variables[x])
                bn.computeCPT(x)
                bn.computeCPT(action.split("_")[2])
                
            without_improvement = 0
        else:
            print("no improvement")
            without_improvement += 1

        """# Check if the score has been changed during the X last iterations
        if without_improvement > 10:
            break"""
        

    return bn, best_score

def find_best_graph(file):
    bn = BayesianNetwork(file)
    var_isolated = [var for var in bn.variables]
    bn, max_score = local_movev4(bn, var_isolated, "")

    #test = bn.score_BIC()
    #bn, max_score = local_movev4(bn, var_isolated, var_isolated.copy())
    bn.write("test.bif")
    bn.plot()
    bn.plot_progression()

    return bn, max_score

#print(find_best_graph("./datasets/mini/dummy.csv"))
print(find_best_graph("./datasets/asia/train.csv"))






