import re
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
import random
import os

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
        self.accprogress=[]
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
        score = self.score()
        num_parameters = 0  # Number of free parameters
        nbr_values = {var.name: len(var.values) for var in self.variables.values()}
        #print(nbr_values)
        num_data_points = len(self.df)



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
        """Return the distribution of Y given the others variables.
        Args:
            Y (str): The name of the variable Y.
            pa (dict): A dictionary of the others variables with their values.
        Returns:
            dict: A dictionary with the distribution of Y and its parent.
        """

        # Possible valuese of Y
        values = self.variables[Y].values

        num = {}
        denominator = 0
        # For each possible value of Y
        for value in values:
            prob = 1
            new_dict = pa.copy()
            new_dict[Y] = value

            # Calculate the probability of Y=y given its parents
            for x in new_dict:
                prob *= self.P_Yisy_given_parents(x, new_dict[x], new_dict)

            num[value] = prob
            denominator += prob

        # Normalize the probabilities
        for value in values:
            if denominator != 0:
                num[value] /= denominator

        return num

    def joint_distrib_double(self, Y, pa={}):
        """Return the joint distribution of Y based on the others variables.
        Args:
            Y (str): The two variables to calculate the joint distribution.
            pa (dict): A dictionary of the others variables with their values.
        Returns:
            dict: A dictionary with the joint distribution of Y given the other variables.
        """

        # Possible values of Y
        values_0 = self.variables[Y[0]].values
        values_1 = self.variables[Y[1]].values

        num = {}
        denominator = 0

        # Iterate over each combinasions of values of Y
        for value_0 in values_0:
            for value_1 in values_1:

                prob = 1
                new_dict = pa.copy()
                new_dict[Y[0]] = value_0
                new_dict[Y[1]] = value_1
                # Calculate the probability of Y[0]=value_0 and Y[1]=value_1 given its parents
                for x in new_dict:
                    prob *= self.P_Yisy_given_parents(x, new_dict[x], new_dict)
                num[(value_0, value_1)] = prob
                denominator += prob
        # Normalize the probabilities
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
    def computeCPT(self, column, alpha=1, K=0):
        K = len(self.variables)
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
        plt.savefig("net_asia3.pdf", bbox_inches='tight')

    def plot_progression(self):
                # Create figure and axes
        fig, ax1 = plt.subplots()

        # Plot the first line using ax1
        ax1.plot(self.progress, color='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel('progress', color='blue')

        # Create a second y-axis
        ax2 = ax1.twinx()

        # Plot the second line using ax2
        ax2.plot(self.accprogress, color='red')
        ax2.set_ylabel('accprogress', color='red')

        # Display the plot
        plt.savefig("progression.pdf", bbox_inches='tight')
        plt.show()
        


    
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

def SGS_local_move(bn, vars, score_function="", max_iterations=50, number_set=1, proportion=0.3):
    """Complex local move using Stochastic Greedy Search
    
    Args:
        bn (BayesianNetwork): The Bayesian Network
        vars (String []): The list of variables
        score_function (String): The score function to use
        max_iterations (int): The maximum number of iterations
        number_set (int): The number of set to test
        proportion (float): The proportion of variables to test in each set

    Returns:
        BayesianNetwork: The Bayesian Network
        float: The score of the Bayesian Network
    """

    assert len(vars) > 1, "There must be at least 2 variables in the network"
    assert proportion > 0 and proportion <= 1, "The proportion must be between 0 and 1"

    if score_function == "BIC":
        best_score = bn.score_BIC()
    else:
        best_score = bn.score()
    without_improvement = 0
    nbr_iter = 0
    while(nbr_iter<max_iterations and without_improvement<10):
        ######## addon ########
        # value_input(bn,  "./datasets/sachs/test_missing.csv", "./datasets/sachs/updated_test.csv")
        # acc = evaluate("./datasets/sachs/test_missing.csv", "./datasets/sachs/updated_test.csv", "./datasets/sachs/test.csv")
        # bn.accprogress.append(acc)
        ######## end addon ########
        nbr_iter += 1
        
        x = random.choice(vars) # String

        score_improvmement = best_score
        action = ""
        
        nodes_tested = []
        for i in range(number_set):
            # Get the set of nodes to test
            count = np.round(proportion * len(vars))
            set_nodes = random.sample(vars, int(count)) # String []
            if (x in set_nodes):
                set_nodes.remove(x)
            
            # Get the set of parents of the node x
            true_parents = [parent.name for parent in bn.variables[x].cpt.parents] # String []

            # Iterate over the nodes in the set previously defined
            for node in set_nodes:
                # Test if we have already tested this node avoiding recomputation
                if node not in nodes_tested:
                    nodes_tested.append(node)

                    # Test if the node is not already a parent of x
                    if node not in true_parents:
                        # Test the Add action
                        # Check if adding the node as a parent of x will create a cycle
                        check = bn.check_cylcle(x, node)
                        if not check:
                            # Add the edge + update the x CPT
                            bn.variables[x].cpt.parents.append(bn.variables[node])
                            bn.computeCPT(x)

                            # Compute the score
                            if score_function == "BIC":
                                tmp_score = bn.score_BIC()
                            else:
                                tmp_score = bn.score()                            

                            # if the score is better than the last one, store the action and the score
                            if tmp_score > score_improvmement:
                                score_improvmement = tmp_score
                                action = "add@" + x + "@" + node

                            # Reset the BN as before the add action
                            bn.variables[x].cpt.parents.remove(bn.variables[node])
                            bn.computeCPT(x)
                        
                    else:
                        # As the node is a parent of x, we can test the remove action and reverse action
                        # Remove the edge + update the x CPT
                        bn.variables[x].cpt.parents.remove(bn.variables[node])
                        bn.computeCPT(x)

                        # compute the score
                        if score_function == "BIC":
                            tmp_score = bn.score_BIC()
                        else:
                            tmp_score = bn.score()

                        # if the score is better than the last one, store the action and the score
                        if tmp_score > score_improvmement:
                            score_improvmement = tmp_score
                            action = "remove@" + x + "@" + node

                        # Reset the BN as before the remove action
                        bn.variables[x].cpt.parents.append(bn.variables[node])
                        bn.computeCPT(x)


                        # Reverse = Remove + add

                        # Remove the edge + update the x CPT
                        bn.variables[x].cpt.parents.remove(bn.variables[node])
                        bn.computeCPT(x)

                        # Check if adding the node as a parent of x will create a cycle
                        check = bn.check_cylcle(node, x)
                        if not check:
                            # Add the edge + update the x CPT
                            bn.variables[node].cpt.parents.append(bn.variables[x])
                            bn.computeCPT(node)

                            # compute the score
                            if score_function == "BIC":
                                tmp_score = bn.score_BIC()
                            else:
                                tmp_score = bn.score()

                            # if the score is better than the last one, store the action and the score
                            if tmp_score > score_improvmement:
                                score_improvmement = tmp_score
                                action = "reverse@" + x + "@" + node
                            
                            # Reset the BN as before the add action
                            bn.variables[node].cpt.parents.remove(bn.variables[x])
                            bn.computeCPT(node)

                        # Reset the BN as before the remove action
                        bn.variables[x].cpt.parents.append(bn.variables[node])
                        bn.computeCPT(x)    
        
        # Display the result
        print(best_score)
        bn.progress.append(best_score)
        


        # Apply the action and update the score if we have found an action
        # that gave a better score
        if score_improvmement > best_score:
            # Update the best score
            best_score = score_improvmement

            # Apply the action
            if action.startswith("add"):
                bn.variables[x].cpt.parents.append(bn.variables[action.split("@")[2]])
                bn.computeCPT(x)
            elif action.startswith("remove"):
                bn.variables[x].cpt.parents.remove(bn.variables[action.split("@")[2]])
                bn.computeCPT(x)
            elif action.startswith("reverse"):
                bn.variables[x].cpt.parents.remove(bn.variables[action.split("@")[2]])
                bn.variables[action.split("@")[2]].cpt.parents.append(bn.variables[x])
                bn.computeCPT(x)
                bn.computeCPT(action.split("@")[2])
            
            # Reset the counter
            without_improvement = 0
        else:
            # Increment the counter
            print("no improvement")
            without_improvement += 1

    return bn, best_score

"""def find_best_graph(file):
    bn = BayesianNetwork(file)
    var_isolated = [var for var in bn.variables]
    #bn, max_score = local_movev4(bn, var_isolated, "")

    #test = bn.score_BIC()
    bn, max_score = local_movev3(bn, var_isolated)
    bn.write("test.bif")
    bn.plot()
    bn.plot_progression()

    return bn, max_score"""

def value_input(bn, file, file_destination):
    """Replace the missing values by the most probable value based on the Bayesian Network
    Args:
        bn (BayesianNetwork): Bayesian Network
        file (string): File path of the test file
        file_destination (string): File path of the file to write
    """
    df = pd.read_csv(file)

    # Iterate over the rows
    for index, row in df.iterrows():

        pa = {}
        input = []
        # Iterate over the columns
   
        for col in df.columns:
            # verify if the value is missing
            if pd.notna(row[col]):
                if type(row[col]) == float:
                    pa[col] = str(int(row[col]))
                else:
                    pa[col] = str(int(row[col]))
                
                
            else:
                input.append(col)
        
        # Compute the probability
        if len(input) == 0:
            continue
            
        elif len(input) == 1:
            distribution = bn.joint_distrib_simple(input[0], pa)
            # Get the key of the max value
            max_key = max(distribution, key=distribution.get)
            # Replace the missing value by the max key
            df.loc[index, input[0]] = max_key

        elif len(input) == 2:
            distribution = bn.joint_distrib_double(input, pa)
            
            # Get the key of the max value
            max_key = max(distribution, key=distribution.get)
            # Replace the missing value by the max key
            df.loc[index, input[0]] = max_key[0]
            df.loc[index, input[1]] = max_key[1]
        else:
            
            print("Not implemented for more than 2 missing values")

    df.to_csv(file_destination, index=False)
    
    return

def main(train_file, test_file, missing_file, netwrok_file):
    """Main function
    Args:
        train_file (string): File path of the train file
        test_file (string): File path of the test file
        missing_file (string): File path of the writed file
        netwrok_file (string): File path of the network file
    """
    # Init the BN
    bn = BayesianNetwork(train_file)
    
    # Learn the structure
    vars = [var for var in bn.variables]
    # bn, _ = SGS_local_move(bn, vars, "", 30, 2, 0.3)
    
    bn,score = local_movev3(bn, vars)
    bn.plot()
    # Input the values
    value_input(bn, test_file, missing_file)

    bn.plot()
    # bn.plot_progression()
    # Write the network
    # bn.write(netwrok_file)

    return 

def evaluate(missing_file, test_file_inputed, test_file):
    """Evaluate the performance of the network on the test set
    Args:
        missing_file (string): File path of the file with the missing values
        test_file_inputed (string): File path of the file with the inputed values
        test_file (string): File path of the file with the correct values
        
    Returns:
        float: Accuracy of the network
    """
    # Read the files
    correct_test = pd.read_csv(test_file)
    inputed_test = pd.read_csv(test_file_inputed)
    missing_file = pd.read_csv(missing_file)

    accuracy = 0
    nbr_missing = 0

    # Iterate over the rows
    for index, row in missing_file.iterrows():
        # Iterate over the columns
        for col in missing_file.columns:
            # verify if the value is missing
            if not(pd.notna(row[col])):
                nbr_missing += 1
                if str(correct_test.loc[index, col]) == str(int(inputed_test.loc[index, col])):
                    accuracy += 1
          
    print("Accuracy: ", accuracy/nbr_missing)

    return accuracy/nbr_missing


listoffile = os.listdir("./datasets")
listoffile.remove("stormofswords")
scores = []
for file in listoffile:
    print(file)
    main("./datasets/"+file+"/train.csv", "./datasets/"+file+"/test_missing.csv", "./datasets/"+file+"/updated_test.csv" , "network.bif")
    score = evaluate("./datasets/"+file+"/test_missing.csv", "./datasets/"+file+"/updated_test.csv", "./datasets/"+file+"/test.csv")
    print(score)
    scores.append(score)
    
print(scores)
    
#BIC [0.7804878048780488, 0.9043062200956937, 0.7512820512820513, 0.902676399026764, 0.4768856447688564, 0.727735368956743, 0.85]
#normal [0.8, 0.9186602870813397, 0.764102564102564, 0.8953771289537713, 0.4768856447688564, 0.7175572519083969, 0.85]