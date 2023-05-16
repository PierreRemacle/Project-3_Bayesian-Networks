import re
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy



"""# create a list of all the possible combination of two list
def Combinaison(l1, l2):
    retour = []
    for i in range(len(l1)):
        for j in range(len(l2)):
            if type(l1[i]) != list:
                retour.append([l1[i], l2[j]])
            else:
                l1copy = l1[i].copy()
                l1copy.append(l2[j])

                retour.append(l1copy)

    return retour


# calculate the probability of a value of a column, given the combination of the parents
def calculation(df, parents, combin, objective, value, alpha, K):
    dfhere = df

    for i in range(len(parents)):
        dfhere = dfhere[dfhere[parents[i].name] == parents[i].valuestoint(combin[i])]
    dfObjective = dfhere[dfhere[objective] == value]

    if dfhere.shape[0] == 0:
        return 0
    else:
        return (dfObjective.shape[0] + alpha) / (dfhere.shape[0] + alpha * K)"""


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
    def copy(self):

        new_cpt = CPT(self.head, self.parents.copy())
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
        new_variable.cpt = self.cpt.copy()
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

        for variable_name, variable in self.variables.items():
            new_variable = Variable(variable.name, variable.values)
            new_variable.cpt = variable.cpt.copy()
            new_network.variables[variable_name] = new_variable

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
            num_parents = len(variable.cpt.parents)
            num_values = len(variable.values)
            num_parameters += (num_values - 1) (num_parents + 1)

        bic = 2 * score + num_parameters * np.log(num_data_points)
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
            
    def creates_cycle(self, variable_name, new_parent_name):
        graph = {}
        visited = set()

        def dfs(node):
            visited.add(node)

            if node == variable_name:
                return True

            for parent in graph.get(node, []):
                if parent not in visited:
                    if dfs(parent):
                        return True

            return False

        # Create a graph representation of the Bayesian network
        for variable in self.variables.values():
            parents = [parent.name for parent in variable.cpt.parents]
            graph[variable.name] = parents

        # Add the new parent to the variable temporarily
        self.variables[variable_name].cpt.parents.append(
            self.variables[new_parent_name]
        )

        # Check if adding the parent creates a cycle
        result = dfs(new_parent_name)

        # Remove the temporary parent
        self.variables[variable_name].cpt.parents.remove(
            self.variables[new_parent_name]
        )

        return result



# Example for how to read a BayesianNetwork
"""bn = BayesianNetwork("./datasets/mini/dummy.csv")
bn.variables["Burglar"].cpt.parents.append(bn.variables["Alarm"])


bn.computeCPT("Burglar")
print(bn.variables["Burglar"].cpt.entries)
bn.write("dummy3.bif")

bn.load("dummy3.bif")
print(bn.variables["Burglar"].cpt.entries)"""
# def test(input , file,output):

#     bn = BayesianNetwork ( input )

#     for var in bn.variables:

#         bn.variables[var].cpt.entries = None

#     bn.computeCPT_init(file)


#     for var in bn.variables:

#         print(bn.variables[var].cpt.entries )

#     bn.write(output)
# test("dummy.bif","./datasets/mini/dummy.csv" , "dummy1.bif")
# test("alarm.bif","./datasets/alarm/train.csv" , "alarm3.bif")

"""
# Example for how to write a BayesianNetwork
bn.write("alarm2.bif")

# Examples for how to get an entry from the CPT

# return P(HISTORY=TRUE|LVFAILURE=TRUE)
print(bn.P_Yisy_given_parents_x("HISTORY","TRUE",("TRUE",)))
# or
print(bn.P_Yisy_given_parents("HISTORY","TRUE",{"LVFAILURE":"TRUE"}))

# return P(HRBP=NORMAL|ERRLOWOUTPUT=TRUE,HR=LOW)
print(bn.P_Yisy_given_parents_x("HRBP","NORMAL",("TRUE","LOW")))
# or
print(bn.P_Yisy_given_parents("HRBP","NORMAL",{"ERRLOWOUTPUT":"TRUE","HR":"LOW"}))

# return P(HYPOVOLEMIA=TRUE)
print(bn.P_Yisy_given_parents_x("HYPOVOLEMIA","TRUE"))
# or
print(bn.P_Yisy_given_parents("HYPOVOLEMIA","TRUE"))

parents=bn.variables["PRESS"].cpt.parents

print("--------------------------------------------------")
"""
# print(bn.joint_distrib_simple('FLU', {'FEVER':'TRUE','FATIGUE': 'TRUE'}))

# print(bn.joint_distrib_double(['FLU', 'FEVER'],{'FATIGUE': 'TRUE'}))


def local_move(bn, var_isolated):
    best_graph = []
    score = bn.score()
    changed = False
    # best_score = max_score
    if len(var_isolated) < 2:
        best_graph = [var for var in var_isolated]
        return best_graph, score
    else:
        for var1 in var_isolated:
            for var2 in var_isolated:
                if var1 != var2:
                    bn.variables[var1].cpt.parents.append(bn.variables[var2])
                    bn.computeCPT(var1)


                    var_isolated.remove(var2)
                    var_isolated.remove(var1)

                    compo, previous_score = local_move(bn, var_isolated)

                    if previous_score > score:
                        best_graph = [(var1, var2), compo]

                        score = previous_score
                        changed = True



                    var_isolated.append(var1)
                    var_isolated.append(var2)

                    bn.variables[var1].cpt.parents.remove(bn.variables[var2])
                    bn.computeCPT(var1)

        if not changed:
            best_graph = [var for var in var_isolated]
            
    return best_graph, score


def find_best_graph(file):
    bn = BayesianNetwork(file)
    var_isolated = [var for var in bn.variables]
    bn, max_score = local_move(bn, var_isolated)

    return bn, max_score


# print(find_best_graph("./datasets/mini/dummy.csv"))
bn = BayesianNetwork("./datasets/mini/dummy.csv")

bn.variables["Burglar"].cpt.parents.append(bn.variables["Alarm"])
bn.variables["Alarm"].cpt.parents.append(bn.variables["Earthquake"])
print(bn.creates_cycle("Earthquake", "Alarm"))
print(bn.creates_cycle("Burglar", "JohnCalls"))




