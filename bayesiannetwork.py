import re
import pandas as pd
import numpy as np


# create a list of all the possible combination of two list
def Combinaison(l1,l2):
    retour=[]
    for i in range (len(l1)):
        
        for j in range (len(l2)):
            if (type(l1[i]) != list):
                retour.append([l1[i], l2[j]])
            else:
                l1copy = l1[i].copy()
                l1copy.append(l2[j])

                retour.append(l1copy)
            
    return retour

#calculate the probability of a value of a column, given the combination of the parents
def calculation(df,parents,combin, objective , value,alpha,K):

    dfhere = df

    for i in range(len(parents)):

        dfhere=dfhere[dfhere[parents[i].name]==parents[i].valuestoint(combin[i])]
    dfObjective = dfhere[dfhere[objective] == value]

    if (dfhere.shape[0]==0):
        return 0
    else:
        return (dfObjective.shape[0]+alpha)/(dfhere.shape[0]+alpha*K)






# A class for representing the CPT of a variable
class CPT:

    def __init__(self, head, parents):
        self.head = head # The variable this CPT belongs to (object)
        self.parents = parents # Parent variables (objects), in order
        self.entries = {} 
          # Entries in the CPT. The key of the dictionnary is an
          # assignment to the parents; the associated value is a dictionnary 
          # itself, reflecting one row in the CPT.
          # For a variable that has no parents, the key is the empty tuple.


    # String representation of the CPT according to the BIF format
    def __str__(self):
        comma = ", "
        if len(self.parents) == 0:
            return f"probability ( {self.head.name} ) {{" + "\n" \
                f"  table {comma.join ( map(str,self.entries[tuple()].values () ))};" + "\n" \
                f"}}" + "\n"
        else:
            return f"probability ( {self.head.name} | {comma.join ( [p.name for p in self.parents ] )} ) {{" + "\n" + \
                "\n".join ( [  \
                  f"  ({comma.join(names)}) {comma.join(map(str,values.values ()))};" \
                    for names,values in self.entries.items () \
                ] ) + "\n}\n" 


# A class for representing a variable
class Variable:
    
    def __init__(self, name, values):
        self.name = name # Name of the variable
        self.values = values 
          # The domain of the variable: names of the values
        self.cpt = None # No CPT initially

    # String representation of the variable according to the BIF format
    def __str__(self):
        comma = ", "
        return f"variable {self.name} {{" + "\n" \
             + f"  type discrete [ {len(self.values)} ] {{ {(comma.join(self.values))} }};" + "\n" \
             + f"}}" + "\n"
    def valuestoint(self , int ):

        return self.values.index(int)
        
        
class BayesianNetwork:

    # Method for reading a Bayesian Network from a BIF file;
    # fills a dictionary 'variables' with variable names mapped to Variable
    # objects having CPT objects.
    def __init__(self, input_file):

        with open(input_file) as f:
            lines = f.readlines()

        self.variables = {} 
          # The dictionary of variables, allowing quick lookup from a variable
          # name.
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace('/', '-')

        # Parsing all the variable definitions
        i = 0
        while not lines[i].startswith("probability"):
            if lines[i].startswith("variable"):
                variable_name = lines[i].rstrip().split(' ')[1]
                i += 1
                variable_def = lines[i].rstrip().split(' ')
                # only discrete BN are supported
                assert(variable_def[1] == 'discrete')
                variable_values = [x for x in variable_def[6:-1]]
                for j in range(len(variable_values)):
                    variable_values[j] = re.sub('\(|\)|,', '', variable_values[j])
                variable = Variable(variable_name, variable_values)
                self.variables[variable_name] = variable
            i += 1

        
        # Parsing all the CPT definitions
        while i < len(lines):
            split = lines[i].split(' ')
            target_variable_name = split[2]
            variable = self.variables[target_variable_name]

            parents = [self.variables[x.rstrip().lstrip().replace(',', '')] for x in split[4:-2]]

            assert(variable.name == split[2])

            cpt = CPT(variable, parents)
            i += 1
            
            nb_lines = 1
            for p in parents:
                nb_lines *= len(p.values)
            for lid in range(nb_lines):
                cpt_line = lines[i].split(' ')
                #parent_values = [parents[j].values[re.sub('\(|\)|,', '', cpt_line[j])] for j in range(len(parents))]
                parent_values = tuple([ re.sub('\(|\)|,', '', cpt_line[j]) for j in range(len(parents)) ])
                probabilities = re.findall("\d\.\d+(?:e-\d\d)?", lines[i])
                cpt.entries[parent_values] = { v:float(p) for v,p in zip(variable.values,probabilities) }
                i += 1
            variable.cpt = cpt
            i += 1

    # Method for writing a Bayesian Network to an output file
    def write(self,filename):
        with open(filename,"w") as file:
            for var in self.variables.values ():
                file.write(str(var))
            for var in self.variables.values ():
                file.write(str(var.cpt))

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and x an assignment to its parents in the network, specified
    # in the correct order of parents.
    def P_Yisy_given_parents_x(self,Y,y,x=tuple()):
        return self.variables[Y].cpt.entries[x][y]

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and pa a dictionnary of assignments to the parents,
    # with for every parent variable its associated assignment.
    def P_Yisy_given_parents(self,Y,y,pa={}):
        x = tuple([ pa[parent.name] for parent in self.variables[Y].cpt.parents ])
        return self.P_Yisy_given_parents_x(Y,y,x)

    def joint_distrib_simple(self, Y,pa={}):
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
                prob *= self.P_Yisy_given_parents(x,new_dict[x], new_dict)

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
                    prob *= self.P_Yisy_given_parents(x,new_dict[x],new_dict)
                num[(value_0,value_1)] = prob
                denominator += prob
        
        for value in num:
            if denominator != 0:
                num[value] /= denominator
                
        return num
    
    
    # calls computeCPT for every column in the data file
    def computeCPT_init(self, file,alpha=0,K=0):
        df = pd.read_csv(file, sep=',')
        for columns in df.columns:
            if (self.variables[columns].cpt.entries == None):
                self.computeCPT(columns, df,alpha,K)
                
    # Method for computing the CPT of a columns, given a data file and the bayesian network
    def computeCPT(self, column , df,alpha,K):
        retour={}
        #if no parents
        if (len(self.variables[column].cpt.parents)==0):
            a=df[column].value_counts(normalize=True)
            a=a.to_dict()
            retour[()]=a
            self.variables[column].cpt.entries = retour
        #if parents
        else:
            here=[]
            parents = self.variables[column].cpt.parents
            
            #get the possible value for each parents and put them in a list
            for parent in parents:
                unique = list(self.variables[parent.name].values)

                here.append(unique)
                


            #get all the possible combination of the parents
            # example :
            #
            #   from [2,1,0] [1,0]
            #
            #   to [[2,1],[2,0],[1,1],[1,0],[0,1],[0,0]]
            
            for i in range(len(here)-1):
                here[0]=Combinaison(here[0],here[i+1])
            if (len(here)==1):
                newhere=[]
                for pospos in range (len(here[0])):
                    newhere.append([here[0][pospos]])
                combins=newhere
            else:
                combins=here[0]
            
            #for each combination, calculate the probability of each value of the column
            for combin in combins:
                here2={}
                for value in df[column].unique():
                    here2[value]=calculation(df,parents,combin,column,value,alpha,K)

                retour[tuple(combin)] = here2

            self.variables[column].cpt.entries = retour


# Example for how to read a BayesianNetwork
bn = BayesianNetwork ( "test.bif" )


def test(input , file,output):
    
    bn = BayesianNetwork ( input )

    for var in bn.variables:
        
        bn.variables[var].cpt.entries = None
    
    bn.computeCPT_init(file)


    for var in bn.variables:

        print(bn.variables[var].cpt.entries )

    bn.write(output)
test("dummy.bif","./datasets/mini/dummy.csv" , "dummy1.bif")
test("alarm.bif","./datasets/alarm/train.csv" , "alarm3.bif")

'''
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
'''
print(bn.joint_distrib_simple('FLU', {'FEVER':'TRUE','FATIGUE': 'TRUE'}))

print(bn.joint_distrib_double(['FLU', 'FEVER'],{'FATIGUE': 'TRUE'}))




