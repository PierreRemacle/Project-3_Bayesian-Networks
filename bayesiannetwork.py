import re
import pandas as pd
import numpy as np

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
    
    def computeCPT_init(self, file):
        df = pd.read_csv(file, sep=',')
        for columns in df.columns:
            if (self.variables[columns].cpt.entries == None):
                self.computeCPT(columns, df)
    
    def computeCPT(self, column , df):
        retour={}

        if (len(self.variables[column].cpt.parents)==0):
            
            a=df[column].value_counts(normalize=True)
            a=a.to_dict()



            retour[()]=a
            self.variables[column].cpt.entries = retour

        else:
            here=[]
            parents = self.variables[column].cpt.parents
            for parent in parents:
            
                unique = list(np.sort(df[parent.name].unique()))
                
                here.append(unique)

            for i in range(len(here)):
                for j in range(len(here[i])):
                    here[i][j]=str(here[i][j])

           
            for i in range(len(here)-1):
                here[0]=Combinaison(here[0],here[i+1])
            if (len(here)==1):
                newhere=[]
                for pospos in range (len(here[0])):
                    newhere.append([here[0][pospos]])
                combins=newhere
            else:
                combins=here[0]
            
            
            for combin in combins:
                here2={}
                for value in df[column].unique():
                    here2[value]=self.calculation(df,parents,combin,column,value)
                    print(here2[value])

                retour[tuple(combin)] = here2

            self.variables[column].cpt.entries = retour
            
    def calculation(self,df,parents,combin, objective , value):

        dfhere = df
        print(combin)
        for i in range(len(parents)):
            dfhere=dfhere[dfhere[parents[i].name]==int(combin[i])]
        dfObjective = dfhere[dfhere[objective] == value]

        if (dfhere.shape[0]==0):
            return 0
        else:
            return dfObjective.shape[0]/dfhere.shape[0]
        
        

        
        
        
                    
        
                
        
    
        


# Example for how to read a BayesianNetwork
bn = BayesianNetwork ( "dummy.bif" )

for var in bn.variables:
    
    bn.variables[var].cpt.entries = None
    
bn.computeCPT_init("./datasets/mini/dummy.csv")


for var in bn.variables:

    print(bn.variables[var].cpt.entries )

bn.write("dummy1.bif")

df = pd.read_csv("./datasets/alarm/train.csv", sep=',')

a=df["HYPOVOLEMIA"]
b=df["LVFAILURE"]
c=df["LVEDVOLUME"]

print(c.value_counts())
print(len(c.value_counts()))

tt=0
tf=0
ft=0
ff=0

count = c.value_counts()[0]
print("------------------")

for i in range (len(b)):

    if(b[i] == 1 and a[i] == 1 ):
        tt=tt+1
    elif(b[i] == 1 and a[i] == 0 ):
        ft=ft+1
    elif(b[i] == 0 and a[i] == 1 ):
        tf=tf+1
    elif(b[i] == 0 and a[i] == 0 ):
        ff=ff+1

for j in range(3): 
    tt2=0
    tf2=0
    ft2=0
    ff2=0     
    for i in range (len(b)):

        if(b[i] == 1 and a[i] == 1 and c[i] == j):
            tt2=tt2+1
        elif(b[i] == 1 and a[i] == 0 and c[i] == j):
            ft2=ft2+1
        elif(b[i] == 0 and a[i] == 1 and c[i] == j):
            tf2=tf2+1
        elif(b[i] == 0 and a[i] == 0 and c[i] == j):
            ff2=ff2+1
    print(np.round((tt2)/(tt),2),np.round((tf2)/(tf),2),np.round((ft2)/(ft),2),np.round((ff2)/(ff),2))

print(tt2,tf2,ft2,ff2)
print(tt,tf,ft,ff)
print(bn.variables["LVEDVOLUME"].cpt)

print(df[df["LVEDVOLUME"]==0].shape[0])

# c=df["LVEDVOLUME"].value_counts(normalize=True)

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





