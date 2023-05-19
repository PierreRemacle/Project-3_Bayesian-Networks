# Project-3_Bayesian-Networks
## groupe 33
## Pierre Merveille Pierre Remacle

the code takes (train, test, updated_test, network) as parameter

- train is the path to the training set file
- test is the path to the missing values file 
- updated_test is the name of the new file that will be generated
- network is a write of the Bayesian netwrok

the test set for "stormofswords" contained unseen values during the training. Therfore it doesn't work with our implementation. It seems that some of the values in the test set of "stormofswords" are miss labeled.

## Terminal line
```
python ./bayesiannetwork.py "./train.csv" "./test_missing.csv" "./test_fill.csv" "./network.bif"
```