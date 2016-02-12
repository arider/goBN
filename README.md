# goBN
Discrete bayesian networks in Go

## Background
There are tradeoffs in using a discrete bayesian network such as this.  On the one hand, the model can approximate arbitrary distributions with multinomials with increasing numbers of buckets. On the other hand, as the number of buckets increases, the number of observations necessary to support the increased complexity grows very quickly.  A continuous bayesian network might reduce the number of necessary observations by offloading some of the intelligence into the distribution types used.

## Features
99.2% test coverage</br>
Use a simple genetic algorithm to select a network topology.</br>
Make arbitrary queries about the posterior distribution given any amount of evidence.</br>

## Example
example.go implements an experiment in which a BN is inferred and used as a classifier for the iris data set.

##### Some notes
The example uses a very simple genetic algorithm to select a network topology.  The provided InferBayesianNetwork function evaluates the current model using the model likelihood.  For a classification problem like this, it would be better to replace the InferBayesianNetwork function with a function that evaluates the model based on how well the class is predicted instead of general model likelihood.  You would also typically use a separate train and testing set, which is not done in the example.

##### Output
Inferred topology:

sepal_width</br>
-> petal_width</br>
petal_width</br>
-> species</br>
species</br>
-> petal_length</br>
petal_length</br>
-> sepal_length</br>
sepal_length

Model log likelihood -409.7831195345815

confusionMatrix:</br>
[50 0 0]</br>
[0 47 3]</br>
[0 1 49]</br>

precision:  [1 0.94 0.98]
