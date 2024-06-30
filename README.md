# Brain-Network-Classification

This repository contains the final program of my degree thesis as well as the dataset required to run it. When the program is launched, it computes the brain classification following the stratified cross-validation strategy and the parameters given by the user.

## Running the program

To run the program:
```
py brain_network_classification.py [groups] [splits] [exclusiveEdges] [model] [centrality] <thetas>
```

For example:
```
 py -3.11 brain_network_classification.py children,adolescents 5 False NN eigenvectors 0,0.5,0.9
```

## Options
- groups: Group(s) of the dataset you want to work with.
  - adolescents
  - eyesclosed
  - male
  - children
- splits: number of splits for the stratified cross-validation.
- exclusiveEdges: whether only exclusive edges of each class are taken into account for metagraph building or not.
  - True
  - False
- model: classifcation tool
  - LR: Logistic Regression
  - NN: Neural Network
- centrality: centrality metrics to be use during classification.
  - metaScores: metagraph's similarity score
  - edgeBetweenness
  - metaScores*edgeBetweenness
  - eigenvectors
  - closeness
- thetas: list of thetas that will define the bounding of the metagraphs (always in range [0, 1))

## Requirements
The program has been developed and tested using Python 3.11.2 and the following versions of its modules:
- Keras 3.3.3.
- NetworkX 3.3.
- Numpy 1.26.4.
- Scikit-learn 1.4.2
- 
Other versions may or may not work.
