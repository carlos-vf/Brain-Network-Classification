# -*- coding: utf-8 -*-

import networkx as nx
import os
import numpy as np
import sys
from itertools import combinations_with_replacement
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense


class GraphClassifier:
    
    def __init__(self, groupA, groupB, theta):
        
        # Attributes
        self.groupA = groupA
        self.groupB = groupB
        self.theta = theta
        
        # Create meta-graphs
        self.metaGraphA, self.metaGraphB = self.createMetaGraphs()      
        self.metaGraphsA, self.metaGraphsB = self.createMetaGraphsCV()

        # Create model
        self.predictors = self.getScores()
        self.tags = self.getTags()
        self.model = LogisticRegression(solver='liblinear', random_state=0).fit(self.predictors, self.tags)
            
    
    def createMetaGraphs(self):
        
        # Compute Meta-Graphs
        metaA = makeMetaGraph(self.groupA)
        metaB = makeMetaGraph(self.groupB)
        
        # Remove edges under Theta
        removeUnderThresholdEdges(metaA, self.theta)
        removeUnderThresholdEdges(metaB, self.theta)
        
        return metaA, metaB
    
    
    def createMetaGraphsCV(self):
        
        # Compute meta-graphs
        metaAlist = makeMetaGraphsCV(self.groupA)
        metaBlist = makeMetaGraphsCV(self.groupB)
        
        # Remove edges under Theta
        for pair in (metaAlist + metaBlist):
            removeUnderThresholdEdges(pair[1], self.theta)
        
        return metaAlist, metaBlist


    def getScores(self):
        
        scores = []
        
        for graph in (self.groupA + self.groupB):
            
            # Select metagraphs
            if graph in self.groupA:
                mgA = [g[1] for g in self.metaGraphsA if g[0] == graph][0]
                mgB = self.metaGraphB
                
            else:
                mgA = self.metaGraphA
                mgB = [g[1] for g in self.metaGraphsB if g[0] == graph][0]

            # Calculate scores
            mgAScore = 0
            mgBScore = 0
            
            onlyAEdges = set(mgA.edges) - set(mgB.edges)
            onlyBEdges = set(mgB.edges) - set(mgA.edges)
            
            for u, v in graph.edges:
                if (u, v) in onlyAEdges:
                #if (u, v) in mgA.edges:
                    mgAScore += float(mgA[u][v]['weight'])
                elif (u, v) in onlyBEdges:
                #elif (u, v) in mgB.edges:
                    mgBScore += float(mgB[u][v]['weight'])
            
            # Save predictors
            scores.append([mgAScore, mgBScore])
            
        return scores

       
    def getScoreNewGraph(self, graph):
               
        # Calculate scores
        mgAScore = 0
        mgBScore = 0
        
        onlyAEdges = set(self.metaGraphA.edges) - set(self.metaGraphB.edges)
        onlyBEdges = set(self.metaGraphB.edges) - set(self.metaGraphA.edges)
        
        for u, v in graph.edges:
            if (u, v) in onlyAEdges:
            #if (u, v) in self.metaGraphA.edges:
                mgAScore += float(self.metaGraphA[u][v]['weight'])
            elif (u, v) in onlyBEdges:
            #elif (u, v) in self.metaGraphB.edges:
                mgBScore += float(self.metaGraphB[u][v]['weight'])
        
            
        return [mgAScore, mgBScore]

        
    def getTags(self):
        
        a = ['A' for g in self.groupA]
        b = ['B' for g in self.groupB]
        
        return a + b
            
        
        
    def getAUC(self):
        
        distances = []
        
        # Cross validation
        for i in range(len(self.groupA + self.groupB)):
            
            # Create model
            pred = self.predictors[:i] + self.predictors[i+1:]
            tags = self.tags[:i] + self.tags[i+1:]
            model = LogisticRegression(solver='liblinear', random_state=0).fit(pred, tags)
            
            #distance = model.decision_function([self.predictors[i]])[0]
            distance = model.predict_proba([self.predictors[i]])[0][1]
            distances.append(distance)
            
            
        area = metrics.roc_auc_score(self.tags, distances)
        
        return area
        
    
    def classify(self, graph):
        
        pred = self.getScoreNewGraph(graph)
        print("Scores: ", pred)
        prediction = self.model.predict([pred])[0]
        distance = self.model.decision_function([pred])[0]
        classProb = self.model.predict_proba([pred])[0]
        print("Class probs: ", classProb)
        
        m = Model()
        m.setPrediction(prediction)
        m.setDistance(distance)
        m.setClassAProbability(classProb[0])
        m.setClassBProbability(classProb[1])
        m.setClassAScore(pred[0])
        m.setClassBScore(pred[1])
        
        return m
        
        
class GraphClassifier2:
     
     def __init__(self, groupA, groupB, exclusiveEdges, model, centrality, thetas=[0.0]):
         
         # Check arguments
         if centrality not in ('metaScores', 'edgeBetweenness', 'metaScores*edgeBetweenness', 'eigenvectors', 'closeness'):
             print("ERROR")
         
         
         # Attributes
         self.groupA = groupA
         self.groupB = groupB
         self.exclusiveEdges = exclusiveEdges
         self.code = model
         self.centrality = centrality
         
         # Create meta-graphs if needed
         if centrality == 'metaScores' or centrality == 'edgeBetweenness' or centrality == 'metaScores*edgeBetweenness':
             self.metaGraphs = []
             for theta in thetas:
                 metaGraphA, metaGraphB = self.createMetaGraphs(theta) 
                 if centrality == 'edgeBetweenness' or centrality == 'metaScores*edgeBetweenness':
                     self.computeEdgeBetweenness(metaGraphA)
                     self.computeEdgeBetweenness(metaGraphB)
                 self.metaGraphs.append([metaGraphA, metaGraphB])

             # Compute metaScores and/or edgeBetweenness scores
             self.predictors = self.getScores()
             
         elif centrality == 'eigenvectors':
             self.predictors = self.computeEigenVectors(normalize=True)
   
         elif centrality == 'closeness':
             self.predictors = self.computeCloseness()

             
         self.tags = self.getTags()
         
         # Model creation
         if model == 'LR':
             self.model = LogisticRegression(solver='liblinear', random_state=0).fit(self.predictors, self.tags)
             
         elif model == "NN":
             self.model = Sequential()
             self.model.add(Dense(units=32, input_dim=len(self.predictors[0]), activation='relu')) 
             self.model.add(Dense(units=128, activation='relu'))
             self.model.add(Dense(units=128, activation='relu'))
             self.model.add(Dense(units=128, activation='relu'))
             self.model.add(Dense(units=32, activation='relu'))
             self.model.add(Dense(units=1, activation='sigmoid'))
             self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
             self.model.summary() 
             self.history = self.model.fit(np.array(self.predictors), np.array(self.tags), batch_size=10 , epochs=50, validation_split=0.2, verbose=1)
         
     
     def createMetaGraphs(self, theta):
         
         # Compute Meta-Graphs
         metaA = makeMetaGraph(self.groupA)
         metaB = makeMetaGraph(self.groupB)
         
         # Remove edges under Theta
         removeUnderThresholdEdges(metaA, theta)
         removeUnderThresholdEdges(metaB, theta)
         
         return metaA, metaB


     def getScores(self):
         
         scoresGlobal = []
         
         for graph in (self.groupA + self.groupB):
             
            scores = []
             
            for metaGraphs in self.metaGraphs:
            
                metaGraphA = metaGraphs[0]
                metaGraphB = metaGraphs[1]
                
                # Calculate scores
                mgAScore = 0
                mgBScore = 0
                
                if self.exclusiveEdges:
                    edgeSetA = set(metaGraphA.edges) - set(metaGraphB.edges)
                    edgeSetB = set(metaGraphB.edges) - set(metaGraphA.edges)
                    
                else:
                    edgeSetA = set(metaGraphA.edges)
                    edgeSetB = set(metaGraphB.edges)
                     
                
                if self.centrality == 'metaScores':
                    for u, v in graph.edges:
                         if (u, v) in edgeSetA:
                             mgAScore += float(metaGraphA[u][v]['weight'])
                         if (u, v) in edgeSetB:
                             mgBScore += float(metaGraphB[u][v]['weight'])
                
                elif self.centrality == 'edgeBetweenness':
                    for u, v in graph.edges:
                         if (u, v) in edgeSetA:
                             mgAScore += float(metaGraphA[u][v]['edgeBetweenness'])
                         if (u, v) in edgeSetB:
                             mgBScore += float(metaGraphB[u][v]['edgeBetweenness'])
                             
                elif self.centrality == 'metaScores*edgeBetweenness':
                    for u, v in graph.edges:
                         if (u, v) in edgeSetA:
                             mgAScore += float(metaGraphA[u][v]['weight']) * float(metaGraphA[u][v]['edgeBetweenness'])
                         if (u, v) in edgeSetB:
                             mgBScore += float(metaGraphB[u][v]['weight']) * float(metaGraphB[u][v]['edgeBetweenness'])

                    
                # Save predictors
                scores.append(mgAScore)
                scores.append(mgBScore)

            scoresGlobal.append(scores)
             
         return scoresGlobal

        
     def getScoreNewGraph(self, graph, metaGraphA, metaGraphB):
                
         # Calculate scores
         mgAScore = 0
         mgBScore = 0
                 
         if self.exclusiveEdges:
            edgeSetA = set(metaGraphA.edges) - set(metaGraphB.edges)
            edgeSetB = set(metaGraphB.edges) - set(metaGraphA.edges)
            
         else:
            edgeSetA = set(metaGraphA.edges)
            edgeSetB = set(metaGraphB.edges)
             
         if self.centrality == 'metaScores':
           for u, v in graph.edges:
                if (u, v) in edgeSetA:
                    mgAScore += float(metaGraphA[u][v]['weight'])
                if (u, v) in edgeSetB:
                    mgBScore += float(metaGraphB[u][v]['weight'])
       
         elif self.centrality == 'edgeBetweenness':
           for u, v in graph.edges:
                if (u, v) in edgeSetA:
                    mgAScore += float(metaGraphA[u][v]['edgeBetweenness'])
                if (u, v) in edgeSetB:
                    mgBScore += float(metaGraphB[u][v]['edgeBetweenness'])
                    
         elif self.centrality == 'metaScores*edgeBetweenness':
           for u, v in graph.edges:
                if (u, v) in edgeSetA:
                    mgAScore += float(metaGraphA[u][v]['weight']) * float(metaGraphA[u][v]['edgeBetweenness'])
                if (u, v) in edgeSetB:
                    mgBScore += float(metaGraphB[u][v]['weight']) * float(metaGraphB[u][v]['edgeBetweenness'])

             
         return [mgAScore, mgBScore]

         
     def computeEigenVectors(self, normalize):
         
         preds = []
         
         for graph in (self.groupA + self.groupB):
             ev = nx.eigenvector_centrality(graph, max_iter=400)
             ev = list(ev.values())
             
             # Normalization
             if normalize:
                 maximum = max(ev)
                 minimum = min(ev)
                 ev = np.array([((x-minimum) / (maximum-minimum)) for x in ev])
             
             preds.append(ev)
                     
         return preds
                
         
     def computeCloseness(self):
         
         preds = []
         
         for graph in (self.groupA + self.groupB):
            cl = nx.closeness_centrality(graph)
            preds.append(list(cl.values()))
                     
         return preds
     
        
     def getTags(self):
         
         if self.code == 'LR':
             a = ['A' for g in self.groupA]
             b = ['B' for g in self.groupB]
             
         elif self.code == 'NN':
             a = [0 for g in self.groupA]
             b = [1 for g in self.groupB]
            
         
         return a + b
             
     
     def classify(self, graph):
         
         if self.centrality == 'metaScores' or self.centrality == 'edgeBetweenness' or self.centrality == 'metaScores*edgeBetweenness':
             preds = []
             for i in range(len(self.metaGraphs)):
                 scores = self.getScoreNewGraph(graph, self.metaGraphs[i][0], self.metaGraphs[i][1])
                 preds.append(scores[0])
                 preds.append(scores[1])
                 
         elif self.centrality == 'eigenvectors':
             ev = list(nx.eigenvector_centrality(graph, max_iter=400).values())
             maximum = max(ev)
             minimum = min(ev)
             preds = [((x-minimum) / (maximum-minimum)) for x in ev]
             
         elif self.centrality == 'closeness':
             preds = list(nx.closeness_centrality(graph).values())
         
            
         if self.code == 'LR':
             prediction = self.model.predict([preds])[0]
             distance = self.model.decision_function([preds])[0]
             classProb = self.model.predict_proba([preds])[0]
             print("Class probs: ", classProb)
             
             m = Model()
             m.setPrediction(prediction)
             m.setDistance(distance)
             m.setClassAProbability(classProb[0])
             m.setClassBProbability(classProb[1])
             m.setClassAScore(preds[0])
             m.setClassBScore(preds[1])
             
         elif self.code == 'NN':
            prediction = self.model.predict([preds])[0]
            classProb = [1-prediction, prediction]
            print("Class probs: ", classProb)
            
            m = Model()
            m.setPrediction(prediction)
            m.setClassAProbability(classProb[0])
            m.setClassBProbability(classProb[1])
            m.setClassAScore(preds[0])
            m.setClassBScore(preds[1])
         
         return m      
     
        
     def computeEdgeBetweenness(self, graph):
         
         edges = nx.edge_betweenness_centrality(graph, k=None, normalized=True, seed=None)
         nx.set_edge_attributes(graph, edges, name='edgeBetweenness')
        
        
    
class Model:
    
    def __init__(self):
        pass
    
    def setName(self, fileName):
        self.name = fileName
        
    def setGraph(self, graph):
        self.graph = graph
        
    def setTrueLabel(self, trueLabel):
        self.trueLabel = trueLabel
    
    def setPrediction(self, prediction):
        self.prediction = prediction
        
    def setClassAProbability(self, probA):
        self.classAProb = probA
        
    def setClassBProbability(self, probB):
        self.classBProb = probB
        
    def setCommonEdgesProportion(self, cep):
        self.cep = cep
        
    def setClassAScore(self, scoreA):
        self.scoreA = scoreA
        
    def setClassBScore(self, scoreB):
        self.scoreB = scoreB
        
    def setDistance(self, distance):
        self.distance = distance
        
    def printModel(self):
        print("Name: ", self.name)
        print("Graph: ", self.graph)
        print("Label: ", self.trueLabel)
        print("Prediction: ", self.prediction)
        print("Score class A: ", self.scoreA)
        print("Score class B: ", self.scoreB)
        print("Probability class A: ", self.classAProb)
        print("Probability class B: ", self.classBProb)
        print("Distance: ", self.distance)
        print("Metagraphs' common edges: ", self.cep)
        print()
        
    


def readGraphs(graphList, meta):
    """
    Summary
    ----------
    Reads graphs from adjacency matrices in txt files and returns them as networkx objects.


    Parameters
    ----------
    graphList: list of routes to the files to be read.
    
    meta: Set true when reading files contain metagraphs. Then, a dictionary is returned
        in the form (key = file_name : value = graph_object).
        Set false when reading files contain regular graphs. Then, a dictionary is returned
            in the form (key = graph_object : value = file_name).
        

    Returns
    -------
    graphs: list of graphs (networkx Graph)
    
    names: dictionary relating file names and graph objects
    """
    
    graphs = []
    names = {}
    
    # Read dataset
    for file in graphList:
        
        with open(file) as f:
            
            # Create adjacency matrices
            firstLine = f.readline().strip().split()
            m = np.array(firstLine)
            
            for line in f:
                tokens = np.array(line.strip().split())
                m = np.vstack((m, tokens))
                
        # Create graphs
        g = nx.from_numpy_array(A=m, parallel_edges=False)
        
        # Remove weight-0 edges
        edges = [(u, v) for (u, v) in g.edges if float(g[u][v]['weight']) == 0]
        g.remove_edges_from(edges)
        
        if meta:
            names[file.split('/')[-1].split('-')[-1]] = g
        else:
            names[g] = file.split('/')[-1].split('-')[-1]
            
        # Add graph to the list
        graphs.append(g)
    
    return graphs, names




def makeMetaGraph(group):  
    """
    Summary
    ----------
    Creates a metagraph given a set of graphs. In this context, a metagraph consists on a 
    graph with all vertices and the union of edges of all graphs in the set. Each edge has a weight
    proportional to the number of times it appears (edge_weight = number of graphs with that edge / 
    number of graphs in the set).


    Parameters
    ----------
    group: list of graphs (networkx Graph).
        

    Returns
    -------
    metaGraph: metagraph built from graphs in group.
    """
    
    metaGraph = nx.Graph()
    weights = {comb:0 for comb in combinations_with_replacement(group[0].nodes, 2)}
    n = len(group)
    
    for graph in group:
        for (u, v) in graph.edges:
            weights[(u, v)] += int(graph[u][v]['weight'])
      
    for (u, v) in weights:
        if weights[(u, v)] != 0:
            weights[(u, v)] = weights[(u, v)] / n
        metaGraph.add_edge(u, v, weight=weights[(u, v)])

    return metaGraph

    



def makeMetaGraphsCV(group):
    """
    Summary
    ----------
    Creates a lists of metagraphs given a set of graphs. In this context, a metagraph consists on a 
    graph with all vertices and the union of edges of all graphs in the set. Each edge has a weight
    proportional to the number of times it appears (edge_weight = number of graphs with that edge / 
    number of graphs in the set). Cross-Validation is applied in the sense that each built 
    metagraph does not contain data about one of the graphs in the set. This way, a list of 
    metagraphs of the same number of graphs is generated.


    Parameters
    ----------
    group: list of graphs (networkx Graph).
        

    Returns
    -------
    metaGraphs: list of pairs (graph, metagraph) built from graphs in group.
    """
    
    n = len(group)
    metaGraphs = []
    
    for graph in group:
        metaGraph = nx.Graph()
        weights = {comb:0 for comb in combinations_with_replacement(group[0].nodes, 2)}
        
        for otherGraph in group:
            if graph != otherGraph:
                for (u, v) in otherGraph.edges:
                    weights[(u, v)] += int(otherGraph[u][v]['weight'])
            
        for (u, v) in weights:
            if weights[(u, v)] != 0:
                weights[(u, v)] = weights[(u, v)] / (n - 1)
            metaGraph.add_edge(u, v, weight=weights[(u, v)])
        
        metaGraphs.append((graph, metaGraph))
        
    return metaGraphs


    
def removeUnderThresholdEdges(graph, theta):
    """
    Summary
    ----------
    Removes all edges from graph whose weight is under/equal theta


    Parameters
    ----------
    graph: graph to be edited (networkx Graph).
    
    theta: threshold value between 0 and 1 (float).
    """
    
    edges = [(u, v) for (u, v) in graph.edges if float(graph[u][v]['weight']) <= theta]
    graph.remove_edges_from(edges)

    
    


###################### classifier ######################

def thetaLOOCV(dirs):
    """
    Summary
    ----------
    Performs a leave-one-out cross-validation over a graph classification for all thetas


    Parameters
    ----------
    dirs: lists of routes to the files to be read ([group0_file0, ..., group0_filen])
    

    Returns
    -------
    results: list of pairs of the form (group name, area under ROC curve after cross-validation)
    """

    aucs = np.matrix([-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
   
    probs = []
    maxThetas = []
    tags = []
    
    # For each graph
    graphsA, namesA = readGraphs(dirs[0], False) 
    graphsB, namesB = readGraphs(dirs[1], False) 
    graphs = graphsA + graphsB
    
    for j in range(len(graphs)):
        
        print("Graph: ", j)
        
        graph = graphs[j]
        groupA = [g for g in graphsA if g != graph]
        groupB = [g for g in graphsB if g != graph]
        
        # For each theta
        theta = 0.0
        models = []
        for k in range(1, 11):
            
            print("Theta: ", theta)
            
            newClassifier = GraphClassifier(groupA, groupB, theta)
            models.append([theta, newClassifier.getAUC(), newClassifier])

            theta = round(theta + 0.1, 1)
        
        # Get max theta
        maxAUC = 0
        index = 0
        for k in range(len(models)):
            if models[k][1] > maxAUC:
                index = k
                maxAUC = models[k][1]
                
        maxThetas.append(models[index][0])      
        print("Models: ", models)
        
        # Classify graph
        classifier = models[index][2]
        result = classifier.classify(graph)
        probs.append(result.classBProb)
        if graph in graphsA:
            tags.append('A')
        else:
            tags.append('B')
            
        print("Prediction: ", result.prediction)
        print("Tag: ", tags[-1])
        print()
            
        row = [j] + [m[1] for m in models]
        aucs = np.vstack([aucs, np.array(row)])
        
    # Calculate AUC
    auc = metrics.roc_auc_score(tags, probs)
    
    print()
    print("AUC: ", auc)
    print("Thetas: ", maxThetas)
    
    np.savetxt("theta_results_auc.csv", aucs, delimiter = ",")
        
    return auc
    

def stratifiedCV(dirs, splits, exclusiveEdges, model, centrality, thetas = [0.0]):
    """
    Summary
    ----------
    Performs a stratified cross-validation over a graph classification


    Parameters
    ----------
    dirs: list of lists of routes to the files to be read ([[group0_file0, ..., group0_filen], ...])
    
    splits: Number of folds in the cross-validation
    
    exclusiveEdges: If True, when calculating scores for classification, only edges exclusive in one 
        of the two metagraphs are taken into account. If False, all edges are used.
    
    model: classification model. Possible codes are:
        LR -> Logistic Regression
        NN -> Neural Network
        
    centrality: centrality metric to be used as predictor for classification. Possible codes are:
        metaScores -> Metagraphs' scores based on edge proportion
        edgeBetweenness -> Edge betweenness' scores
        metaScores*edgeBetweenness -> Product of both metrics metaScores * edgeBetweenness
        eigenvectors -> Eigenvectors of graph's nodes
        closeness -> Closeness of graph's nodes
        
    thetas: List of thetas which define the metagraphs to be computed for classification
    

    Returns
    -------
    results: list of pairs of the form (group name, area under ROC curve after cross-validation)
    """
    
    results = []

    for group in dirs:
        
        groupName = group[0][0].split('/')[1]
        print("GROUP: ", groupName)
        
        probs = []
        tags = []
        
        # Read graphs
        graphsA, namesA = readGraphs(group[0], False) 
        graphsB, namesB = readGraphs(group[1], False) 
        graphs = graphsA + graphsB
        
        # Separate testing - training
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=123)
        skfSplit = skf.split(graphs, [0 for g in graphsA] + [1 for g in graphsB])
        
        for i, (train_index, test_index) in enumerate(skfSplit):
            
            print(f"Fold {i}:")
            
            trainAThisFold = [graphs[j] for j in train_index if graphs[j] in graphsA]
            trainBThisFold = [graphs[j] for j in train_index if graphs[j] in graphsB]
            
            classifier = GraphClassifier2(trainAThisFold, trainBThisFold, exclusiveEdges, model, centrality, thetas)
            
            # Classification  
            for j in test_index:
                result = classifier.classify(graphs[j])
                probs.append(result.classBProb)
                if graphs[j] in graphsA:
                    tags.append('A')
                else:
                    tags.append('B')
                    
                print("Prediction: ", result.prediction)
                print("Tag: ", tags[-1])
                print()
                    
        auc = metrics.roc_auc_score(tags, probs)
        print("AUC: ", auc)
        
        results.append([groupName, auc])
      
    print("Thetas: ", thetas)
    print("Results: ", results)
        
    return results





###################### ARGUMENTS ####################
args = sys.argv
d = list(args[1].split(","))
s = int(args[2])
if args[3].lower() in ["true", "1"]:
    e = True
else:
    e = False
m = args[4]
c = args[5]
if len(args) == 7:
    t = [float(x) for x in list(args[6].split(","))]
else:
    t = [0]
    


###################### DATASET ######################
dirs = []

if 'adolescents' in d:
    dir1 = "datasets/adolescents/td/"
    c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]
    dir2 = "datasets/adolescents/asd/"
    c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]
    dirs.append([c1, c2])

if 'eyesclosed' in d:
    dir1 = "datasets/eyesclosed/td/"
    c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]
    dir2 = "datasets/eyesclosed/asd/"
    c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]
    dirs.append([c1, c2])

if 'male' in d:
    dir1 = "datasets/male/td/"
    c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]
    dir2 = "datasets/male/asd/"
    c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]
    dirs.append([c1, c2])

if 'children' in d:
    dir1 = "datasets/children/td/"
    c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]
    dir2 = "datasets/children/asd/"
    c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]
    dirs.append([c1, c2])



results = stratifiedCV(dirs=dirs, splits=s, exclusiveEdges=e, model=m, centrality=c, thetas=t)
print(results)

