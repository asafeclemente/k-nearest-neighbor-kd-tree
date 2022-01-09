from os import PRIO_PGRP
import kdtree
import numpy as np
import multiprocessing
from utils import euclideanDistance
from utils import heapPut
from utils import mescle
from utils import confusion_matrix
import statistics
import random

def selectNeighbor(neighbors):
    y =  statistics.multimode(map(lambda xneighbor : xneighbor.value, neighbors))
    y = random.choice(y)
    return y

def nearestsNeighbors(node, point, neighbors, number_neighb):

    if (node.isLeaf()):
        dist = euclideanDistance(point, node.point)
        neighbors = heapPut(neighbors, (-dist, node), number_neighb)
        return neighbors

    next_node = None; outro_lado = None
    
    # Procurar o lado em que potencialmente os knns estão
    if (point[node.current_axis] <= node.key):
        next_node = node.left
        outro_lado = node.right
    else: 	     
        next_node = node.right 
        outro_lado = node.left
	
    neighbors = nearestsNeighbors(next_node, point, neighbors, number_neighb)

    dist = abs(point[node.current_axis] - node.key) #distancia é do ponto até o eixo 

    # Se o vizinho mais longe estiver a uma distância maior que até outro lado da árvore precisamos verificar o outro lado
    # ou caso não tenha vizinhos suficientes ainda
    if ((-neighbors[0][0] >= dist) or (len(neighbors) < number_neighb)):
        neighbors = nearestsNeighbors(outro_lado, point, neighbors, number_neighb)
	
    return neighbors


def classifyPoint(point):
    knns = nearestsNeighbors(aux_model.kd_tree, point, neighbors=[], number_neighb=aux_model.n_neighbors)
    y = selectNeighbor(np.array(knns)[:,1])
    return y

def classify(model):

    # predict = []
    global aux_model
    aux_model = model

    pool = multiprocessing.Pool()
    predict = pool.map(classifyPoint, model.test)

    # for point in aux_model.test:
    #     y = classifyPoint(point)
    #     predict.append(y)
    
    return predict

class x_NN:
  
    # recebe um conjunto de pontos de treino no constrói a árvore kd 
    # e outro de teste, em que aplica o KNN para cada ponto.

    def __init__(self, train, y_train, test, y_test, n_neighbors):
        # Creates a new node for a kd-tree.
        self.train = train
        self.test = test
        self.n_neighbors = n_neighbors
        self.kd_tree = kdtree.createKdTree(mescle(train, y_train), train.shape[1])
        self.y_test = y_test
        self.predict = classify(self)
        self.cm = confusion_matrix(y_test, self.predict)
    
    def getAccuracy(self):
        d = self.cm[0][0] + self.cm[1][1]
        fp = self.cm[1][0]
        fn  = self.cm[0][1]
        return d / (d + fp + fn)
        
    def getPrecision(self):
        vp = self.cm[0][0]
        fp = self.cm[1][0]
        return vp / (vp + fp)

    def getRecall(self):
        vp = self.cm[0][0]
        fn  = self.cm[0][1]
        return vp / (vp + fn)

