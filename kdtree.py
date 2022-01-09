import math

class KDNode:

    # Nó de uma kd-tree

    def __init__(self, key=None, point=None, value=None, left=None, right=None, current_axis=None):
        self.key = key
        self.point = point
        self.value = value
        self.left = left
        self.right = right
        self.current_axis = current_axis
    
    def isLeaf(self):
    # Cria uma kd-tree a partir de uma lista de pontos
        return self.point is not None

    def __lt__(self, other):
        return self.value < other.value

def createKdTree(points, dimensions, axis=0):

    # Cria uma kd-tree a partir de uma lista de pontos

    if len(points) == 1:
        return KDNode(point=points[0]['point'], value=points[0]['_class'], current_axis=axis)

    else:
        # Ordena a lista de pontos pela dimensão correspondente
        points.sort(key=lambda point: point['point'][axis])

        # Escolhendo a mediana como elemento central
        median = math.ceil(len(points) / 2)
        loc = points[median-1]

        root = KDNode(key=loc['point'][axis], current_axis=axis)

        # Os filhos corresponderão a um divisão no eixo seguinte
        axis =  (axis + 1) % dimensions

        root.left = createKdTree(points[:median], dimensions, axis=axis)
        root.right = createKdTree(points[median:], dimensions, axis=axis)

        return root




