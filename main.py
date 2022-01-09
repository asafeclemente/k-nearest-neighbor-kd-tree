import pandas as pd
import utils
from knn import x_NN
import time

NUMBER_NEIGHBORS = [1,2,3,5,7,9,13]
FILES = ['banana', 'coil2000', 'haberman', 'magic', 'phoneme', 'pima' ,'ring', 'spambase', 'titanic', 'twonorm']

def dataProcessing(file):
# Recebe o nome do arquivo e retorna conjunto de treino e teste normalizado e dividido com 30% para teste

    df = utils.open_dat_df("./datasets/" + file + ".dat")

    x = df.copy().drop('_class', axis=1)
    classes = df['_class']

    X_train, X_test, y_train, y_test = utils.trainTestSplit(x, classes, test_size=0.3)
    
    Z_train = (X_train - X_train.mean()) / X_train.std(ddof=1)

    # Normalizando o conjunto de teste com média de desvio do conjunto de treino
    Z_test = (X_test -  X_train.mean()) / X_train.std(ddof=1) 

    return Z_train.values, y_train.values, Z_test.values, y_test.values

def main():

    for file_name in FILES:
        
        results = []
        
        for k in NUMBER_NEIGHBORS:
            print(file_name, k)

            D_train, y_train, D_test, y_test = dataProcessing(file_name)
            # Criar modelo knn
            custom = x_NN(D_train, y_train, D_test, y_test, n_neighbors=k)

            # Salvar Acurácia, Recall e Precisão
            results.append((custom.getAccuracy(), custom.getRecall(), custom.getPrecision()))

        results = pd.DataFrame(results, columns=['accuracy', 'recall', 'precision'])

        utils.save_result(file_name, results, NUMBER_NEIGHBORS)

if __name__ == "__main__":
    main()

    