import numpy as np
import heapq as hq
import pandas as pd
import  csv
import string
import matplotlib.pyplot as plt

def heapPut(nestneighs, tuple_d, n_neighbors):
    if (len(nestneighs) < n_neighbors):
        hq.heappush(nestneighs, tuple_d)
    else: 
        hq.heappushpop(nestneighs, tuple_d)
    return nestneighs

def euclideanDistance(a,b):
    return np.linalg.norm(np.array(a)- np.array(b))

def trainTestSplit(X, y, test_size=0.3):

    n_train = int((1 - test_size) * X.shape[0]) 
    random_order = np.random.permutation(X.shape[0])

    X_train, X_test = np.split(np.take(X,random_order,axis=0), [n_train])
    y_train, y_test = np.split(np.take(y,random_order), [n_train])

    return X_train, X_test, y_train, y_test

def mescle(points, y):
    m = []
    for i in range(len(points)):
        dic = {}
        dic['point'] = points[i]
        dic['_class'] = y[i]
        m.append(dic)
    return m 

def open_dat_df(file_name):
    
    text = open(file_name, "r").read()
    text_split = text.split('@data')
    lines = text_split[1].replace(" ", "").splitlines()
    reader = csv.reader(lines)
    parsed_csv = list(reader)

    splitado = text_split[0].split('@attribute')
    columns = []

    for i in range (len(splitado) - 1):
        if (i == 0): pass
        else:
            name = splitado[i].split(' ')[1]
            columns.append(name)
    columns.append('_class')
    df = pd.DataFrame(parsed_csv, columns=columns).dropna().apply(pd.to_numeric, errors='ignore')
    
    return df

def confusion_matrix(real, predict):
        fp = 0; fn = 0; tp = 0; tn = 0

        for actual_value, predicted_value in zip(real, predict):
            if predicted_value == actual_value:
                if predicted_value in ['positive', 1, 1.0, 'tested_positive', 'h']:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted_value in ['positive', 1, 1.0, 'tested_positive', 'h']:
                    fp += 1
                else:
                    fn += 1
                    
        return [[tp, fn],
                [fp, tn]]

def bar_plot(ax, data, total_width=0.8, single_width=1):

    plt.style.use('grayscale')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []

    for i, (_, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        bars.append(bar[0])    
    
    ax.set_ylabel('Scores')
    ax.set_xlabel('Número de vizinhos')

    ax.legend(bars, data.keys())
    

def save_result(file_name, results, NUMBER_NEIGHBORS):

    results.to_csv( "./results/" + file_name + '.csv', index=False)

    neighbors = [0]
    neighbors.extend(NUMBER_NEIGHBORS)

    _, ax = plt.subplots()
    bar_plot(ax, results.to_dict('list'), total_width=.8, single_width=.9)

    ax.set_xticklabels(neighbors)
    ax.set_title(string.capwords(file_name))
    ax.set(ylim=(0.2, 1))

    plt.savefig( "./results/" +  file_name + '.png')