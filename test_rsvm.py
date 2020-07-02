import pandas as pd
import numpy as np
import random
from lib.rsvm.trainer import Trainer
from lib.rsvm.predictor import Predictor
import pickle
from sklearn.metrics.pairwise import rbf_kernel

def train_zone():
    A = pd.read_csv('sample_data_300.csv')
    A = A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

    m, n = A.shape
    A_value = np.array(A.values[:,range(3)])

    class_label = np.array([[]])
    for i in range(m):
        if A.iloc[i]['Dominasi Zona'] == "Atas":
            class_label = np.append(class_label, [1])
        elif A.iloc[i]['Dominasi Zona'] == "Tengah":
            class_label = np.append(class_label, [2])
        elif A.iloc[i]['Dominasi Zona'] == "Bawah":
            class_label = np.append(class_label, [3])


    A_value = np.column_stack((A_value, class_label.astype(int)))

    trainer = Trainer(A_value, 3)
    trainer.make(r = 0.1 , v = 1)
    trainer.tune(c = 100, g = 0.1, k = 1, s = 0)
    trainer.train()

    subset_A = trainer.get_subset_data()
    model = trainer.set_model()
    
    model_top_middle_zone = model.get(1).get(2).get('model')
    model_top_bottom_zone = model.get(1).get(3).get('model')
    model_middle_bottom_zone = model.get(2).get(3).get('model')

    return [model_top_middle_zone, model_top_bottom_zone, model_middle_bottom_zone, subset_A]

def kernel_trick(x, subset_A):
    return rbf_kernel(x, subset_A, gamma = 0.1)

def predict(x, subset_A, w, b):
    return np.array(np.sign(kernel_trick(x, subset_A) * w-b))[0][0]

def predict_zone(model):
    x = [[63,23,20]]
    pred = []
    for i in range(3):
        pred = pred + [predict(x, model[3], model[i].get('w'), model[i].get('b'))]

    print(pred)

x = train_zone()
predict_zone(x)