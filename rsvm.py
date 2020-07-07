import pandas as pd
import numpy as np
import random
import os
from lib.rsvm.trainer import Trainer
from lib.rsvm.predictor import Predictor
import pickle
from sklearn.metrics.pairwise import rbf_kernel
from statistics import mean

def train_zone(A):
    if os.path.exists("model_zone.pkl"):
        os.remove("model_zone.pkl")

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

    numfold = 5

    trainer = Trainer(A_value, 3)
    trainer.make(r = 0.1, v = numfold)
    trainer.tune(c = 100, g = 0.0001, k = 1, s = 0)
    trainer.train()
    trainer.save(fname='model_zone')
    
    accuracy = trainer.get_accuracy()
    train_acc = sum(accuracy[0][0])/numfold
    test_acc = sum(accuracy[1][0])/numfold

    model = trainer.set_model()
    
    model_top_middle_zone = model.get(1).get(2).get('model')
    model_top_bottom_zone = model.get(1).get(3).get('model')
    model_middle_bottom_zone = model.get(2).get(3).get('model')

    return [model_top_middle_zone, model_top_bottom_zone, model_middle_bottom_zone, train_acc, test_acc]

def train_pressure(A):
    if os.path.exists("model_pressure.pkl"):
        os.remove("model_pressure.pkl")

    A = A[['Rerata', 'Persentase', 'Tekanan Tulisan']]

    m, n = A.shape
    A_value = np.array(A.values[:,range(2)])

    class_label = np.array([[]])
    for i in range(m):
        if A.iloc[i]['Tekanan Tulisan'] == "Kuat":
            class_label = np.append(class_label, [1])
        elif A.iloc[i]['Tekanan Tulisan'] == "Sedang":
            class_label = np.append(class_label, [2])
        elif A.iloc[i]['Tekanan Tulisan'] == "Ringan":
            class_label = np.append(class_label, [3])

    A_value = np.column_stack((A_value, class_label.astype(int)))

    numfold = 5

    trainer = Trainer(A_value, 2)
    trainer.make(r = 0.1 , v = numfold)
    trainer.tune(c = 100, g = 0.0001, k = 1, s = 0)
    trainer.train()
    trainer.save(fname='model_pressure')

    accuracy = trainer.get_accuracy()
    train_acc = sum(accuracy[0][0])/numfold
    test_acc = sum(accuracy[1][0])/numfold

    model = trainer.set_model()
    
    model_top_middle_zone = model.get(1).get(2).get('model')
    model_top_bottom_zone = model.get(1).get(3).get('model')
    model_middle_bottom_zone = model.get(2).get(3).get('model')

    return [model_top_middle_zone, model_top_bottom_zone, model_middle_bottom_zone, train_acc, test_acc]

def predict_zone(x):
    predictor = Predictor('model_zone.pkl')
    return predictor.predict(x)

def predict_pressure(x):
    predictor = Predictor('model_pressure.pkl')
    return predictor.predict(x)

def result_zone(_class):
    personality = ""
    if _class == 1:
        personality = "Penulis lebih memperhatikan aspek spiritual, impian, harapan, dan ambisi dalam hidupnya. Penulis lebih suka melakukan kegiatan berpikir dan memikirkan masa depannya"
    elif _class == 2:
        personality = "Penulis lebih mementingkan kehidupannya saat ini dan sulit untuk membuat rencana jangka panjang mereka"
    elif _class == 3:
        personality = "Penulis lebih mementingkan aspek fisik kehidupan dan lebih mengandalkan ototnya daripada otaknya"

    return personality

def result_pressure(_class):
    personality = ""
    if _class == 1:
        personality = "Penulis memiliki tingkat emosional yang tinggi, sulit beradaptasi, selalu serius akan segala sesuatu, tegas, dan memiliki keinginan yang kuat"
    elif _class == 2:
        personality = "Penulis memiliki kemampuan untuk mengontrol emosinya dengan baik, nyaman, dan tidak suka memendam kemarahan"
    elif _class == 3:
        personality = "Penulis memiliki kepribadian yang tenang dan santai, lebih sensitif, pengertian, dan sulit mengambil keputusan karena mudah terpengaruh"

    return personality


#A = pd.read_csv('dataset_csv/dataset.csv')
#train_zone(A)

#x = np.array([[37,42,52]])
#print(predict_zone(x))
