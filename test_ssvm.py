import numpy as np
import pandas as pd
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances

A = pd.read_csv('sample_data.csv') # A = matriks dataset

alpha = 5
v = 2

def getnext_w(current_w, step, step_direction):
    return current_w + step * step_direction[0:-1]

def getnext_gamma(current_gamma, step, step_direction):
    return current_gamma + step * step_direction[-1]

def smoothing_function(x):
    x = np.array(x, dtype=np.float)
    return x + 1/alpha * np.log(1 + expit(-alpha * x))

def objective_function(e, D, A, w, gamma):
    plus = plus_function(e, D, A, w, gamma)
    p = smoothing_function(plus)
    return 0.5 * (v*math.pow(np.linalg.norm(p), 2) + 1/2 * ((np.transpose(w) * w)[0, 0] + math.pow(gamma, 2)))

def kernel_trick(A, subset_A):
    return rbf_kernel(A, subset_A, gamma = 2)

def plus_function(e, D, A, w, gamma):
    return e-D*(A*w-e*gamma)

def get_gradient(w, A, D, e, gamma, plus_function):
    return np.vstack((
        w - v * np.transpose(D * A) * plus_function,
        gamma + v * np.transpose(D * e) * plus_function
    ))

def calculate_hessian(plus_function, A, e):
    s = 0.5 * np.sign(plus_function)

    diag_s = np.identity(s.shape[0])
    
    h11 = np.transpose(A) * diag_s * A
    h12 = -np.transpose(A) * diag_s * e
    h21 = -np.transpose(e) * diag_s * A
    h22 = np.transpose(e) * diag_s * e
    
    h = np.vstack((
            np.hstack((h11,h12)),
            np.hstack((h21,h22))
        ))
    I = np.identity(h.shape[0])
    
    return I+v*h

def get_next_armijo_step(e, D, A_value, w, gamma, step_direction, step_gap):
    step = 1
    objective = objective_function(e, D, A_value, w, gamma)
        
    next_w = getnext_w(w, step, step_direction)
    next_gamma = getnext_gamma(gamma, step, step_direction)
    next_objective = objective_function(e, D, A_value, next_w, next_gamma)
    objective_difference = objective - next_objective

    while objective_difference < -0.05 * step * step_gap:
        step *= 0.5
        next_w = getnext_w(w, step, step_direction)
        next_gamma = getnext_gamma(gamma, step, step_direction)
        next_objective = objective_function(e, D, A_value, next_w, next_gamma)
        objective_difference = objective - next_objective

    return step

def train_zone(A, class1, class2):
    A = A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]
    A = A.loc[(A['Dominasi Zona'] == class1) | (A['Dominasi Zona'] == class2)]
    
    m, n = A.shape

    class_label_full = []
    for i in range(m):
        if A.iloc[i]['Dominasi Zona'] == class1:
            class_label_full += [1]
        else:
            class_label_full += [-1]

    D = np.mat(np.reshape(np.diag(class_label_full),(m,m)))

    e = np.mat(np.reshape([[1 for x in range(m)] for y in range(1)],(m,1)))
    w = np.mat(np.zeros((n-1, 1)))
    gamma = 0
    A_value = np.mat(np.reshape(A.values[:,range(3)],(m,n-1)))

    plus = plus_function(e, D, A_value, w, gamma)
    plus = (plus < 0).choose(plus, 0)
    
    gradientmatrix = get_gradient(w, A_value, D, e, gamma, plus)
    gradient = (np.transpose(gradientmatrix) * gradientmatrix)[0,0]

    hessian = calculate_hessian(plus, A_value, e)
    hessian = np.mat(hessian, dtype=float)
        
    step_direction = np.linalg.inv(hessian)*(-1*gradientmatrix)
    step_gap = np.transpose(step_direction) * gradientmatrix
    step = get_next_armijo_step(e, D, A_value, w, gamma, step_direction, step_gap)

    convergence = 0.01
    while gradient >= convergence:
        w = getnext_w(w, step, step_direction)
        gamma = getnext_gamma(gamma, step, step_direction)

        hessian = calculate_hessian(plus, A_value, e)
        hessian = np.mat(hessian, dtype=float)
        
        gradientmatrix = get_gradient(w, A_value, D, e, gamma, plus)

        step_direction = np.linalg.inv(hessian)*(-1*gradientmatrix)
        step_gap = np.transpose(step_direction) * gradientmatrix
        step = get_next_armijo_step(e, D, A_value, w, gamma, step_direction, step_gap)

        gradient = step * (np.transpose(gradientmatrix) * gradientmatrix)[0, 0]

    return [w, gamma]

def classifier():
    zone_class = ["Atas", "Tengah", "Bawah"]
    pressure_class = ["Kuat, Sedang", "Ringan"]

    zonemodel = []

    zonemodel = zonemodel + [train_zone(A, zone_class[0], zone_class[1])]
    zonemodel = zonemodel + [train_zone(A, zone_class[0], zone_class[2])]
    zonemodel = zonemodel + [train_zone(A, zone_class[1], zone_class[2])]

    x = [25,23,81]

    pred = []
    for i in range(3):
        pred = pred + [predict(x, zonemodel[i][0], zonemodel[i][1])]

    print(pred)
1
def predict(x, w, gamma):
    return np.array(np.sign(np.transpose(x)*w-gamma))[0][0]

def result_zone(_class):
    personality = ""
    if _class == "Atas":
        personality = "Penulis lebih memperhatikan aspek spiritual, impian, harapan, dan ambisi dalam hidupnya. Penulis lebih suka melakukan kegiatan berpikir dan memikirkan masa depannya"
    elif _class == "Tengah":
        personality = "Penulis lebih mementingkan kehidupannya saat ini dan sulit untuk membuat rencana jangka panjang mereka"
    else:
        personality = "Penulis lebih mementingkan aspek fisik kehidupan dan lebih mengandalkan ototnya daripada otaknya"

    return personality

def result_pressure(_class):
    personality = ""
    if _class == "Kuat":
        personality = "Penulis memiliki tingkat emosional yang tinggi, sulit beradaptasi, selalu serius akan segala sesuatu, tegas, dan memiliki keinginan yang kuat"
    elif _class == "Sedang":
        personality = "Penulis memiliki kemampuan untuk mengontrol emosinya dengan baik, nyaman, dan tidak suka memendam kemarahan"
    else:
        personality = "Penulis memiliki kepribadian yang tenang dan santai, lebih sensitif, pengertian, dan sulit mengambil keputusan karena mudah terpengaruh"

    return personality
    
x = train_zone(A, "Atas", "Bawah")
print(x)

classifier()