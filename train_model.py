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
percentage = 10 #persentase subset matriks yang akan digunakan

def get_subset_matrix(A, percent):
    count = int(len(A) * percent / 100)
    return A.sample(count)
    #return A.iloc[[0,1,5]]

subset_A = get_subset_matrix(A, percentage)
subset_A_pressure = subset_A[['Rerata', 'Persentase', 'Tekanan Tulisan']]
subset_A_zone = subset_A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

def get_next_u(current_u, step, step_direction, n):
    return current_u + step * step_direction[0:n-1]

def get_next_gamma(current_gamma, step, step_direction, n):
    return current_gamma + step * step_direction[n-1, 0]

def smoothing_function(x):
    x = np.array(x, dtype=np.float)
    return x + 1/alpha * np.log(1 + expit(-alpha * x))

def objective_function(e, D, subset_D, A, subset_A, u, gamma):
    plus = plus_function(e, D, subset_D, A, subset_A, u, gamma)
    p = smoothing_function(plus)
    return v/2 * math.pow(np.linalg.norm(p), 2) + 1/2 * ((np.transpose(u) * u)[0, 0] + math.pow(gamma, 2))

def kernel_trick(A, subset_A):
    return rbf_kernel(A, subset_A, gamma = 2)

def plus_function(e, D, subset_D, A, subset_A, u, gamma):
    return e-D*(kernel_trick(A, subset_A)*subset_D*u-e*gamma)

def get_gradient(u, A, D, e, gamma, plus_function):
    return np.vstack((
        u - v * np.transpose(A) * D * plus_function,
        gamma + v * np.transpose(e) * D * plus_function
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

def train_zone(A):
    A = A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

    subset_A = get_subset_matrix(A, percentage)
    subset_A_zone = subset_A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

    _m, _n = A.shape
    m, n = subset_A_zone.shape

    class_label_full = []
    for i in range(_m):
        if A.iloc[i]['Dominasi Zona'] == 'Atas':
            class_label_full += [1]
        else:
            class_label_full += [-1]

    class_label_subset = []
    for i in range(m):
        if subset_A_zone.iloc[i]['Dominasi Zona'] == 'Atas':
            class_label_subset += [1]
        else:
            class_label_subset += [-1]

    D = np.mat(np.reshape(np.diag(class_label_full),(_m,_m)))
    subset_D = np.mat(np.reshape(np.diag(class_label_subset),(m,m)))

    e = np.mat(np.reshape([[1 for x in range(_m)] for y in range(1)],(_m,1)))
    u = np.mat(np.reshape(1/8 * e[:m],(m,1)))
    gamma = 0
    
    A_value = np.mat(np.reshape(A.values[:,range(3)],(_m,_n-1)))
    subset_A_value = np.mat(np.reshape(subset_A.values[:,range(3)],(m,n-1)))

    plus = plus_function(e, D, subset_D, A_value, subset_A_value, u, gamma)
    plus = (plus < 0).choose(plus, 0)
    
    gradient_matrix = get_gradient(u, A_value, D, e, gamma, plus)
    gradient = (np.transpose(gradient_matrix) * gradient_matrix)[0,0]

    convergence = 0.01
    while gradient > convergence:
        hessian = calculate_hessian(plus, A_value, e)
        hessian = np.mat(hessian, dtype=float)
        
        step_direction = np.linalg.inv(hessian)*(-1*gradient_matrix)
        step_gap = np.transpose(step_direction) * gradient_matrix
        step = 1
        
        objective = objective_function(e, D, subset_D, A_value, subset_A_value, u, gamma)

        next_u = get_next_u(u, step, step_direction, n)
        next_gamma = get_next_gamma(gamma, step, step_direction, n)
        next_objective = objective_function(e, D, subset_D, A_value, subset_A_value, next_u, next_gamma)

        objective_difference = objective - next_objective
        
        while objective_difference < -0.05 * step * step_gap:
            step *= 0.5
            next_u = get_next_u(u, step, step_direction, n)
            next_gamma = get_next_gamma(gamma, step, step_direction, n)
            next_objective = objective_function(e, D, subset_D, A_value, subset_A_value, next_u, next_gamma)
            objective_difference = objective - next_objective
        
        gradient = step * (np.transpose(gradient_matrix) * gradient_matrix)[0, 0]
        
    return [subset_A_value, subset_D, next_u, next_gamma]

def predict(x, subset_A, subset_D, u, gamma):
    return np.sign(kernel_trick(x, subset_A)*subset_D*u-gamma)

def result(_class):
    personality = ""
    if _class == 1:
        personality = "Lebih memperhatikan aspek spiritual, impian, harapan, dan ambisi dalam hidupnya. Penulisa lebih suka melakukan kegiatan berpikir dan memikirkan masa depannya"
    else:
        personality = "Lebih mementingkan aspek fisik kehidupan dan lebih mengandalkan ototnya daripada otaknya"

    return personality
    
train_zone(A)