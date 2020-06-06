import numpy as np
import pandas as pd
import scipy.linalg as la
import math
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel

A = pd.read_csv('sample_data.csv') # A = matriks dataset

alpha = 5
e = []
u = []
m = 0 #baris matriks/jumlah data
n = 0 #kolom matriks/jumlah fitur
gamma = 0
v = 2

percentage = 10 #persentase subset matriks yang akan digunakan

def get_subset_matrix(A, percent):
    count = int(len(A) * percent / 100)
    return A.sample(count)

subset_A = get_subset_matrix(A, percentage)
subset_A_pressure = subset_A[['Rerata', 'Persentase', 'Tekanan Tulisan']]
subset_A_zone = subset_A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

def get_next_u(current_u, step, step_direction):
    return current_u + step * step_direction[0:n]

def get_next_gamma(current_gamma, step, step_direction):
    return current_gamma + step * step_direction[n, 0]

def smoothing_function(x):
    x = np.array(x, dtype=np.float)
    return x + 1/alpha * np.log(1 + expit(-alpha * x))

def objective_function(e, D, subset_D, A, subset_A, u, gamma):
    plus = plus_function(e, D, subset_D, A, subset_A, u, gamma)
    p = smoothing_function(plus)
    return v/2 * math.pow(np.linalg.norm(p), 2) + 1/2 * ((np.transpose(u) * u)[0, 0] + math.pow(gamma, 2))

def kernel_trick(A, subset_A):
    return rbf_kernel(A, subset_A.T, gamma = 2)

def plus_function(e, D, subset_D, A, subset_A, u, gamma):
    return e-D*(kernel_trick(A, subset_A)*subset_D*u-e*gamma)

def get_gradient(w, A, D, e, gamma, plus_function):
    return np.vstack((
        w - v * np.transpose(A) * D * plus_function,
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
    
    gradient_matrix = get_gradient(u, A_value, D, e, gamma, plus)
    gradient = np.transpose(gradient_matrix) * gradient_matrix

    if(gradient > 0):
        hessian = calculate_hessian(plus, A_value, e)
        hessian_lu = la.lu(hessian)[0]
        
        di = np.linalg.inv(hessian_lu)*(-1*gradient_matrix)
        obj = objective_function(e, D, subset_D, A_value, subset_A_value, u, gamma)
        
        x = np.array([[60,20,20]])
        g = kernel_trick(x, subset_A_value)*subset_D*di[:-1]-di[-1]
        
        #predict = np.sign(g)
        #print(class_label_subset, predict)
        
    
train_zone(A)