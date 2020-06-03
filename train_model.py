import numpy as np
import pandas as pd

A = pd.read_csv('sample_data.csv') # A = matriks dataset

alpha = 5
e = []
u = []
m = 0 #baris matriks/jumlah data
n = 0 #kolom matriks/jumlah fitur
gamma = 0

percentage = 10 #persentase subset matriks yang akan digunakan

def get_subset_matrix(A, percent):
    count = int(len(A) * percent / 100)
    return A.sample(count)

subset_A = get_subset_matrix(A, percentage)
subset_A_pressure = subset_A[['Rerata', 'Persentase', 'Tekanan Tulisan']]
subset_A_zone = subset_A[['Zona Atas', 'Zona Tengah', 'Zona Bawah', 'Dominasi Zona']]

def smoothing_function(x):
    return x + 1/alpha * np.log(1 + np.exp(-alpha * x))

def get_next_u(current_u, step, step_direction):
    return current_u + step * step_direction[0:n]

def get_next_gamma(current_gamma, step, step_direction):
    return current_gamma + step * step_direction[n, 0]

def kernel_trick(A, subset_A):
    return A.transpose()*A

def plus_function(e, D, subset_D, A, subset_A, u, gamma):
    return e-D*(kernel_trick(A, subset_A)*subset_D*u-e*gamma)

def get_gradient(w, A, D, e, gamma, plus_function):
    v = 5
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
    v = 5
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

    #ukuran e ganti full
    e = np.mat(np.reshape([[1 for x in range(m)] for y in range(1)],(m,1)))
    u = np.mat(np.reshape(1/8 * e[:m],(m,1)))
    gamma = 0
    
    A_value = np.mat(np.reshape(A.values[:,range(3)],(_m,_n-1)))
    subset_A_value = np.mat(np.reshape(subset_A.values[:,range(3)],(m,n-1)))
    
    #subset_D ganti dengan D
    plus = plus_function(e, subset_D, subset_D, A_value, subset_A_value, u, gamma)
    #subset A ganti dengan A
    gradient_matrix = get_gradient(u, subset_A_value, subset_D, e, gamma, plus)
    gradient = np.transpose(gradient_matrix) * gradient_matrix
    
    if(gradient > 0):
        print(calculate_hessian(plus, subset_A_value, e))




train_zone(A)
