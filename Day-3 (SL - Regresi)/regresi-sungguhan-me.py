import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_csv("https://drive.google.com/uc?id=1AN-f6TzO_QxOiFGFI9FYpQauecp7I0mb", header=None)

x = data[[0, 1]].to_numpy()
y = data[2].to_numpy()

def cost(y,pred):
  return ((pred-y)**2).sum(axis=0) / (2*y.shape[0])

def predict(w, x):
    return w.T @ x

def add_bias(x):
    bias = np.ones((x.shape[0]))
    return np.c_[bias, x]

def update_bobot(w, xb, y, alpha):
    output = predict(w, xb)
    error = output - y
    m = y.shape[0]
    # Karena xb sekarang (3, 47) dan error (47,), 
    # maka gunakan xb @ error -> (3, 47) @ (47,) -> (3,)
    gradient = (xb @ error) / m
    w = w - alpha * gradient
    return w

# Inisialisasi
w = np.array([0.5, 1.0, 0.03])
alpha = 0.01
iterasi = 1
xb = add_bias(x)
xb = xb.T

print(xb)
print(y)

print(iterasi)
output = predict(w, xb)
print("cost sebelumnya = ", cost(y, output))
w = update_bobot(w, xb, y, alpha)
output = predict(w, xb)
print("cost setelahnya = ", cost(y, output))
iterasi += 1