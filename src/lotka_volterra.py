import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

alpha = 1. #mortality rate due to predators
beta = 0.5
delta = 1.
gamma = 0.5
x0 = 1.
y0 = 3.

def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

Nt = 1000
tmax = 15.
t = np.linspace(0.,tmax, Nt)
X0 = [x0, y0]
res = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
x, y = res.T

plt.figure()
plt.grid()
plt.title("Lotka-Volterra concentration")
plt.plot(t, x, 'xb', label = 'Deer', color = "#0073E6", alpha = 0.5)
plt.plot(t, y, '+r', label = "Wolves", color = "#B51963", alpha = 0.5)
plt.xlabel('Time t, [days]')
plt.ylabel('Concentration')
plt.legend()

plt.savefig("visualizations/lv_intervention.pdf", format = "pdf", bbox_inches = "tight")

plt.show()