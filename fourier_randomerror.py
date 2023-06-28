import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import time
import random

def integrate(f, a, b, n):

    h = (b - a) / n  
    x = np.linspace(a, b, n+1)  
    y = f(x)  
    integral = h * (np.sum(y) - (y[0] + y[-1]) / 2) 
    return integral
    
def coeffs(func, N, st, T):
    l1 = []
    for n in range(N + 1):
        a_n = (2 / T) * integrate(lambda t: func(t) * np.cos(n * (2 * np.pi * (1 / T) * t)), st, st + T,200)
        b_n = (2 / T) * integrate(lambda t: func(t) * np.sin(n * (2 * np.pi * (1 / T) * t)), st, st + T,200)
        l1.append((a_n, b_n))
    return np.array(l1)


def fourier(t, AB_coeffs, T):
    fourier = 0
    a = AB_coeffs[:, 0]
    b = AB_coeffs[:, 1]
    for n in range(len(AB_coeffs)):
        if n > 0:
            fourier += a[n] * np.cos(2 * np.pi * n * t / T) + b[n] * np.sin(2 * np.pi * n * t / T)
        else:
            fourier += a[0] / 2.
    return fourier

def main(a,e):
    acc = a
    T = 5
    a_f = -5
    f = lambda x: (x ** 2 + (a_f) - 11) ** 2 + (x + (a_f) ** 2 - 7) ** 2 + np.random.uniform(-e,e,size=len(x))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Comparison between plots')
    
    xm = np.linspace(-5, 5, 200)
    ym = f(xm)
    ax1.plot(xm, ym)
    ax1.set_title("Original Himmelblau Function at y = 0")
    
    x_range_1 = np.linspace(-4.9, -0.1, 100)
    coeffs_1 = coeffs(f, acc, -5, T)
    y1 = fourier(x_range_1, coeffs_1, T)
    ax2.plot(x_range_1, y1)

    
    x_range_2 = np.linspace(0.1, 4.9, 100)
    coeffs_2 = coeffs(f, acc, 0, T)
    y2 = fourier(x_range_2, coeffs_2, T)
    ax2.plot(x_range_2, y2)
    ax2.set_title("Fourier Regression at y = 0")


    diff = []
    for i in range(100):
        v = ym[i] - y1[i]
        diff.append(v)
    for i in range(100):
        v = ym[100+i] - y2[i]
        diff.append(v)


    plx = np.linspace(-5,5,len(diff))
    ax3.plot(plx,diff)
    ax3.set_title("Variance")
    plt.show()
    
    return np.mean(diff)


    







