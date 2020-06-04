import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as si

cubic = False
extrapolate = False

# maturité du swap : 18/12/2020
# pricé au 23/01/2020

# vol du 23/01/2020 :

vols_input = [
        38.2005,
        34.2896,
        26.9257,
        19.8784,
        16.5607,
        15.0182,
        14.3032,
        13.631,
        13.0096,
        12.451,
        11.57,
        10.771,
        12.2317,
        16.7258,
        21.3792,
    ]

strikes_input = [0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]


spot = 3749
forward  = 3646
T = 330/365
r = -0.00324
div = 3.2271/100


######## Interpolation ########
if extrapolate:
    fill = "extrapolate"
else:
    fill = (vols_input[0], vols_input[-1])

if cubic:
    vol = interp1d([e * spot for e in strikes_input], vols_input, kind='cubic', fill_value=fill, bounds_error=False)
else:
    vol = interp1d([e * spot for e in strikes_input], vols_input, fill_value=fill, bounds_error=False)



######## Intégration de la Nappe ########


def put(k, sigma):
    sigma = sigma/100
    d1 = (np.log(forward / k) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(forward / k) + (- 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (k * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - spot * np.exp(-div * T) * si.norm.cdf(-d1, 0.0, 1.0))

def call(k, sigma):
    sigma = sigma/100
    d1 = (np.log(forward / k) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(forward / k) + (- 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (spot * np.exp(-div * T) * si.norm.cdf(d1, 0.0, 1.0) - k * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))


def f(x):
    if x < np.log(forward):
        return ( put(np.exp(x), vol(np.exp(x))) - put(np.exp(x), vol(forward))) /np.exp(x)
    else:
        return ( call(np.exp(x), vol(np.exp(x))) - call(np.exp(x), vol(forward))) /np.exp(x)


def g(x):
    if x < np.log(forward):
        return put(np.exp(x), vol(np.exp(x))) /np.exp(x)
    else:
        return call(np.exp(x), vol(np.exp(x))) /np.exp(x)


b = np.log(forward) + 10 * vol(forward)/100 * np.sqrt(T)
a = np.log(forward) - 10 * vol(forward)/100 * np.sqrt(T)


Nx = 50
X = np.linspace(a, b, Nx, endpoint=True)




integral = 0
integral_ = 0


for jx in range(Nx-1):
    integral += (f(X[jx]) + f(X[jx+1])) / 2 * (X[jx+1] - X[jx])
    integral_ += (g(X[jx]) + g(X[jx+1])) / 2 * (X[jx+1] - X[jx])


# Avec variable de contrôle
var = (vol(forward)/100)**2 + 2/T*integral
var = np.sqrt(var) * np.exp(-r*T)
print(f"Fair price of the var with control variate, {Nx} steps : {var}")

# Sans
var = (2/T*integral_)
var = np.sqrt(var) * np.exp(-r*T)
print(f"Fair price of the var without control variate, {Nx} steps : {var}")




