import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from newfunctions import*
from numpy.linalg import norm

tol = 1.e-6

ts,ps = solve_ode_pressure(f= ode_model_pressure, t0 = 0,t1= 1,dt= 0.1,p0=  2,pars = [0.01,0.03,0.05,0.01])
Psol = 1.818801157
print(ps[1])
assert norm(ps[1]-Psol) < tol

ts, Ts= solve_ode_temperature(f= ode_model_temperature, t0=0, t1=1, dt=0.1, p0=2, pars=[0.01,0.03,0.05,0.01])
Tsol = 1
assert norm(Ts - Tsol) < tol
