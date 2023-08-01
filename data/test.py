
from main import*

if __name__ == "__main__":
    a = 1.2603295448322993e-07
    b = 0.03252493165262612
    c = 1.64710150e-05
    # t, p = solve_ode_pressure(f=ode_model_pressure, t0=1950, t1=2020, dt=0.1, p0=0.05, pars=[a, b, 0.05, c])
    # print(np.interp(1967,t,p))
    PRt, PRr, PR2t, PR2r, WLt, WLr, Tt, Tte = load_data()
    print(np.interp(1967,Tt, Tte))
