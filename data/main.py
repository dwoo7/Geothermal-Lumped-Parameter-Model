import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from statistics import stdev


def load_data():
    """ Return the given data.

            Parameters:
            -----------

            Returns:
            --------
            PRt, PRr, PR2t, PR2r, WLt, WLr, Tt, Tte : float array
                Arrays containing given data

            Notes:
            ------
            None
        """
    # import data sets
    PRt, PRr = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=True).T
    PR2t, PR2r = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=True).T
    WLt, WLr = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=True).T
    Tt, Tte = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=True).T

    return PRt, PRr, PR2t, PR2r, WLt, WLr, Tt, Tte


def pressure(WLr):
    """ Return the data values for pressure
        Parameters:
        -----------
        WLr,WLt : float array
                  given water level data
        Returns:
        --------
        P : float array
            Pressure evaluated 
    """

    # convert water level to pressure array
    # range = max(WLr) - min(WLr)
    # shrink = 0.31 / range
    # P = (WLr * shrink)
    # P = P - max(P) + 0.05

    reference_point = WLr[len(WLr) - 1]
    P = np.zeros(len(WLr))
    for i in range(len(WLr)):
        height = WLr[i] - reference_point
        P[i] = 930 + (930 * 9.81 * height)  # bernoulli's eqn
        P[i] = P[i] * 10 ** -5

    return P


def find_q(t):
    """ Return the values for q

        Parameters:
        -----------
        t : float
            Independent variable.

        Returns:
        --------
        qi : float
             q values interpolated at t
        dqdti : float
                dqdt interpolated at t
    """

    # read values
    tq, q = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=True).T

    # calculate dqdt at all points in array using differences
    dqdt = 0. * q  # allocate derivative vector
    dqdt[1:-1] = (q[2:] - q[:-2]) / (tq[2:] - tq[:-2])  # central differences
    dqdt[0] = (q[1] - q[0]) / (tq[1] - tq[0])  # forward difference
    dqdt[-1] = (q[-1] - q[-2]) / (tq[-1] - tq[-2])  # backward difference

    # interpolate at t
    qi = np.interp(t, tq, q)
    dqdti = np.interp(t, tq, dqdt)

    return qi, dqdti


def find_q_predict(t, q_const):
    """ Return the values for q with a constant production rate (predictions)

        Parameters:
        -----------
        t : float
            Independent variable.
        q_const : float
            Constant extraction rate used in prediction.

        Returns:
        --------
        dqdti : float
                dqdt interpolated at t
    """

    # read values
    tq, q = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=True).T

    # append prediction values to data
    q = np.append(q, q_const)
    tq = np.append(tq, t)

    # calculate dqdt at all points in array using differences
    dqdt = 0. * q  # allocate derivative vector
    dqdt[1:-1] = (q[2:] - q[:-2]) / (tq[2:] - tq[:-2])  # central differences
    dqdt[0] = (q[1] - q[0]) / (tq[1] - tq[0])  # forward difference
    dqdt[-1] = (q[-1] - q[-2]) / (tq[-1] - tq[-2])  # backward difference

    # interpolate at t
    dqdti = np.interp(t, tq, dqdt)

    return dqdti


def ode_model_pressure(t, p, a, b, p0, c):
    """ Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        c : float
            Slow drainage parameter.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dqdt : float
            Derivative of dependent variable with respect to independent variable.
    """

    # find q and dqdt at given time
    q, dqdt = find_q(t)

    return -a * q - b * (p - p0) - c * dqdt


def ode_model_pressure_q(t, p, a, b, p0, c, q):
    """ Return the derivative dP/dt at time, t, for given parameters.
        Used for predictions with constant q

        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        c : float
            Slow drainage parameter.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.
        q : float
            Constant production rate.

        Returns:
        --------
        dqdt : float
            Derivative of dependent variable with respect to independent variable.
    """

    # find q and dqdt at given time
    if t <= 2021:
        q, dqdt = find_q(t)
    else:
        q = q
        dqdt = find_q_predict(t, q)

    return -a * q - b * (p - p0) - c * dqdt



def solve_ode_pressure(f, t0, t1, dt, p0, pars):
    """ Solve pressure ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        p0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        ps : array-like
            Dependent variable solution vector.
    """
    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    ps = 0. * ts  # array to store solution
    ps[0] = p0  # set initial value

    # use improved euler to solve
    for k in range(nt):
        f_0 = f(ts[k], ps[k], *pars)
        f_1 = f(ts[k] + dt, ps[k] + dt * f_0, *pars)
        ps[k + 1] = ps[k] + dt * 0.5 * (f_0 + f_1)

    return ts, ps


def ode_model_temperature(t, T, p, a, b, p0, T0, at, bt):
    """ Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        p : float
            Dependent variable. (From pressure ODE)
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable. (For pressure ODE)
        T0 : float
            Ambient value of dependent variable.
        at : float
            Source/sink strength parameter. (cold water inflow)
        bt : float
            Recharge strength parameter. (conduction)

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.
    """

    # Temperature depends on direction of flow
    if p > p0:
        td = T
    else:
        td = 30

    return -at * (b / a) * (p - p0) * (td - T) - bt * (T - T0)


def solve_ode_temperature(f, t0, t1, dt, T0, pars):
    """ Solve temperature ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        T0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        Ts : array-like
            Dependent variable solution vector.
    """
    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    Ts = 0. * ts  # array to store solution
    Ts[0] = T0  # set initial value

    # get values of p
    a = 1.2603295448322993e-07
    b = 0.03252493165262612
    c = 1.64710150e-05
    t, p = solve_ode_pressure(f=ode_model_pressure, t0=1950, t1=2050, dt=0.1, p0=0.05, pars=[a, b, 0.05, c])

    # use improved euler to solve
    for k in range(nt):
        f_0 = f(ts[k], Ts[k], p[k], *pars)
        f_1 = f(ts[k] + dt, Ts[k] + dt * f_0, p[k], *pars)
        Ts[k + 1] = Ts[k] + dt * 0.5 * (f_0 + f_1)

    return ts, Ts


def solve_ode_temperature_predict(f, t0, t1, dt, T0, pars, q):
    """ Solve temperature ODE numerically.
        Used for predictions with constant production rate.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        T0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.
        q : float
            Constant extraction rate for prediction.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        Ts : array-like
            Dependent variable solution vector.
    """
    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    Ts = 0. * ts  # array to store solution
    Ts[0] = T0  # set initial value

    # get values of p
    t, p = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                              pars=[pars[0], pars[1], 0.05, c, q])

    # use improved euler to solve
    for k in range(nt):
        f_0 = f(ts[k], Ts[k], p[k], *pars)
        f_1 = f(ts[k] + dt, Ts[k] + dt * f_0, p[k], *pars)
        Ts[k + 1] = Ts[k] + dt * 0.5 * (f_0 + f_1)

    return ts, Ts


def fit_pressure(t, a, b, c):
    """ Fit parameters to pressure ODE .

            Parameters:
            -----------
            t : array-like
                Array of Independent variables
            a : float
                parameter for pressure ODE to be calibrated
            b : float
                parameter for pressure ODE to be calibrated
            c : float
                parameter for pressure ODE to be calibrated

            Returns:
            --------
            p : array-like
                Pressure variable solution vector.
    """

    # solve for pressure using ODE solver
    tx, p = solve_ode_pressure(f=ode_model_pressure, t0=t[0], t1=t[-1], dt=0.1, p0=0.05, pars=[a, b, 0.05, c])

    return p


def fit_temperature(t, at, bt):
    """ Fit parameters to pressure ODE .

                Parameters:
                -----------
                t : array-like
                    Array of Independent variables
                at : float
                    parameter for temperature ODE to be calibrated
                bt : float
                    parameter for temperature ODE to be calibrated

                Returns:
                --------
                X : array-like
                    Temperature variable solution vector.
    """

    # hard coded values fitted from pressure ODE

    # solve for temperature using ODE solver
    a = 1.2603295448322993e-07
    b = 0.03252493165262612
    T, X = solve_ode_temperature(f=ode_model_temperature, t0=1950, t1=2020, dt=0.1, T0=148,
                                 pars=[a, b, 0.05, 149., at, bt])

    return X


def ode_model_pressure_benchmark(t, p, a, b, p0, c):
    """ Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        c : float
            Slow drainage parameter.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dqdt : float
            Derivative of dependent variable with respect to independent variable.
    """

    # constant q and dqdt for benchmarking
    q = 1
    dqdt = 0

    return -a * q - b * (p - p0) - c * dqdt


def plot_benchmark_pressure():
    """ Compare analytical and numerical solutions.
        Parameters:
        -----------
        none
        Returns:
        --------
        none
        Notes:
        ------
        This function called within if __name__ == "__main__":
        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.

    """
    # initialise variables
    a = 1
    b = 1
    c = 0
    q = 1
    p0 = 0

    tA = np.linspace(0, 10, 100)
    xA = np.zeros(100)

    x_Error = np.zeros(len(xA))
    inverse_stepsize = np.linspace(1, 5, 21)
    x_Convergence = np.zeros(len(inverse_stepsize))

    # solve numerically
    tN, pN = solve_ode_pressure(f=ode_model_pressure_benchmark, t0=0, t1=10, dt=0.1, p0=0, pars=[a, b, 0, c])

    # solve analytically and find error
    for i in range(len(tA)):
        xA[i] = -((a * q) / b) * (1 - math.exp(-tA[i])) + p0
        x_Error[i] = abs(pN[i] - xA[i])

    # timestep convergence
    for i in range(0, len(inverse_stepsize)):
        tConv, xConv = solve_ode_pressure(f=ode_model_pressure_benchmark, t0=0, t1=10, dt=(inverse_stepsize[i]) ** (-1),
                                          p0=0, pars=[a, b, 0, c])
        x_Convergence[i] = xConv[-1]

    # plot solutions
    plt.subplot(1, 3, 1)
    plt.plot(tA, xA, 'b--', label='analytical solution')
    plt.plot(tN, pN, 'ro', label='numerical solution')
    plt.title('Pressure Bench')
    plt.xlabel('t')
    plt.ylabel('X')
    plt.legend()

    # plot error
    plt.subplot(1, 3, 2)
    plt.plot(tA, x_Error, 'k-')
    plt.title('Error Analysis')
    plt.xlabel('t')
    plt.ylabel('Relative Error Against Benchmark')

    # plot convergence
    plt.subplot(1, 3, 3)
    plt.plot(inverse_stepsize, x_Convergence, 'bx')
    plt.title('Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('X(t=2020)')

    plt.show()
    plt.savefig('Benchmark for pressure', dpi=600)


def ode_model_temperature_benchmark(t, T, p, a, b, p0, T0, at, bt):
    """ Return the derivative dT/dt at time, t, for given benchmark parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        p : float
            Dependent variable. (From pressure ODE)
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable. (For pressure ODE)
        T0 : float
            Ambient value of dependent variable.
        at : float
            Source/sink strength parameter. (cold water inflow)
        bt : float
            Recharge strength parameter. (conduction)

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.
    """

    # values for benchmarking
    if p > p0:
        td = T
    else:
        td = 3

    return -at * (b / a) * (p - p0) * (td - T) - bt * (T - T0)


def solve_ode_temperature_benchmark(f, t0, t1, dt, T0, pars):
    """ Solve temperature ODE numerically for benchmark.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        T0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        Ts : array-like
            Dependent variable solution vector.
    """
    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    Ts = 0. * ts  # array to store solution
    Ts[0] = T0  # set initial value

    # get values of p from benchmark pressure ode
    t, p = solve_ode_pressure(f=ode_model_pressure_benchmark, t0=0, t1=10, dt=0.1, p0=3, pars=[1, 0.5, 3, 0])

    # use improved euler to solve
    for k in range(nt):
        f_0 = f(ts[k], Ts[k], p[k], *pars)
        f_1 = f(ts[k] + dt, Ts[k] + dt * f_0, p[k], *pars)
        Ts[k + 1] = Ts[k] + dt * 0.5 * (f_0 + f_1)

    return ts, Ts


def plot_benchmark_temperature():
    """ Compare analytical and numerical solutions.
        Parameters:
        -----------
        none
        Returns:
        --------
        none
        Notes:
        ------
        This function called within if __name__ == "__main__":
        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.

    """
    # initialise variables
    tA = np.linspace(0, 10, 100)
    T = np.zeros(100)
    a = 1
    b = 0.5
    at = 1
    bt = 0
    q0 = 1
    Tc = 3
    p0 = 3
    T0 = 5

    x_Error = np.zeros(len(tA))
    inverse_stepsize = np.linspace(1, 5, 21)
    x_Convergence = np.zeros(len(inverse_stepsize))

    # find numerical solution
    tN, TN = solve_ode_temperature_benchmark(f=ode_model_temperature_benchmark, t0=0, t1=10, dt=0.1, T0=5,
                                             pars=[a, b, p0, 5, at, bt])

    # find analytical solution and error
    for i in range(len(tA)):
        T[i] = Tc + (T0 - Tc) * math.exp(-((at * q0) / b) * ((math.exp(-b * tA[i])) + b * tA[i] - 1))
        x_Error[i] = abs(TN[i] - T[i])

    # find timestep convergence
    for i in range(0, len(inverse_stepsize)):
        tConv, xConv = solve_ode_temperature_benchmark(f=ode_model_temperature_benchmark, t0=0, t1=10,
                                                       dt=(inverse_stepsize[i]) ** (-1), T0=5,
                                                       pars=[a, b, p0, 5, at, bt])
        x_Convergence[i] = xConv[-1]

    # plot solutions
    plt.subplot(1, 3, 1)
    plt.plot(tA, T, 'b--', label='analytical solution')
    plt.plot(tN, TN, 'ro', label='numerical solution')
    plt.legend()
    plt.title('Temperature Bench')
    plt.xlabel('t')
    plt.ylabel('X')

    # plot error
    plt.subplot(1, 3, 2)
    plt.plot(tA, x_Error, 'k-')
    plt.title('Error Analysis')
    plt.xlabel('t')
    plt.ylabel('Relative Error Against Benchmark')

    # plot timestep convergence
    plt.subplot(1, 3, 3)
    plt.plot(inverse_stepsize, x_Convergence, 'bx')
    plt.title('Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('X(t=2020)')

    plt.show()
    plt.savefig('Benchmark for temperatrue', dpi=600)


if __name__ == "__main__":

    # load data
    PRt, PRr, PR2t, PR2r, WLt, WLr, Tt, Tte = load_data()
    P = pressure(WLr)

    # set-up for euler ODE solve
    nt = int(np.ceil((2020 - 1950) / 0.1))  # compute number of Euler steps to take
    t = 1950 + np.arange(nt + 1) * 0.1  # x array

    # interpolate pressure, temperature
    pi = np.interp(t, WLt, P)
    Ti = np.interp(t, Tt, Tte)

    # print('Calculating and Plotting ODEs')

    # fit parameters and construct covariance matrix
    sigma = [1] * len(pi)
    pc, cov = curve_fit(fit_pressure, t, pi, p0=[1.91766614e-07, 3.73582596e-02, 1.75772607e-05], sigma=sigma)
    Tc, covT = curve_fit(f=fit_temperature, xdata=t, ydata=Ti, p0=[0.000002, 1], sigma = sigma)
    print('Parameters for pressure ODE (a, b, c): ' + str(pc))
    print('Parameters for temperature ODE (at, bt): ' + str(Tc))
    a = pc[0]
    b = pc[1]
    c = pc[2]
    at = Tc[0]
    bt = Tc[1]

    # booleans to indicate which code to run
    plot_model = False
    plot_benchmarks = False
    plot_future = True
    plot_uncertainty = True

    if plot_model is True:
        # set up plot
        f, ax1 = plt.subplots(nrows=1, ncols=1)
        ax2 = ax1.twinx()
        ax1.set_ylabel('pressure (Bars)')
        ax1.set_xlabel('time (t)')
        ax2.set_ylabel('temperature (degC)')

        # Pressure
        ax1.plot(WLt, P, 'k.', label='pressure data')
        t, p = solve_ode_pressure(f=ode_model_pressure, t0=1950, t1=2020, dt=0.1, p0=0.05, pars=[a, b, 0.05, c])
        ax1.plot(t, p, 'k-', label='pressure best fit')

        # Temperature
        T, X = solve_ode_temperature(f=ode_model_temperature, t0=1950, t1=2020, dt=0.1, T0=149,
                                     pars=[a, b, 0.05, 149, at, bt])
        ax2.plot(Tt, Tte, 'r.', label='temperature data')
        ax2.plot(T, X, 'r-', label='temperature best fit')

        ax1.legend(loc=3)
        ax2.legend(loc=4)
        ax2.set_title('ODE Model Best Fits vs Data')

        plt.show()
        plt.savefig('LPM best fit', dpi=600)

        # plot misfit
        f, (ax3, ax4) = plt.subplots(1, 2)

        # pressure
        pressure_misfit = P - np.interp(WLt, t, p)
        ax3.plot(WLt, pressure_misfit, 'rx')
        ax3.plot(WLt, np.zeros(len(WLt)), 'k.')
        ax3.set_title("Pressure Misfit")
        ax3.set_ylabel('pressure (bars)')
        ax3.set_xlabel('time (t)')

        # temperature
        temp_misfit = Tte - np.interp(Tt, T, X)
        ax4.plot(Tt, temp_misfit, 'rx')
        ax4.plot(Tt, np.zeros(len(Tt)), 'k.')
        ax4.set_title("Temperature Misfit")
        ax4.set_ylabel('temperature (degC)')
        ax4.set_xlabel('time (t)')
        plt.show()
        plt.savefig('misfit graphs', dpi=600)

    if plot_benchmarks is True:
        plot_benchmark_pressure()
        plot_benchmark_temperature()

    if plot_future is True:
        # create samples for uncertainty
        ps = np.random.multivariate_normal(pc, cov, 100)  # samples from posterior

        # future predictions (Pressure)
        f2, ax3 = plt.subplots(nrows=1, ncols=1)

        # current observed data
        ax3.plot(WLt, P, 'k.', label='pressure data')
        v = stdev(P) # standard deviation
        ax3.errorbar(WLt, P, yerr = v, fmt = '.r')

        # half the current extraction rate at 4,500 tonnes per day
        t1, p1 = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                    pars=[a, b, 0.05, c, 4500])
        ax3.plot(t1, p1, 'g-', label='half current extraction')

        values = []
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                             pars=[pi[0], pi[1], 0.05, pi[2], 4500])
                ax3.plot(t1[710:], p_co[710:], 'g-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Pressure haf extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax3.plot([], [], 'g-', lw=0.5)

        # zero extraction rate at 0 tonnes per day
        t2, p2 = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                    pars=[a, b, 0.05, c, 0])
        ax3.plot(t2, p2, 'r-', label='zero extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                             pars=[pi[0], pi[1], 0.05, pi[2], 0])
                ax3.plot(t1[710:], p_co[710:], 'r-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Pressure zero extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax3.plot([], [], 'r-', lw=0.5)

        # double the current extraction rate at 18,000 tonnes per day
        t3, p3 = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                    pars=[a, b, 0.05, c, 18000])
        ax3.plot(t3, p3, 'b-', label='double current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                             pars=[pi[0], pi[1], 0.05, pi[2], 18000])
                ax3.plot(t1[710:], p_co[710:], 'b-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Pressure double extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax3.plot([], [], 'b-', lw=0.5)

        # triple the current extraction rate at 27,000 tonnes per day
        t4, p4 = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                    pars=[a, b, 0.05, c, 27000])
        ax3.plot(t4, p4, 'm-', label='triple current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_pressure(f=ode_model_pressure_q, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                             pars=[pi[0], pi[1], 0.05, pi[2], 27000])
                ax3.plot(t1[710:], p_co[710:], 'm-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Pressure triple extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax3.plot([], [], 'm-', lw=0.5)

        # The current extraction rate at around 9000 tonnes per day
        t, p = solve_ode_pressure(f=ode_model_pressure, t0=1950, t1=2050, dt=0.1, p0=0.05, pars=[a, b, 0.05, c])
        ax3.plot(t, p, 'k-', label='current extraction')

        # plot threshold reference
        healthyPressure = np.interp(1967,t, p)
        ax3.plot(t,np.full(len(t),healthyPressure), 'k--')

        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_pressure(f=ode_model_pressure, t0=1950, t1=2050, dt=0.1, p0=0.05,
                                             pars=[pi[0], pi[1], 0.05, pi[2]])
                # ax3.plot(t1[710:], p_co[710:], 'k-', alpha=0.2, lw=0.5)
                ax3.plot(t1, p_co, 'k-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Pressure current extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax3.plot([], [], 'k-', lw=0.5)

        ax3.legend(loc=2, prop={'size': 8})
        ax3.set_ylabel('pressure (Bars)')
        ax3.set_xlabel('time (t)')
        ax3.set_title('Forecasted Pressure for Different Production Rates vs Time')
        plt.show()
        plt.savefig("Future pressure predictions", dpi=600)

        # future predictions (Temperature)
        f3, ax4 = plt.subplots(nrows=1, ncols=1)
        ps = np.random.multivariate_normal(Tc, covT, 100)  # samples from posterior
        a = 1.2603295448322993e-07
        b = 0.03252493165262612

        # current observed data
        ax4.plot(Tt, Tte, 'k.', label='temperature data')
        vt = stdev(Tte) # standard deviation
        ax4.errorbar(Tt, Tte, yerr = vt, fmt = '.r')

        # half the current extraction rate at 5,000 tonnes per day
        t1, p1 = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                               pars=[a, b, 0.05, 149, at, bt], q=5000)
        ax4.plot(t1, p1, 'g-', label='half current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                                        pars=[a, b, 0.05, 149, pi[0], pi[1]], q=5000)
                ax4.plot(t1[710:], p_co[710:], 'g-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Temperature half extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax4.plot([], [], 'g-', lw=0.5)
        # zero extraction rate at 0 tonnes per day
        t2, p2 = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                               pars=[a, b, 0.05, 149, at, bt], q=0)
        ax4.plot(t2, p2, 'r-', label='zero extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                                        pars=[a, b, 0.05, 149, pi[0], pi[1]], q=0)
                ax4.plot(t1[710:], p_co[710:], 'r-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Temperature zero extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax4.plot([], [], 'r-', lw=0.5)
        # double the current extraction rate at 18,000 tonnes per day
        t3, p3 = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                               pars=[a, b, 0.05, 149, at, bt], q=18000)
        ax4.plot(t3, p3, 'b-', label='double current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                                        pars=[a, b, 0.05, 149, pi[0], pi[1]], q=18000)
                ax4.plot(t1[710:], p_co[710:], 'b-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Temperature double extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax4.plot([], [], 'b-', lw=0.5)
        # triple the current extraction rate at 27,000 tonnes per day
        t4, p4 = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                               pars=[a, b, 0.05, 149, at, bt], q=27000)
        ax4.plot(t4, p4, 'm-', label='triple current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_temperature_predict(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                                        pars=[a, b, 0.05, 149, pi[0], pi[1]], q=27000)
                ax4.plot(t1[710:], p_co[710:], 'm-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Temperature triple extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax4.plot([], [], 'm-', lw=0.5)
        # The current extraction rate at around 9000 tonnes per day
        T, X = solve_ode_temperature(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                     pars=[a, b, 0.05, 149, at, bt])

        # plot threshold temp
        healthyTemperature = np.interp(1967,Tt, Tte)
        ax4.plot(T,np.full(len(T),healthyTemperature), 'k--')

        ax4.plot(T, X, 'k-', label='current extraction')
        if plot_uncertainty is True:
            for pi in ps:
                _, p_co = solve_ode_temperature(f=ode_model_temperature, t0=1950, t1=2050, dt=0.1, T0=149,
                                                pars=[a, b, 0.05, 149, pi[0], pi[1]])
                ax4.plot(t1, p_co, 'k-', alpha=0.2, lw=0.5)
                values.append(p_co[-1])
            print("Temperature current extraction: ", np.percentile(values,5), " , ", np.percentile(values,95))
            values = []
            ax4.plot([], [], 'k-', lw=0.5)

        ax4.legend(loc=2, prop={'size': 8})
        ax4.set_ylabel('Temperature (degC)')
        ax4.set_xlabel('time (t)')
        ax4.set_title('Forecasted Temperature for Different Production Rates vs Time')
        plt.show()
        plt.savefig("Future temperature predictions", dpi=600)
