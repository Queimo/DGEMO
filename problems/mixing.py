import numpy as np
from .problem import Problem
from scipy.optimize import fsolve
from scipy.integrate import odeint


class MixingProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=[0.4,0.1], xu=[20,10])
        self.run_simulation_vectorized = np.vectorize(self.run_simulation)
    
    def _evaluate_F(self, x):
        Q_gas, Q_liquid = x[:, 0], x[:, 1]
        # Z = self.run_simulation_vectorized(Q_gas, Q_liquid)
        # return np.array(Z).transpose()
        # loop instead of vectorized
        Z = np.zeros((x.shape[0], 2))
        for i in range(x.shape[0]):
            Z[i, :] = self.run_simulation(Q_gas[i], Q_liquid[i])
        return Z
    
    def _calc_pareto_front(self, n_pareto_points=100):
        
        raise Exception("Pareto front is not known for this problem!")
        from .common import generate_initial_samples, get_problem
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args
         
        prob = get_problem('mixingproblem')
        X_init, Y_init = generate_initial_samples(prob, n_pareto_points)
        
        #namespace to dict
        solver_args = vars(get_solver_args())
        
        
        solver = NSGA2Solver(**solver_args)

        # find Pareto front
        solution = solver.solve(prob, X_init, Y_init)
        
        Y_paretos = solution['y']
        return Y_paretos

        
    def run_simulation(self, Q_gas=0.5, Q_liquid=8, R=0.5e-3, L=0.05):
        
        
        mixing, res, tr_i, dP_t = self.mixing_time(Q_gas, Q_liquid, R, L)
        
        # Initial conditions
        C_I_0 = 0.05  # Initial concentration of KI in stream 1 (M)
        C_IO3_0 = 0.01  # Initial concentration of KIO3 in stream 1 (M)
        C_H3BO3_0 = 0.25  # Initial concentration of *added* H3BO3 in stream 1 (M)
        C_H2BO3_0 = C_H3BO3_0 / 2
        C_H_0 = 0.05625  # Initial concentration of H+ in stream 2 (M)
        
        f_result, time_result, k2_result, Da2_result = self.solve_model(mixing.item(), C_H_0=C_H_0, C_I_0=C_I_0, C_IO3_0=C_IO3_0, C_H3BO3_0=C_H3BO3_0)
        
        I3_result = f_result[4][-1] * C_H_0

        f_H = f_result[0]
        f_I = f_result[1]
        f_IO3 = f_result[2]
        f_I2 = f_result[3]
        f_I3 = f_result[4]
        
        # MATLAB Code
        # Y = 2*V2*(I2_end)/(V1*C_H_0);
        # Yst = 6*C_IO3_0/C_H2BO3_0/(6*C_IO3_0/C_H2BO3_0+1);
        # Xs = Y/Yst;
        
        #Python Code
        Y = 2 * 0.01 * f_I2[-1] / (0.01 * C_H_0)
        Yst = 6 * C_IO3_0 / C_H2BO3_0 / (6 * C_IO3_0 / C_H2BO3_0 + 1)
        Xs = Y/Yst
        I3_result = f_result[4][-1] * C_H_0
        
        # print(f"Mixing time =  {mixing.item()}")
        # print(f"[I3] after reaction = {I3_result}")
        # print(f"Interfacial shear rate = {tr_i}")
        # print(f"Pressure drop = {dP_t}")

        return Xs, dP_t

    def mixing_time(self, Q_gas, Q_liquid, R=0.5e-3, L=0.05):
        """
        The function is used to solve the film thickness based on physical parameters.
        Since the film thickness is a function of Reynolds numbers and Reynolds numbers are a function of film thickess H,
        an initial film thickness needs to be assumed and solved iteratively.

        Film thickness depends on channel radius and physical properties of the gas and liquid.

        R = tube radius (m)
        m_g = inlet gas mass flow rate (kg/s)
        m_l = inlet liquid mass flow rate (kg/s)
        rho_g = density of the gas (kg/m3)
        rho_l = density of the liquid (kg/m3)
        mu_g = viscosity of the gas (Pa.s)
        mu_l = viscosity of the liquid (Pa.s)
        H = initial film thickness estimation (m)

        Quantities calculated by the model:

        Liquid film thickess (m)
        Gas and liquid superficial velcocity(m/s)
        Gas and liquid film velocirt (m/s)
        Gas and liquid Reynolds number (-)
        Shear rate (1/s)
        Gas and liquid phase residence time (s)
        Pressure drop (Pa)
        Gas and liquid Weber number (-)
        Energy dissipation rate (W/kg)
        Mixing time based on engulfment model (s)

        Input: liquid (ml/min) and gas (L/min) flow rate
        Output: mixing time and liquid residence time


        """

        # Tube dimensions
        # K10 tube dimensions
        # R = 3.5E-3/2  # m
        # L = 100E-3  # m

        # Tube dimensions
        # K1 tube dimensions
        # R = 0.5E-3  # m
        # L = 0.05  # m

        # Fluid parameters
        rho_g = 1.184  # gas phase density (kg/m^3)
        rho_l = 998  # liquid phase density (kg/m^3)
        mu_g = 1.81e-5  # gas phase viscosity (Pa.s)
        mu_l = 8.90e-4  # liquid phase viscosity (Pa.s)

        # Flowrates
        q_g = Q_gas  # gas mass flowrate L/min
        q_l = Q_liquid  # liquid mass flowrate mL/min
        m_g = q_g / 60 * 0.001 * rho_g  # gas mass flowrate (kg/s)
        m_l = q_l / 60 * 0.001 * 0.001 * rho_l  # liquid mass flowrate (kg/s)

        # Define the function for fsolve
        def annular_flow_eq(H):
            return self.film_thickness(R, m_g, m_l, rho_g, rho_l, mu_g, mu_l, H) - H

        # Solve for liquid film thickness
        H, info, exitflag, msg = fsolve(
            annular_flow_eq, 1.6e-5, xtol=1e-8, maxfev=int(1e4), full_output=True
        )

        if exitflag == 1 or exitflag == 5 or True:
            U_g = (m_g / rho_g) / (np.pi * (R - H) ** 2)  # gas film velocity (m/s)
            U_l = (m_l / rho_l) / (
                np.pi * (R**2 - (R - H) ** 2)
            )  # liquid film velocity (m/s)

            Re_g = rho_g * U_g * 2 * (R - H) / mu_g  # gas film Reynolds number
            Re_l = rho_l * U_l * 2 * H / mu_l  # liquid film Reynolds number

            # Friction factor calculation (unmodified)
            f_g = np.where(Re_g < 2300, 16 / Re_g, 0.079 / (Re_g**0.25))
            f_l = np.where(Re_l < 2300, 16 / Re_l, 0.079 / (Re_l**0.25))

            # Shear calculation
            t_i = 0.5 * f_g * rho_g * (U_g - U_l) ** 2  # interfacial shear stress (Pa)
            tr_i = t_i / mu_l  # liquid shear rate (1/s)

            t_w = 0.5 * f_l * rho_l * U_l**2  # wall shear stress (Pa)
            tr_w = t_w / mu_l  # wall shear rate (1/s)

            # Bulk residence time calculation
            tres_l = L / U_l  # liquid phase residence time (s)
            tres_g = L / U_g  # gas phase residence time (s)

            # Pressure drop calculations
            P_a = 101325  # atmospheric pressure (Pa)
            x = m_g / (m_g + m_l)  # gas mass fraction
            G = (m_g + m_l) / (np.pi * R**2)  # flow quality
            dP_l = L / (2 * R) * (4 * f_l * G**2 * x**2) / (2 * rho_l)  # liquid phase
            dP_g = (
                np.sqrt(
                    L / (2 * R) * (4 * f_g * G**2 * x**2) / (2 * rho_g) * 2 * P_a
                    + P_a**2
                )
                - P_a
            )  # gas phase
            X = np.sqrt(dP_l / dP_g)

            # Weir number calculations
            C = np.where(
                (Re_g < 2300) & (Re_l < 2300),
                5,
                np.where(
                    (Re_g < 2300) & (Re_l >= 2300),
                    10,
                    np.where((Re_g >= 2300) & (Re_l < 2300), 12, 20),
                ),
            )

            We_g = rho_g * U_g**2 * (R - H) * 2 / 0.072
            We_l = rho_l * U_l**2 * (R * 2 - (R - H) * 2) / 0.072

            psi_l = np.sqrt(1 + C / X + 1 / X**2)
            dP_t = psi_l**2 * dP_l  # total pressure drop (Pa)
            # minimize the pressure drop

            U_gs = m_g / rho_g / (np.pi * R**2)
            U_ls = m_l / rho_l / (np.pi * R**2)
            e = dP_t / rho_l * U_l / L
            um = 17.2 * np.sqrt(mu_l / rho_l / e)

        else:
            print(msg)
            print(info)
            
            print("error in solver")
            print(exitflag)

        return um, tres_l, tr_i, dP_t

    def film_thickness(self, R, m_g, m_l, rho_g, rho_l, mu_g, mu_l, H):
        # Calculate the flow characteristics of 2-phase annular flow

        # Gas film velocity
        U_g = (m_g / rho_g) / (np.pi * (R - H) ** 2)

        # Liquid film velocity
        U_l = (m_l / rho_l) / (np.pi * (R**2 - (R - H) ** 2))

        # Gas film Reynolds number
        Re_g = rho_g * U_g * 2 * (R - H) / mu_g

        # Liquid film Reynolds number
        Re_l = rho_l * U_l * 2 * H / mu_l

        if Re_g < 2300:
            if m_g < 1.9733e-05:
                H_real = R * (1 - Re_l / Re_g * (rho_g * U_g**2) / (rho_l * U_l**2))
            else:
                H_real = R * (
                    1
                    - 0.288
                    * Re_l**1.39
                    / Re_g**0.69
                    * (rho_g * U_g**2)
                    / (rho_l * U_l**2)
                )
        else:
            H_real = R * (
                1
                - 0.0548
                * Re_l**1.39
                / Re_g**0.47
                * (rho_g * U_g**2)
                / (rho_l * U_l**2)
            )

        return H_real

    def k2_calculate(self, theta, f_H, f_I, f_IO3, f_I2, f_I3, C_H0):
        """
        C_H0 is the initial concentration of acid,
        f_H is the time dependent concentration of acid
        All the charges are -1. Therefore no term of Z^2
        0.5 multiplier comes from equation

        Somehow, they both need to be there
        exp(theta) describes the dilution of the stream due to volume increase
        mu is not constant since the concentration of ions change as reaction progresses
        mu is updated at every integration step
        """

        mu = np.sum([f_H, f_I, f_IO3, f_I2, f_I3]) / np.exp(theta) * C_H0 * 0.5

        if f_H <= 0:
            k = 0
        else:
            if mu < 0.166:
                k = 10 ** (9.28105 - 3.664 * np.sqrt(mu))
            else:
                k = 10 ** (8.383 - 1.5112 * np.sqrt(mu) + 0.23689 * mu)

        return k
    
    def k2_calculate_vectorized(self, theta, f_H, f_I, f_IO3, f_I2, f_I3, C_H0):
        """
        Vectorized function to calculate k values for arrays of data.

        Parameters:
        theta, f_H, f_I, f_IO3, f_I2, f_I3 are all arrays or scalars.
        C_H0 is the initial concentration of acid, scalar or an array of the same length as the other arrays.

        Returns:
        k - array of k values computed element-wise.
        """
        # Compute mu element-wise
        mu = (f_H + f_I + f_IO3 + f_I2 + f_I3) / np.exp(theta) * C_H0 * 0.5

        # Vectorized conditional computation for k
        k = np.where(
            f_H <= 0,
            0,  # If condition is true (f_H <= 0), set k to 0
            np.where(
                mu < 0.166,
                10 ** (9.28105 - 3.664 * np.sqrt(mu)),
                10 ** (8.383 - 1.5112 * np.sqrt(mu) + 0.23689 * mu)
            )
        )

        return k


    def VD_model(self, y, theta, tm, C):
        """
        Villermaux-Dushman reaction with incorporation mixing model

        Model reactions
        H2B03- + H+ ---> H3BO3 (1)
        5I- + IO3- + 6H+ ---> 3I2 + 3H20 (2)
        I- + I2 <===> I3- (3)

        The first reaction is infinetly fast, do not care about the mass balance of the first reaction
        Only considering first reaction here since both reaction (1) and (2) consume H+

        Function has placeholders, it returns a system of differential equations as a function of
        Damkohler numbers and initial concentrations which need to be evaluated later
        Differential equations are in non-dimensional form with respect to theta where
        theta = t/tm
            t = actual time
            tm = characteristic mixing time
        """
        # Time-dependent concentration of species
        f_H = y[0]
        f_I = y[1]
        f_IO3 = y[2]
        f_I2 = y[3]
        f_I3 = y[4]

        # Initial concentration of species
        C_I_0 = C[0]
        C_IO3_0 = C[1]
        C_H3BO3_0 = C[2]
        C_H_0 = C[3]
        C_H2BO3_0 = C_H3BO3_0 / 2

        # Dimensionless concentration ratios for the model equations
        P = C_H2BO3_0 / C_H_0
        Q1 = C_I_0 / C_H_0
        Q2 = C_IO3_0 / C_H_0

        # Damkohler numers, non-dimensional time scale comparison
        # Concentrations are the initial concentration of H+, rate constants are time dependent
        # due to time dependence of k2 on ionic strenght which changes with concentration
        # This part need more investigation.
        k3_f = 5.6e9  # Forward rate of reaction (3)
        k3_r = 7.5e9  # Backward rate of reaction (3)
        Da2 = (
            tm * self.k2_calculate_vectorized(theta, f_H, f_I, f_IO3, f_I2, f_I3, C_H_0) * C_H_0**4
        )  # Da of reaction (2)
        Da3 = tm * k3_f * C_H_0  # Da of forward reaction (3)
        Da3_r = tm * k3_r  # Da of backward reaction (3)

        # System of non-dimensional ODEs to describe the change in the concentration in Fluid 2 (acid phase)
        dHdtheta = -6 * Da2 * (f_I**2 * f_IO3 * f_H**2) / np.exp(
            theta
        ) ** 4 - P * np.exp(theta) * (f_H > 0)
        dIdtheta = (
            -5 * Da2 * (f_I**2 * f_IO3 * f_H**2) / np.exp(theta) ** 4
            - Da3 * f_I * f_I2 / np.exp(theta)
            + Da3_r * f_I3
            + Q1 * np.exp(theta)
        )
        dIO3dtheta = -1 * Da2 * (f_I**2 * f_IO3 * f_H**2) / np.exp(
            theta
        ) ** 4 + Q2 * np.exp(theta)
        dI2dtheta = (
            3 * Da2 * (f_I**2 * f_IO3 * f_H**2) / np.exp(theta) ** 4
            - Da3 * f_I * f_I2 / np.exp(theta)
            + Da3_r * f_I3
        )
        dI3dtheta = Da3 * f_I * f_I2 / np.exp(theta) - Da3_r * f_I3

        dydtheta = [dHdtheta, dIdtheta, dIO3dtheta, dI2dtheta, dI3dtheta]

        return dydtheta


    def solve_model(self, tm, C_H_0=0.05625, C_I_0=0.05, C_IO3_0=0.01, C_H3BO3_0=0.25):
        """
        Main function to solve the ODEs for given initial conditions.
        Function accepts mixing time (tm) since it changes with reaction conditions
        First, the function needs to be evaluated with the initial conditiosn then solved

        It returns all the non-dimensional non zero concentrations and dimensionless time
        """
        # It is unclear why this part exists. The final time proposed by this is too much.
        V1 = 0.01  # Borate/Iodide stream volume (L)
        V2 = 0.01  # Acid stream volume (L)
        t_final = np.log((V1 + V2) / V2)  # Assumed time of mixing completion

        # Define upper and lower limits of solution time
        # odeint cannot pick the timesteps automatically
        time = np.linspace(0, 0.4, num=100)
        odefun = lambda t, y: self.VD_model(
            t, y, tm, [C_I_0, C_IO3_0, C_H3BO3_0, C_H_0]
        )  # Evaluate the function with system parameters
        f_values = odeint(odefun, [1, 0, 0, 0, 0], time, rtol=1e-5, atol=1e-5).T

        # Remove the negative concentration values since we do not know when the reaction ends
        boolArr = f_values[0] > 0
        count = np.count_nonzero(boolArr)
        f_cleaned = f_values[:, :count]
        time_cleaned = time[:count]

        k2_values = []
        Da2_values = []

        # Calculate the rate constant k2 and Da2 for every time step of the integration
        # for t, f_H, f_I, f_IO3, f_I2, f_I3 in zip(time_cleaned, f_cleaned[0], f_cleaned[1], f_cleaned[2], f_cleaned[3], f_cleaned[4]):
        #     k2 = self.k2_calculate(t / tm, f_H, f_I, f_IO3, f_I2, f_I3, C_H_0)
        #     Da2 = tm * k2 * C_H_0
        #     k2_values.append(k2)
        #     Da2_values.append(Da2)

        # replace the loop with a vectorized function
        k2_values = self.k2_calculate_vectorized(
            time_cleaned / tm,
            f_cleaned[0],
            f_cleaned[1],
            f_cleaned[2],
            f_cleaned[3],
            f_cleaned[4],
            C_H_0,
        )
        Da2_values = tm * k2_values * C_H_0

        return (
            f_cleaned,
            time_cleaned,
            k2_values,
            Da2_values,
        )


#test pareto front calculation
if __name__ == "__main__":
    prob = MixingProblem()
    true_front = prob.pareto_front()
    print(true_front)