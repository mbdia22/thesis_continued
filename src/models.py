import numpy as np


class Vasicek:
    """
    Vasicek Model for Interest Rate Simulation.
    Parameters from Table 5.1 of the previous thesis.
    """
    def __init__(self, r0=0.04, kappa=0.15, sigma=0.01, theta=0.0522, 
                 q=0.03, T=10, L=0.4, lambda_0=0.025, lambda_1=0.0):
        self.r0 = r0
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.q = q
        self.T = T
        self.L = L
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1

    def hazard_rate(self, r):
        """
        Calculates the hazard rate lambda(t) based on the short rate r(t).
        lambda(t) = lambda_0 + lambda_1 * r(t)
        """
        return self.lambda_0 + self.lambda_1 * r

    def zero_bond_price(self, t, T, r_t):
        """
        Calculates the price of a zero-coupon bond P(t, T) under the Vasicek model.
        P(t, T) = A(t, T) * exp(-B(t, T) * r(t))
        """
        tau = T - t
        B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        A = np.exp((self.theta - self.sigma**2 / (2 * self.kappa**2)) * 
                   (B - tau) - (self.sigma**2 / (4 * self.kappa)) * B**2)
        
        return A * np.exp(-B * r_t)

    def duration(self, t, T):
        """
        Calculates the stochastic duration (sensitivity to r) of the bond.
        D(t, T) = - (1/P) * dP/dr = B(t, T)
        """
        tau = T - t
        B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        return B

    def simulate_path(self, steps, dt):
        """
        Simulates a path of the short rate r(t) using the Euler-Maruyama method.
        """
        r = np.zeros(steps)
        r[0] = self.r0
        for i in range(1, steps):
            dr = self.kappa * (self.theta - r[i-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            r[i] = r[i-1] + dr
        return r
