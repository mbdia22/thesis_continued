import numpy as np
from utils import simulate_paths

def price_bond_rmv(model, T, dt=1/252, n_sims=10000):
    """
    Calculates the price of a defaultable bond under Recovery of Market Value (RMV).
    
    P_RMV(0, T) = E[ exp( - integral_0^T (r(u) + L * lambda(u)) du ) ]
    
    Args:
        model: An instance of the Vasicek model.
        T (float): Maturity in years.
        dt (float): Time step.
        n_sims (int): Number of simulations.
        
    Returns:
        float: The estimated bond price.
    """
    # 1. Simulate short rate paths
    r_paths, t_grid = simulate_paths(model, T, dt, n_sims)
    
    # 2. Calculate hazard rate paths lambda(t)
    # lambda(t) = lambda_0 + lambda_1 * r(t)
    lambda_paths = model.hazard_rate(r_paths)
    
    # 3. Calculate the instantaneous discount rate R(t)
    # R(t) = r(t) + L * lambda(t)
    R_paths = r_paths + model.L * lambda_paths
    
    # 4. Integrate R(t) over [0, T] using trapezoidal rule
    # axis=1 integrates over time
    integral_R = np.trapz(R_paths, dx=dt, axis=1)
    
    # 5. Calculate discount factors
    discount_factors = np.exp(-integral_R)
    
    # 6. Average over simulations
    price = np.mean(discount_factors)
    
    return price
