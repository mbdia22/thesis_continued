import numpy as np

def simulate_paths(model, T, dt, n_sims):
    """
    Simulates multiple paths of the short rate r(t) using the Euler-Maruyama method.
    
    Args:
        model: An instance of a short rate model (e.g., Vasicek).
        T (float): Time horizon in years.
        dt (float): Time step size.
        n_sims (int): Number of simulation paths.
        
    Returns:
        np.ndarray: A matrix of shape (n_sims, steps) containing the simulated rate paths.
        np.ndarray: A vector of time points.
    """
    steps = int(T / dt) + 1
    t_grid = np.linspace(0, T, steps)
    r = np.zeros((n_sims, steps))
    r[:, 0] = model.r0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(1, steps):
        # Euler-Maruyama discretization
        # dr = kappa * (theta - r) * dt + sigma * dW
        # We use the vectorized version for efficiency
        
        drift = model.kappa * (model.theta - r[:, i-1]) * dt
        diffusion = model.sigma * sqrt_dt * np.random.normal(size=n_sims)
        
        r[:, i] = r[:, i-1] + drift + diffusion
        
    return r, t_grid
