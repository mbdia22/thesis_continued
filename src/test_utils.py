"""
Comprehensive tests for the utils module.

Tests for the simulate_paths function which implements the Euler-Maruyama
discretization scheme for simulating interest rate paths under the Vasicek model.
"""

import numpy as np
import pytest
from models import Vasicek
from utils import simulate_paths


class TestSimulatePaths:
    """Test suite for the simulate_paths function."""

    def test_output_shape(self):
        """Test that simulate_paths returns arrays of the correct shape."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 5.0
        dt = 0.1
        n_sims = 100

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        expected_steps = int(T / dt) + 1

        # Check shape of rate paths
        assert r_paths.shape == (n_sims, expected_steps), \
            f"Expected shape ({n_sims}, {expected_steps}), got {r_paths.shape}"

        # Check shape of time grid
        assert t_grid.shape == (expected_steps,), \
            f"Expected time grid shape ({expected_steps},), got {t_grid.shape}"

    def test_initial_conditions(self):
        """Test that all paths start at r0."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.01
        n_sims = 50

        r_paths, _ = simulate_paths(model, T, dt, n_sims)

        # All paths should start at r0
        assert np.allclose(r_paths[:, 0], model.r0), \
            f"All paths should start at r0={model.r0}, but got {r_paths[:, 0]}"

    def test_time_grid_correctness(self):
        """Test that the time grid is correctly generated."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 2.0
        dt = 0.25
        n_sims = 10

        _, t_grid = simulate_paths(model, T, dt, n_sims)

        # Check start and end times
        assert t_grid[0] == 0.0, "Time grid should start at 0"
        assert np.isclose(t_grid[-1], T), f"Time grid should end at T={T}"

        # Check uniform spacing
        expected_grid = np.linspace(0, T, int(T / dt) + 1)
        assert np.allclose(t_grid, expected_grid), \
            "Time grid should be uniformly spaced"

    def test_deterministic_convergence(self):
        """Test that when sigma=0, the process converges to theta."""
        r0 = 0.02
        theta = 0.05
        kappa = 0.2

        # Zero volatility - deterministic mean reversion
        model = Vasicek(r0=r0, kappa=kappa, sigma=0.0, theta=theta)
        T = 20.0  # Long time horizon for convergence
        dt = 0.1
        n_sims = 5

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # All paths should be identical (deterministic)
        for i in range(1, n_sims):
            assert np.allclose(r_paths[0, :], r_paths[i, :], atol=1e-10), \
                "With sigma=0, all paths should be identical"

        # Final value should be close to theta
        final_value = r_paths[0, -1]
        # Analytical solution: r(T) = theta + (r0 - theta) * exp(-kappa * T)
        expected_final = theta + (r0 - theta) * np.exp(-kappa * T)

        assert np.isclose(final_value, expected_final, rtol=1e-3), \
            f"Final value {final_value} should converge to {expected_final}"

    def test_mean_reversion_property(self):
        """Test that the process exhibits mean reversion towards theta."""
        r0 = 0.01  # Start below theta
        theta = 0.05
        kappa = 0.3
        sigma = 0.005

        model = Vasicek(r0=r0, kappa=kappa, sigma=sigma, theta=theta)
        T = 10.0
        dt = 0.01
        n_sims = 5000

        # Set seed for reproducibility
        np.random.seed(42)

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # Average across simulations
        mean_path = np.mean(r_paths, axis=0)

        # The mean should move from r0 towards theta
        assert mean_path[0] < theta, "Initial value should be below theta"

        # The mean should increase over time (mean reversion)
        # Check that the mean at the end is higher than at the start
        assert mean_path[-1] > mean_path[0], \
            "Mean should increase due to mean reversion towards theta"

        # Check that final mean is close to theta (may not reach exactly due to finite T)
        # Analytical mean: E[r(t)] = theta + (r0 - theta) * exp(-kappa * t)
        expected_mean_final = theta + (r0 - theta) * np.exp(-kappa * T)

        assert np.isclose(mean_path[-1], expected_mean_final, atol=0.005), \
            f"Final mean {mean_path[-1]} should be close to analytical mean {expected_mean_final}"

    def test_variance_scaling(self):
        """Test that paths with higher sigma have higher variance."""
        model_low = Vasicek(r0=0.04, kappa=0.15, sigma=0.001, theta=0.05)
        model_high = Vasicek(r0=0.04, kappa=0.15, sigma=0.02, theta=0.05)

        T = 5.0
        dt = 0.01
        n_sims = 1000

        # Set seed for fair comparison
        np.random.seed(123)
        r_paths_low, _ = simulate_paths(model_low, T, dt, n_sims)

        np.random.seed(123)
        r_paths_high, _ = simulate_paths(model_high, T, dt, n_sims)

        # Calculate variance at final time
        var_low = np.var(r_paths_low[:, -1])
        var_high = np.var(r_paths_high[:, -1])

        # Higher sigma should produce higher variance
        assert var_high > var_low, \
            f"Higher sigma should produce higher variance: {var_high} vs {var_low}"

    def test_analytical_variance(self):
        """Test that the simulated variance matches analytical variance."""
        r0 = 0.04
        kappa = 0.2
        sigma = 0.01
        theta = 0.05

        model = Vasicek(r0=r0, kappa=kappa, sigma=sigma, theta=theta)
        T = 5.0
        dt = 0.01
        n_sims = 10000

        # Set seed for reproducibility
        np.random.seed(456)

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # Calculate empirical variance at final time
        empirical_var = np.var(r_paths[:, -1])

        # Analytical variance for Vasicek model:
        # Var[r(t)] = (sigma^2 / (2 * kappa)) * (1 - exp(-2 * kappa * t))
        analytical_var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * T))

        # Allow 10% relative tolerance due to finite sampling
        assert np.isclose(empirical_var, analytical_var, rtol=0.1), \
            f"Empirical variance {empirical_var} should match analytical {analytical_var}"

    def test_different_time_steps(self):
        """Test that the function works with different time step sizes."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        n_sims = 10

        # Test various dt values
        dt_values = [1.0, 0.5, 0.1, 0.01, 1/252]  # Including daily (252 trading days)

        for dt in dt_values:
            r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

            expected_steps = int(T / dt) + 1
            assert r_paths.shape == (n_sims, expected_steps), \
                f"Failed for dt={dt}: expected shape ({n_sims}, {expected_steps}), got {r_paths.shape}"

            # All paths should start at r0
            assert np.allclose(r_paths[:, 0], model.r0), \
                f"Failed for dt={dt}: paths don't start at r0"

    def test_small_time_horizon(self):
        """Test behavior with very small time horizons."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 0.1  # Very small T
        dt = 0.01
        n_sims = 10

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # Should still work and return valid output
        assert r_paths.shape[0] == n_sims
        assert r_paths.shape[1] > 0
        assert np.all(np.isfinite(r_paths)), "All values should be finite"

    def test_large_time_horizon(self):
        """Test behavior with large time horizons."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 30.0  # 30 years
        dt = 0.25  # Quarterly
        n_sims = 50

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # Should still work and return valid output
        expected_steps = int(T / dt) + 1
        assert r_paths.shape == (n_sims, expected_steps)
        assert np.all(np.isfinite(r_paths)), "All values should be finite"

        # Mean should converge to theta over long horizons
        mean_final = np.mean(r_paths[:, -1])
        assert np.isclose(mean_final, model.theta, atol=0.01), \
            f"Over long horizons, mean should converge to theta={model.theta}"

    def test_single_simulation(self):
        """Test that the function works with n_sims=1."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.1
        n_sims = 1

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        expected_steps = int(T / dt) + 1
        assert r_paths.shape == (1, expected_steps)
        assert r_paths[0, 0] == model.r0

    def test_many_simulations(self):
        """Test that the function can handle many simulations efficiently."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.1
        n_sims = 10000  # Large number

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        expected_steps = int(T / dt) + 1
        assert r_paths.shape == (n_sims, expected_steps)
        assert np.all(np.isfinite(r_paths)), "All values should be finite"

    def test_no_negative_explosion(self):
        """Test that rates don't explode to unrealistic values."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 10.0
        dt = 0.01
        n_sims = 100

        np.random.seed(789)
        r_paths, _ = simulate_paths(model, T, dt, n_sims)

        # Check that rates stay in a reasonable range
        # For Vasicek, rates can go negative, but shouldn't explode
        min_rate = np.min(r_paths)
        max_rate = np.max(r_paths)

        # Reasonable bounds: typically rates shouldn't deviate too much
        # Allow for 5-6 standard deviations from theta (99.9999% of values)
        # The Vasicek model has a stationary distribution, so extreme outliers are rare but possible
        std_dev = np.sqrt((model.sigma**2 / (2 * model.kappa)) * (1 - np.exp(-2 * model.kappa * T)))
        lower_bound = model.theta - 6 * std_dev
        upper_bound = model.theta + 6 * std_dev

        assert min_rate > lower_bound, \
            f"Minimum rate {min_rate} is unrealistically low (< {lower_bound})"
        assert max_rate < upper_bound, \
            f"Maximum rate {max_rate} is unrealistically high (> {upper_bound})"

    def test_euler_maruyama_discretization(self):
        """Test that the Euler-Maruyama discretization is correctly implemented."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.1
        n_sims = 1

        # Set seed for reproducibility
        np.random.seed(999)

        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        # Manually verify the first step
        np.random.seed(999)

        # Initialize
        r_manual = np.zeros(len(t_grid))
        r_manual[0] = model.r0
        sqrt_dt = np.sqrt(dt)

        # First step
        drift = model.kappa * (model.theta - r_manual[0]) * dt
        diffusion = model.sigma * sqrt_dt * np.random.normal()
        r_manual[1] = r_manual[0] + drift + diffusion

        # Compare first step
        assert np.isclose(r_paths[0, 1], r_manual[1]), \
            f"First step mismatch: {r_paths[0, 1]} vs {r_manual[1]}"

    def test_different_kappa_values(self):
        """Test behavior with different mean reversion speeds."""
        r0 = 0.02
        theta = 0.05
        sigma = 0.01
        T = 10.0
        dt = 0.1
        n_sims = 1000

        # Fast mean reversion
        model_fast = Vasicek(r0=r0, kappa=0.5, sigma=sigma, theta=theta)
        np.random.seed(100)
        r_paths_fast, _ = simulate_paths(model_fast, T, dt, n_sims)

        # Slow mean reversion
        model_slow = Vasicek(r0=r0, kappa=0.05, sigma=sigma, theta=theta)
        np.random.seed(100)
        r_paths_slow, _ = simulate_paths(model_slow, T, dt, n_sims)

        # Fast mean reversion should reach theta quicker
        mean_fast = np.mean(r_paths_fast[:, -1])
        mean_slow = np.mean(r_paths_slow[:, -1])

        # Fast kappa should be closer to theta
        dist_fast = abs(mean_fast - theta)
        dist_slow = abs(mean_slow - theta)

        assert dist_fast < dist_slow, \
            f"Faster mean reversion should be closer to theta: {dist_fast} vs {dist_slow}"

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same random seed."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.1
        n_sims = 10

        # First run
        np.random.seed(42)
        r_paths_1, _ = simulate_paths(model, T, dt, n_sims)

        # Second run with same seed
        np.random.seed(42)
        r_paths_2, _ = simulate_paths(model, T, dt, n_sims)

        # Should be identical
        assert np.allclose(r_paths_1, r_paths_2), \
            "Results should be reproducible with the same random seed"


class TestSimulatePathsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_kappa(self):
        """Test behavior when kappa=0 (no mean reversion)."""
        # When kappa=0, the Vasicek model becomes dr = sigma * dW (Brownian motion)
        model = Vasicek(r0=0.04, kappa=0.0, sigma=0.01, theta=0.05)
        T = 1.0
        dt = 0.01
        n_sims = 10

        # Should still work without division by zero
        r_paths, t_grid = simulate_paths(model, T, dt, n_sims)

        assert r_paths.shape == (n_sims, int(T / dt) + 1)
        assert np.all(np.isfinite(r_paths)), "All values should be finite"

    def test_high_volatility(self):
        """Test behavior with very high volatility."""
        model = Vasicek(r0=0.04, kappa=0.15, sigma=0.1, theta=0.05)  # High sigma
        T = 1.0
        dt = 0.01
        n_sims = 100

        r_paths, _ = simulate_paths(model, T, dt, n_sims)

        # Should still produce finite values
        assert np.all(np.isfinite(r_paths)), "All values should be finite even with high volatility"

        # Variance should be high
        final_var = np.var(r_paths[:, -1])
        assert final_var > 0.001, "High volatility should produce high variance"

    def test_extreme_parameters(self):
        """Test with extreme but valid parameters."""
        model = Vasicek(r0=0.001, kappa=1.0, sigma=0.05, theta=0.10)
        T = 2.0
        dt = 0.05
        n_sims = 50

        r_paths, _ = simulate_paths(model, T, dt, n_sims)

        assert np.all(np.isfinite(r_paths)), "Should handle extreme parameters"
        assert r_paths.shape == (n_sims, int(T / dt) + 1)


def test_integration_with_pricing():
    """Integration test: verify simulate_paths works correctly with pricing functions."""
    from pricing_rmv import price_bond_rmv

    # Use parameters from Table 5.1
    model = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.0522,
                    L=0.4, lambda_0=0.025, lambda_1=0.0)

    T = 5.0

    # This should work without errors
    np.random.seed(12345)
    price = price_bond_rmv(model, T, dt=1/252, n_sims=1000)

    # Price should be positive and less than 1 (for a zero-coupon bond)
    assert 0 < price < 1, f"Bond price should be in (0, 1), got {price}"

    # For defaultable bonds, price should be less than risk-free
    model_riskfree = Vasicek(r0=0.04, kappa=0.15, sigma=0.01, theta=0.0522,
                             L=0.0, lambda_0=0.0, lambda_1=0.0)

    np.random.seed(12345)
    price_riskfree = price_bond_rmv(model_riskfree, T, dt=1/252, n_sims=1000)

    assert price < price_riskfree, \
        "Defaultable bond price should be less than risk-free bond price"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
