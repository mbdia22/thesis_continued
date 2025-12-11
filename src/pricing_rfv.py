import numpy as np
from scipy.integrate import quad

class VasicekRFV:
    """
    Vasicek Model with Recovery of Face Value (RFV) logic.
    """
    def __init__(self, r0, kappa, sigma, theta, lambda_0, lambda_1, L):
        self.r0 = r0
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.L = L

    def _b_star(self, tau):
        """
        Calculates B*(tau) for the affine survival probability.
        B*(tau) = ((1 + Lambda_1) / kappa) * (1 - exp(-kappa * tau))
        """
        if self.kappa == 0:
            return (1 + self.lambda_1) * tau
        return ((1 + self.lambda_1) / self.kappa) * (1 - np.exp(-self.kappa * tau))

    def _a_star(self, tau):
        """
        Calculates A*(tau) for the affine survival probability.
        A*(tau) = integral_0^tau (kappa*theta*B*(s) - 0.5*sigma^2*B*(s)^2 - Lambda_0) ds
        """
        def integrand(s):
            bs = self._b_star(s)
            return self.kappa * self.theta * bs - 0.5 * self.sigma**2 * bs**2 - self.lambda_0
        
        res, _ = quad(integrand, 0, tau)
        return res

    def survival_value(self, t, T, r=None):
        """
        Calculates V_surv(t, T) = E[exp(-integral_t^T k(u) du)]
        V_surv(t, T) = exp(A*(tau) - B*(tau) * r(t))

        Args:
            t: Current time
            T: Maturity time
            r: Current short rate r(t). If None, uses self.r0
        """
        tau = T - t
        a_val = self._a_star(tau)
        b_val = self._b_star(tau)
        rate = self.r0 if r is None else r
        return np.exp(a_val - b_val * rate)

    def expected_lambda(self, t, u):
        """
        Calculates E^Q_u [lambda(u)] = Lambda_0 + Lambda_1 * E^Q_u [r(u)]
        """
        # Mean^P(r(u))
        mean_p = self.r0 * np.exp(-self.kappa * (u - t)) + \
                 self.theta * (1 - np.exp(-self.kappa * (u - t)))
        
        # Shift(t, u)
        def shift_integrand(s):
            # s goes from 0 to u-t
            # integrand: exp(-kappa*(u-t-s)) * sigma^2 * B*(s)
            return np.exp(-self.kappa * (u - t - s)) * self.sigma**2 * self._b_star(s)
        
        shift, _ = quad(shift_integrand, 0, u - t)
        
        e_r_u = mean_p - shift
        return self.lambda_0 + self.lambda_1 * e_r_u

    def price(self, t, T, r=None):
        """
        Calculates the RFV bond price: Survival Value + Recovery Value.

        Args:
            t: Current time
            T: Maturity time
            r: Current short rate r(t). If None, uses self.r0
        """
        # 1. Survival Value
        v_surv = self.survival_value(t, T, r)

        # 2. Recovery Value
        # V_rec = (1-L) * integral_t^T V_surv(t, u) * E[lambda(u)] du
        w = 1 - self.L

        def recovery_integrand(u):
            # V_surv(t, u) - use same rate for consistency
            surv_u = self.survival_value(t, u, r)
            # E[lambda(u)]
            e_lam = self.expected_lambda(t, u)
            return surv_u * e_lam

        v_rec_integral, _ = quad(recovery_integrand, t, T)
        v_rec = w * v_rec_integral

        return v_surv + v_rec

    def duration(self, t, T):
        """
        Calculates duration using central finite difference.
        Duration = -(1/P) * (dP/dr) where the derivative is approximated
        by shifting r by +/- 1 basis point.
        """
        shift = 0.0001  # 1 basis point

        # Calculate prices at shifted rates
        p_up = self.price(t, T, r=self.r0 + shift)
        p_down = self.price(t, T, r=self.r0 - shift)
        p_base = self.price(t, T, r=self.r0)

        # Central finite difference: dP/dr ≈ (P(r+ε) - P(r-ε)) / (2ε)
        dp_dr = (p_up - p_down) / (2 * shift)

        # Duration = -(1/P) * (dP/dr)
        return -(1 / p_base) * dp_dr

    def convexity(self, t, T):
        """
        Calculates convexity using central finite difference.
        Convexity = (1/P) * (d²P/dr²) where the second derivative is
        approximated using: (P(r+ε) - 2P(r) + P(r-ε)) / ε²
        """
        epsilon = 1e-4  # Small perturbation

        # Calculate prices at shifted rates
        p_up = self.price(t, T, r=self.r0 + epsilon)
        p_down = self.price(t, T, r=self.r0 - epsilon)
        p_base = self.price(t, T, r=self.r0)

        # Central finite difference: d²P/dr² ≈ (P(r+ε) - 2P(r) + P(r-ε)) / ε²
        d2p_dr2 = (p_up - 2 * p_base + p_down) / (epsilon ** 2)

        # Convexity = (1/P) * (d²P/dr²)
        return (1 / p_base) * d2p_dr2

if __name__ == "__main__":
    # Parameters for Verification
    r0 = 0.04
    kappa = 0.15
    sigma = 0.01
    theta = 0.0522
    L = 0.4
    lambda_0 = 0.025
    lambda_1 = 0.0 # Critical for benchmarking
    T = 10
    
    model = VasicekRFV(r0, kappa, sigma, theta, lambda_0, lambda_1, L)
    
    print(f"--- Vasicek RFV Model Verification ---")
    print(f"Parameters: r0={r0}, kappa={kappa}, sigma={sigma}, theta={theta}")
    print(f"            L={L}, lambda_0={lambda_0}, lambda_1={lambda_1}")
    
    price = model.price(0, T)
    dur = model.duration(0, T)
    
    print(f"\nPrice(0, {T}): {price:.6f}")
    print(f"Duration(0, {T}): {dur:.6f}")
    
    expected_dur = 5.179
    print(f"Expected Duration: ~{expected_dur}")
    
    if abs(dur - expected_dur) < 0.01:
        print("\n[SUCCESS] Duration matches analytical benchmark.")
    else:
        print("\n[WARNING] Duration mismatch.")
