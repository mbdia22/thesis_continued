from models import Vasicek
from pricing_rmv import price_bond_rmv
from pricing_rfv import price_bond_rfv
import numpy as np

def test_pricing():
    # 1. Test Risk-Free Case (L=0, lambda=0)
    # In this case, RMV and RFV should match Vasicek Zero Bond Price
    print("--- Testing Risk-Free Case (L=0, lambda=0) ---")
    v_riskfree = Vasicek(lambda_0=0, lambda_1=0, L=0)
    T = 5
    r0 = 0.04
    
    # Analytical Price
    price_analytical = v_riskfree.zero_bond_price(0, T, r0)
    print(f"Analytical Price: {price_analytical:.6f}")
    
    # MC RMV
    price_rmv = price_bond_rmv(v_riskfree, T, n_sims=5000)
    print(f"MC RMV Price:     {price_rmv:.6f}")
    
    # MC RFV
    price_rfv = price_bond_rfv(v_riskfree, T, n_sims=5000)
    print(f"MC RFV Price:     {price_rfv:.6f}")
    
    # Check if close
    assert np.abs(price_rmv - price_analytical) < 1e-3, "RMV failed risk-free test"
    assert np.abs(price_rfv - price_analytical) < 1e-3, "RFV failed risk-free test"
    print("Risk-Free tests passed!")
    
    # 2. Test Defaultable Case
    print("\n--- Testing Defaultable Case (Table 5.1 Params) ---")
    v_default = Vasicek() # Uses default params from Table 5.1
    
    price_rmv_def = price_bond_rmv(v_default, T, n_sims=5000)
    print(f"MC RMV Price: {price_rmv_def:.6f}")
    
    price_rfv_def = price_bond_rfv(v_default, T, n_sims=5000)
    print(f"MC RFV Price: {price_rfv_def:.6f}")
    
    print("Defaultable prices calculated successfully.")

if __name__ == "__main__":
    test_pricing()
