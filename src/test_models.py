from models import Vasicek
import numpy as np

def test_vasicek():
    v = Vasicek()
    r_t = 0.04
    t = 0
    T = 10
    price = v.zero_bond_price(t, T, r_t)
    print(f"Vasicek Zero Bond Price (t={t}, T={T}, r={r_t}): {price}")
    
    hazard = v.hazard_rate(r_t)
    print(f"Hazard Rate at r={r_t}: {hazard}")

if __name__ == "__main__":
    test_vasicek()
