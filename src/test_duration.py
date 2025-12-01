from models import Vasicek

def test_duration():
    # Case: Lambda_1 = 0
    v = Vasicek(lambda_1=0.0)
    t = 0
    T = 10
    
    # Calculate duration
    dur = v.duration(t, T)
    print(f"Duration (Lambda_1=0, t={t}, T={T}): {dur}")
    
    # Verify against manual calculation for Vasicek B(t, T)
    # B = (1 - exp(-kappa*tau)) / kappa
    kappa = v.kappa
    tau = T - t
    expected_dur = (1 - 2.718281828**(-kappa * tau)) / kappa
    print(f"Expected Duration: {expected_dur}")

if __name__ == "__main__":
    test_duration()
