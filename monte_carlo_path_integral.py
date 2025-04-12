import numpy as np

# Parameters
num_paths = 1000  # Number of sampled paths
dt = 0.1  # Time step
T = 1.0  # Total evolution time
k = 1.0  # Potential strength
hbar = 1.0  # Normalized hbar

# Initial and target states (simplified as scalars for toy model)
phi_i = 0.0  # Represents |00>
phi_f = 1.0  # Represents Bell state (simplified)

# Action calculation for a path
def action(phi, dt, T, k):
    S = 0.0
    for t in range(1, len(phi)):
        dphi_dt = (phi[t] - phi[t-1]) / dt  # Derivative
        kinetic = 0.5 * dphi_dt**2
        potential = k * (phi[t] - phi_f)**2
        S += (kinetic - potential) * dt
    return S

# Monte Carlo sampling of paths
np.random.seed(42)
Z = 0.0
N = int(T / dt)
for _ in range(num_paths):
    # Generate random path from phi_i to phi_f
    phi = np.linspace(phi_i, phi_f, N) + np.random.normal(0, 0.1, N)
    phi[0], phi[-1] = phi_i, phi_f  # Fix endpoints
    S = action(phi, dt, T, k)
    Z += np.exp(1j * S / hbar)

# Transition probability
P = np.abs(Z / num_paths)**2
print(f"Transition probability P(|00> -> Bell): {P:.4f}")