import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    V = np.array(V, dtype=float)
    
    # Copy to avoid modifying original
    V_new = V.copy()
    
    # TD error
    delta = r + gamma * V[s_next] - V[s]
    
    # Update value
    V_new[s] = V[s] + alpha * delta
    
    return V_new