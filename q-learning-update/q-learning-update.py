import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    Q = np.array(Q, dtype=float)
    
    # Copy Q-table
    Q_new = Q.copy()
    
    # TD target
    target = r + gamma * np.max(Q[s_next])
    
    # Update rule
    Q_new[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    
    return Q_new.tolist()