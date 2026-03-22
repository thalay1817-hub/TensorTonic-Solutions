def value_iteration_step(values, transitions, rewards, gamma):
    n_states = len(values)
    n_actions = len(transitions[0])
    
    new_values = []
    
    for s in range(n_states):
        best = float('-inf')  # max over actions
        
        for a in range(n_actions):
            q = rewards[s][a]
            
            # Expected future value
            future = 0.0
            for s_next in range(n_states):
                future += transitions[s][a][s_next] * values[s_next]
            
            q += gamma * future
            
            if q > best:
                best = q
        
        new_values.append(float(best))
    
    return new_values