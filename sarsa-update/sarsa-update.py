def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    # Deep copy of Q-table
    new_q = [row[:] for row in q_table]

    # Compute TD error using original Q-table
    td = reward + gamma * q_table[next_state][next_action] - q_table[state][action]

    # Update Q-value
    new_q[state][action] += alpha * td

    return new_q


# Example 1
q_table = [[0, 0], [0, 0]]
print(sarsa_update(q_table, 0, 1, 1.0, 1, 0, 0.1, 0.9))

# Example 2
q_table = [[1, 2], [3, 4]]
print(sarsa_update(q_table, 0, 0, 5.0, 1, 1, 0.5, 0.9))