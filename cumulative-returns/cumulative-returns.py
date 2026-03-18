def cumulative_returns(returns):
    result = []
    cum = 1.0  # initial wealth factor
    
    for r in returns:
        cum *= (1 + r)          # compound
        result.append(cum - 1)  # cumulative return
    
    return result


# Example usage
returns1 = [0.1, 0.05, -0.02]
print(cumulative_returns(returns1))

returns2 = [-0.5, 1.0]
print(cumulative_returns(returns2))