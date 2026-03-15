def linear_interpolation(values):
    result = values.copy()
    n = len(result)
    
    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            
            # find next known value
            right = i
            while right < n and result[right] is None:
                right += 1
            
            v_left = result[left]
            v_right = result[right]
            
            # fill missing values
            for j in range(left + 1, right):
                result[j] = v_left + (j - left) / (right - left) * (v_right - v_left)
            
            i = right
        else:
            i += 1
    
    return result