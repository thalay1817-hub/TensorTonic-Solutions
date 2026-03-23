import numpy as np

def calculate_eigenvalues(matrix):
    try:
        A = np.asarray(matrix, dtype=float)
        
        # Check if 2D and square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return None
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(A)
        
        # Sort by real part, then imaginary part
        idx = np.lexsort((eigvals.imag, eigvals.real))
        eigvals = eigvals[idx]
        
        return eigvals  # return NumPy array
    
    except Exception:
        return None