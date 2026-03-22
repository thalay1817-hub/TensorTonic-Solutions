import math

def sobel_edges(image):
    H = len(image)
    W = len(image[0])
    
    # Sobel kernels
    Kx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
    Ky = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    # Zero padding
    padded = [[0]*(W+2) for _ in range(H+2)]
    for i in range(H):
        for j in range(W):
            padded[i+1][j+1] = image[i][j]
    
    # Output
    output = [[0.0 for _ in range(W)] for _ in range(H)]
    
    # Convolution
    for i in range(H):
        for j in range(W):
            Gx = 0
            Gy = 0
            
            for a in range(3):
                for b in range(3):
                    val = padded[i+a][j+b]
                    Gx += val * Kx[a][b]
                    Gy += val * Ky[a][b]
            
            # Gradient magnitude
            output[i][j] = math.sqrt(Gx**2 + Gy**2)
    
    return output