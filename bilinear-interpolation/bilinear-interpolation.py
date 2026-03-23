import math

def bilinear_resize(image, new_h, new_w):
    H = len(image)
    W = len(image[0])
    
    # Initialize output
    output = [[0.0 for _ in range(new_w)] for _ in range(new_h)]
    
    for i in range(new_h):
        for j in range(new_w):
            
            # Handle edge case when new_h or new_w = 1
            if new_h == 1:
                src_y = 0
            else:
                src_y = i * (H - 1) / (new_h - 1)
                
            if new_w == 1:
                src_x = 0
            else:
                src_x = j * (W - 1) / (new_w - 1)
            
            # Integer parts
            y0 = int(math.floor(src_y))
            x0 = int(math.floor(src_x))
            
            # Fractional parts
            dy = src_y - y0
            dx = src_x - x0
            
            # Neighbor indices (boundary safe)
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # Bilinear interpolation
            val = (
                image[y0][x0] * (1 - dy) * (1 - dx) +
                image[y1][x0] * dy * (1 - dx) +
                image[y0][x1] * (1 - dy) * dx +
                image[y1][x1] * dy * dx
            )
            
            output[i][j] = val
    
    return output