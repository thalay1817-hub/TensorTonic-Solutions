import math

def rotate_image(image, angle_degrees):
    H = len(image)
    W = len(image[0])
    
    # Output image filled with 0
    output = [[0 for _ in range(W)] for _ in range(H)]
    
    # Convert angle to radians
    theta = math.radians(angle_degrees)
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Image center
    cy = (H - 1) / 2
    cx = (W - 1) / 2
    
    for i in range(H):
        for j in range(W):
            dy = i - cy
            dx = j - cx
            
            # Inverse rotation
            src_y = cy + dy * cos_t + dx * sin_t
            src_x = cx - dy * sin_t + dx * cos_t
            
            sy = round(src_y)
            sx = round(src_x)
            
            # Check bounds
            if 0 <= sy < H and 0 <= sx < W:
                output[i][j] = image[sy][sx]
            else:
                output[i][j] = 0
    
    return output